# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepare independent GGUF projections with the canonical kernel lifecycle."""

from collections.abc import Callable
from dataclasses import asdict, fields, replace

import numpy as np
import torch
from torch.nn import Module, Parameter

from vllm.model_executor.kernels.gguf import (
    GGUFOperatorCapability,
    dense_fp16_cache_capabilities,
)
from vllm.model_executor.kernels.linear import (
    Sm70GgufAffineConfig,
    Sm70GgufLatticeConfig,
    Sm70GgufLut4Config,
    choose_mp_linear_kernel,
)
from vllm.model_executor.layers.quantization.gguf_lattice_transcode import (
    LATTICE_TYPES,
    LatticeGGUFProjection,
    transcode_lattice,
)
from vllm.model_executor.layers.quantization.gguf_lut_transcode import (
    LUT4_TYPES,
    Lut4GGUFProjection,
    transcode_lut4,
)
from vllm.model_executor.layers.quantization.gguf_native import pad_weight_tail
from vllm.model_executor.layers.quantization.gguf_transcode import (
    AFFINE_BITPLANE_TYPES,
    AFFINE_GROUP32_TYPES,
    AFFINE_U2_TYPES,
    AffineGGUFProjection,
    transcode_affine,
)
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types
from vllm.transformers_utils.gguf_tensor_reader import quant_size, quant_type_name

_AFFINE_TYPES = AFFINE_GROUP32_TYPES | AFFINE_U2_TYPES | AFFINE_BITPLANE_TYPES


class GGUFPreparedProjection(Module):
    """One mixed projection; canonical preparation never changes its row order."""

    def __init__(self, weight, source_type, act_dtype, enabled, prefill_min_m):
        super().__init__()
        self.source_type = int(source_type)
        self.enabled = enabled
        self.prefill_min_m = prefill_min_m
        self.kernel = None
        self.logical_output_size = weight.shape[0]
        self.output_padding = 0
        self.cache_capabilities: tuple[GGUFOperatorCapability, ...] = ()
        self.register_parameter("fp16_cache", None)
        self.rejection_reason = self._prepare(weight, act_dtype)
        if self.kernel is None:
            self.register_parameter(
                "weight",
                Parameter(
                    pad_weight_tail(weight, self.source_type) if enabled else weight,
                    requires_grad=False,
                ),
            )

    def _prepare(self, weight, act_dtype):
        if not self.enabled:
            return "disabled_by_kernel_config"
        if weight.device.type != "cuda":
            return "device_not_cuda"
        if current_platform.get_device_capability(weight.device.index) != (7, 0):
            return "requires_sm70"
        if act_dtype != torch.float16:
            return "requires_fp16_activations"
        config_class: type[
            Sm70GgufAffineConfig | Sm70GgufLut4Config | Sm70GgufLatticeConfig
        ]
        transcode: Callable[
            [np.ndarray, int],
            AffineGGUFProjection | Lut4GGUFProjection | LatticeGGUFProjection,
        ]
        if self.source_type in _AFFINE_TYPES:
            config_class, transcode = Sm70GgufAffineConfig, transcode_affine
        elif self.source_type in LUT4_TYPES:
            config_class, transcode = Sm70GgufLut4Config, transcode_lut4
        elif self.source_type in LATTICE_TYPES:
            config_class, transcode = Sm70GgufLatticeConfig, transcode_lattice
        else:
            return "source_format_codec_unavailable"
        block, size = quant_size(self.source_type)
        if weight.ndim != 2 or weight.shape[1] % size:
            return "incomplete_source_projection"
        n, k = weight.shape[0], weight.shape[1] // size * block
        try:
            canonical = transcode(weight.detach().cpu().numpy(), self.source_type)
        except ValueError as error:
            return f"canonical_transcode_rejected:{error}"
        padding = -n % 32
        if padding:
            canonical = replace(
                canonical,
                **{
                    field.name: np.pad(
                        getattr(canonical, field.name), ((0, padding), (0, 0))
                    )
                    for field in fields(canonical)
                    if isinstance(getattr(canonical, field.name), np.ndarray)
                },
            )
        config = config_class(
            full_weight_shape=(k, n),
            partition_weight_shape=(k, n + padding),
            weight_type=getattr(scalar_types, f"uint{canonical.bits}"),
            act_type=act_dtype,
            group_size=canonical.group_size,
            zero_points=config_class is Sm70GgufAffineConfig,
            has_g_idx=False,
            source_type=self.source_type,
            enabled=self.enabled,
        )
        try:
            kernel_class = choose_mp_linear_kernel(config, compute_capability=70)
        except ValueError as error:
            return f"canonical_kernel_rejected:{error}"
        if isinstance(canonical, LatticeGGUFProjection):
            codes, stats = canonical.mma884_storage()
            stats = stats.view({2: np.int16, 4: np.int32, 8: np.int64}[stats.itemsize])
        else:
            codes, stats = canonical.codes, canonical.scales
        self.register_parameter(
            "codes", Parameter(torch.from_numpy(codes).to(weight.device), False)
        )
        self.register_parameter(
            "stats", Parameter(torch.from_numpy(stats).to(weight.device), False)
        )
        if config.zero_points:
            assert isinstance(canonical, AffineGGUFProjection)
            self.register_parameter(
                "mins",
                Parameter(torch.from_numpy(canonical.mins).to(weight.device), False),
            )
        self.kernel = kernel_class(
            config, "codes", "stats", "mins" if config.zero_points else None
        )
        self.kernel.process_weights_after_loading(self)
        self.output_padding = padding
        self.cache_capabilities = dense_fp16_cache_capabilities(
            self.source_type, k, n, act_dtype, self.enabled
        )
        if any(c.reason is None for c in self.cache_capabilities):
            cached = canonical.dequantize()[:n].astype(np.float16)
            self.register_parameter(
                "fp16_cache",
                Parameter(torch.from_numpy(cached).to(weight.device), False),
            )
        return None

    def admission(self):
        result = {
            "source_type": quant_type_name(self.source_type),
            "reason": self.rejection_reason,
        }
        if self.kernel is not None:
            result["kernel"] = type(self.kernel).__name__
            result["local_weight_shape"] = list(
                self.kernel.config.partition_weight_shape
            )
            result["logical_output_size"] = self.logical_output_size
            result["zero_padded_output_rows"] = self.output_padding
            result["operators"] = [
                asdict(c)
                for c in (
                    *getattr(
                        self.kernel,
                        "operator_capabilities",
                        (self.kernel.capability,),
                    ),
                    *self.cache_capabilities,
                )
            ]
        return result

    def forward(self, x):
        if self.kernel is not None:
            rows = x.numel() // x.shape[-1]
            if self.fp16_cache is not None and any(
                c.reason is None and c.supports_m(rows) for c in self.cache_capabilities
            ):
                output = torch.mm(x.reshape(-1, x.shape[-1]), self.fp16_cache.T)
                return output.reshape(*x.shape[:-1], self.logical_output_size)
            output = self.kernel.apply_weights(self, x)
            return (
                output[..., : self.logical_output_size]
                if self.output_padding
                else output
            )
        # Imported lazily because the GGUF method owns fallback dispatch.
        from vllm.model_executor.layers.quantization.gguf import fused_mul_mat_gguf

        return fused_mul_mat_gguf(
            x, self.weight, self.source_type, self.enabled, self.prefill_min_m
        )
