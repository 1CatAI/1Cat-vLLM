# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Canonical GGUF affine weights in the shared mixed-precision lifecycle."""

import weakref
from dataclasses import dataclass

import torch

from vllm._sm70.policy import NativeBindings
from vllm.config.sm70_native import capture_linear_native_config
from vllm.model_executor.kernels.gguf import GGUFDecoderFamily, GGUFOperatorCapability
from vllm.model_executor.kernels.linear.sm70_provider import (
    flatten_linear_input,
    restore_linear_output,
)
from vllm.model_executor.layers.quantization.gguf_transcode import (
    AFFINE_BITPLANE_TYPES,
    AFFINE_GROUP32_TYPES,
    AFFINE_U2_TYPES,
)
from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.model_executor.layers.quantization.utils.sm70_layer_workspaces import (
    workspace_pool,
)
from vllm.scalar_type import scalar_types
from vllm.transformers_utils.gguf_tensor_reader import quant_type_name

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig

# Calibrated physical layouts, (bits, group, K, N) -> minimum prefill M.
# The values come from the matched full/TP4 projection sweeps in
# docs/design/gguf_turbomind_prefill.md. Unmeasured layouts retain fused MMA.
_AFFINE_BLAS_MIN_M = {
    (3, 16, 5120, 17408): 512,
    (3, 16, 5120, 4352): 512,
    (4, 32, 5120, 4352): 512,
    (4, 32, 5120, 2560): 2048,
    (4, 32, 5120, 4096): 512,
    (4, 32, 5120, 8704): 512,
    (5, 32, 5120, 6144): 2048,
    (5, 32, 5120, 1536): 2048,
    (5, 32, 5120, 4096): 512,
    (5, 32, 5120, 2560): 2048,
    (6, 16, 6144, 5120): 2048,
    (6, 16, 1536, 5120): 2048,
    # Flash-Next TP4 Q5_K/Q6_K projections; dequantization is included in
    # the paired timings in benchmarks/flashnext_prefill/OPTIMIZATION_20261014.md.
    (5, 32, 2560, 2560): 2048,
    (6, 16, 2560, 2560): 2048,
    (6, 16, 2560, 1536): 2048,
    (6, 16, 1536, 2560): 2048,
    # Same-codec GDN QKV/gate and QSA Q/K/V segments after TP4 merging.
    (5, 32, 2560, 4096): 2048,
    (6, 16, 2560, 4096): 2048,
    (5, 32, 2560, 3328): 2048,
    (5, 32, 2560, 3072): 4096,
    # Remaining Flash-Next layouts, measured with segment restoration and
    # the bounded native NN algorithm window (query rows4096--8192).
    (4, 32, 2560, 1536): 4096,
    (4, 32, 1536, 2560): 4096,
    (4, 32, 2560, 3072): 4096,
    (4, 32, 2560, 4096): 4096,
    (5, 32, 2560, 1536): 4096,
    (5, 32, 1536, 2560): 4096,
    # Q8 output projections: include restoration and DQ in paired timings.
    (8, 32, 1536, 2560): 4096,
    (8, 32, 160, 2560): 6144,
}
_affine_blas_workspaces: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
_affine_blas_views: dict = {}


def _get_affine_blas_workspace(
    weight: torch.Tensor, elements: int | None = None
) -> torch.Tensor | None:
    """The engine's shared FP16 scratch of the admitted large-M BLAS path.

    Prepare-time callers pass their projection's k * n: the scratch grows to
    the largest projection actually admitted on this device, not the largest
    calibrated shape of any model (5120 x 17408 = 170 MiB). Views bound with
    `_bind_affine_blas_workspace` follow it when it grows. Runtime callers
    (elements=None) only borrow the scratch their prepared shape reserved.
    """
    key = (weight.device, torch.float16)
    pool = workspace_pool("gguf_affine_blas", _affine_blas_workspaces)
    workspace = pool.get(key)
    if elements is None or (workspace is not None and workspace.numel() >= elements):
        return workspace
    try:
        grown = torch.empty(elements, dtype=torch.float16, device=weight.device)
    except torch.OutOfMemoryError:
        return None
    pool[key] = grown
    views = workspace_pool("gguf_affine_blas_views", _affine_blas_views)
    for ref, k, n in views.get(key, ()):
        layer = ref()
        if layer is not None:
            layer.gguf_tm_blas_workspace = grown[: k * n].view(k, n)
    return grown


def _bind_affine_blas_workspace(
    layer: torch.nn.Module, weight: torch.Tensor, k: int, n: int
) -> bool:
    """Reserve k * n scratch elements and give `layer` its (k, n) view."""
    workspace = _get_affine_blas_workspace(weight, k * n)
    if workspace is None:
        return False
    key = (weight.device, torch.float16)
    views = workspace_pool("gguf_affine_blas_views", _affine_blas_views)
    views.setdefault(key, []).append((weakref.ref(layer), k, n))
    layer.gguf_tm_blas_workspace = workspace[: k * n].view(k, n)
    return True


@dataclass
class Sm70GgufAffineConfig(MPLinearLayerConfig):
    source_type: int = 0
    enabled: bool = True


class TurboMindGgufAffineKernel(MPLinearKernel):
    """Only native FP16 activation/MMA compute; no activation quantization."""

    config: Sm70GgufAffineConfig

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    @classmethod
    def can_implement(cls, config: MPLinearLayerConfig) -> tuple[bool, str | None]:
        if not isinstance(config, Sm70GgufAffineConfig):
            return False, "requires_canonical_gguf_affine_storage"
        if not config.enabled:
            return False, "disabled_by_kernel_config"
        if (
            config.source_type
            not in AFFINE_GROUP32_TYPES | AFFINE_U2_TYPES | AFFINE_BITPLANE_TYPES
        ):
            return False, "source_format_codec_unavailable"
        if config.act_type != torch.float16:
            return False, "requires_fp16_activations"
        expected_type = (
            {
                6: scalar_types.uint5,
                7: scalar_types.uint5,
                11: scalar_types.uint3,
                13: scalar_types.uint5,
                14: scalar_types.uint6,
            }[config.source_type]
            if config.source_type in AFFINE_BITPLANE_TYPES
            else (
                scalar_types.uint2
                if config.source_type in AFFINE_U2_TYPES
                else (
                    scalar_types.uint8
                    if config.source_type == 8
                    else scalar_types.uint4
                )
            )
        )
        if config.weight_type != expected_type:
            return False, "canonical_integer_width_unavailable"
        expected_group = 16 if config.source_type in (10, 11, 14) else 32
        if config.group_size != expected_group or config.has_g_idx:
            return False, "canonical_group_or_activation_order_not_supported"
        k, n = config.partition_weight_shape
        if k <= 0 or n <= 0 or k % config.group_size or n % 32:
            return False, "local_shape_cuts_canonical_group_or_output_pack"
        for name in ("gguf_affine_sm70_prepare", "gguf_affine_gemm_sm70_out"):
            if not hasattr(torch.ops._C, name):
                return False, f"operator_missing:{name}"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self.native_ops = NativeBindings(capture_linear_native_config("gguf").values)
        if getattr(layer, "_gguf_tm_affine_prepared", False):
            return
        codes, scales, mins, _ = self._get_weight_params(layer)
        assert mins is not None
        bits = self.config.weight_type.size_bits
        weight, stats, meta = torch.ops._C.gguf_affine_sm70_prepare(
            codes, scales, mins, bits, self.config.group_size
        )
        replace_parameter(layer, self.w_q_name, weight)
        replace_parameter(layer, self.w_s_name, stats)
        if self.w_zp_name is not None:
            layer.register_parameter(self.w_zp_name, None)
        layer.gguf_tm_k_ld, layer.gguf_tm_q_ld = meta.tolist()
        self.bits = bits
        self.capability = GGUFOperatorCapability(
            GGUFDecoderFamily.AFFINE,
            quant_type_name(self.config.source_type),
            "gguf_affine_gemm_sm70_out",
            True,
        )
        k, n = self.config.partition_weight_shape
        minimum_m = _AFFINE_BLAS_MIN_M.get((bits, self.config.group_size, k, n))
        reason = None
        if not hasattr(torch.ops._C, "gguf_affine_blas_sm70_out"):
            reason = "operator_missing:gguf_affine_blas_sm70_out"
        elif minimum_m is None:
            reason = "local_shape_has_no_prefill_calibration"
        elif not _bind_affine_blas_workspace(layer, weight, k, n):
            reason = "prefill_workspace_allocation_failed"
        self.prefill_capability = GGUFOperatorCapability(
            GGUFDecoderFamily.AFFINE,
            quant_type_name(self.config.source_type),
            "gguf_affine_blas_sm70_out",
            True,
            min_m=minimum_m or 512,
            reason=reason,
        )
        self.operator_capabilities = (self.capability, self.prefill_capability)
        layer._gguf_tm_affine_prepared = True

    def apply_weights(self, layer, x, bias=None):
        if not self.capability.supports_m(x.numel() // x.shape[-1]):
            raise ValueError("M outside the GGUF affine operator capability")
        n = self.config.partition_weight_shape[1]
        rows = flatten_linear_input(x).contiguous()
        output = torch.empty((rows.shape[0], n), dtype=x.dtype, device=x.device)
        if (
            self.prefill_capability.reason is None
            and self.prefill_capability.supports_m(rows.shape[0])
        ):
            torch.ops._C.gguf_affine_blas_sm70_out(
                output,
                rows,
                getattr(layer, self.w_q_name),
                getattr(layer, self.w_s_name),
                self.bits,
                layer.gguf_tm_blas_workspace,
                self.config.group_size,
            )
        else:
            self.native_ops.gguf_affine_gemm_sm70_out(
                output,
                rows,
                getattr(layer, self.w_q_name),
                getattr(layer, self.w_s_name),
                self.bits,
                layer.gguf_tm_k_ld,
                layer.gguf_tm_q_ld,
                self.config.group_size,
            )
        if bias is not None:
            output.add_(bias)
        return restore_linear_output(output, x, n)
