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
    decoder_family,
    dense_fp16_cache_capabilities,
)
from vllm.model_executor.kernels.linear import (
    Sm70GgufAffineConfig,
    Sm70GgufLatticeConfig,
    Sm70GgufLut4Config,
    choose_mp_linear_kernel,
)
from vllm.model_executor.kernels.linear.mixed_precision.sm70_gguf import (
    _get_affine_blas_workspace,
)
from vllm.model_executor.layers.quantization.gguf_lattice_transcode import (
    LATTICE_TYPES,
    LatticeGGUFProjection,
    transcode_lattice,
)
from vllm.model_executor.layers.quantization.gguf_layout import GGUFHeadTilingLayout
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
from vllm.utils.torch_utils import direct_register_custom_op

_AFFINE_TYPES = AFFINE_GROUP32_TYPES | AFFINE_U2_TYPES | AFFINE_BITPLANE_TYPES


def _supports_band(rows: int, bands: list[int]) -> bool:
    return any(
        rows >= bands[i] and (bands[i + 1] < 0 or rows <= bands[i + 1])
        for i in range(0, len(bands), 2)
    )


def _prepared_gguf_projection(
    x: torch.Tensor,
    codes: torch.Tensor,
    stats: torch.Tensor,
    cache: torch.Tensor | None,
    family: int,
    decoder: int,
    group_size: int,
    k_ld: int,
    q_ld: int,
    output_size: int,
    logical_size: int,
    cache_bands: list[int],
    blas_bands: list[int],
) -> torch.Tensor:
    # Keep M-dependent policy behind an opaque op: vLLM's range compilation
    # drops Dynamo guards and would otherwise freeze the prefill branch.
    rows = x.reshape(-1, x.shape[-1]).contiguous()
    if cache is not None and _supports_band(rows.shape[0], cache_bands):
        return torch.mm(rows, cache.T).reshape(*x.shape[:-1], logical_size)
    output = torch.empty((rows.shape[0], output_size), dtype=x.dtype, device=x.device)
    if _supports_band(rows.shape[0], blas_bands):
        # The already prepared shared scratch is internal to this op. Passing
        # its aliased views through Dynamo would materialize full-size clones
        # and copybacks even when decode never touches the scratch.
        workspace = _get_affine_blas_workspace(codes)
        if workspace is None:
            raise RuntimeError("Admitted GGUF BLAS workspace is unavailable")
        scratch = workspace[: rows.shape[1] * output_size].view(
            rows.shape[1], output_size
        )
        if family == 0:
            torch.ops._C.gguf_affine_blas_sm70_out(
                output, rows, codes, stats, decoder, scratch, group_size
            )
        else:
            torch.ops._C.gguf_lattice_blas_sm70_out(
                output, rows, codes, stats, decoder, scratch, group_size
            )
    elif family == 0:
        torch.ops._C.gguf_affine_gemm_sm70_out(
            output, rows, codes, stats, decoder, k_ld, q_ld, group_size
        )
    elif family == 1:
        torch.ops._C.gguf_lut4_gemm_sm70_out(
            output, rows, codes, stats, decoder, k_ld, q_ld, group_size
        )
    else:
        torch.ops._C.gguf_lattice_gemm_sm70_out(
            output, rows, codes, stats, decoder, k_ld, q_ld, group_size
        )
    # Return a separate contiguous result so the fake and real strides agree
    # even when the prepared output pack contains padding.
    return output[:, :logical_size].contiguous().reshape(*x.shape[:-1], logical_size)


def _prepared_gguf_projection_fake(
    x: torch.Tensor,
    codes: torch.Tensor,
    stats: torch.Tensor,
    cache: torch.Tensor | None,
    family: int,
    decoder: int,
    group_size: int,
    k_ld: int,
    q_ld: int,
    output_size: int,
    logical_size: int,
    cache_bands: list[int],
    blas_bands: list[int],
) -> torch.Tensor:
    return torch.empty((*x.shape[:-1], logical_size), dtype=x.dtype, device=x.device)


direct_register_custom_op(
    op_name="prepared_gguf_projection",
    op_func=_prepared_gguf_projection,
    fake_impl=_prepared_gguf_projection_fake,
)


def _prepared_gguf_mixed_projection(
    x: torch.Tensor,
    codes: list[torch.Tensor],
    stats: list[torch.Tensor],
    caches: list[torch.Tensor | None],
    descriptors: list[int],
    cache_bands: list[int],
    blas_bands: list[int],
) -> torch.Tensor:
    rows = x.reshape(-1, x.shape[-1]).contiguous()
    specs = [descriptors[i : i + 9] for i in range(0, len(descriptors), 9)]
    cache_offset = blas_offset = 0
    policies = []
    for spec in specs:
        cache_count, blas_count = spec[-2:]
        policies.append(
            (
                cache_bands[cache_offset : cache_offset + cache_count],
                blas_bands[blas_offset : blas_offset + blas_count],
            )
        )
        cache_offset += cache_count
        blas_offset += blas_count
    # Retain the calibrated per-projection policy outside measured target
    # verification sizes. This decision must use actual M inside the op.
    direct = rows.shape[0] in (5, 20) and all(
        not _supports_band(rows.shape[0], cb) and not _supports_band(rows.shape[0], bb)
        for cb, bb in policies
    )
    if not direct:
        outputs = [
            _prepared_gguf_projection(
                x,
                c,
                s,
                cache,
                spec[0],
                spec[1],
                spec[2],
                spec[3],
                spec[4],
                spec[5],
                spec[6],
                cb,
                bb,
            )
            for c, s, cache, spec, (cb, bb) in zip(
                codes, stats, caches, specs, policies
            )
        ]
        return torch.cat(outputs, dim=-1)
    total = sum(spec[6] for spec in specs)
    output = torch.empty((rows.shape[0], total), dtype=x.dtype, device=x.device)
    offset = 0
    for c, s, spec in zip(codes, stats, specs):
        family, decoder, group, k_ld, q_ld, output_size, logical_size = spec[:7]
        destination = output[:, offset : offset + logical_size]
        if family == 0:
            torch.ops._C.gguf_affine_gemm_sm70_out(
                destination, rows, c, s, decoder, k_ld, q_ld, group
            )
        elif family == 1:
            torch.ops._C.gguf_lut4_gemm_sm70_out(
                destination, rows, c, s, decoder, k_ld, q_ld, group
            )
        else:
            torch.ops._C.gguf_lattice_gemm_sm70_out(
                destination, rows, c, s, decoder, k_ld, q_ld, group
            )
        offset += logical_size
    return output.reshape(*x.shape[:-1], total)


def _prepared_gguf_mixed_projection_fake(
    x: torch.Tensor,
    codes: list[torch.Tensor],
    stats: list[torch.Tensor],
    caches: list[torch.Tensor | None],
    descriptors: list[int],
    cache_bands: list[int],
    blas_bands: list[int],
) -> torch.Tensor:
    total = sum(descriptors[6::9])
    return torch.empty((*x.shape[:-1], total), dtype=x.dtype, device=x.device)


direct_register_custom_op(
    op_name="prepared_gguf_mixed_projection",
    op_func=_prepared_gguf_mixed_projection,
    fake_impl=_prepared_gguf_mixed_projection_fake,
)


def mixed_projection_capabilities(projections):
    """Declare output-view eligibility independently of quantization family."""
    if len(projections) < 2 or any(p.kernel is None for p in projections):
        # An imported fallback format may not have a canonical family at all.
        # Its existing projection admission carries the rejection reason.
        return ()
    capabilities = []
    for projection in projections:
        reason = projection.rejection_reason
        if projection.kernel is None:
            reason = reason or "projection_not_prepared"
        elif projection.output_padding:
            reason = "mixed_output_has_padded_projection"
        elif projection.logical_output_size % 32:
            reason = "mixed_output_cuts_output_pack"
        for m in (5, 20):
            capabilities.append(
                GGUFOperatorCapability(
                    decoder_family(projection.source_type),
                    quant_type_name(projection.source_type),
                    "prepared_gguf_mixed_projection",
                    True,
                    min_m=m,
                    max_m=m,
                    reason=reason,
                )
            )
    return tuple(capabilities)


def apply_prepared_gguf_projections(x, projections):
    capabilities = mixed_projection_capabilities(projections)
    if not capabilities or any(c.reason is not None for c in capabilities):
        outputs = [projection(x) for projection in projections]
        return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=-1)
    codes: list[torch.Tensor] = []
    stats: list[torch.Tensor] = []
    caches: list[torch.Tensor | None] = []
    descriptors: list[int] = []
    cache_bands: list[int] = []
    blas_bands: list[int] = []
    for projection in projections:
        kernel = projection.kernel
        config = kernel.config
        if isinstance(config, Sm70GgufAffineConfig):
            family, decoder = 0, kernel.bits
        elif isinstance(config, Sm70GgufLut4Config):
            family, decoder = 1, kernel.lut_id
        else:
            family, decoder = 2, kernel.source_type
        cb = _admitted_bands(projection.cache_capabilities)
        bb = _admitted_bands(
            tuple(
                c
                for c in getattr(kernel, "operator_capabilities", ())
                if "blas" in c.operator
            )
        )
        codes.append(projection.codes)
        stats.append(projection.stats)
        caches.append(projection.fp16_cache)
        descriptors.extend(
            (
                family,
                decoder,
                config.group_size,
                projection.gguf_tm_k_ld,
                projection.gguf_tm_q_ld,
                config.partition_weight_shape[1],
                projection.logical_output_size,
                len(cb),
                len(bb),
            )
        )
        cache_bands.extend(cb)
        blas_bands.extend(bb)
    return torch.ops.vllm.prepared_gguf_mixed_projection(
        x, codes, stats, caches, descriptors, cache_bands, blas_bands
    )


def _admitted_bands(capabilities):
    return [
        bound
        for capability in capabilities
        if capability.reason is None
        for bound in (capability.min_m, capability.max_m or -1)
    ]


def prepare_gguf_projections(
    sources, act_dtype, enabled, prefill_min_m, input_layout=None
):
    """Coalesce adjacent compatible shards without changing projection order."""
    groups: list[tuple[list[torch.Tensor], int]] = []
    for weight, source_type in sources:
        if (
            groups
            and source_type in _AFFINE_TYPES | LUT4_TYPES | LATTICE_TYPES
            and source_type == groups[-1][1]
            and weight.shape[1:] == groups[-1][0][0].shape[1:]
            and weight.dtype == groups[-1][0][0].dtype
            and weight.device == groups[-1][0][0].device
        ):
            groups[-1][0].append(weight)
        else:
            groups.append(([weight], source_type))
    projections = []
    for weights, source_type in groups:
        projection = GGUFPreparedProjection(
            weights[0] if len(weights) == 1 else torch.cat(weights, dim=0),
            source_type,
            act_dtype,
            enabled,
            prefill_min_m,
            input_layout=input_layout,
        )
        projection.source_output_sizes = tuple(weight.shape[0] for weight in weights)
        projections.append(projection)
    if input_layout is not None and not all(
        projection.input_layout_restored for projection in projections
    ):
        # Keep every shard in the same input order when one cannot be restored.
        reasons = [
            p.rejection_reason for p in projections if not p.input_layout_restored
        ]
        fallback = prepare_gguf_projections(sources, act_dtype, enabled, prefill_min_m)
        for projection in fallback:
            projection.input_layout_rejection_reasons = reasons
        return fallback
    return projections


class GGUFPreparedProjection(Module):
    """One mixed projection; canonical preparation never changes its row order."""

    def __init__(
        self, weight, source_type, act_dtype, enabled, prefill_min_m, input_layout=None
    ):
        super().__init__()
        self.source_type = int(source_type)
        self.enabled = enabled
        self.prefill_min_m = prefill_min_m
        self.kernel = None
        self.input_layout = input_layout
        self.input_layout_restored = False
        self.input_layout_rejection_reasons = []
        self.logical_output_size = weight.shape[0]
        self.source_output_sizes = (self.logical_output_size,)
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
        if self.input_layout is not None:
            if not isinstance(self.input_layout, GGUFHeadTilingLayout):
                return "input_layout_codec_unavailable"
            if not isinstance(canonical, AffineGGUFProjection):
                return "input_layout_requires_affine_groups"
            head_span, remainder = divmod(
                self.input_layout.head_dim, canonical.group_size
            )
            if remainder:
                return "input_layout_cuts_canonical_group"
            canonical = replace(
                canonical,
                codes=self.input_layout.weight_to_vllm(
                    torch.from_numpy(canonical.codes), dim=1
                ).numpy(),
                scales=self.input_layout.weight_to_vllm(
                    torch.from_numpy(canonical.scales), dim=1, head_dim=head_span
                ).numpy(),
                mins=self.input_layout.weight_to_vllm(
                    torch.from_numpy(canonical.mins), dim=1, head_dim=head_span
                ).numpy(),
            )
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
        self.input_layout_restored = self.input_layout is not None
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
            "input_layout_restored": self.input_layout_restored,
            "input_layout_rejection_reasons": self.input_layout_rejection_reasons,
            "source_output_sizes": list(self.source_output_sizes),
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
            kernel = self.kernel
            config = kernel.config
            if isinstance(config, Sm70GgufAffineConfig):
                family, decoder = 0, kernel.bits
            elif isinstance(config, Sm70GgufLut4Config):
                family, decoder = 1, kernel.lut_id
            else:
                family, decoder = 2, kernel.source_type
            capabilities = getattr(kernel, "operator_capabilities", ())
            blas = tuple(c for c in capabilities if "blas" in c.operator)
            return torch.ops.vllm.prepared_gguf_projection(
                x,
                self.codes,
                self.stats,
                self.fp16_cache,
                family,
                decoder,
                config.group_size,
                self.gguf_tm_k_ld,
                self.gguf_tm_q_ld,
                config.partition_weight_shape[1],
                self.logical_output_size,
                _admitted_bands(self.cache_capabilities),
                _admitted_bands(blas),
            )
        # Imported lazily because the GGUF method owns fallback dispatch.
        from vllm.model_executor.layers.quantization.gguf import fused_mul_mat_gguf

        return fused_mul_mat_gguf(
            x, self.weight, self.source_type, self.enabled, self.prefill_min_m
        )
