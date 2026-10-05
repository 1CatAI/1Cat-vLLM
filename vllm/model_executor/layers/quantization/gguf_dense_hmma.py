# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One resident GGUF segment bank and bounded graph-stable scratch."""

import numpy as np
import torch

from vllm.config import get_current_vllm_config_or_none
from vllm.model_executor.kernels.gguf import GGUFOperatorCapability, decoder_family
from vllm.model_executor.layers.quantization.gguf_dense_hmma_formats import pack
from vllm.transformers_utils.gguf_tensor_reader import quant_type_name
from vllm.utils.torch_utils import direct_register_custom_op

_FORMATS = {12: 0, 13: 1, 14: 2, 20: 3, 23: 3, 8: 4}
# The two streams may overlap shared and routed work. Keep shared scratch
# separate from ordinary dense projections; layers within each stream are ordered.
_workspaces = {}


def workspace(device, shared=False):
    key = (device, shared)
    if key not in _workspaces:
        _workspaces[key] = dict(
            weight=torch.empty(16 * 1024 * 1024, device=device, dtype=torch.uint8),
            stats=torch.empty(8 * 1024 * 1024, device=device, dtype=torch.uint8),
            partial=torch.empty(1024 * 1024, device=device, dtype=torch.float32),
            counters=torch.zeros(1024, device=device, dtype=torch.int32),
            h=torch.empty((8, 160), device=device, dtype=torch.float16),
            gate=torch.empty(8, device=device, dtype=torch.float16),
        )
    return _workspaces[key]


def prepare_segment_bank(projection, canonical, device):
    cfg = get_current_vllm_config_or_none()
    source = projection.source_type
    k, n = projection.kernel.config.partition_weight_shape
    reason = None
    if cfg is None or not cfg.kernel_config.sm70_gguf.small_m_hmma:
        reason = "disabled_by_kernel_config"
    elif source not in _FORMATS:
        reason = "segment_format_not_supported"
    elif (k, n) not in (
        {(2560, width) for width in (128, 160, 256, 320, 1536, 2560, 3072, 3328, 4096)}
        | {(160, 2560), (1536, 2560)}
    ):
        reason = "segment_shape_has_no_calibration"
    elif projection.output_padding:
        reason = "segment_output_padding_not_qualified"
    elif not all(
        hasattr(torch.ops._C, op)
        for op in (
            "gguf_dense_segments_sm70_out",
            "gguf_dense_restore_canonical_sm70_out",
            "gguf_shared_gate_up_sm70_out",
        )
    ):
        reason = "packaged_segment_operator_missing"
    projection.segment_capability = GGUFOperatorCapability(
        decoder_family(source),
        quant_type_name(source),
        "gguf_dense_segments_sm70_out",
        True,
        min_m=1,
        max_m=8,
        reason=reason,
    )
    if reason:
        return False
    fmt = _FORMATS[source]
    codes = canonical.codes
    if source == 8:
        codes = ((codes.astype(np.int16) - 128) & 255).astype(np.uint8)
    minimum = canonical.mins if fmt in (0, 1) else None
    payload = pack(
        fmt,
        codes,
        canonical.scales.astype(np.float32),
        minimum.astype(np.float32) if minimum is not None else None,
        canonical.group_size,
    )
    projection.codes = torch.nn.Parameter(
        torch.from_numpy(payload[0]).to(device), False
    )
    projection.stats = torch.nn.Parameter(
        torch.from_numpy(payload[2]).to(device), False
    )
    projection.register_parameter(
        "segment_high",
        torch.nn.Parameter(torch.from_numpy(payload[1]).to(device), False),
    )
    projection.segment_format = fmt
    projection.cache_capabilities = ()
    workspace(device, k == 160 or n <= 320)
    return True


def apply_segments(rows, codes, scales, high, formats, ns, output, views):
    k = rows.shape[1]
    shared = k == 160 or sum(ns) <= 320
    scratch = workspace(rows.device, shared)
    if 1 <= rows.shape[0] <= 8:
        warps = 4 if k == 2560 and sum(ns) > 320 else 8
        if k == 1536:
            warps = 8
        split = 5 if k == 2560 and sum(ns) <= 320 else 1
        torch.ops._C.gguf_dense_segments_sm70_out(
            rows,
            codes,
            high,
            scales,
            views,
            formats,
            ns,
            k,
            split,
            warps,
            scratch["partial"],
            scratch["counters"],
            None,
        )
        return output
    for c, s, h, fmt, n, destination in zip(codes, scales, high, formats, ns, views):
        restore_and_apply(rows, c, s, h, fmt, n, destination)
    return output


def restore_and_apply(rows, codes, scale, high, fmt, n, output, k_ld=None, q_ld=None):
    k = rows.shape[1]
    scratch = workspace(rows.device, k == 160 or n <= 320)
    bits = (4, 5, 6, 4, 8)[fmt]
    group = 16 if fmt == 2 else 32
    count = k * n // (4 if fmt == 4 else 8)
    weight = scratch["weight"][: count * 4].view(torch.int32).view(k, -1)
    dtype = torch.int16 if fmt == 3 else torch.int64 if fmt in (1, 2) else torch.int32
    stats_count = k // group * n
    stats = (
        scratch["stats"][
            : stats_count * {torch.int16: 2, torch.int32: 4, torch.int64: 8}[dtype]
        ]
        .view(dtype)
        .view(k // group, n)
    )
    torch.ops._C.gguf_dense_restore_canonical_sm70_out(
        weight, stats, codes, high, scale, fmt, k, n
    )
    # The descriptor is carried from the original converter, before its bank
    # is replaced. Defaults are supplied only by the multi-segment wrapper.
    if k_ld is None:
        k_ld, q_ld = descriptor(k, n, fmt)
    if fmt == 3:
        torch.ops._C.gguf_lut4_gemm_sm70_out(
            output, rows, weight, stats, 0, k_ld, q_ld, group
        )
    else:
        torch.ops._C.gguf_affine_gemm_sm70_out(
            output, rows, weight, stats, bits, k_ld, q_ld, group
        )


_descriptors = {}


def remember_descriptor(k, n, fmt, k_ld, q_ld):
    _descriptors[(k, n, fmt)] = (k_ld, q_ld)


def descriptor(k, n, fmt):
    return _descriptors[(k, n, fmt)]


def _shared_expert(
    x: torch.Tensor,
    gate_codes: torch.Tensor,
    gate_high: torch.Tensor,
    gate_scale: torch.Tensor,
    up_codes: torch.Tensor,
    up_high: torch.Tensor,
    up_scale: torch.Tensor,
    down_codes: torch.Tensor,
    down_high: torch.Tensor,
    down_scale: torch.Tensor,
    weight_gate: torch.Tensor,
    formats: list[int],
    gu_codes: list[torch.Tensor],
    gu_stats: list[torch.Tensor],
    gu_caches: list[torch.Tensor | None],
    gu_specs: list[int],
    gu_cache_bands: list[int],
    gu_blas_bands: list[int],
    down_specs: list[int],
) -> torch.Tensor:
    from .gguf_turbomind import _prepared_gguf_mixed_projection

    rows = x.reshape(-1, x.shape[-1]).contiguous()
    if rows.shape[0] > 8:
        gu = _prepared_gguf_mixed_projection(
            rows, gu_codes, gu_stats, gu_caches, gu_specs, gu_cache_bands, gu_blas_bands
        )
        h = torch.empty((rows.shape[0], 160), device=x.device, dtype=x.dtype)
        torch.ops._C.silu_and_mul(h, gu)
        out = torch.empty((rows.shape[0], 2560), device=x.device, dtype=x.dtype)
        restore_and_apply(
            h,
            down_codes,
            down_scale,
            down_high,
            formats[2],
            2560,
            out,
            down_specs[0],
            down_specs[1],
        )
        g = torch.mm(rows, weight_gate.reshape(-1, 1).to(rows.dtype))
        return (out * torch.sigmoid(g)).reshape(*x.shape[:-1], 2560)
    scratch = workspace(x.device, True)
    h = scratch["h"][: rows.shape[0]]
    gate = scratch["gate"]
    out = torch.empty((rows.shape[0], 2560), device=x.device, dtype=x.dtype)
    torch.ops._C.gguf_shared_gate_up_sm70_out(
        rows,
        [gate_codes, gate_high, gate_scale],
        [up_codes, up_high, up_scale],
        formats[:2],
        weight_gate.reshape(-1),
        h,
        gate,
        scratch["partial"],
        scratch["counters"],
        5,
        8,
    )
    torch.ops._C.gguf_dense_segments_sm70_out(
        h,
        [down_codes],
        [down_high],
        [down_scale],
        [out],
        [formats[2]],
        [2560],
        160,
        1,
        8,
        scratch["partial"],
        scratch["counters"],
        gate,
    )
    return out.reshape(*x.shape[:-1], 2560)


def _shared_expert_fake(
    x: torch.Tensor,
    gate_codes: torch.Tensor,
    gate_high: torch.Tensor,
    gate_scale: torch.Tensor,
    up_codes: torch.Tensor,
    up_high: torch.Tensor,
    up_scale: torch.Tensor,
    down_codes: torch.Tensor,
    down_high: torch.Tensor,
    down_scale: torch.Tensor,
    weight_gate: torch.Tensor,
    formats: list[int],
    gu_codes: list[torch.Tensor],
    gu_stats: list[torch.Tensor],
    gu_caches: list[torch.Tensor | None],
    gu_specs: list[int],
    gu_cache_bands: list[int],
    gu_blas_bands: list[int],
    down_specs: list[int],
) -> torch.Tensor:
    return torch.empty((*x.shape[:-1], 2560), device=x.device, dtype=x.dtype)


direct_register_custom_op(
    "prepared_gguf_shared_expert", _shared_expert, fake_impl=_shared_expert_fake
)


def maybe_apply_shared_expert(layer, x):
    from .gguf_turbomind import prepared_projection_arguments

    if layer.expert_gate is None:
        return None
    gu = getattr(layer.gate_up_proj, "gguf_tm_projections", None)
    down = getattr(layer.down_proj, "gguf_tm_projections", None)
    weight = getattr(layer.expert_gate, "weight", None)
    if (
        gu is None
        or down is None
        or len(down) != 1
        or not all(hasattr(p, "segment_format") for p in (*gu, *down))
        or weight is None
        or weight.dtype not in (torch.float16, torch.float32)
        or weight.numel() != 2560
        or not weight.is_contiguous()
        or sum(p.logical_output_size for p in gu) != 320
        or down[0].logical_output_size != 2560
        or any(p.kernel.config.partition_weight_shape[0] != 2560 for p in gu)
        or down[0].kernel.config.partition_weight_shape[0] != 160
    ):
        return None
    pairs = []
    for p in gu:
        tile = 0
        for n in p.source_output_sizes:
            if n != 160:
                return None
            groups = 20
            cstride = groups * 4 * 512 * (2 if p.segment_format == 4 else 1)
            hstride = (
                groups
                * 512
                * (1 if p.segment_format == 1 else 2 if p.segment_format == 2 else 0)
            )
            sstride = groups * 512
            pairs.append(
                (
                    p.codes[tile * cstride : (tile + 5) * cstride],
                    p.segment_high[tile * hstride : (tile + 5) * hstride],
                    p.stats[tile * sstride : (tile + 5) * sstride],
                    p.segment_format,
                )
            )
            tile += 5
    if len(pairs) != 2:
        return None
    p = down[0]
    return torch.ops.vllm.prepared_gguf_shared_expert(
        x,
        *pairs[0][:3],
        *pairs[1][:3],
        p.codes,
        p.segment_high,
        p.stats,
        weight,
        [pairs[0][3], pairs[1][3], p.segment_format],
        *prepared_projection_arguments(gu),
        [p.gguf_tm_k_ld, p.gguf_tm_q_ld],
    )
