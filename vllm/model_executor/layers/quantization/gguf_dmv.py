# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One resident M8 projection plane bank with exact canonical restoration."""

import numpy as np
import torch
from torch.nn import Parameter

from vllm.config import get_current_vllm_config_or_none
from vllm.model_executor.kernels.gguf import GGUFOperatorCapability, decoder_family
from vllm.model_executor.layers.quantization import gguf_dmv_formats as iq
from vllm.model_executor.layers.quantization.gguf_dense_hmma import workspace
from vllm.model_executor.layers.quantization.gguf_dense_hmma_formats import decode, pack
from vllm.transformers_utils.gguf_tensor_reader import quant_size, quant_type_name
from vllm.utils.torch_utils import direct_register_custom_op

FORMATS = {12: 0, 23: 3, 21: 5, 18: 6, **iq.IQ2_FORMATS}
_tables = {}
_reverse_tables = {}


def table(device):
    if device not in _tables:
        _tables[device] = torch.from_numpy(iq.tables()).to(device)
    return _tables[device]


def reverse_table(kind, device):
    key = (kind, device)
    if key not in _reverse_tables:
        _reverse_tables[key] = torch.from_numpy(iq.iq2_reverse_table(kind)).to(device)
    return _reverse_tables[key]


def eligible_sources(sources, prefix):
    """Restrict storage replacement to complete, measured TP4 projection shapes."""
    cfg = get_current_vllm_config_or_none()
    if cfg is None or not cfg.kernel_config.sm70_gguf.projection_planes:
        return False
    scope = cfg.kernel_config.sm70_gguf.projection_plane_scope
    if scope == "gated_pair" and not prefix.endswith(".gate_up_proj"):
        return False
    if scope == "iq3_xxs" and not any(t == 18 for _, t in sources):
        return False
    quantized = [(w, t) for w, t in sources if t not in (1, 30)]
    if not quantized or len({t for _, t in quantized}) > 2:
        return False
    if any(t in iq.IQ2_FORMATS for _, t in quantized):
        if not cfg.kernel_config.sm70_gguf.iq2_signed_nibbles:
            return False
        # Only the measured gate/up and down shapes admit the new decoder.
        if not prefix.endswith((".gate_up_proj", ".down_proj")):
            return False
        if any(t == 12 for _, t in quantized):
            return False
    for w, t in quantized:
        if t not in FORMATS or w.dtype != torch.uint8 or w.ndim != 2:
            return False
        block, size = quant_size(t)
        if w.shape[1] % size:
            return False
        k = w.shape[1] // size * block
        if (k, w.shape[0]) not in {
            (5120, 4352),
            (4352, 5120),
            (1536, 5120),
            (5120, 512),
            (5120, 1536),
            (5120, 3072),
            (5120, 256),
        }:
            return False
    return all(
        t not in (1, 30) or (w.dtype == torch.float16 and w.ndim == 2)
        for w, t in sources
    )


def prepare_bank(projection, raw_weight, canonical):
    kind = projection.source_type
    k, n = projection.kernel.config.partition_weight_shape
    if kind not in FORMATS or projection.output_padding:
        return False
    # The IQ3 byte-to-half trick uses a 1024*scale cancellation term.
    # Preserve canonical arithmetic for coefficients whose cancellation term
    # would overflow, even when the reconstructed weight itself is finite.
    if (
        kind in (18, 21)
        and np.max(np.abs(canonical.scales.astype(np.float32))) > 63.96875
    ):
        projection.dmv_rejection_reason = "iq3_scale_exceeds_decode_cancellation_range"
        return False
    if not all(
        hasattr(torch.ops._C, op)
        for op in (
            "gguf_dmv_sm70_out",
            "gguf_dmv_restore_sm70_out",
        )
    ):
        return False
    if kind in iq.IQ2_FORMATS and not hasattr(
        torch.ops._C, "gguf_dmv_restore_iq2_sm70_out"
    ):
        return False
    raw = raw_weight.detach().cpu().numpy()
    if kind in iq.IQ2_FORMATS:
        fmt, codes, scale = iq.pack_iq2(raw, kind)
        high = None
    elif kind in (18, 21):
        fmt, codes, scale = iq.pack(raw, kind)
        high = np.empty(0, dtype=np.uint8)
    else:
        fmt, q, s, m, gs = decode(raw, kind)
        codes, high, scale = pack(fmt, q, s, m, gs)
        if fmt == 3:
            scale = iq.compact_lut4_scale(scale)
    # GDN's input heads are permuted in units of 128, matching plane K groups.
    layout = projection.input_layout
    if layout is not None:
        order = (
            layout.weight_to_vllm(
                torch.arange(k // 128).reshape(1, -1),
                dim=1,
                head_dim=layout.head_dim // 128,
            )
            .numpy()
            .reshape(-1)
        )

        def reorder(a):
            return np.ascontiguousarray(
                a.reshape(n // 32, k // 128, -1)[:, order]
            ).reshape(-1)

        codes, scale = reorder(codes), reorder(scale)
    projection.codes = Parameter(torch.from_numpy(codes).to(raw_weight.device), False)
    projection.stats = Parameter(torch.from_numpy(scale).to(raw_weight.device), False)
    projection.register_parameter(
        "dmv_high",
        Parameter(
            reverse_table(kind, raw_weight.device)
            if kind in iq.IQ2_FORMATS
            else torch.from_numpy(high).to(raw_weight.device),
            False,
        ),
    )
    projection.dmv_format = fmt
    projection.cache_capabilities = ()
    projection.dmv_capability = GGUFOperatorCapability(
        decoder_family(kind),
        quant_type_name(kind),
        "gguf_dmv_sm70_out",
        True,
        min_m=8,
        max_m=8,
    )
    scratch = workspace(raw_weight.device)
    # The largest coalesced TP4 gate/up bank is N8704,K5120. Grow only
    # during loading, before any layer captures the shared scratch in a graph.
    for name, size in (("weight", 24 * 1024 * 1024), ("stats", 12 * 1024 * 1024)):
        if scratch[name].numel() < size:
            scratch[name] = torch.empty(
                size, dtype=torch.uint8, device=raw_weight.device
            )
    table(raw_weight.device)
    return True


def restore(
    rows, codes, scales, high, fmt, n, k_ld, q_ld, out, cache_bands, blas_bands
):
    k = rows.shape[1]
    scratch = workspace(rows.device)
    count = k * n // (8 if fmt in (0, 3) else 16)
    weight = scratch["weight"][: count * 4].view(torch.int32).view(k, -1)
    kind = {7: 16, 8: 17, 9: 22}.get(fmt)
    group = 16 if kind in (17, 22) else 32
    dtype = (
        torch.int16 if fmt == 3 else torch.int32 if fmt == 0 or kind else torch.int64
    )
    size = 2 if fmt == 3 else 4 if fmt == 0 or kind else 8
    stats = scratch["stats"][: k // group * n * size].view(dtype).view(k // group, n)
    if kind is not None:
        torch.ops._C.gguf_dmv_restore_iq2_sm70_out(
            weight, stats, codes, scales, high, kind, k, n
        )
    elif fmt == 0:
        torch.ops._C.gguf_dense_restore_canonical_sm70_out(
            weight,
            stats,
            codes,
            high,
            scales,
            fmt,
            k,
            n,
        )
    else:
        torch.ops._C.gguf_dmv_restore_sm70_out(weight, stats, codes, scales, fmt, k, n)
    from .gguf_turbomind import _prepared_gguf_projection

    family = 0 if fmt == 0 else 1 if fmt == 3 else 2
    decoder = (
        kind
        if kind is not None
        else 4
        if fmt == 0
        else 0
        if fmt == 3
        else 21
        if fmt == 5
        else 18
    )
    result = _prepared_gguf_projection(
        rows,
        weight,
        stats,
        None,
        family,
        decoder,
        group,
        k_ld,
        q_ld,
        n,
        n,
        cache_bands,
        blas_bands,
    )
    out.copy_(result)


def prepare_layer(layer, projections):
    quantized = [p for p in projections if p.kernel is not None]
    if not quantized or not all(hasattr(p, "dmv_format") for p in quantized):
        return {
            "reason": next(
                (
                    reason
                    for p in projections
                    for reason in getattr(p, "dmv_rejection_reasons", ())
                ),
                "projection_planes_not_qualified",
            )
        }
    # Retain source boundaries even when the canonical loader coalesces them.
    codes, scales, high, fmts, ns = [], [], [], [], []
    for p in quantized:
        tile = 0
        groups = p.kernel.config.partition_weight_shape[0] // 128
        cstride = groups * 512 * (3 if p.dmv_format in (5, 6) else 4)
        sstride = groups * (
            128 if p.dmv_format == 6 else 512 if p.dmv_format == 0 else 256
        )
        for n in p.source_output_sizes:
            end = tile + n // 32
            codes.append(p.codes[tile * cstride : end * cstride])
            scales.append(p.stats[tile * sstride : end * sstride])
            high.append(p.dmv_high)
            fmts.append(p.dmv_format)
            ns.append(n)
            tile = end
    if len(codes) > 4:
        return {"reason": "too_many_projection_segments"}
    pair = layer.prefix.endswith(".gate_up_proj")
    kw, tn, split = (4, 4, 1) if pair else (4, 2, 1)
    # Narrow segments share one launch with wider projections; split2 is used
    # only for an independently launched narrow matrix.
    if not pair and sum(ns) <= 1536:
        split = 2
    tiles = (
        ns[0] // 32 // (tn // 2)
        if pair
        else sum((n + tn * 32 - 1) // (tn * 32) for n in ns)
    )
    device = quantized[0].codes.device
    layer.register_buffer(
        "gguf_dmv_partials",
        torch.empty(tiles * split * tn * 256, dtype=torch.float32, device=device),
        persistent=False,
    )
    layer.register_buffer(
        "gguf_dmv_counters",
        torch.zeros(tiles, dtype=torch.int32, device=device),
        persistent=False,
    )
    gdn_heads = any(getattr(p, "dmv_gdn_heads", False) for p in projections)
    layer.gguf_dmv_operands = (
        codes,
        high,
        scales,
        fmts,
        ns,
        kw,
        tn,
        split,
        pair,
        gdn_heads,
    )
    floating = [p.weight for p in projections if p.kernel is None]
    if floating:
        combined = floating[0] if len(floating) == 1 else torch.cat(floating, dim=0)
        layer.register_parameter("gguf_dmv_floating", Parameter(combined, False))
        offset = 0
        for projection in projections:
            if projection.kernel is None:
                n = projection.weight.shape[0]
                projection.weight = Parameter(combined.narrow(0, offset, n), False)
                offset += n
    return {
        "reason": None,
        "operator": "gguf_dmv_sm70_out",
        "min_m": 8,
        "max_m": 8,
        "kw": kw,
        "tn": tn,
        "split": split,
        "pair": pair,
        "source_types": [p.source_type for p in quantized],
        "resident_bytes": sum(p.codes.numel() + p.stats.numel() for p in quantized),
        "shared_codebook_bytes": 3072,
        "fallback": "restore_canonical",
    }


def _project(
    x: torch.Tensor,
    codes: list[torch.Tensor],
    high: list[torch.Tensor],
    scales: list[torch.Tensor],
    formats: list[int],
    widths: list[int],
    kw: int,
    tn: int,
    split: int,
    pair: bool,
    gdn_heads: bool,
    partials: torch.Tensor,
    counters: torch.Tensor,
    floating: torch.Tensor | None,
    fallback_floating: list[torch.Tensor],
    fallback_codes: list[torch.Tensor],
    fallback_stats: list[torch.Tensor],
    fallback_caches: list[torch.Tensor | None],
    descriptors: list[int],
    cache_bands: list[int],
    blas_bands: list[int],
) -> torch.Tensor:
    from .gguf_turbomind import _prepared_gguf_mixed_projection

    rows = x.reshape(-1, x.shape[-1]).contiguous()
    width = (
        widths[0]
        if pair
        else sum(widths) + (floating.shape[0] if floating is not None else 0)
    )
    if rows.shape[0] != 8:
        if gdn_heads:
            rows = (
                rows.reshape(-1, 4, 3, 128)
                .transpose(1, 2)
                .reshape(-1, 1536)
                .contiguous()
            )
        result = _prepared_gguf_mixed_projection(
            rows,
            fallback_codes,
            fallback_stats,
            fallback_caches,
            descriptors,
            cache_bands,
            blas_bands,
        )
        if fallback_floating:
            values = [
                torch.ops.vllm.prepared_gguf_fp16_projection(rows, w, 30, True)
                for w in fallback_floating
            ]
            result = torch.cat((result, *values), dim=1)
        if pair:
            out = rows.new_empty((rows.shape[0], width))
            torch.ops._C.silu_and_mul(out, result)
            result = out
        return result.reshape(*x.shape[:-1], width)
    out = rows.new_empty((8, width))
    if pair:
        # Pair mode never writes the segment outputs; alias the final output.
        views = [out, out]
        ab_out = None
    else:
        sizes = widths + ([floating.shape[0]] if floating is not None else [])
        segments = list(out.split(sizes, dim=1))
        views, ab_out = (
            segments[: len(widths)],
            segments[-1] if floating is not None else None,
        )
    torch.ops._C.gguf_dmv_sm70_out(
        rows,
        codes,
        high,
        scales,
        views,
        formats,
        widths,
        rows.shape[1],
        split,
        kw,
        partials,
        counters,
        tn,
        None,
        table(rows.device),
        floating,
        ab_out,
        out if pair else None,
        gdn_heads,
    )
    return out.reshape(*x.shape[:-1], width)


def _project_fake(
    x,
    codes,
    high,
    scales,
    formats,
    widths,
    kw,
    tn,
    split,
    pair,
    gdn_heads,
    partials,
    counters,
    floating,
    fallback_floating,
    fallback_codes,
    fallback_stats,
    fallback_caches,
    descriptors,
    cache_bands,
    blas_bands,
):
    n = (
        widths[0]
        if pair
        else sum(widths) + (floating.shape[0] if floating is not None else 0)
    )
    return x.new_empty((*x.shape[:-1], n))


direct_register_custom_op(
    "gguf_dmv_projection",
    _project,
    fake_impl=_project_fake,
    mutates_args=["partials", "counters"],
)


def apply_layer(layer, x, fused=False):
    from .gguf_turbomind import prepared_projection_arguments

    codes, high, scales, fmts, ns, kw, tn, split, pair, gdn_heads = (
        layer.gguf_dmv_operands
    )
    pair = pair and fused
    return torch.ops.vllm.gguf_dmv_projection(
        x,
        codes,
        high,
        scales,
        fmts,
        ns,
        kw,
        tn,
        split,
        pair,
        gdn_heads,
        layer.gguf_dmv_partials,
        layer.gguf_dmv_counters,
        getattr(layer, "gguf_dmv_floating", None),
        [p.weight for p in layer.gguf_tm_projections if p.kernel is None],
        *prepared_projection_arguments(
            [p for p in layer.gguf_tm_projections if p.kernel is not None]
        ),
    )
