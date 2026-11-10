# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measured joint attention inputs with opaque runtime-M fallback."""

from dataclasses import asdict

import torch
from torch.nn import Parameter

from vllm.model_executor.kernels.gguf import native_qkv_capabilities
from vllm.model_executor.layers.quantization.gguf_native_pair import (
    _SOURCE_BLOCK_BYTES,
    _SOURCE_PACKERS,
)
from vllm.model_executor.layers.quantization.gguf_qkvz import prepared_source_views
from vllm.model_executor.layers.quantization.gguf_turbomind import (
    _prepared_gguf_mixed_projection,
    _prepared_gguf_projection,
    prepared_projection_arguments,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op


def _native_qkv(
    x: torch.Tensor,
    weights: list[torch.Tensor],
    scales: list[torch.Tensor],
    types: list[int],
    partials: torch.Tensor,
    counters: torch.Tensor,
    codes: list[torch.Tensor],
    stats: list[torch.Tensor],
    caches: list[torch.Tensor | None],
    descriptors: list[int],
    cache_bands: list[int],
    blas_bands: list[int],
) -> torch.Tensor:
    rows = x.reshape(-1, x.shape[-1]).contiguous()
    if rows.shape[0] == 8:
        out = rows.new_empty((8, 3584))
        torch.ops._C.gguf_qkv_sm70_out(
            out, rows, weights, scales, types, partials, counters
        )
    elif len(codes) == 1:
        out = _prepared_gguf_projection(
            rows,
            codes[0],
            stats[0],
            caches[0],
            descriptors[0],
            descriptors[1],
            descriptors[2],
            descriptors[3],
            descriptors[4],
            descriptors[5],
            descriptors[6],
            cache_bands,
            blas_bands,
        )
    else:
        out = _prepared_gguf_mixed_projection(
            rows, codes, stats, caches, descriptors, cache_bands, blas_bands
        )
    return out.reshape(*x.shape[:-1], 3584)


def _native_qkv_fake(
    x: torch.Tensor,
    weights: list[torch.Tensor],
    scales: list[torch.Tensor],
    types: list[int],
    partials: torch.Tensor,
    counters: torch.Tensor,
    codes: list[torch.Tensor],
    stats: list[torch.Tensor],
    caches: list[torch.Tensor | None],
    descriptors: list[int],
    cache_bands: list[int],
    blas_bands: list[int],
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], 3584))


direct_register_custom_op(
    op_name="gguf_native_qkv",
    op_func=_native_qkv,
    mutates_args=["partials", "counters"],
    fake_impl=_native_qkv_fake,
)


def prepare_native_qkv(layer, sources, projections, enabled):
    source_types = tuple(kind for _, kind in sources)
    device = sources[0][0].device if sources else torch.device("cpu")
    hardware = (
        current_platform.get_device_capability(device.index)
        if device.type == "cuda"
        else None
    )
    kernel = projections[0].kernel if projections else None
    shape = kernel.config.partition_weight_shape if kernel else (0, 0)
    capabilities = native_qkv_capabilities(
        source_types,
        shape[0],
        sum(p.logical_output_size for p in projections),
        kernel.config.act_type if kernel else None,
        enabled,
        hardware.to_int() if hardware else 0,
    )
    reason = next(
        (c.reason for c in capabilities if c.reason),
        None if capabilities else "qkv_shape_or_source_has_no_calibration",
    )
    widths = (3072, 256, 256)
    if reason is None:
        if not layer.prefix.endswith(".self_attn.qkv_proj"):
            reason = "requires_combined_attention_input_projection"
        elif tuple(w.shape[0] for w, _ in sources) != widths:
            reason = "qkv_shape_or_source_has_no_calibration"
        elif any(
            w.dtype != torch.uint8 or w.shape[1] != 20 * _SOURCE_BLOCK_BYTES[k]
            for w, k in sources
        ):
            reason = "qkv_operand_storage_not_supported"
        elif any(p.kernel is None for p in projections) or (
            tuple(s for p in projections for s in p.source_output_sizes) != widths
        ):
            reason = "canonical_fallback_unavailable"
    if reason is None:
        weights, scales, types = [], [], []
        views = prepared_source_views(projections)
        for (weight, kind), view in zip(sources, views):
            if kind in (10, 12):
                assert view is not None
                codes, stats = view
                weights.append(codes)
                scales.append(stats)
                types.append(102 if kind == 10 else 104)
            else:
                weights.append(
                    torch.from_numpy(
                        _SOURCE_PACKERS[kind](weight.detach().cpu().numpy())
                    ).to(device)
                )
                scales.append(torch.empty(0, dtype=torch.int32, device=device))
                types.append(kind)
        layer.gguf_qkv_weights = torch.nn.ParameterList(
            Parameter(w, False) for w in weights
        )
        layer.gguf_qkv_scales = torch.nn.ParameterList(
            Parameter(s, False) for s in scales
        )
        layer.gguf_qkv_types = types
        layer.register_buffer(
            "gguf_qkv_partials",
            torch.empty((56, 2, 512), dtype=torch.float32, device=device),
            persistent=False,
        )
        layer.register_buffer(
            "gguf_qkv_counters",
            torch.zeros(56, dtype=torch.int32, device=device),
            persistent=False,
        )
    return {
        "operator": "gguf_qkv_sm70_out",
        "source_types": source_types,
        "operators": [asdict(c) for c in capabilities],
        "min_m": 8,
        "max_m": 8,
        "n": 3584,
        "k": 5120,
        "graph_safe": True,
        "reason": reason,
        "fallback": "canonical",
    }


def apply_native_qkv(layer, x):
    return torch.ops.vllm.gguf_native_qkv(
        x,
        list(layer.gguf_qkv_weights),
        list(layer.gguf_qkv_scales),
        layer.gguf_qkv_types,
        layer.gguf_qkv_partials,
        layer.gguf_qkv_counters,
        *prepared_projection_arguments(layer.gguf_tm_projections),
    )
