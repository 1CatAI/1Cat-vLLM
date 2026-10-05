# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measured one-launch GDN inputs with runtime-M canonical fallback."""

from dataclasses import asdict, replace

import torch
from torch.nn import Parameter

from vllm.model_executor.kernels.gguf import native_qkvz_capabilities
from vllm.model_executor.layers.quantization.gguf_native_pair import (
    _SOURCE_BLOCK_BYTES,
    _SOURCE_PACKERS,
)
from vllm.model_executor.layers.quantization.gguf_turbomind import (
    _prepared_gguf_mixed_projection,
    prepared_projection_arguments,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op


def _native_qkvz(
    x: torch.Tensor,
    weights: list[torch.Tensor],
    scales: list[torch.Tensor],
    types: list[int],
    floating: list[torch.Tensor],
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
        out = rows.new_empty((8, 4120))
        torch.ops._C.gguf_qkvz_sm70_out(
            out, rows, weights, scales, types, partials, counters
        )
    else:
        quantized = _prepared_gguf_mixed_projection(
            rows, codes, stats, caches, descriptors, cache_bands, blas_bands
        )
        outputs = [quantized]
        for weight in floating:
            outputs.append(
                torch.ops.vllm.prepared_gguf_fp16_projection(rows, weight, 30, True)
            )
        out = torch.cat(outputs, dim=-1)
    return out.reshape(*x.shape[:-1], 4120)


def _native_qkvz_fake(
    x: torch.Tensor,
    weights: list[torch.Tensor],
    scales: list[torch.Tensor],
    types: list[int],
    floating: list[torch.Tensor],
    partials: torch.Tensor,
    counters: torch.Tensor,
    codes: list[torch.Tensor],
    stats: list[torch.Tensor],
    caches: list[torch.Tensor | None],
    descriptors: list[int],
    cache_bands: list[int],
    blas_bands: list[int],
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], 4120))


direct_register_custom_op(
    op_name="gguf_native_qkvz",
    op_func=_native_qkvz,
    mutates_args=["partials", "counters"],
    fake_impl=_native_qkvz_fake,
)


def prepared_source_views(projections):
    """Alias each canonical source's N tiles and preserve the full stat stride."""
    result: list[tuple[torch.Tensor, torch.Tensor] | None] = []
    for projection in projections:
        offset = 0
        for width in projection.source_output_sizes:
            if projection.source_type in (10, 12):
                k = projection.kernel.config.partition_weight_shape[0]
                bits = 2 if projection.source_type == 10 else 4
                count = width * k * bits // 32
                codes = projection.codes.reshape(-1).narrow(
                    0, offset * k * bits // 32, count
                )
                result.append(
                    (
                        codes.view(k, width * bits // 32),
                        projection.stats[:, offset : offset + width],
                    )
                )
            else:
                result.append(None)
            offset += width
    return result


def prepare_native_qkvz(layer, sources, projections, enabled: bool):
    source_types = tuple(kind for _, kind in sources)
    device = sources[0][0].device if sources else torch.device("cpu")
    hardware = (
        current_platform.get_device_capability(device.index)
        if device.type == "cuda"
        else None
    )
    dtype = (
        projections[0].kernel.config.act_type
        if projections and projections[0].kernel
        else None
    )
    capabilities = native_qkvz_capabilities(
        source_types, 5120, 4120, dtype, enabled, hardware.to_int() if hardware else 0
    )
    reason = next(
        (c.reason for c in capabilities if c.reason),
        None if capabilities else "qkvz_shape_or_source_has_no_calibration",
    )
    widths = (512, 512, 1536, 1536, 12, 12)
    if reason is None:
        if not layer.prefix.endswith(".in_proj_qkvz"):
            reason = "requires_combined_gdn_input_projection"
        elif tuple(w.shape[0] for w, _ in sources) != widths:
            reason = "qkvz_shape_or_source_has_no_calibration"
        elif any(
            w.dtype != torch.uint8 or w.shape[1] != 20 * _SOURCE_BLOCK_BYTES[kind]
            for w, kind in sources[:4]
        ) or any(
            w.dtype != torch.float16 or w.shape[1] != 5120 for w, _ in sources[4:]
        ):
            reason = "qkvz_operand_storage_not_supported"
        elif any(
            p.source_type not in (1, 30) and p.kernel is None for p in projections
        ) or (
            tuple(size for p in projections for size in p.source_output_sizes) != widths
        ):
            reason = "canonical_fallback_unavailable"
    if reason is None:
        views = prepared_source_views(projections)
        weights, scales, types = [], [], []
        for index, (weight, kind) in enumerate(sources[:4]):
            if kind in (10, 12):
                view = views[index]
                assert view is not None
                codes, stats = view
                weights.append(codes)
                scales.append(stats)
                types.append(102 if kind == 10 else 104)
            else:
                records = _SOURCE_PACKERS[kind](weight.detach().cpu().numpy())
                weights.append(torch.from_numpy(records).to(device))
                scales.append(torch.empty(0, dtype=torch.int32, device=device))
                types.append(kind)
        floating = torch.cat([weight.detach().cpu() for weight, _ in sources[4:]])
        padded = torch.zeros(64, 5120, dtype=torch.float16)
        padded[:24] = floating
        weights.append(
            padded.reshape(2, 32, 40, 16, 8)
            .permute(0, 2, 3, 1, 4)
            .contiguous()
            .to(device)
        )
        scales.append(torch.empty(0, dtype=torch.int32, device=device))
        types.append(1)
        layer.gguf_qkvz_weights = torch.nn.ParameterList(
            Parameter(w, False) for w in weights
        )
        layer.gguf_qkvz_scales = torch.nn.ParameterList(
            Parameter(s, False) for s in scales
        )
        layer.gguf_qkvz_types = types
        layer.gguf_qkvz_floating = torch.nn.ParameterList(
            Parameter(p.weight, False) for p in projections if p.kernel is None
        )
        layer.register_buffer(
            "gguf_qkvz_partials",
            torch.empty((65, 2, 512), dtype=torch.float32, device=device),
            persistent=False,
        )
        layer.register_buffer(
            "gguf_qkvz_counters",
            torch.zeros(65, dtype=torch.int32, device=device),
            persistent=False,
        )
    return {
        "operator": "gguf_qkvz_sm70_out",
        "source_types": list(source_types),
        "operators": [asdict(replace(c, reason=reason)) for c in capabilities],
        "min_m": 8,
        "max_m": 8,
        "n": 4120,
        "k": 5120,
        "graph_safe": True,
        "reason": reason,
        "fallback": "canonical",
    }


def apply_native_qkvz(layer, x):
    return torch.ops.vllm.gguf_native_qkvz(
        x,
        list(layer.gguf_qkvz_weights),
        list(layer.gguf_qkvz_scales),
        layer.gguf_qkvz_types,
        list(layer.gguf_qkvz_floating),
        layer.gguf_qkvz_partials,
        layer.gguf_qkvz_counters,
        *prepared_projection_arguments(
            [p for p in layer.gguf_tm_projections if p.kernel is not None]
        ),
    )
