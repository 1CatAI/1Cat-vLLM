# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measured source-sized mixed gated pairs with runtime canonical fallback."""

from dataclasses import asdict, replace

import torch
from torch.nn import Parameter

from vllm.model_executor.kernels.gguf import native_gated_pair_capabilities
from vllm.model_executor.layers.quantization.gguf_iq2_s_records import (
    pack_iq2_s_records,
)
from vllm.model_executor.layers.quantization.gguf_iq2_xs_records import (
    pack_iq2_xs_records,
)
from vllm.model_executor.layers.quantization.gguf_iq3_records import (
    signed_index_records,
)
from vllm.model_executor.layers.quantization.gguf_iq3_xxs_records import (
    pack_iq3_xxs_records,
)
from vllm.model_executor.layers.quantization.gguf_iq4_native import pack_iq4_xs_records
from vllm.model_executor.layers.quantization.gguf_q4_k_records import pack_q4_k_records
from vllm.model_executor.layers.quantization.gguf_turbomind import (
    _prepared_gguf_mixed_projection,
    prepared_projection_arguments,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

_SOURCE_BLOCK_BYTES = {12: 144, 17: 74, 18: 98, 21: 110, 22: 82, 23: 136}
_SOURCE_PACKERS = {
    12: pack_q4_k_records,
    17: pack_iq2_xs_records,
    18: pack_iq3_xxs_records,
    21: signed_index_records,
    22: pack_iq2_s_records,
    23: pack_iq4_xs_records,
}


def _native_gated_pair(
    x: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    gate_type: int,
    up_type: int,
    codes: list[torch.Tensor],
    stats: list[torch.Tensor],
    caches: list[torch.Tensor | None],
    descriptors: list[int],
    cache_bands: list[int],
    blas_bands: list[int],
) -> torch.Tensor:
    rows = x.reshape(-1, x.shape[-1]).contiguous()
    output = rows.new_empty((rows.shape[0], 4352))
    if rows.shape[0] == 8:
        torch.ops._C.gguf_native_pair_sm70_out(
            output, rows, gate, up, gate_type, up_type
        )
    else:
        pair = _prepared_gguf_mixed_projection(
            rows, codes, stats, caches, descriptors, cache_bands, blas_bands
        )
        torch.ops._C.silu_and_mul(output, pair)
    return output.reshape(*x.shape[:-1], 4352)


def _native_gated_pair_fake(
    x: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    gate_type: int,
    up_type: int,
    codes: list[torch.Tensor],
    stats: list[torch.Tensor],
    caches: list[torch.Tensor | None],
    descriptors: list[int],
    cache_bands: list[int],
    blas_bands: list[int],
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], 4352))


direct_register_custom_op(
    op_name="gguf_native_gated_pair",
    op_func=_native_gated_pair,
    fake_impl=_native_gated_pair_fake,
)


def prepare_native_gated_pair(layer, sources, projections, enabled: bool):
    source_types = tuple(t for _, t in sources)
    device_capability = (
        current_platform.get_device_capability(sources[0][0].device.index)
        if sources and sources[0][0].device.type == "cuda"
        else None
    )
    capabilities = native_gated_pair_capabilities(
        source_types,
        5120,
        4352,
        projections[0].kernel.config.act_type
        if projections and projections[0].kernel is not None
        else None,
        enabled,
        compute_capability=device_capability.to_int() if device_capability else 0,
    )
    reason = next(
        (c.reason for c in capabilities if c.reason),
        None if capabilities else "gated_pair_shape_or_source_has_no_calibration",
    )
    if reason is None:
        if not layer.prefix.endswith(".gate_up_proj"):
            reason = "requires_gate_up_projection_pair"
        elif any(
            w.dtype != torch.uint8 or w.shape != (4352, 20 * _SOURCE_BLOCK_BYTES[t])
            for w, t in sources
        ):
            reason = "gated_pair_shape_or_source_has_no_calibration"
        elif len(projections) != 2 or any(
            p.kernel is None or p.logical_output_size != 4352 for p in projections
        ):
            reason = "canonical_fallback_unavailable"
    if reason is None:
        layer.gguf_native_gated_records = torch.nn.ParameterList(
            Parameter(
                torch.from_numpy(_SOURCE_PACKERS[t](w.detach().cpu().numpy())).to(
                    w.device
                ),
                requires_grad=False,
            )
            for w, t in sources
        )
        layer.gguf_native_gated_types = source_types
    return {
        "operator": "gguf_native_pair_sm70_out",
        "source_types": list(source_types),
        "operators": [asdict(replace(c, reason=reason)) for c in capabilities],
        "min_m": 8,
        "max_m": 8,
        "n": 4352,
        "k": 5120,
        "graph_safe": True,
        "reason": reason,
        "fallback": "canonical",
    }


def apply_native_gated_pair(layer, x):
    return torch.ops.vllm.gguf_native_gated_pair(
        x,
        layer.gguf_native_gated_records[0],
        layer.gguf_native_gated_records[1],
        layer.gguf_native_gated_types[0],
        layer.gguf_native_gated_types[1],
        *prepared_projection_arguments(layer.gguf_tm_projections),
    )
