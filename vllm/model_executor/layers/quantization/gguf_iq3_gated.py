# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Calibrated IQ3_S gated pairs with runtime-M canonical fallback."""

import torch
from torch.nn import Parameter

from vllm.model_executor.kernels.gguf import iq3_gated_pair_capability
from vllm.model_executor.layers.quantization.gguf_iq3_records import (
    signed_index_records,
)
from vllm.model_executor.layers.quantization.gguf_turbomind import (
    _admitted_bands,
    _prepared_gguf_projection,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op


def _iq3_gated_pair(
    x: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    codes: list[torch.Tensor],
    stats: list[torch.Tensor],
    k_ld: list[int],
    q_ld: list[int],
    blas_bands: list[int],
) -> torch.Tensor:
    rows = x.reshape(-1, x.shape[-1]).contiguous()
    output = torch.empty((rows.shape[0], 4352), dtype=x.dtype, device=x.device)
    if rows.shape[0] == 8:
        torch.ops._C.gguf_iq3_gated_sm70_out(output, rows, gate, up)
    else:
        projections = [
            _prepared_gguf_projection(
                rows,
                codes[i],
                stats[i],
                None,
                2,
                21,
                32,
                k_ld[i],
                q_ld[i],
                4352,
                4352,
                [],
                blas_bands,
            )
            for i in range(2)
        ]
        torch.ops._C.silu_and_mul(output, torch.cat(projections, dim=-1))
    return output.reshape(*x.shape[:-1], 4352)


def _iq3_gated_pair_fake(
    x: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    codes: list[torch.Tensor],
    stats: list[torch.Tensor],
    k_ld: list[int],
    q_ld: list[int],
    blas_bands: list[int],
) -> torch.Tensor:
    return torch.empty((*x.shape[:-1], 4352), dtype=x.dtype, device=x.device)


direct_register_custom_op(
    op_name="gguf_iq3_gated_pair",
    op_func=_iq3_gated_pair,
    fake_impl=_iq3_gated_pair_fake,
)


def prepare_iq3_gated_pair(layer, sources, projections, enabled: bool):
    device_capability = (
        current_platform.get_device_capability(sources[0][0].device.index)
        if sources and sources[0][0].device.type == "cuda"
        else None
    )
    capability = iq3_gated_pair_capability(
        tuple(t for _, t in sources),
        5120,
        4352,
        projections[0].kernel.config.act_type
        if projections and projections[0].kernel is not None
        else None,
        enabled,
        compute_capability=device_capability.to_int() if device_capability else 0,
    )
    reason = capability.reason
    if reason is None:
        if len(sources) != 2 or not layer.prefix.endswith(".gate_up_proj"):
            reason = "requires_gate_up_projection_pair"
        elif any(w.dtype != torch.uint8 or w.shape != (4352, 2200) for w, _ in sources):
            reason = "gated_pair_shape_or_source_has_no_calibration"
        elif len(projections) != 2 or any(p.kernel is None for p in projections):
            reason = "canonical_fallback_unavailable"
        elif any(p.codes.device.type != "cuda" for p in projections):
            reason = "requires_cuda_fp16_activations"
    if reason is None:
        layer.gguf_iq3_gated_records = torch.nn.ParameterList(
            Parameter(
                torch.from_numpy(signed_index_records(w.detach().cpu().numpy())).to(
                    w.device
                ),
                requires_grad=False,
            )
            for w, _ in sources
        )
    return {
        "operator": "gguf_iq3_gated_sm70_out",
        "source_type": "IQ3_S",
        "family": "lattice_codebook",
        "min_m": 8,
        "max_m": 8,
        "n": 4352,
        "k": 5120,
        "graph_safe": True,
        "reason": reason,
        "fallback": "canonical",
    }


def apply_iq3_gated_pair(layer, x):
    projections = layer.gguf_tm_projections
    capabilities = getattr(projections[0].kernel, "operator_capabilities", ())
    blas_bands = _admitted_bands(tuple(c for c in capabilities if "blas" in c.operator))
    return torch.ops.vllm.gguf_iq3_gated_pair(
        x,
        *layer.gguf_iq3_gated_records,
        [p.codes for p in projections],
        [p.stats for p in projections],
        [p.gguf_tm_k_ld for p in projections],
        [p.gguf_tm_q_ld for p in projections],
        blas_bands,
    )
