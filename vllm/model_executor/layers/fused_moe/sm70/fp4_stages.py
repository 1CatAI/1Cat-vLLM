# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared direct and routed FP4 stage order, without format-module callbacks."""

import torch

from vllm import _sm70_ops as ops
from vllm.model_executor.layers.fused_moe.sm70.declarations import FP4_STAGE_BINDINGS
from vllm.model_executor.layers.fused_moe.sm70.fp4_codec import Fp4MoECodec
from vllm.model_executor.layers.fused_moe.sm70.reduction import (
    _mtp_weighted_reduce,
    _single_token_weighted_reduce,
)
from vllm.model_executor.layers.quantization.sm70_moe_router import Sm70MoeRoutePlan

FUSED_W13 = frozenset(
    mode
    for (_, stage, mode), binding in FP4_STAGE_BINDINGS.items()
    if stage == "w13" and "activation" in binding[1]
)
FUSED_W2 = frozenset(
    mode
    for (_, stage, mode), binding in FP4_STAGE_BINDINGS.items()
    if stage == "w2" and "reduce" in binding[1]
)


def apply_swiglu(out, gate_up, limit=None, *, interleaved=False):
    if interleaved:
        if limit is not None:
            raise RuntimeError(
                "Interleaved SM70 NVFP4 SwiGLU does not support clamping."
            )
        torch.ops._C.silu_and_mul_interleaved(out, gate_up)
    elif limit is None:
        torch.ops._C.silu_and_mul(out, gate_up)
    else:
        torch.ops._C.silu_and_mul_with_clamp(out, gate_up, float(limit))


def execute_fp4(
    codec: Fp4MoECodec,
    plan: Sm70MoeRoutePlan,
    buffers: dict[str, torch.Tensor],
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    offsets: torch.Tensor | None = None,
    expert_ids: torch.Tensor | None = None,
    expert_count: int = 0,
    unpermute_offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    """W13 → optional activation → W2 → optional reduction.

    Routing is already prepared by the original selector. Fusion covers
    adjacent stages with its original layout and FP16 rounding boundaries.
    """
    route_ids = (
        topk_ids.view(-1)
        if plan.w13.value
        in {"qpn", "fused_qpn", "fused_batch_qpn", "glm_qpn", "active_grouped"}
        else None
    )
    codec.gemm_w13(
        plan, buffers, x, route_ids, offsets, expert_ids, expert_count, topk_ids
    )
    if plan.w13.value not in FUSED_W13:
        apply_swiglu(
            buffers["intermediate"],
            buffers["gate_up"],
            codec.swiglu_limit,
            interleaved=plan.interleaved,
        )
    codec.gemm_w2(
        plan, buffers, route_ids, offsets, expert_ids, expert_count, topk_weights
    )
    output = buffers["output"]
    if plan.w2.value in FUSED_W2:
        return output
    if plan.reduction == "triton_single":
        _single_token_weighted_reduce(buffers["sorted_output"], topk_weights, output)
    elif plan.reduction == "triton_batch":
        _mtp_weighted_reduce(buffers["sorted_output"], topk_weights, output)
    elif plan.reduction == "native_weighted":
        ops.awq_moe_single_token_weighted_reduce_out(
            buffers["sorted_output"],
            topk_weights,
            buffers["token_expert_indices"],
            output,
            topk_weights.shape[1],
            codec.dimensions.hidden_size,
        )
    else:
        torch.ops._moe_C.moe_unpermute(
            buffers["sorted_output"],
            topk_weights,
            buffers["inv_permuted_idx"],
            unpermute_offsets,
            topk_weights.shape[1],
            output,
        )
    return output
