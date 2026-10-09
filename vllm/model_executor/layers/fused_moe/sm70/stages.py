# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared ordered stages for routed TurboMind MoE.

Routing metadata and weights are borrowed from their existing layer owners.
There are no policy reads or allocations in the common stage dispatcher.
"""

from typing import Any

import torch

from vllm import _sm70_ops as ops
from vllm.model_executor.layers.fused_moe.sm70.single_token import Activation
from vllm.model_executor.layers.fused_moe.sm70.weight_codec import Sm70MoEWeightCodec
from vllm.model_executor.layers.quantization.sm70_moe_router import Sm70MoeRoutePlan


def execute_gemm(
    codec: Sm70MoEWeightCodec,
    stage: str,
    mode: str,
    layer: Any,
    buffers: dict[str, torch.Tensor],
    group_size: int,
    dense_ids: torch.Tensor,
    x: torch.Tensor,
) -> None:
    w13 = stage == "w13"
    out = buffers["gate_up" if w13 else "sorted_output"]
    inp = buffers["permuted_input" if w13 else "intermediate"]
    weights = getattr(layer, stage + "_strided_ptrs_w")
    scales = getattr(layer, stage + "_strided_ptrs_s")
    k = getattr(layer, "sm70_" + stage + "_k_dim")
    n = getattr(layer, "sm70_" + stage + "_n_dim")
    gemm = codec.gemm_w13 if w13 else codec.gemm_w2
    if mode == "indexed_prefill":
        gemm(
            mode,
            out,
            x,
            buffers["input_row_indices"],
            buffers["expert_offsets"],
            dense_ids,
            weights,
            scales,
            layer.sm70_num_experts,
            k,
            n,
            group_size,
        )
    elif mode == "active_grouped":
        gemm(
            mode,
            out,
            inp,
            buffers["permuted_experts_id"],
            buffers["active_expert_offsets"],
            buffers["sorted_expert_ids"],
            weights,
            scales,
            out.shape[0],
            k,
            n,
            group_size,
        )
    elif mode in ("batched", "per_expert_dispatch"):
        if mode == "per_expert_dispatch":
            codec.log(
                "MoE batched %s using per-expert dispatch selection (experts=%d).",
                stage.upper(),
                layer.sm70_num_experts,
            )
        gemm(
            mode,
            out,
            inp,
            buffers["expert_offsets"],
            weights,
            scales,
            layer.sm70_num_experts,
            k,
            n,
            group_size,
            False,
        )
    else:
        if w13:
            codec.log(
                "MoE CUDA-graph-safe dense-stage path enabled (experts=%d).",
                layer.sm70_num_experts,
            )
        gemm(
            "dense",
            out,
            inp,
            buffers["expert_offsets"],
            dense_ids,
            weights,
            scales,
            layer.sm70_num_experts,
            k,
            n,
            group_size,
        )


def execute_routed(
    codec: Sm70MoEWeightCodec,
    plan: Sm70MoeRoutePlan,
    layer: Any,
    buffers: dict[str, torch.Tensor],
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    group_size: int,
    dense_ids: torch.Tensor,
    *,
    activation: Activation | None = None,
    observer: Any = None,
    trim_output: bool = False,
) -> torch.Tensor:
    """W13 → activation → W2 → weighted reduction, also for fused stages."""
    if plan.use_batched_strict_w13:
        codec.log(
            "MoE batched path using strict dense-stage for multi-token "
            "shapes (experts=%d).",
            layer.sm70_num_experts,
        )
    execute_gemm(codec, "w13", plan.w13, layer, buffers, group_size, dense_ids, x)
    if observer is not None:
        observer.after_w13(buffers)
    if activation is None:
        torch.ops._C.silu_and_mul(buffers["intermediate"], buffers["gate_up"])
    else:
        activation(layer, buffers["intermediate"], buffers["gate_up"])
    if observer is not None:
        observer.after_activation(buffers)
    if plan.w2 == "chunked":
        ops.awq_moe_chunked_w2_sm70_out(
            buffers["output"],
            buffers["sorted_output"],
            buffers["intermediate"],
            buffers["expert_offsets"],
            buffers["permuted_idx"],
            topk_weights,
            buffers["chunk_expert_offsets"],
            buffers["chunk_range_begin"],
            buffers["chunk_range_end"],
            buffers["chunk_a_indices"],
            buffers["chunk_inv_permuted_idx"],
            layer.w2_strided_ptrs_w,
            layer.w2_strided_ptrs_s,
            x.shape[0],
            topk_weights.shape[1],
            layer.sm70_num_experts,
            layer.sm70_w2_k_dim,
            layer.sm70_w2_n_dim,
            layer.sm70_hidden_logical_size,
            group_size,
            plan.chunk_tokens,
        )
        return (
            observer.finish(buffers["output"], chunked=True)
            if observer is not None
            else buffers["output"]
        )
    if plan.use_batched_exact_w2 and plan.w2 == "dense":
        codec.log(
            "MoE batched path using exact dense-stage W2 (experts=%d).",
            layer.sm70_num_experts,
        )
    if plan.w2 == "active_grouped":
        codec.log(
            "MoE batched path using grouped-active exact W2 (routes=%d).",
            topk_weights.numel(),
        )
    execute_gemm(codec, "w2", plan.w2, layer, buffers, group_size, dense_ids, x)
    if observer is not None:
        observer.after_w2(buffers)
    sorted_output = buffers["sorted_output"]
    if trim_output:
        sorted_output = sorted_output[:, : layer.sm70_hidden_logical_size]
    torch.ops._moe_C.moe_unpermute(
        sorted_output,
        topk_weights,
        buffers["inv_permuted_idx"],
        buffers["expert_offsets64"],
        topk_weights.shape[1],
        buffers["output"],
    )
    return (
        observer.finish(buffers["output"])
        if observer is not None
        else buffers["output"]
    )
