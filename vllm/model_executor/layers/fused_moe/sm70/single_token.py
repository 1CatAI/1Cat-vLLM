# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One active-expert W13/activation/W2/reduce flow shared by AWQ and FP8."""

from collections.abc import Callable
from typing import Any

import torch

from vllm.model_executor.layers.fused_moe.sm70.weight_codec import Sm70MoEWeightCodec
from vllm.model_executor.layers.quantization.sm70_moe_router import (
    Sm70MoeRoutePlan,
)

Observer = Callable[[Any, torch.Tensor, str], torch.Tensor]
Activation = Callable[[Any, torch.Tensor, torch.Tensor], None]


def execute_single_token(
    codec: Sm70MoEWeightCodec,
    plan: Sm70MoeRoutePlan,
    layer: Any,
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    ids: torch.Tensor,
    buffers: dict[str, torch.Tensor],
    group_size: int,
    *,
    activation: Activation | None = None,
    observe: Observer | None = None,
    trim_output: bool = False,
) -> torch.Tensor:
    top_k = ids.shape[1]
    output = buffers["output"]
    codec.log(
        "MoE single-token active-expert dense path enabled (top_k=%d, experts=%d).",
        top_k,
        layer.sm70_num_experts,
    )
    if plan.strict:
        codec.log(
            "MoE batched path using strict single-token decode route (top_k=%d).", top_k
        )
    if plan.batched_indexed:
        codec.log(
            "MoE batched path using single-token indexed dense-stage route (top_k=%d).",
            top_k,
        )
    if plan.w13 == "indexed" or plan.w2 == "indexed":
        codec.log(
            "MoE single-token indexed dense-stage path enabled "
            "(top_k=%d, w13=%s, w2=%s).",
            top_k,
            plan.w13 == "indexed",
            plan.w2 == "indexed",
        )
    if plan.w13 == "compact":
        codec.log(
            "MoE single-token compact grouped W13 path enabled (top_k=%d).", top_k
        )
        codec.gemm_w13(
            "compact",
            buffers["gate_up"],
            buffers["permuted_input"],
            x,
            ids,
            layer.w13_strided_ptrs_w,
            layer.w13_strided_ptrs_s,
            buffers["compact_w13_ptrs_w"],
            buffers["compact_w13_ptrs_s"],
            buffers["expert_offsets"],
            buffers["expert_offsets64"],
            buffers["inv_permuted_idx"],
            buffers["sorted_expert_ids"],
            layer.sm70_w13_k_dim,
            layer.sm70_w13_n_dim,
            group_size,
            layer.sm70_hidden_logical_size,
        )
    else:
        codec.gemm_w13(
            plan.w13,
            buffers["gate_up"],
            buffers["permuted_input"],
            x,
            ids,
            layer.w13_strided_ptrs_w,
            layer.w13_strided_ptrs_s,
            buffers["expert_offsets"],
            buffers["expert_offsets64"],
            buffers["inv_permuted_idx"],
            buffers["sorted_expert_ids"],
            layer.sm70_w13_k_dim,
            layer.sm70_w13_n_dim,
            group_size,
            layer.sm70_hidden_logical_size,
        )
    if observe is not None:
        for name, label in (
            ("expert_offsets", "st_expert_offsets"),
            ("sorted_expert_ids", "st_sorted_expert_ids"),
            ("inv_permuted_idx", "st_inv_permuted_idx"),
            ("gate_up", "st_w13_out"),
        ):
            buffers[name] = observe(layer, buffers[name], label)
    if activation is None:
        torch.ops._C.silu_and_mul(buffers["intermediate"], buffers["gate_up"])
    else:
        activation(layer, buffers["intermediate"], buffers["gate_up"])
    if observe is not None:
        buffers["intermediate"] = observe(layer, buffers["intermediate"], "st_silu_out")
    codec.gemm_w2(
        plan.w2,
        buffers["sorted_output"],
        buffers["intermediate"],
        buffers["expert_offsets"],
        buffers["sorted_expert_ids"],
        layer.w2_strided_ptrs_w,
        layer.w2_strided_ptrs_s,
        top_k,
        layer.sm70_w2_k_dim,
        layer.sm70_w2_n_dim,
        group_size,
    )
    if observe is not None:
        buffers["sorted_output"] = observe(layer, buffers["sorted_output"], "st_w2_out")
    sorted_output = buffers["sorted_output"]
    if trim_output:
        sorted_output = sorted_output[:, : layer.sm70_hidden_logical_size]
    if plan.weighted_reduce:
        codec.log("MoE single-token weighted-reduce path enabled (top_k=%d).", top_k)
        codec.operators.awq_moe_single_token_weighted_reduce_out(
            sorted_output,
            topk_weights,
            buffers["inv_permuted_idx"],
            output,
            top_k,
            layer.sm70_hidden_logical_size,
        )
    else:
        codec.operators.moe_unpermute(
            sorted_output,
            topk_weights,
            buffers["inv_permuted_idx"],
            buffers["expert_offsets64"][: top_k + 1],
            top_k,
            output,
        )
    if observe is not None:
        output = observe(layer, output, "st_output")
    return output
