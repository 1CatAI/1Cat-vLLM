# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ordered FP32 reductions, retaining the native NVFP4 launch contracts."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _single_token_weighted_reduce_kernel(
    expert_output_ptr,
    topk_weights_ptr,
    output_ptr,
    HIDDEN: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < HIDDEN
    acc = tl.zeros((BLOCK,), tl.float32)
    for slot in tl.static_range(0, TOP_K):
        values = tl.load(
            expert_output_ptr + slot * HIDDEN + offsets,
            mask=mask,
            other=0.0,
        )
        weight = tl.load(topk_weights_ptr + slot)
        acc += values.to(tl.float32) * weight
    tl.store(output_ptr + offsets, acc, mask=mask)


def _single_token_weighted_reduce(
    expert_output: torch.Tensor,
    topk_weights: torch.Tensor,
    output: torch.Tensor,
) -> None:
    top_k, hidden = expert_output.shape
    if tuple(topk_weights.shape) != (1, top_k) or tuple(output.shape) != (1, hidden):
        raise ValueError("SM70 NVFP4 direct weighted-reduce shape mismatch.")
    block = 256
    _single_token_weighted_reduce_kernel[(triton.cdiv(hidden, block),)](
        expert_output,
        topk_weights,
        output,
        HIDDEN=hidden,
        TOP_K=top_k,
        BLOCK=block,
        num_warps=4,
    )


@triton.jit
def _mtp_weighted_reduce_kernel(
    expert_output_ptr,
    topk_weights_ptr,
    output_ptr,
    HIDDEN: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < HIDDEN
    acc = tl.zeros((BLOCK,), tl.float32)
    for slot in tl.static_range(0, TOP_K):
        route = token * TOP_K + slot
        values = tl.load(
            expert_output_ptr + route * HIDDEN + offsets,
            mask=mask,
            other=0.0,
        )
        weight = tl.load(topk_weights_ptr + route)
        acc += values.to(tl.float32) * weight
    tl.store(output_ptr + token * HIDDEN + offsets, acc, mask=mask)


def _mtp_weighted_reduce(
    expert_output: torch.Tensor,
    topk_weights: torch.Tensor,
    output: torch.Tensor,
) -> None:
    tokens, top_k = topk_weights.shape
    hidden = expert_output.shape[1]
    if tuple(expert_output.shape) != (tokens * top_k, hidden):
        raise ValueError("SM70 NVFP4 MTP direct expert-output shape mismatch.")
    if tuple(output.shape) != (tokens, hidden):
        raise ValueError("SM70 NVFP4 MTP direct weighted-reduce shape mismatch.")
    block = 256
    _mtp_weighted_reduce_kernel[(tokens, triton.cdiv(hidden, block))](
        expert_output,
        topk_weights,
        output,
        HIDDEN=hidden,
        TOP_K=top_k,
        BLOCK=block,
        num_warps=4,
    )


def weighted_reduce_rows(
    expert_rows: torch.Tensor,
    topk_weights: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """GGUF/skinny slot-major reduction with their existing FP32 semantics.

    Keep multiply before sum and cast only the final sum; this is deliberately
    separate from the ordered Triton and FP16 per-expert index-add paths.
    """
    return (expert_rows.float() * topk_weights[..., None].float()).sum(1).to(dtype)
