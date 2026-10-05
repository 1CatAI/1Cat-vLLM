# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark-only INT8 expert weights with FP16 operands and FP32 sums."""

import torch


@torch.inference_mode()
def prepare_int8_expert_weight(weight, *, block32=False):
    """Pack row or input-block-scaled integers into the 32-by-16 MMA layout.

    Zero rows use scale one. Startup FP32 scratch is bounded to 4096 rows.
    This helper installs no serving dispatch and never replaces FP16 weights.
    """
    if weight.ndim != 2 or weight.dtype != torch.float16:
        raise ValueError("Expected a two-dimensional FP16 weight")
    rows, width = weight.shape
    if rows % 32 or width % (32 if block32 else 16):
        raise ValueError("Expert weight must align to 32 output and 16 input rows")
    integer = torch.empty_like(weight, dtype=torch.int8)
    groups = width // 32 if block32 else 1
    scales = weight.new_empty((groups, rows)) if block32 else weight.new_empty(rows)
    for begin in range(0, rows, 4096):
        values = weight[begin : begin + 4096].float()
        if block32:
            values = values.reshape(values.shape[0], groups, 32)
        scale = values.abs().amax(-1, keepdim=True) / 127
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        integer[begin : begin + 4096].copy_(
            (values / scale).round().clamp(-127, 127).to(torch.int8).reshape(-1, width)
        )
        if block32:
            scales[:, begin : begin + 4096].copy_(scale.squeeze(-1).t())
        else:
            scales[begin : begin + 4096].copy_(scale.flatten())
    lane = torch.arange(32, device=weight.device)
    columns = ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) >> 2)
    physical = torch.tensor(
        [0, 2, 4, 6, 1, 3, 5, 7, 8, 10, 12, 14, 9, 11, 13, 15],
        device=weight.device,
    )
    packed = integer.reshape(rows // 32, 32, width // 16, 16)
    packed = packed.index_select(1, columns).index_select(3, physical)
    return packed.permute(0, 2, 1, 3).contiguous().view(torch.uint8), scales


def packed_int8_fallback_experts(
    x, codes13, scales13, codes2, scales2, topk_weights, topk_ids
):
    """Research-only larger-batch fallback reading the native INT8 store.

    FP16 activations and block32 dequantization use the existing W8A16 Triton
    implementation and its configuration selection. No serving dispatch or
    replacement checkpoint parameter is installed by this helper.
    """
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        invoke_fused_moe_wna16_triton_kernel,
        try_get_optimal_moe_config,
    )
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )
    from vllm.triton_utils import tl

    if x.dtype != torch.float16 or x.ndim != 2 or x.shape[1] != 2560:
        raise ValueError("Packed draft experts require FP16 [M,2560] activations")
    experts = scales13.shape[1]
    w13 = codes13.view(torch.int8).view(experts, 320, 2560)
    w2 = codes2.view(torch.int8).view(experts, 2560, 160)
    if scales13.shape != (80, experts, 320) or scales2.shape != (5, experts, 2560):
        raise ValueError("Expected input-block-32 draft expert scales")
    if topk_ids.shape != topk_weights.shape or topk_ids.shape != (x.shape[0], 10):
        raise ValueError("Packed draft experts require ten routes per token")
    config = try_get_optimal_moe_config(
        w13.shape, w2.shape, 10, "int8_w8a16", x.shape[0], block_shape=[0, 32]
    )
    sorted_ids, expert_ids, total = moe_align_block_size(
        topk_ids, config["BLOCK_SIZE_M"], experts, ignore_invalid_experts=True
    )
    m = x.shape[0]
    # Reuse the same workspace lifetime as the existing fused-experts path.
    cache = x.new_empty(m * 10 * 2560)
    up = cache[: m * 10 * 320].view(m, 10, 320)
    down = cache.view(m, 10, 2560)
    activation = x.new_empty(m * 10, 160)
    invoke_fused_moe_wna16_triton_kernel(
        x,
        w13,
        up,
        scales13.permute(1, 2, 0),
        None,
        topk_weights,
        sorted_ids,
        expert_ids,
        total,
        False,
        10,
        config,
        tl.float16,
        True,
        False,
        [0, 32],
        packed_int8=True,
    )
    torch.ops._C.silu_and_mul(activation, up.view(-1, 320))
    invoke_fused_moe_wna16_triton_kernel(
        activation,
        w2,
        down,
        scales2.permute(1, 2, 0),
        None,
        topk_weights,
        sorted_ids,
        expert_ids,
        total,
        True,
        1,
        config,
        tl.float16,
        True,
        False,
        [0, 32],
        packed_int8=True,
    )
    output = torch.empty_like(x)
    ops.moe_sum(down, output)
    return output
