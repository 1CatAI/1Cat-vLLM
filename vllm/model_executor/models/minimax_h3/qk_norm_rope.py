# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""D128 Q/K RMSNorm and partial RoPE with explicit FP16 rounding boundaries."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _qk_norm_rope_kernel(
    X,
    W,
    ROPE,
    OUT,
    HEADS: tl.constexpr,
    ROWS: tl.constexpr,
    STRIDE_T: tl.constexpr,
    STRIDE_H: tl.constexpr,
    HALF: tl.constexpr,
    EPS: tl.constexpr,
    BR: tl.constexpr = 4,
    D: tl.constexpr = 128,
):
    row = tl.program_id(0) * BR + tl.arange(0, BR)
    d = tl.arange(0, D)
    token, head = row // HEADS, row % HEADS
    offsets = token[:, None] * STRIDE_T + head[:, None] * STRIDE_H + d[None, :]
    x = tl.load(X + offsets, row[:, None] < ROWS, 0).to(tl.float32)
    weight = tl.load(W + d).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 1) / D + EPS)
    normalized = ((x * inv[:, None]) * weight[None, :]).to(tl.float16)
    partner = tl.where(d < HALF, d + HALF, tl.where(d < HALF * 2, d - HALF, d))
    rotated = tl.gather(normalized, tl.broadcast_to(partner[None, :], (BR, D)), 1)
    rd = d % HALF
    cos = tl.load(
        ROPE + token[:, None] * (HALF * 2) + rd[None, :], row[:, None] < ROWS, 0
    ).to(tl.float16)
    sin = tl.load(
        ROPE + token[:, None] * (HALF * 2) + HALF + rd[None, :],
        row[:, None] < ROWS,
        0,
    ).to(tl.float16)
    # The reference rounds normalized values and each rotary product to FP16
    # before adding/subtracting. Do not contract these into an FP32 FMA.
    first = (normalized.to(tl.float32) * cos.to(tl.float32)).to(tl.float16)
    second = (rotated.to(tl.float32) * sin.to(tl.float32)).to(tl.float16)
    result = tl.where(
        d[None, :] < HALF,
        first.to(tl.float32) - second.to(tl.float32),
        first.to(tl.float32) + second.to(tl.float32),
    ).to(tl.float16)
    result = tl.where(d[None, :] < HALF * 2, result, normalized)
    tl.store(OUT + row[:, None] * D + d[None, :], result, row[:, None] < ROWS)


def qk_norm_rope(q, k, q_weight, k_weight, rope_table, eps):
    outputs = []
    for x, weight in ((q, q_weight), (k, k_weight)):
        output = torch.empty(x.shape, device=x.device, dtype=x.dtype)
        rows = x.shape[0] * x.shape[1]
        _qk_norm_rope_kernel[(triton.cdiv(rows, 4),)](
            x,
            weight,
            rope_table,
            output,
            x.shape[1],
            rows,
            x.stride(0),
            x.stride(1),
            rope_table.shape[-1] // 2,
            eps,
            num_warps=4,
            enable_fp_fusion=False,
        )
        outputs.append(output)
    return tuple(outputs)
