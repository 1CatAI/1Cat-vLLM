# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepare H3's wide-range gated activation without an FP32 temporary."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _silu_prepare_kernel(H, Y, S, K: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    gate = tl.load(H + row * (2 * K) + cols, cols < K, other=0).to(tl.float32)
    up = tl.load(H + row * (2 * K) + K + cols, cols < K, other=0).to(tl.float32)
    value = (gate * (1.0 / (1.0 + tl.exp(-gate)))) * up
    magnitude = tl.abs(value)
    magnitude = tl.where(magnitude != magnitude, float("inf"), magnitude)
    maximum = tl.max(magnitude, 0)
    exponent = (maximum.to(tl.int32, bitcast=True) >> 23) & 255
    scale_exp = tl.where(exponent == 255, 0, tl.maximum(exponent - 137, 0))
    scale = ((scale_exp + 127) << 23).to(tl.float32, bitcast=True)
    tl.store(Y + row * K + cols, value / scale, cols < K)
    tl.store(S + row, scale)


def silu_prepare_fp16(hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Preserve FP32 SiLU/product and power-of-two scaling before rounding.

    The scale leaves the same ConvRot256 headroom as ``prepare_fp16``. Disable
    FMA contraction to retain the established SiLU arithmetic order.
    """
    if not hidden.is_cuda or hidden.dtype != torch.float16:
        raise ValueError("H3 fused SiLU preparation requires CUDA FP16 input")
    if hidden.ndim < 2 or hidden.shape[-1] % 2 or hidden.numel() == 0:
        raise ValueError("H3 fused SiLU requires non-empty [gate, up] rows")
    flat = hidden.reshape(-1, hidden.shape[-1]).contiguous()
    rows, width = flat.shape
    channels = width // 2
    if channels > 16384:
        raise ValueError("H3 fused SiLU supports at most 16384 channels")
    values = torch.empty((rows, channels), dtype=torch.float16, device=hidden.device)
    scale = torch.empty((rows, 1), dtype=torch.float32, device=hidden.device)
    _silu_prepare_kernel[(rows,)](
        flat,
        values,
        scale,
        channels,
        triton.next_power_of_2(channels),
        num_warps=8,
        enable_fp_fusion=False,
    )
    return values.reshape(*hidden.shape[:-1], channels), scale
