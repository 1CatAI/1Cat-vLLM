# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One-launch FP32 RMS arithmetic for V4.1's short FP16 rows."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _short_rms(x, weight, out, H: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    at = tl.arange(0, BLOCK)
    value = tl.load(x + row * H + at, at < H, 0).to(tl.float32)
    variance = tl.sum(value * value, 0) / H
    result = value * tl.rsqrt(variance + EPS) * tl.load(weight + at, at < H, 0)
    tl.store(out + row * H + at, result, at < H)


def short_rms_norm(x, weight, eps):
    if (
        x.ndim != 2
        or not x.is_cuda
        or x.dtype != torch.float16
        or not 1 <= x.shape[0] <= 6
        or x.shape[1] not in (128, 512, 1280, 5120)
        or not x.is_contiguous()
        or weight.shape != (x.shape[1],)
        or weight.dtype != torch.float32
        or weight.device != x.device
        or not weight.is_contiguous()
    ):
        raise ValueError("V4.1 short RMS requires FP16 M1..6 and FP32 weights")
    output = torch.empty_like(x)
    _short_rms[(x.shape[0],)](
        x,
        weight,
        output,
        H=x.shape[1],
        EPS=eps,
        BLOCK=triton.next_power_of_2(x.shape[1]),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output
