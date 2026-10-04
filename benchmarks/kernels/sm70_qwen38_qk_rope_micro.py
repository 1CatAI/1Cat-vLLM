# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Research kernel for the gated Qwen3.5 6Q/1K, D256, rotary64 verifier.

No model dispatch is connected until cold whole-layer and route admission.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _qk_norm_rope(
    QKV,
    QW,
    KW,
    Pos,
    Cache,
    QOut,
    KOut,
    ROW: tl.constexpr,
    POS_PLANES: tl.constexpr,
    EPS: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    col = tl.arange(0, 256)
    if head < 6:
        source = QKV + token * ROW + head * 512
        weight = QW
        destination = QOut + token * 1536 + head * 256
    else:
        source = QKV + token * ROW + 3072
        weight = KW
        destination = KOut + token * 256
    values = tl.load(source + col).to(tl.float32)
    variance = tl.sum(values * values, 0) / 256.0
    inverse = tl.rsqrt(variance + EPS)
    w = tl.load(weight + col).to(tl.float32) + 1.0
    normalized = (values * inverse * w).to(tl.float16)
    # Reload only the rotary partner. Its norm shares the same variance,
    # while its learned channel weight can differ.
    partner = tl.where(col < 32, col + 32, tl.where(col < 64, col - 32, col))
    pv = tl.load(source + partner).to(tl.float32)
    pw = tl.load(weight + partner).to(tl.float32) + 1.0
    pn = (pv * inverse * pw).to(tl.float16)
    frequency = col % 32
    plane = tl.full((256,), 0, tl.int32)
    if POS_PLANES == 3:
        plane = tl.where((frequency % 3 == 1) & (frequency < 33), 1, plane)
        plane = tl.where((frequency % 3 == 2) & (frequency < 30), 2, plane)
    position = tl.load(Pos + plane * 8 + token)
    cosine = tl.load(Cache + position * 64 + frequency)
    sine = tl.load(Cache + position * 64 + 32 + frequency)
    # Preserve the native FP16 multiplication boundaries; neither head nor
    # drafter precision is changed by this preprocessing experiment.
    first = (normalized.to(tl.float32) * cosine.to(tl.float32)).to(tl.float16)
    second = (pn.to(tl.float32) * sine.to(tl.float32)).to(tl.float16)
    sign = tl.where(col < 32, -1.0, 1.0)
    rotated = (first.to(tl.float32) + sign * second.to(tl.float32)).to(tl.float16)
    tl.store(destination + col, tl.where(col < 64, rotated, normalized))


def qk_norm_rope(qkv, q_weight, k_weight, positions, cache, eps=1e-6):
    assert qkv.shape == (8, 3584) and qkv.dtype == torch.float16
    assert qkv.stride(1) == 1
    assert q_weight.shape == k_weight.shape == (256,)
    assert q_weight.is_contiguous() and k_weight.is_contiguous()
    assert cache.ndim == 2 and cache.shape[1] == 64 and cache.is_contiguous()
    assert cache.dtype == torch.float16
    assert positions.shape in ((8,), (3, 8)) and positions.is_contiguous()
    assert positions.dtype == torch.int64
    q = torch.empty((8, 1536), device=qkv.device, dtype=qkv.dtype)
    k = torch.empty((8, 256), device=qkv.device, dtype=qkv.dtype)
    _qk_norm_rope[(8, 7)](
        qkv,
        q_weight,
        k_weight,
        positions,
        cache,
        q,
        k,
        qkv.stride(0),
        1 if positions.ndim == 1 else 3,
        eps,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return q, k
