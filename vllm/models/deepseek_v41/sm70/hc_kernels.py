# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused Triton kernels of the DeepSeek-V4.1 SM70 hyper-connections (common/hc.py; DeepSeek's Mega-mHC, report §2.4.1).

* ``sinkhorn_split``: one program per token turns the 24 HC mix logits into ``pre`` / ``post`` / ``comb`` (sigmoid,
  softmax, 20 Sinkhorn iterations on the 4x4 ``comb``) in registers. The torch reference launches ~80 kernels per call
  (twice per layer), which dominated the graph-mode decode profile on gpubox.

Bit-exact with ``common.hc.sinkhorn_split``: FP32, libdevice ``exp`` (torch's ``expf``), round-to-nearest divisions,
no FMA contraction (torch rounds ``x * scale`` and ``+ base`` separately), and the 4-term sums in torch's order
(``ROW_ORDER`` / ``COL_ORDER``, measured by tests). The 4-term sums add one exactly-extracted element at a time.
"""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton

from ..common.contracts import HC, HC_EPS, HC_SINKHORN_ITERS

MIX_HC = (2 + HC) * HC
# 0: ((a + b) + c) + d   1: (a + b) + (c + d)   2: (a + c) + (b + d)
SUM_ORDERS = (0, 1, 2)
ROW_ORDER = 2   # torch reduces the contiguous last dim as (a + c) + (b + d)
COL_ORDER = 0   # and the strided dim -2 left to right (measured: tests bit-exact, 200k tokens x 8 real sites)


@triton.jit
def _sum4(a, b, c, d, ORDER: tl.constexpr):
    if ORDER == 0:
        return ((a + b) + c) + d
    elif ORDER == 1:
        return (a + b) + (c + d)
    else:
        return (a + c) + (b + d)


@triton.jit
def _pick(x, idx, i: tl.constexpr, axis: tl.constexpr):
    """x[:, i] (axis 1) or x[i, :] (axis 0), exactly: one non-zero term per reduction."""
    return tl.sum(tl.where(idx == i, x, 0.0), axis=axis)


@triton.jit
def _row_sum(c, kk, ORDER: tl.constexpr):   # sum over the last index k -> [4] indexed by j
    return _sum4(_pick(c, kk, 0, 1), _pick(c, kk, 1, 1), _pick(c, kk, 2, 1), _pick(c, kk, 3, 1), ORDER)


@triton.jit
def _col_sum(c, jj, ORDER: tl.constexpr):   # sum over the first index j -> [4] indexed by k
    return _sum4(_pick(c, jj, 0, 0), _pick(c, jj, 1, 0), _pick(c, jj, 2, 0), _pick(c, jj, 3, 0), ORDER)


@triton.jit
def _sigmoid(x):
    return tl.div_rn(1.0, 1.0 + tl.extra.cuda.libdevice.exp(-x))


@triton.jit
def _sinkhorn_split_kernel(mix_ptr, scale_ptr, base_ptr, pre_ptr, post_ptr, comb_ptr, eps, ITERS: tl.constexpr,
                           ROW: tl.constexpr, COL: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    i4 = tl.arange(0, 4)
    s0 = tl.load(scale_ptr)
    s1 = tl.load(scale_ptr + 1)
    s2 = tl.load(scale_ptr + 2)
    pre = _sigmoid(tl.load(mix_ptr + t * 24 + i4) * s0 + tl.load(base_ptr + i4)) + eps
    post = 2.0 * _sigmoid(tl.load(mix_ptr + t * 24 + 4 + i4) * s1 + tl.load(base_ptr + 4 + i4))
    jj = i4[:, None] + 0 * i4[None, :]
    kk = 0 * i4[:, None] + i4[None, :]
    flat = i4[:, None] * 4 + i4[None, :]
    c = tl.load(mix_ptr + t * 24 + 8 + flat) * s2 + tl.load(base_ptr + 8 + flat)
    # softmax over k, + eps
    mx = tl.max(c, axis=1)
    c = tl.extra.cuda.libdevice.exp(c - mx[:, None])
    c = tl.div_rn(c, _row_sum(c, kk, ROW)[:, None]) + eps
    c = tl.div_rn(c, (_col_sum(c, jj, COL) + eps)[None, :])
    for _ in tl.static_range(ITERS - 1):
        c = tl.div_rn(c, (_row_sum(c, kk, ROW) + eps)[:, None])
        c = tl.div_rn(c, (_col_sum(c, jj, COL) + eps)[None, :])
    tl.store(pre_ptr + t * 4 + i4, pre)
    tl.store(post_ptr + t * 4 + i4, post)
    tl.store(comb_ptr + t * 16 + flat, c)


def sinkhorn_split(mixes: torch.Tensor, scale: torch.Tensor, base: torch.Tensor, iters: int = HC_SINKHORN_ITERS,
                   eps: float = HC_EPS, *, row_order: int = ROW_ORDER, col_order: int = COL_ORDER
                   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """mixes [T,24] f32 -> pre [T,4], post [T,4], comb [T,4,4] f32; same contract as ``common.hc.sinkhorn_split``."""
    if mixes.dtype != torch.float32 or mixes.dim() != 2 or mixes.shape[1] != MIX_HC or not mixes.is_contiguous():
        raise TypeError(f"sinkhorn_split needs contiguous f32 [T, {MIX_HC}] mixes, got {mixes.dtype} {tuple(mixes.shape)}")
    num_tokens = mixes.shape[0]
    pre = torch.empty((num_tokens, HC), dtype=torch.float32, device=mixes.device)
    post = torch.empty((num_tokens, HC), dtype=torch.float32, device=mixes.device)
    comb = torch.empty((num_tokens, HC, HC), dtype=torch.float32, device=mixes.device)
    if num_tokens:
        _sinkhorn_split_kernel[(num_tokens,)](
            mixes, scale.float().contiguous(), base.float().contiguous(), pre, post, comb, float(eps),
            ITERS=int(iters), ROW=row_order, COL=col_order, num_warps=1, enable_fp_fusion=False)
    return pre, post, comb
