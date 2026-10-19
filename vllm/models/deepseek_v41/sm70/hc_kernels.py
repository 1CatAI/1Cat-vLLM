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


# =====================================================================================
# fused hyper-connection kernels (gpubox lane): hc_pre, hc_post, rmsnorm_to_act, hc_mixes
# =====================================================================================
#
# ``hc_pre`` and ``hc_post`` are pure 4-term combinations and reproduce torch bit for bit (summation order and
# FMA use measured by tests, as for the Sinkhorn split). ``rmsnorm_to_act`` and ``hc_mixes`` replace a torch
# reduction over 5120 / 20480 values (and a cuBLAS GEMV) with a blocked FP32 reduction, so they are exact to
# ~1 ulp of FP32 but not bit-identical; they are gated by the KL panel, not the greedy byte check.

HIDDEN = tl.constexpr(5120)
HC_PRE_ORDER = 0     # torch.sum over the strided dim of size 4 is sequential (measured: bit-exact)
HC_POST_PAIRWISE = False   # cuBLAS K=4 bmm: (c0r0 + c1r1) + (c2r2 + c3r3) vs sequential; measured by tests
HC_POST_BMM_FMA = True     # products accumulated with FMAs
HC_POST_ADD_FMA = False    # addcmul(mixed, post, x): product rounded, then added (measured: bit-exact with torch)
_BLOCK_H = 1024


@triton.jit
def _bf16_to_f32(v):
    """BF16 bits (int16 tensor) -> FP32 value: sign-extend, shift into the high half, bitcast."""
    return (v.to(tl.int32) << 16).to(tl.float32, bitcast=True)


@triton.jit
def _bf16_rne(x):
    """FP32 -> BF16 bits (round to nearest even), done in integer arithmetic (no native cvt on SM70)."""
    bits = x.to(tl.int32, bitcast=True)
    lsb = (bits >> 16) & 1
    rounded = bits + 0x7FFF + lsb
    nan = x != x
    return tl.where(nan, (bits >> 16) | 0x40, rounded >> 16).to(tl.int16)


@triton.jit
def _hc_pre_kernel(stream_ptr, pre_ptr, out_ptr, ORDER: tl.constexpr, BLOCK: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    h0 = tl.program_id(1) * BLOCK
    d = h0 + tl.arange(0, BLOCK)
    mask = d < HIDDEN
    base = stream_ptr + t * 4 * HIDDEN
    s0 = _bf16_to_f32(tl.load(base + d, mask=mask, other=0))
    s1 = _bf16_to_f32(tl.load(base + HIDDEN + d, mask=mask, other=0))
    s2 = _bf16_to_f32(tl.load(base + 2 * HIDDEN + d, mask=mask, other=0))
    s3 = _bf16_to_f32(tl.load(base + 3 * HIDDEN + d, mask=mask, other=0))
    p0 = tl.load(pre_ptr + t * 4)
    p1 = tl.load(pre_ptr + t * 4 + 1)
    p2 = tl.load(pre_ptr + t * 4 + 2)
    p3 = tl.load(pre_ptr + t * 4 + 3)
    a = p0 * s0
    b = p1 * s1
    c = p2 * s2
    e = p3 * s3
    if ORDER == 0:
        y = ((a + b) + c) + e
    elif ORDER == 1:
        y = (a + b) + (c + e)
    else:
        y = (a + c) + (b + e)
    tl.store(out_ptr + t * HIDDEN + d, _bf16_rne(y), mask=mask)


def hc_pre_fused(stream: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
    T = stream.shape[0]
    out = torch.empty((T, HIDDEN), dtype=torch.bfloat16, device=stream.device)
    if T:
        _hc_pre_kernel[(T, triton.cdiv(HIDDEN, _BLOCK_H))](
            stream.view(torch.int16), pre.contiguous(), out.view(torch.int16), ORDER=HC_PRE_ORDER, BLOCK=_BLOCK_H,
            num_warps=4, enable_fp_fusion=False)
    return out


@triton.jit
def _hc_post_kernel(x_ptr, res_ptr, post_ptr, comb_ptr, out_ptr, PAIRWISE: tl.constexpr, BMM_FMA: tl.constexpr,
                    ADD_FMA: tl.constexpr, BLOCK: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    h0 = tl.program_id(1) * BLOCK
    d = h0 + tl.arange(0, BLOCK)
    mask = d < HIDDEN
    base = res_ptr + t * 4 * HIDDEN
    r0 = _bf16_to_f32(tl.load(base + d, mask=mask, other=0))
    r1 = _bf16_to_f32(tl.load(base + HIDDEN + d, mask=mask, other=0))
    r2 = _bf16_to_f32(tl.load(base + 2 * HIDDEN + d, mask=mask, other=0))
    r3 = _bf16_to_f32(tl.load(base + 3 * HIDDEN + d, mask=mask, other=0))
    x = tl.load(x_ptr + t * HIDDEN + d, mask=mask, other=0.0)
    for c in tl.static_range(4):
        # comb[t, j, c]: the FIRST comb index contracts
        c0 = tl.load(comb_ptr + t * 16 + 0 * 4 + c)
        c1 = tl.load(comb_ptr + t * 16 + 1 * 4 + c)
        c2 = tl.load(comb_ptr + t * 16 + 2 * 4 + c)
        c3 = tl.load(comb_ptr + t * 16 + 3 * 4 + c)
        p = tl.load(post_ptr + t * 4 + c)
        if PAIRWISE:
            if BMM_FMA:
                acc = tl.fma(c1, r1, c0 * r0) + tl.fma(c3, r3, c2 * r2)
            else:
                acc = (c0 * r0 + c1 * r1) + (c2 * r2 + c3 * r3)
        else:
            if BMM_FMA:
                acc = tl.fma(c3, r3, tl.fma(c2, r2, tl.fma(c1, r1, c0 * r0)))
            else:
                acc = ((c0 * r0 + c1 * r1) + c2 * r2) + c3 * r3
        if ADD_FMA:
            y = tl.fma(p, x, acc)
        else:
            y = acc + p * x
        tl.store(out_ptr + t * 4 * HIDDEN + c * HIDDEN + d, _bf16_rne(y), mask=mask)


def hc_post_fused(x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor) -> torch.Tensor:
    T = residual.shape[0]
    out = torch.empty_like(residual)
    if T:
        _hc_post_kernel[(T, triton.cdiv(HIDDEN, _BLOCK_H))](
            x.float().contiguous(), residual.view(torch.int16), post.contiguous(), comb.contiguous(),
            out.view(torch.int16), PAIRWISE=HC_POST_PAIRWISE, BMM_FMA=HC_POST_BMM_FMA, ADD_FMA=HC_POST_ADD_FMA,
            BLOCK=_BLOCK_H, num_warps=4, enable_fp_fusion=False)
    return out


@triton.jit
def _rmsnorm_act_kernel(x_ptr, w_ptr, out_ptr, eps, X_BF16: tl.constexpr, BLOCK: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for h0 in tl.static_range(0, HIDDEN, BLOCK):
        d = h0 + tl.arange(0, BLOCK)
        if X_BF16:
            x = _bf16_to_f32(tl.load(x_ptr + t * HIDDEN + d))
        else:
            x = tl.load(x_ptr + t * HIDDEN + d)
        acc += x * x
    var = tl.sum(acc, axis=0) / HIDDEN
    r = tl.math.rsqrt(var + eps)
    for h0 in tl.static_range(0, HIDDEN, BLOCK):
        d = h0 + tl.arange(0, BLOCK)
        if X_BF16:
            x = _bf16_to_f32(tl.load(x_ptr + t * HIDDEN + d))
        else:
            x = tl.load(x_ptr + t * HIDDEN + d)
        w = tl.load(w_ptr + d)
        tl.store(out_ptr + t * HIDDEN + d, (w * (x * r)).to(tl.float16))


@triton.jit
def _normalize_act_kernel(x_ptr, r_ptr, w_ptr, out_ptr, X_BF16: tl.constexpr, BLOCK: tl.constexpr):
    """out[t] = (w * (x[t] * r[t])).to(fp16) with torch's two fp32 roundings (no FMA / fusion)."""
    t = tl.program_id(0).to(tl.int64)
    r = tl.load(r_ptr + t)
    for h0 in tl.static_range(0, HIDDEN, BLOCK):
        d = h0 + tl.arange(0, BLOCK)
        if X_BF16:
            x = _bf16_to_f32(tl.load(x_ptr + t * HIDDEN + d))
        else:
            x = tl.load(x_ptr + t * HIDDEN + d)
        w = tl.load(w_ptr + d)
        tl.store(out_ptr + t * HIDDEN + d, (w * (x * r)).to(tl.float16))


def rmsnorm_to_act_exact(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Bit-identical to the torch rmsnorm_to_act: variance + rsqrt by torch (same reduction kernel, same rounding),
    the normalize + weight + fp16 cast as one kernel (6 launches vs 8; the fully fused kernel's own variance sum
    differs in the last bit, which flips fp16 outputs and shifts the KL panel by ~0.013)."""
    T = x.shape[0]
    out = torch.empty((T, HIDDEN), dtype=torch.float16, device=x.device)
    if T:
        x32 = x.float()
        r = torch.rsqrt(x32.square().mean(-1) + eps)                          # [T] fp32, torch's exact values
        bf16 = x.dtype == torch.bfloat16
        xin = x.view(torch.int16) if bf16 else x32
        _normalize_act_kernel[(T,)](xin.contiguous(), r.contiguous(), weight.float().contiguous(), out, X_BF16=bf16,
                                    BLOCK=_BLOCK_H, num_warps=4, enable_fp_fusion=False)
    return out


def rmsnorm_to_act_fused(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    T = x.shape[0]
    out = torch.empty((T, HIDDEN), dtype=torch.float16, device=x.device)
    if T:
        bf16 = x.dtype == torch.bfloat16
        xin = x.view(torch.int16) if bf16 else x.float()
        _rmsnorm_act_kernel[(T,)](xin.contiguous(), weight.float().contiguous(), out, float(eps), X_BF16=bf16,
                                  BLOCK=_BLOCK_H, num_warps=4, enable_fp_fusion=False)
    return out


MIX_CHUNK = 1024
MIX_NCHUNK = tl.constexpr(4 * 5120 // MIX_CHUNK)     # 20 programs per token


@triton.jit
def _hc_mixes_partial_kernel(stream_ptr, fn_ptr, part_ptr, BLOCK: tl.constexpr):
    """Partial sums per (token, chunk): part[t, ch, 0..23] = x_chunk . fn[m]_chunk, part[t, ch, 24] = sum x_chunk^2."""
    t = tl.program_id(0).to(tl.int64)
    ch = tl.program_id(1)
    N: tl.constexpr = 4 * HIDDEN
    d = ch * BLOCK + tl.arange(0, BLOCK)
    x = _bf16_to_f32(tl.load(stream_ptr + t * N + d))
    m = tl.arange(0, 32)
    mmask = m < 24
    f = tl.load(fn_ptr + m[:, None].to(tl.int64) * N + d[None, :], mask=mmask[:, None], other=0.0)
    dots = tl.sum(f * x[None, :], axis=1)                       # [32]
    tl.store(part_ptr + (t * MIX_NCHUNK + ch) * 32 + m, dots, mask=mmask)
    tl.store(part_ptr + (t * MIX_NCHUNK + ch) * 32 + 24, tl.sum(x * x, axis=0))


@triton.jit
def _hc_mixes_finish_sinkhorn_kernel(part_ptr, scale_ptr, base_ptr, pre_ptr, post_ptr, comb_ptr, eps_hc, eps_norm,
                                     ITERS: tl.constexpr, ROW: tl.constexpr, COL: tl.constexpr):
    """Reduce the MIX_NCHUNK partials of one token in fixed order, apply rsqrt(mean(x^2) + eps), then the Sinkhorn
    split (same math as _sinkhorn_split_kernel)."""
    t = tl.program_id(0).to(tl.int64)
    N: tl.constexpr = 4 * HIDDEN
    m = tl.arange(0, 32)
    acc = tl.zeros((32,), dtype=tl.float32)
    for ch in tl.static_range(MIX_NCHUNK):
        acc += tl.load(part_ptr + (t * MIX_NCHUNK + ch) * 32 + m)
    sumsq = tl.sum(tl.where(m == 24, acc, 0.0), axis=0)
    r = tl.math.rsqrt(sumsq / N + eps_norm)
    mix = acc * r                                                # [32], rows 0..23 valid
    i4 = tl.arange(0, 4)
    s0 = tl.load(scale_ptr)
    s1 = tl.load(scale_ptr + 1)
    s2 = tl.load(scale_ptr + 2)
    mix_pre = tl.sum(tl.where(m[None, :] == i4[:, None], mix[None, :], 0.0), axis=1)
    mix_post = tl.sum(tl.where(m[None, :] == 4 + i4[:, None], mix[None, :], 0.0), axis=1)
    pre = _sigmoid(mix_pre * s0 + tl.load(base_ptr + i4)) + eps_hc
    post = 2.0 * _sigmoid(mix_post * s1 + tl.load(base_ptr + 4 + i4))
    jj = i4[:, None] + 0 * i4[None, :]
    kk = 0 * i4[:, None] + i4[None, :]
    flat = i4[:, None] * 4 + i4[None, :]
    mix_c = tl.sum(tl.where(m[None, None, :] == 8 + flat[:, :, None], mix[None, None, :], 0.0), axis=2)
    c = mix_c * s2 + tl.load(base_ptr + 8 + flat)
    mx = tl.max(c, axis=1)
    c = tl.extra.cuda.libdevice.exp(c - mx[:, None])
    c = tl.div_rn(c, _row_sum(c, kk, ROW)[:, None]) + eps_hc
    c = tl.div_rn(c, (_col_sum(c, jj, COL) + eps_hc)[None, :])
    for _ in tl.static_range(ITERS - 1):
        c = tl.div_rn(c, (_row_sum(c, kk, ROW) + eps_hc)[:, None])
        c = tl.div_rn(c, (_col_sum(c, jj, COL) + eps_hc)[None, :])
    tl.store(pre_ptr + t * 4 + i4, pre)
    tl.store(post_ptr + t * 4 + i4, post)
    tl.store(comb_ptr + t * 16 + flat, c)


def hc_mixes_fused(stream: torch.Tensor, fn: torch.Tensor, scale: torch.Tensor, base: torch.Tensor, eps: float
                   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    T = stream.shape[0]
    dev = stream.device
    pre = torch.empty((T, HC), dtype=torch.float32, device=dev)
    post = torch.empty((T, HC), dtype=torch.float32, device=dev)
    comb = torch.empty((T, HC, HC), dtype=torch.float32, device=dev)
    if T:
        part = torch.empty((T, MIX_NCHUNK, 32), dtype=torch.float32, device=dev)
        _hc_mixes_partial_kernel[(T, MIX_NCHUNK)](stream.view(torch.int16).contiguous(), fn.contiguous(), part,
                                                  BLOCK=MIX_CHUNK, num_warps=4, enable_fp_fusion=False)
        _hc_mixes_finish_sinkhorn_kernel[(T,)](part, scale.float().contiguous(), base.float().contiguous(), pre, post,
                                               comb, float(HC_EPS), float(eps), ITERS=int(HC_SINKHORN_ITERS),
                                               ROW=ROW_ORDER, COL=COL_ORDER, num_warps=1, enable_fp_fusion=False)
    return pre, post, comb
