# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode GEMV fast paths for DeepSeek-V4.1 on SM70 (lane L-MOE; PORT_DESIGN §2.2, §4.1).

Small-M (decode, M <= 8) matrix-vector products with FP16 operands and FP32 accumulation, reading each weight
element once and reusing it for every row of ``x`` from registers. They are bandwidth-bound on V100, so a token
costs ``N * K * 2`` bytes per projection. Shape-generic (K a multiple of 16), used for

* the router gate (N = 384 or 128, K = 5120) with an exponent-biased FP16 weight (``alpha`` = 2^-k, exact);
* the shared expert: ``gate_up_swiglu`` (N = 2 * I, K = 5120) fuses the SwiGLU into the GEMV epilogue and
  ``down_combine`` (N = 5120, K = I) adds the routed-expert combine in its epilogue (FP32 out);
* small projections of other lanes (compressor ``wkv``/``wgate``, indexer ``weights_proj``) through ``gemv``.

Semantics match the unfused path exactly where a rounding is involved: ``gate_up_swiglu`` rounds the gate and
up projections to FP16 before the activation, as an FP16 GEMM output would be, so decode (fused) and prefill
(cuBLAS + ``swiglu_fp32``) differ only in FP32 accumulation order. ``down_combine`` uses the combine order of
``moe_kernels.combine_reference`` (slot order, no FMA) and adds the shared output last.
"""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton

MAX_GEMV_ROWS = 8


_TILE_ELEMS = 8192  # FP32 accumulator elements per program (128 threads x 64 registers)


def _config(n: int, k: int, m_pad: int, weights_per_row: int = 1) -> tuple[int, int]:
    """(BLOCK_N, BLOCK_K) from the SM70 decode sweep (L-MOE progress, sub-item 4; L2-busting weight rotation).
    The accumulator ``[M_PAD, BLOCK_N, BLOCK_K]`` (x ``weights_per_row``) stays within ``_TILE_ELEMS``."""
    if k >= 1024:
        block_n = {1: 2 if n <= 512 else 4, 2: 1, 4: 2}.get(m_pad, 2 if n <= 512 else 4)
        block_k = 1024
    else:
        block_n, block_k = 8, 1024
    while block_k > 16 and (k % block_k or m_pad * block_n * block_k * weights_per_row > _TILE_ELEMS):
        block_k //= 2
    while block_n > 1 and m_pad * block_n * block_k * weights_per_row > _TILE_ELEMS:
        block_n //= 2
    if k % block_k:
        raise ValueError(f"GEMV needs K to be a multiple of 16, got K={k}")
    return block_n, block_k


@triton.jit
def _dot_rows(x_ptr, stride_xm, w_ptr, rows, row_mask, K, M, M_PAD: tl.constexpr, BLOCK_N: tl.constexpr,
              BLOCK_K: tl.constexpr):
    """[M_PAD, BLOCK_N] FP32 = x[:M] . W[rows]^T; partial products accumulate elementwise and are reduced once."""
    ks = tl.arange(0, BLOCK_K)
    if M_PAD == 1:
        acc2 = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            w = tl.load(w_ptr + rows[:, None] * K + (k0 + ks)[None, :], mask=row_mask[:, None], other=0.0)
            x = tl.load(x_ptr + k0 + ks)
            acc2 += w.to(tl.float32) * x.to(tl.float32)[None, :]
        return tl.sum(acc2, axis=1)[None, :]
    else:
        ms = tl.arange(0, M_PAD)
        acc3 = tl.zeros((M_PAD, BLOCK_N, BLOCK_K), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            w = tl.load(w_ptr + rows[:, None] * K + (k0 + ks)[None, :], mask=row_mask[:, None], other=0.0)
            x = tl.load(x_ptr + ms[:, None] * stride_xm + (k0 + ks)[None, :], mask=(ms < M)[:, None], other=0.0)
            acc3 += x.to(tl.float32)[:, None, :] * w.to(tl.float32)[None, :, :]
        return tl.sum(acc3, axis=2)


@triton.jit
def _gemv_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    M,
    N,
    K,
    alpha,
    stride_xm,
    stride_om,
    M_PAD: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    OUT_FP16: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < N
    ms = tl.arange(0, M_PAD)
    acc = _dot_rows(x_ptr, stride_xm, w_ptr, rows, row_mask, K, M, M_PAD, BLOCK_N, BLOCK_K) * alpha
    out_ptrs = out_ptr + ms[:, None] * stride_om + rows[None, :]
    mask = (ms < M)[:, None] & row_mask[None, :]
    if OUT_FP16:
        tl.store(out_ptrs, acc.to(tl.float16), mask=mask)
    else:
        tl.store(out_ptrs, acc, mask=mask)


def gemv(
    x: torch.Tensor, weight: torch.Tensor, *, out_dtype: torch.dtype = torch.float32, alpha: float = 1.0
) -> torch.Tensor:
    """``alpha * x @ weight.T`` for x [M <= 8, K] fp16, weight [N, K] fp16; FP32 accumulate; out fp32/fp16."""
    _check_rows(x, weight.shape[1])
    if weight.dtype != torch.float16 or weight.ndim != 2 or not weight.is_contiguous():
        raise TypeError("gemv: weight must be a contiguous fp16 [N, K] matrix")
    if out_dtype not in (torch.float32, torch.float16):
        raise TypeError(f"gemv: out_dtype {out_dtype} not supported")
    m, k = x.shape
    n = weight.shape[0]
    out = torch.empty((m, n), dtype=out_dtype, device=x.device)
    if m == 0:
        return out
    m_pad = _m_pad(m)
    block_n, block_k = _config(n, k, m_pad)
    _gemv_kernel[(triton.cdiv(n, block_n),)](
        x, weight, out, m, n, k, float(alpha), x.stride(0), out.stride(0),
        M_PAD=m_pad, BLOCK_N=block_n, BLOCK_K=block_k, OUT_FP16=out_dtype == torch.float16, num_warps=4,
    )
    return out


@triton.jit
def _gate_up_swiglu_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    M,
    I,
    K,
    limit,
    stride_xm,
    M_PAD: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < I
    ms = tl.arange(0, M_PAD)
    acc_g = _dot_rows(x_ptr, stride_xm, w_ptr, rows, row_mask, K, M, M_PAD, BLOCK_N, BLOCK_K)
    acc_u = _dot_rows(x_ptr, stride_xm, w_ptr, rows + I, row_mask, K, M, M_PAD, BLOCK_N, BLOCK_K)
    # an unfused path stores both projections in FP16 before the activation; do the same rounding
    gate = tl.minimum(acc_g.to(tl.float16).to(tl.float32), limit)
    up = tl.minimum(tl.maximum(acc_u.to(tl.float16).to(tl.float32), -limit), limit)
    act = tl.div_rn(gate, 1.0 + tl.extra.cuda.libdevice.exp(-gate)) * up
    mask = (ms < M)[:, None] & row_mask[None, :]
    tl.store(out_ptr + ms[:, None] * I + rows[None, :], act.to(tl.float16), mask=mask)


def gate_up_swiglu(x: torch.Tensor, w13: torch.Tensor, limit: float) -> torch.Tensor:
    """x [M <= 8, K] fp16, w13 [2I, K] fp16 (gate rows then up rows) -> SwiGLU [M, I] fp16."""
    _check_rows(x, w13.shape[1])
    if w13.dtype != torch.float16 or w13.ndim != 2 or w13.shape[0] % 2 or not w13.is_contiguous():
        raise TypeError("gate_up_swiglu: w13 must be a contiguous fp16 [2I, K] matrix")
    m, k = x.shape
    inter = w13.shape[0] // 2
    out = torch.empty((m, inter), dtype=torch.float16, device=x.device)
    if m == 0:
        return out
    m_pad = _m_pad(m)
    block_n, block_k = _config(2 * inter, k, m_pad)
    block_n = max(1, block_n // 2)  # each program reads a gate row and its up row
    _gate_up_swiglu_kernel[(triton.cdiv(inter, block_n),)](
        x, w13, out, m, inter, k, float(limit), x.stride(0),
        M_PAD=m_pad, BLOCK_N=block_n, BLOCK_K=block_k, num_warps=4,
    )
    return out


@triton.jit
def _down_combine_kernel(
    act_ptr,
    w_ptr,
    y_ptr,
    tw_ptr,
    out_ptr,
    M,
    N,
    K,
    M_PAD: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    TOP_K: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < N
    ms = tl.arange(0, M_PAD)
    acc = _dot_rows(act_ptr, K, w_ptr, rows, row_mask, K, M, M_PAD, BLOCK_N, BLOCK_K)
    mask = (ms < M)[:, None] & row_mask[None, :]
    if TOP_K > 0:
        routed = tl.zeros((M_PAD, BLOCK_N), dtype=tl.float32)
        for j in tl.static_range(TOP_K):
            wj = tl.load(tw_ptr + ms * TOP_K + j, mask=ms < M, other=0.0)
            yj = tl.load(y_ptr + (ms * TOP_K + j)[:, None] * N + rows[None, :], mask=mask, other=0.0)
            if j == 0:
                routed = wj[:, None] * yj.to(tl.float32)
            else:
                routed = routed + wj[:, None] * yj.to(tl.float32)
        acc = routed + acc
    tl.store(out_ptr + ms[:, None] * N + rows[None, :], acc, mask=mask)


def down_combine(
    act: torch.Tensor,
    w2: torch.Tensor,
    y_slots: torch.Tensor | None = None,
    topk_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """FP32 ``act @ w2.T`` (+ the routed combine ``sum_j w[t,j] y[t*k+j]`` added before it, slot order)."""
    _check_rows(act, w2.shape[1])
    if w2.dtype != torch.float16 or w2.ndim != 2 or not w2.is_contiguous():
        raise TypeError("down_combine: w2 must be a contiguous fp16 [N, K] matrix")
    m, k = act.shape
    n = w2.shape[0]
    top_k = 0
    if y_slots is not None:
        if topk_weights is None or topk_weights.dtype != torch.float32 or not topk_weights.is_contiguous():
            raise TypeError("down_combine: topk_weights must be contiguous float32 [M, k]")
        top_k = topk_weights.shape[1]
        if (y_slots.dtype != torch.float16 or tuple(y_slots.shape) != (m * top_k, n)
                or not y_slots.is_contiguous()):
            raise TypeError(f"down_combine: y_slots must be contiguous fp16 [{m * top_k}, {n}]")
    if not act.is_contiguous():
        raise TypeError("down_combine: act must be contiguous")
    out = torch.empty((m, n), dtype=torch.float32, device=act.device)
    if m == 0:
        return out
    m_pad = _m_pad(m)
    block_n, block_k = _config(n, k, m_pad)
    _down_combine_kernel[(triton.cdiv(n, block_n),)](
        act, w2, y_slots if y_slots is not None else act, topk_weights if topk_weights is not None else act,
        out, m, n, k, M_PAD=m_pad, BLOCK_N=block_n, BLOCK_K=block_k, TOP_K=top_k,
        num_warps=4, enable_fp_fusion=False,
    )
    return out


def _m_pad(m: int) -> int:
    return max(1, triton.next_power_of_2(m))


def _check_rows(x: torch.Tensor, k: int) -> None:
    if x.dtype != torch.float16 or x.ndim != 2 or x.stride(1) != 1:
        raise TypeError("GEMV input must be a row-major fp16 [M, K] matrix")
    if not 0 <= x.shape[0] <= MAX_GEMV_ROWS:
        raise ValueError(f"GEMV serves at most {MAX_GEMV_ROWS} rows, got {x.shape[0]}")
    if x.shape[1] != k:
        raise ValueError(f"GEMV K mismatch: x has {x.shape[1]}, weight has {k}")
