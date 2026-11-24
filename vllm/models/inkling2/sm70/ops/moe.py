# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Inkling2 MoE kernels for SM70.

- `softcapped_swiglu_`: `gcap tanh(g / gcap) * sigmoid(g) * ucap tanh(u / ucap)`
  on `[gate | up]` halves (libdevice tanh: `tanh.approx` needs SM75+).
- `moe_gemv`: per-slot expert GEMV `out[s] = W[ids[s]] @ x[s // x_div]` for
  fp16 experts. Memory bound, so scalar FMA (all Triton dots are FMA on SM70)
  is fine; CUDA-graph safe (no host sync).
"""

import torch

from vllm.triton_utils import tl, tldevice, triton


@triton.jit
def _softcapped_swiglu_kernel(
    GateUp,
    Out,
    inter,
    stride_in,
    stride_out,
    GCAP: tl.constexpr,
    UCAP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = col < inter
    g = tl.load(GateUp + row * stride_in + col, mask, other=0.0).to(tl.float32)
    u = tl.load(GateUp + row * stride_in + inter + col, mask, other=0.0).to(tl.float32)
    gf = g
    if GCAP > 0:
        gf = GCAP * tldevice.tanh(g / GCAP)
    if UCAP > 0:
        u = UCAP * tldevice.tanh(u / UCAP)
    sig = 1.0 / (1.0 + tl.exp(-g))
    tl.store(
        Out + row * stride_out + col, (gf * sig * u).to(Out.dtype.element_ty), mask
    )


def softcapped_swiglu_(
    out: torch.Tensor, gate_up: torch.Tensor, gate_cap: float, up_cap: float
) -> torch.Tensor:
    """`out [S, I]` <- softcapped SwiGLU of `gate_up [S, 2I]` (halves)."""
    rows, inter = out.shape
    assert gate_up.shape[0] >= rows and gate_up.shape[1] == 2 * inter
    assert gate_up.stride(1) == 1 and out.stride(1) == 1
    if rows:
        block = 512
        _softcapped_swiglu_kernel[(rows, triton.cdiv(inter, block))](
            gate_up,
            out,
            inter,
            gate_up.stride(0),
            out.stride(0),
            GCAP=float(gate_cap),
            UCAP=float(up_cap),
            BLOCK=block,
        )
    return out


@triton.jit
def _moe_gemv_kernel(
    X,
    W,
    Ids,
    Out,
    N,
    K,
    stride_x,
    stride_we,
    stride_wn,
    stride_out,
    X_DIV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    slot = tl.program_id(0).to(tl.int64)
    n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n < N
    expert = tl.load(Ids + slot).to(tl.int64)
    x_row = X + (slot // X_DIV) * stride_x
    w_rows = W + expert * stride_we + n[:, None] * stride_wn
    acc = tl.zeros([BLOCK_N], tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        k_mask = k < K
        x = tl.load(x_row + k, mask=k_mask, other=0.0).to(tl.float32)
        w = tl.load(
            w_rows + k[None, :], mask=n_mask[:, None] & k_mask[None, :], other=0.0
        )
        acc += tl.sum(w.to(tl.float32) * x[None, :], axis=1)
    tl.store(Out + slot * stride_out + n, acc.to(Out.dtype.element_ty), mask=n_mask)


def moe_gemv(
    x: torch.Tensor,
    weight: torch.Tensor,
    ids: torch.Tensor,
    out: torch.Tensor,
    x_div: int,
) -> torch.Tensor:
    """`out[s] = weight[ids[s]] @ x[s // x_div]`; `weight [E, N, K]`, `ids [S]`."""
    slots = ids.shape[0]
    _, n, k = weight.shape
    assert weight.stride(2) == 1 and x.stride(1) == 1 and out.stride(1) == 1
    if slots:
        block_n = 32
        _moe_gemv_kernel[(slots, triton.cdiv(n, block_n))](
            x,
            weight,
            ids,
            out,
            n,
            k,
            x.stride(0),
            weight.stride(0),
            weight.stride(1),
            out.stride(0),
            X_DIV=x_div,
            BLOCK_N=block_n,
            BLOCK_K=128,
            num_warps=4,
        )
    return out


@triton.jit
def _moe_gemv_swiglu_kernel(
    X,
    W,
    Ids,
    Out,
    inter,
    K,
    stride_x,
    stride_we,
    stride_wn,
    stride_out,
    X_DIV: tl.constexpr,
    GCAP: tl.constexpr,
    UCAP: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """W13 GEMV of `_moe_gemv_kernel` and `_softcapped_swiglu_kernel` in one launch: a program owns
    the gate rows n and the up rows inter + n of one slot. Every row's dot product is the one the
    two-launch path computes (same tile shape, same reduction), the gate / up values are rounded to
    fp16 as `gate_up` is, then the softcapped SwiGLU in fp32 as in the swiglu kernel."""
    slot = tl.program_id(0).to(tl.int64)
    n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n < inter
    expert = tl.load(Ids + slot).to(tl.int64)
    x_row = X + (slot // X_DIV) * stride_x
    g_rows = W + expert * stride_we + n[:, None] * stride_wn
    u_rows = W + expert * stride_we + (n[:, None] + inter) * stride_wn
    acc_g = tl.zeros([BLOCK_N], tl.float32)
    acc_u = tl.zeros([BLOCK_N], tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        k_mask = k < K
        x = tl.load(x_row + k, mask=k_mask, other=0.0).to(tl.float32)
        wg = tl.load(
            g_rows + k[None, :], mask=n_mask[:, None] & k_mask[None, :], other=0.0
        )
        wu = tl.load(
            u_rows + k[None, :], mask=n_mask[:, None] & k_mask[None, :], other=0.0
        )
        acc_g += tl.sum(wg.to(tl.float32) * x[None, :], axis=1)
        acc_u += tl.sum(wu.to(tl.float32) * x[None, :], axis=1)
    # fp16 gate_up boundary, as the two-launch path stores it
    g = acc_g.to(tl.float16).to(tl.float32)
    u = acc_u.to(tl.float16).to(tl.float32)
    gf = g
    if GCAP > 0:
        gf = GCAP * tldevice.tanh(g / GCAP)
    if UCAP > 0:
        u = UCAP * tldevice.tanh(u / UCAP)
    sig = 1.0 / (1.0 + tl.exp(-g))
    tl.store(
        Out + slot * stride_out + n, (gf * sig * u).to(Out.dtype.element_ty), n_mask
    )


def moe_gemv_swiglu(
    x: torch.Tensor,
    weight: torch.Tensor,
    ids: torch.Tensor,
    out: torch.Tensor,
    x_div: int,
    gate_cap: float,
    up_cap: float,
    block_n: int = 32,
    num_warps: int = 4,
) -> torch.Tensor:
    """`out[s] = softcapped_swiglu(weight[ids[s]] @ x[s // x_div])` for `weight [E, 2I, K]` with
    `[gate; up]` halves, `out [S, I]`: `moe_gemv` + `softcapped_swiglu_` in one launch."""
    slots = ids.shape[0]
    _, n2, k = weight.shape
    inter = n2 // 2
    assert (
        weight.stride(2) == 1
        and x.stride(1) == 1
        and out.stride(1) == 1
        and out.shape[1] == inter
    )
    if slots:
        _moe_gemv_swiglu_kernel[(slots, triton.cdiv(inter, block_n))](
            x,
            weight,
            ids,
            out,
            inter,
            k,
            x.stride(0),
            weight.stride(0),
            weight.stride(1),
            out.stride(0),
            X_DIV=x_div,
            GCAP=float(gate_cap),
            UCAP=float(up_cap),
            BLOCK_N=block_n,
            BLOCK_K=128,
            num_warps=num_warps,
        )
    return out


@triton.jit
def _moe_weighted_sum_kernel(
    Down, Wt, Out, H, TOPK: tl.constexpr, BLOCK: tl.constexpr, ORDER: tl.constexpr
):
    """`out[t] = sum_s down[t * TOPK + s] * w[t, s]` (fp32 products, fp16 result): the
    `(down.view(T, K, H) * w[..., None]).sum(1).to(fp16)` of the two-launch path in one launch."""
    t = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = col < H
    base = Down + t * TOPK * H + col
    if ORDER == 0:  # sequential
        acc = tl.load(base, mask, other=0.0) * tl.load(Wt + t * TOPK)
        for s in range(1, TOPK):
            acc += tl.load(base + s * H, mask, other=0.0) * tl.load(Wt + t * TOPK + s)
    else:  # four interleaved partial sums (torch's strided reduction), then combined
        a0 = tl.load(base, mask, other=0.0) * tl.load(Wt + t * TOPK)
        a1 = tl.load(base + H, mask, other=0.0) * tl.load(Wt + t * TOPK + 1)
        a2 = tl.load(base + 2 * H, mask, other=0.0) * tl.load(Wt + t * TOPK + 2)
        a3 = tl.load(base + 3 * H, mask, other=0.0) * tl.load(Wt + t * TOPK + 3)
        a0 += tl.load(base + 4 * H, mask, other=0.0) * tl.load(Wt + t * TOPK + 4)
        a1 += tl.load(base + 5 * H, mask, other=0.0) * tl.load(Wt + t * TOPK + 5)
        a2 += tl.load(base + 6 * H, mask, other=0.0) * tl.load(Wt + t * TOPK + 6)
        a3 += tl.load(base + 7 * H, mask, other=0.0) * tl.load(Wt + t * TOPK + 7)
        acc = ((a0 + a1) + a2) + a3
    tl.store(Out + t * H + col, acc.to(Out.dtype.element_ty), mask)


def moe_weighted_sum(
    down: torch.Tensor, weights: torch.Tensor, out: torch.Tensor, order: int = 0
) -> torch.Tensor:
    """`down [T * 8, H]` fp32, `weights [T, 8]` fp32 -> `out [T, H]`."""
    t, top_k = weights.shape
    h = down.shape[1]
    block = 512
    _moe_weighted_sum_kernel[(t, triton.cdiv(h, block))](
        down,
        weights,
        out,
        h,
        TOPK=top_k,
        BLOCK=block,
        ORDER=order,
        num_warps=4,
        enable_fp_fusion=False,  # separate fp32 products, as the torch mul + sum
    )
    return out
