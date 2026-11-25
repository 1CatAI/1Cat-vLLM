# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM70 kernels of the Inkling2 DSpark draft.

Port of upstream `vllm/models/inkling2/nvidia/ops/dspark.py` with the bf16
hard-coding removed: activations are fp16 (the target's dtype on V100), the
Markov gate and the softcap run in fp32 and keep fp32 outputs. The draft is
trained in bf16, but it only proposes tokens: the target verifies them, so
the draft's rounding changes the acceptance rate, never the output.

Also holds the paged-KV writer fused with the per-head q/k RMSNorm + RoPE
(the draft's attention reads the port's `(blocks, page, H_kv, 2 * D)` cache)
and the gate-free split merge for the draft's block attention.
"""

import torch

from vllm.triton_utils import tl, tldevice, triton

from .rel_attention import MODE_PARTIAL, rel_attention


@triton.jit
def _mix_rmsnorm_kernel(
    X,
    ALPHA,
    WEIGHT,
    OUT,
    ROW_STRIDE: tl.constexpr,
    TAP_STRIDE: tl.constexpr,
    HIDDEN_STRIDE: tl.constexpr,
    TAPS: tl.constexpr,
    HIDDEN: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    h = tl.arange(0, BLOCK)
    mixed = tl.full((BLOCK,), 0, tl.float32)
    for tap in tl.static_range(TAPS):
        x = tl.load(
            X + row * ROW_STRIDE + tap * TAP_STRIDE + h * HIDDEN_STRIDE,
            h < HIDDEN,
            0,
        ).to(tl.float32)
        alpha = tl.load(ALPHA + tap).to(tl.float32)
        mixed += x * alpha
    inv_rms = tl.rsqrt(tl.sum(mixed * mixed, 0) / HIDDEN + EPS)
    weight = tl.load(WEIGHT + h, h < HIDDEN, 0).to(tl.float32)
    tl.store(OUT + row * HIDDEN + h, mixed * inv_rms * weight, h < HIDDEN)


def mix_rmsnorm(
    targets: torch.Tensor, alpha: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    """RMSNorm of the `alpha`-weighted sum of `targets[:, i]` (fp32 math).

    `targets` is `(rows, taps, hidden)`; the output has `targets.dtype`.
    """
    rows, taps, hidden = targets.shape
    assert alpha.shape == (taps,) and alpha.is_contiguous()
    assert weight.shape == (hidden,) and weight.is_contiguous()
    out = targets.new_empty((rows, hidden))
    if rows:
        _mix_rmsnorm_kernel[(rows,)](
            targets,
            alpha,
            weight,
            out,
            *targets.stride(),
            taps,
            hidden,
            eps,
            triton.next_power_of_2(hidden),
            num_warps=8,
        )
    return out


@triton.jit
def _qkv_prep_kernel(
    X,
    Q_WEIGHT,
    K_WEIGHT,
    COS_SIN,
    POS,
    SLOT,
    CACHE,
    row_stride,
    stride_blk,
    stride_slot,
    stride_head,
    NHQ: tl.constexpr,
    NHK: tl.constexpr,
    D: tl.constexpr,
    PAGE: tl.constexpr,
    EPS: tl.constexpr,
    STORE_V: tl.constexpr,
):
    """Per (token, head): q heads are RMS-normed and rotated in place; k heads
    are normed, rotated and written to the cache's K half; v heads (if
    `STORE_V`) are copied to its V half. `X` is `(T, (NHQ + 2 * NHK) * D)`
    (`NHQ = 0` for context K/V)."""
    t = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1)
    pos = tl.load(POS + t).to(tl.int64)
    slot = tl.load(SLOT + t)
    HALF: tl.constexpr = D // 2
    half_ar = tl.arange(0, HALF)
    d_ar = tl.arange(0, D)
    row = X + t * row_stride + h * D
    if h < NHQ + NHK:
        is_q = h < NHQ
        cos = tl.load(COS_SIN + pos * D + half_ar).to(tl.float32)
        sin = tl.load(COS_SIN + pos * D + HALF + half_ar).to(tl.float32)
        x = tl.load(row + d_ar).to(tl.float32)
        inv = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
        w1 = tl.where(
            is_q,
            tl.load(Q_WEIGHT + half_ar).to(tl.float32),
            tl.load(K_WEIGHT + half_ar).to(tl.float32),
        )
        w2 = tl.where(
            is_q,
            tl.load(Q_WEIGHT + HALF + half_ar).to(tl.float32),
            tl.load(K_WEIGHT + HALF + half_ar).to(tl.float32),
        )
        x1 = tl.load(row + half_ar).to(tl.float32) * inv * w1
        x2 = tl.load(row + HALF + half_ar).to(tl.float32) * inv * w2
        # Round the norm output to the activation dtype, as the unfused ops.
        x1 = x1.to(X.dtype.element_ty).to(tl.float32)
        x2 = x2.to(X.dtype.element_ty).to(tl.float32)
        o1 = (x1 * cos - x2 * sin).to(X.dtype.element_ty)
        o2 = (x2 * cos + x1 * sin).to(X.dtype.element_ty)
        if is_q:
            tl.store(row + half_ar, o1)
            tl.store(row + HALF + half_ar, o2)
        elif slot >= 0:
            blk = slot // PAGE
            off = slot % PAGE
            kvh = h - NHQ
            dst = CACHE + blk * stride_blk + off * stride_slot + kvh * stride_head
            tl.store(dst + half_ar, o1)
            tl.store(dst + HALF + half_ar, o2)
    elif STORE_V and slot >= 0:
        blk = slot // PAGE
        off = slot % PAGE
        kvh = h - NHQ - NHK
        v = tl.load(row + d_ar)
        dst = CACHE + blk * stride_blk + off * stride_slot + kvh * stride_head + D
        tl.store(dst + d_ar, v)


def qkv_prep_(
    x: torch.Tensor,
    positions: torch.Tensor,
    slots: torch.Tensor,
    q_weight: torch.Tensor | None,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    kv_cache: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    eps: float,
) -> None:
    """In-place per-head q norm + RoPE of the leading `num_q_heads` of the
    fused `x` rows; k norm + RoPE + paged write of the next `num_kv_heads`
    heads (K half of `kv_cache`) and the paged write of the last
    `num_kv_heads` heads (V half). Rows with a negative slot write nothing.
    `num_q_heads == 0` handles context K/V (`q_weight` unused)."""
    head_dim = cos_sin_cache.shape[-1]
    page = kv_cache.shape[1]
    num_tokens = x.shape[0]
    assert x.stride(1) == 1 and kv_cache.stride(3) == 1
    if num_tokens:
        _qkv_prep_kernel[(num_tokens, num_q_heads + 2 * num_kv_heads)](
            x,
            q_weight if q_weight is not None else k_weight,
            k_weight,
            cos_sin_cache,
            positions,
            slots,
            kv_cache,
            x.stride(0),
            kv_cache.stride(0),
            kv_cache.stride(1),
            kv_cache.stride(2),
            NHQ=num_q_heads,
            NHK=num_kv_heads,
            D=head_dim,
            PAGE=page,
            EPS=eps,
            STORE_V=True,
        )


@triton.jit
def _block_combine_kernel(
    OPart,
    LsePart,
    SeqLens,
    Y,
    num_tokens,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    offs_d = tl.program_id(2) * BLOCK_D + tl.arange(0, BLOCK_D)
    y_ptrs = Y + row * H * D + head * D + offs_d
    if tl.load(SeqLens + row // BLOCK) == 0:
        tl.store(y_ptrs, tl.zeros([BLOCK_D], tl.float32).to(Y.dtype.element_ty))
        return
    splits = tl.arange(0, BLOCK_S)
    part = (splits * num_tokens + row) * H + head
    lse_s = tl.load(LsePart + part, mask=splits < NUM_SPLITS, other=float("-inf"))
    lse_max = tl.max(lse_s)
    lse_max = tl.where(lse_max == float("-inf"), 0.0, lse_max)
    w = tl.exp2(lse_s - lse_max)
    w_sum = tl.sum(w)
    o_s = tl.load(
        OPart + part[:, None] * D + offs_d[None, :],
        mask=(w > 0.0)[:, None],
        other=0.0,
    )
    o = tl.sum(o_s * (w / tl.where(w_sum == 0.0, 1.0, w_sum))[:, None], 0)
    tl.store(y_ptrs, o.to(Y.dtype.element_ty))


def block_attention(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    out: torch.Tensor,
    *,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    block: int,
    softmax_scale: float,
    window: int,
    num_splits: int,
) -> None:
    """Bidirectional block attention of the DSpark draft: every query of a
    request sees the keys `[seqlen - window, seqlen)` (`window` 0: all), with
    no relative bias, gate or sink. `q` is `(T, H, D)`, `out` `(T, H * D)`.
    Requests with a zero length write zeros. A single split or split-KV
    partials merged by `_block_combine_kernel`."""
    num_tokens, num_heads, head_dim = q.shape
    o_part = torch.empty(
        (num_splits, num_tokens, num_heads, head_dim),
        dtype=torch.float32,
        device=q.device,
    )
    lse_part = torch.empty(
        (num_splits, num_tokens, num_heads), dtype=torch.float32, device=q.device
    )
    rel_attention(
        q,
        key_cache,
        value_cache,
        o_part,
        lse_part,
        None,
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        max_query_len=block,
        softmax_scale=softmax_scale,
        window=window,
        num_splits=num_splits,
        mode=MODE_PARTIAL,
    )
    block_d = 64
    _block_combine_kernel[(num_tokens, num_heads, head_dim // block_d)](
        o_part,
        lse_part,
        cache_seqlens,
        out,
        num_tokens,
        H=num_heads,
        D=head_dim,
        BLOCK=block,
        NUM_SPLITS=num_splits,
        BLOCK_S=triton.next_power_of_2(num_splits),
        BLOCK_D=block_d,
    )


@triton.jit
def _markov_gate_kernel(RAW, STATE, OUT, LATENT, RANK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, RANK)
    raw = RAW + row * 3 * RANK + c
    gate = tl.load(raw)
    cand = tl.load(raw + RANK)
    out = tl.load(raw + 2 * RANK)
    state = tl.load(STATE + row * RANK + c).to(tl.float32)
    gate = 1.0 / (1.0 + tldevice.exp(-gate))
    cand = tldevice.tanh(cand)
    tl.store(OUT + row * RANK + c, gate * state + (1.0 - gate) * cand)
    tl.store(LATENT + row * RANK + c, tldevice.tanh(out).to(LATENT.dtype.element_ty))


def markov_gate(
    raw: torch.Tensor, state: torch.Tensor, latent_dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gating of the recurrent head's fp32 `rnn_proj` output `raw`
    `(rows, 3 * rank)`: returns the fp32 state
    `gate * state + (1 - gate) * tanh(cand)` and `tanh(out)` as `latent_dtype`."""
    rows, rank = state.shape
    assert raw.shape == (rows, 3 * rank) and raw.is_contiguous()
    assert state.is_contiguous() and rank & (rank - 1) == 0
    out = torch.empty_like(state)
    latent = torch.empty(rows, rank, dtype=latent_dtype, device=state.device)
    if rows:
        _markov_gate_kernel[(rows,)](raw, state, out, latent, rank)
    return out, latent


@triton.jit
def _add_softcap_kernel(A, B, OUT, stride_a, stride_b, softcap, N, BLOCK: tl.constexpr):
    row = tl.program_id(1).to(tl.int64)
    cols = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < N
    a = tl.load(A + row * stride_a + cols, mask, other=0).to(tl.float32)
    b = tl.load(B + row * stride_b + cols, mask, other=0).to(tl.float32)
    x = tldevice.tanh((a + b) / softcap) * softcap
    tl.store(OUT + row * N + cols, x, mask)


def add_softcap(a: torch.Tensor, b: torch.Tensor, softcap: float) -> torch.Tensor:
    """fp32 `softcap * tanh((a + b) / softcap)` of two 2D tensors (the
    `(1, N)` bias broadcasts over the rows of `a` when `b` has one row)."""
    rows, n = a.shape
    assert a.stride(1) == 1 and b.stride(1) == 1
    out = torch.empty((rows, n), dtype=torch.float32, device=a.device)
    stride_b = 0 if b.shape[0] == 1 and rows > 1 else b.stride(0)
    if rows:
        _add_softcap_kernel[(triton.cdiv(n, 1024), rows)](
            a, b, out, a.stride(0), stride_b, softcap, n, 1024
        )
    return out


@triton.jit
def _add_rms_norm_kernel(
    X, R, W, OUT, stride_x, stride_r, N: tl.constexpr, EPS, BLOCK: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * stride_x + cols, mask, other=0).to(tl.float32)
    r = tl.load(R + row * stride_r + cols, mask, other=0).to(tl.float32)
    s = x + r
    tl.store(R + row * stride_r + cols, s.to(R.dtype.element_ty), mask)
    inv = tl.rsqrt(tl.sum(s * s, 0) / N + EPS)
    w = tl.load(W + cols, mask, other=0).to(tl.float32)
    tl.store(OUT + row * N + cols, (s * inv * w).to(OUT.dtype.element_ty), mask)


def add_rms_norm(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    """In place `residual += x`; returns `rms_norm(residual + x) * weight`
    (the norm reads the unrounded fp32 sum)."""
    rows, n = x.shape
    assert x.stride(1) == 1 and residual.is_contiguous()
    out = torch.empty_like(x)
    if rows:
        block = triton.next_power_of_2(n)
        _add_rms_norm_kernel[(rows,)](
            x,
            residual,
            weight,
            out,
            x.stride(0),
            residual.stride(0),
            n,
            eps,
            BLOCK=block,
            num_warps=min(max(block // 512, 1), 16),
        )
    return out


@triton.jit
def _silu_mul_kernel(X, OUT, stride_x, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < N
    gate = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(X + row * stride_x + N + cols, mask=mask, other=0.0).to(tl.float32)
    y = gate / (1.0 + tl.exp(-gate)) * up
    tl.store(OUT + row * N + cols, y.to(OUT.dtype.element_ty), mask=mask)


def silu_mul(gate_up: torch.Tensor) -> torch.Tensor:
    """`silu(gate) * up` of the `[gate | up]` halves of `gate_up` `(rows,
    2 * n)` in one pass (fp32 math, one rounding)."""
    rows, width = gate_up.shape
    n = width // 2
    assert gate_up.stride(1) == 1 and width == 2 * n
    out = torch.empty((rows, n), dtype=gate_up.dtype, device=gate_up.device)
    if rows:
        block = 1024
        _silu_mul_kernel[(rows, triton.cdiv(n, block))](
            gate_up, out, gate_up.stride(0), N=n, BLOCK=block, num_warps=4
        )
    return out
