# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 attention epilogue and log scaling for SM70.

From upstream `vllm/models/inkling2/nvidia/ops/attention.py` (`gate_sink`,
`compute_log_scaling_tau`) without the SM90 PDL hooks. The attention itself is
`rel_attention` (upstream's Triton SM90/SM12x kernel, retuned for SM70).
"""

import torch

from vllm.triton_utils import tl, tldevice, triton


@triton.jit
def _sigmoid(x):
    return 1.0 / (1.0 + tl.exp2(-x * 1.4426950408889634))


@triton.jit
def _gate_sink_kernel(
    X,
    Gate,
    LSE,
    Sink,
    Y,
    stride_x,
    stride_gate,
    stride_lse_t,
    stride_lse_h,
    stride_sink,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = col < H * D
    head = col // D
    x = tl.load(X + row * stride_x + col, mask, other=0)
    gate = tl.load(Gate + row * stride_gate + col, mask, other=0).to(tl.float32)
    lse = tl.load(LSE + row * stride_lse_t + head * stride_lse_h, mask, other=0)
    sink = tl.load(Sink + row * stride_sink + head, mask, other=0).to(tl.float32)
    result = x.to(tl.float32) * _sigmoid(gate)
    result = result * _sigmoid(lse.to(tl.float32) - sink)
    tl.store(Y + row * H * D + col, result.to(Y.dtype.element_ty), mask)


def gate_sink(
    output: torch.Tensor,
    gate: torch.Tensor,
    lse: torch.Tensor,
    sinks: torch.Tensor,
    result: torch.Tensor,
) -> None:
    """Write `output * sigmoid(gate) * sigmoid(lse - sink)` per head to
    `result` (model dtype).

    `gate` and `result` are `(T, H * D)`, `sinks` and `lse` are `(T, H)` and
    `output` is `(T, H, D)` (fp32).
    """
    num_tokens, num_heads = sinks.shape
    if num_tokens == 0:
        return
    assert lse.shape == (num_tokens, num_heads)
    assert output.shape[1] == num_heads and output.is_contiguous()
    assert gate.stride(1) == sinks.stride(1) == 1
    assert result.shape == gate.shape and result.is_contiguous()
    width = gate.shape[1]
    _gate_sink_kernel[(num_tokens, triton.cdiv(width, 512))](
        output,
        gate,
        lse,
        sinks,
        result,
        output.stride(0),
        gate.stride(0),
        lse.stride(0),
        lse.stride(1),
        sinks.stride(0),
        num_heads,
        width // num_heads,
        512,
        enable_fp_fusion=False,
    )


@triton.jit(do_not_specialize_on_alignment=["n"])
def _log_scaling_tau_kernel(
    Pos,
    Out,
    n,
    FLOOR: tl.constexpr,
    ALPHA: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    pos = tl.load(Pos + i, i < n, other=0)
    ratio = tl.maximum((pos + 1).to(tl.float32) * (1.0 / FLOOR), 1.0)
    tau = 1.0 + ALPHA * tldevice.log(ratio)
    tl.store(Out + i, tau, i < n)


def compute_log_scaling_tau(
    positions: torch.Tensor, n_floor: int, alpha: float
) -> torch.Tensor:
    """Return fp32 `1 + alpha * log(max((positions + 1) / n_floor, 1))`."""
    output = torch.empty(positions.shape, device=positions.device, dtype=torch.float32)
    n = positions.numel()
    if n:
        _log_scaling_tau_kernel[(triton.cdiv(n, 256),)](
            positions, output, n, n_floor, alpha, 256, enable_fp_fusion=False
        )
    return output


# ---------------------------------------------------------------- decode
# Upstream's non-SM100 decode/verify route: `rel_attention` writes split
# partials (MODE_PARTIAL) and `_rel_decode_combine_kernel` merges them and
# applies the output gate and sink (or MODE_GATED for a single split).

# Decode/verify path when a request has at most this many query tokens per
# 64-row GQA tile (upstream `_VERIFY_BLOCK_M // group`).
DECODE_BLOCK_M = 64


def max_decode_query_len(num_heads: int, num_kv_heads: int) -> int:
    return DECODE_BLOCK_M // (num_heads // num_kv_heads)


def plan_decode_attention(
    is_local: bool,
    num_reqs: int,
    num_kv_heads: int,
    rel_extent: int,
    num_sms: int = 80,
    block_n: int = 32,
) -> int:
    """KV split count of the decode path. Depends only on the batch (not the
    KV lengths), so it holds for every replay of a CUDA graph. Local layers
    split their window into a few key blocks; full layers give each request
    and KV head about `num_sms / 8` splits (fewer once the batch fills the
    GPU), never less than one CTA per SM."""
    work = num_reqs * num_kv_heads
    if is_local:
        if work <= 16:
            return min(8, triton.cdiv(rel_extent, block_n))
        return 4 if work <= 32 else 2 if work <= 64 else 1
    return max(1, max(num_sms // work, min(num_sms // 8, 6 * num_sms // work)))


@triton.jit
def _rel_decode_combine_kernel(
    OPart,
    LsePart,
    Gate,
    Sink,
    Y,
    num_tokens,
    num_actual_tokens,
    stride_gate,
    stride_sink,
    H: tl.constexpr,
    D: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    offs_d = tl.program_id(2) * BLOCK_D + tl.arange(0, BLOCK_D)
    y_ptrs = Y + row * H * D + head * D + offs_d
    if row >= num_actual_tokens:
        tl.store(y_ptrs, tl.zeros([BLOCK_D], tl.float32).to(Y.dtype.element_ty))
        return
    gate = tl.load(Gate + row * stride_gate + head * D + offs_d).to(tl.float32)
    sink = tl.load(Sink + row * stride_sink + head).to(tl.float32)
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
    lse = (tl.log2(tl.where(w_sum == 0.0, 1.0, w_sum)) + lse_max) * 0.6931471805599453
    o = o * _sigmoid(gate)
    o = o * _sigmoid(lse - sink)
    tl.store(y_ptrs, o.to(Y.dtype.element_ty))


def rel_decode_attention(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    rel_logits: torch.Tensor,
    gate: torch.Tensor,
    sinks: torch.Tensor,
    out: torch.Tensor,
    *,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_query_len: int,
    num_actual_tokens: int,
    softmax_scale: float,
    window: int,
    num_splits: int,
) -> None:
    """Gated `(T, H * D)` attention output for decode/verify batches: split-KV
    `rel_attention` partials merged by `_rel_decode_combine_kernel` (or the
    gated single-split launch). Rows from `num_actual_tokens` on are zeroed."""
    from .rel_attention import MODE_GATED, MODE_PARTIAL, rel_attention

    num_tokens, num_heads, head_dim = q.shape
    common = dict(
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        max_query_len=max_query_len,
        softmax_scale=softmax_scale,
        window=window,
    )
    if num_splits == 1:
        if num_actual_tokens < num_tokens:
            out[num_actual_tokens:].zero_()
        rel_attention(
            q,
            key_cache,
            value_cache,
            out,
            None,
            rel_logits,
            gate,
            sinks,
            num_splits=1,
            mode=MODE_GATED,
            **common,
        )
        return
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
        rel_logits,
        num_splits=num_splits,
        mode=MODE_PARTIAL,
        **common,
    )
    block_d = 64
    _rel_decode_combine_kernel[(num_tokens, num_heads, head_dim // block_d)](
        o_part,
        lse_part,
        gate,
        sinks,
        out,
        num_tokens,
        num_actual_tokens,
        gate.stride(0),
        sinks.stride(0),
        H=num_heads,
        D=head_dim,
        NUM_SPLITS=num_splits,
        BLOCK_S=triton.next_power_of_2(num_splits),
        BLOCK_D=block_d,
    )
