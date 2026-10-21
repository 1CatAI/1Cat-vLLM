# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sparse MLA prefill attention as one gather and two batched matmuls.

The Triton prefill kernel scatters the key reads head by head. Every head of a
token reads the same keys, so gathering them once per token and handing the
rest to the BLAS library does the same work 5-10x faster on V100 and RTX 8000
(64 heads, 64-256 query tokens, 384-640 keys) and lands closer to an fp64
reference than the Triton kernel does.
"""

import math

import torch

from vllm.triton_utils import tl, tldevice, triton

WorkspaceSpec = tuple[tuple[int, ...], torch.dtype]

# Query tokens attended per pass. The buffers grow with this number, not with
# the prefill chunk: at 64 heads and 640 keys they take 110 MiB for 128 tokens,
# and a 256-token chunk reserving twice the former 140 MiB ran the V100 stages
# of a pipeline out of memory. A second pass costs two more BLAS calls per layer.
MAX_TOKENS_PER_PASS = 128

# Slots one gather program copies: eight 1 KiB rows of a 512-wide fp16 key.
_GATHER_SLOTS = 8


def sparse_prefill_bmm_workspace_specs(
    num_tokens: int,
    num_heads: int,
    head_dim: int,
    width: int,
    dtype: torch.dtype,
) -> list[WorkspaceSpec]:
    """Buffers of `sparse_attn_prefill_bmm`, in its argument order."""
    num_tokens = min(num_tokens, MAX_TOKENS_PER_PASS)
    return [
        ((num_tokens, width, head_dim), dtype),
        ((num_tokens, num_heads, width), torch.float32),
        ((num_tokens, num_heads, width), dtype),
    ]


def _fit(buffer: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    # Buffers are reserved for the widest index tensor; a narrower one must
    # still see contiguous rows, so cut the flat storage and reshape.
    return buffer.view(-1)[: math.prod(shape)].view(shape)


@triton.jit
def _gather_rows_kernel(
    out_ptr,
    src_ptr,
    indices_ptr,
    unused_ptr,
    width,
    src_stride,
    indices_stride,
    unused_stride,
    ROW: tl.constexpr,
    BLOCK_ROW: tl.constexpr,
    BLOCK_SLOTS: tl.constexpr,
):
    token = tl.program_id(0)
    slots = tl.program_id(1) * BLOCK_SLOTS + tl.arange(0, BLOCK_SLOTS)
    columns = tl.arange(0, BLOCK_ROW)
    in_width = slots < width
    skip = tl.load(
        unused_ptr + token * unused_stride + slots, mask=in_width, other=1
    ).to(tl.int1)
    rows = tl.load(
        indices_ptr + token * indices_stride + slots, mask=in_width & ~skip, other=0
    ).to(tl.int64)
    in_row = columns[None, :] < ROW
    # An unused slot gets zeros, not the row its stale index names: that row
    # may be unwritten workspace holding NaN, and 0 * NaN in PV is NaN.
    values = tl.load(
        src_ptr + rows[:, None] * src_stride + columns[None, :],
        mask=(~skip)[:, None] & in_row,
        other=0.0,
    )
    out = out_ptr + (token.to(tl.int64) * width + slots)[:, None] * ROW
    tl.store(out + columns[None, :], values, mask=in_width[:, None] & in_row)


def gather_rows(
    keys: torch.Tensor, src: torch.Tensor, indices: torch.Tensor, unused: torch.Tensor
) -> None:
    """keys [T, W, D] = src [S, D] rows named by indices [T, W], zero where
    unused [T, W] is set; unused indices are never read."""
    num_tokens, width = indices.shape
    row = src.shape[-1]
    assert keys.is_contiguous() and keys.shape == (num_tokens, width, row)
    assert src.stride(-1) == 1 and indices.stride(-1) == 1 and unused.stride(-1) == 1
    _gather_rows_kernel[(num_tokens, triton.cdiv(width, _GATHER_SLOTS))](
        keys,
        src,
        indices,
        unused,
        width,
        src.stride(0),
        indices.stride(0),
        unused.stride(0),
        ROW=row,
        BLOCK_ROW=triton.next_power_of_2(row),
        BLOCK_SLOTS=_GATHER_SLOTS,
    )


@triton.jit
def _sink_softmax_kernel(
    scores_ptr,
    unused_ptr,
    sink_ptr,
    probs_ptr,
    scale,
    num_heads,
    width,
    unused_stride,
    BLOCK_W: tl.constexpr,
):
    row = tl.program_id(0)
    token = row // num_heads
    slots = tl.arange(0, BLOCK_W)
    in_width = slots < width
    skip = tl.load(
        unused_ptr + token * unused_stride + slots, mask=in_width, other=1
    ).to(tl.int1)
    logits = tl.load(
        scores_ptr + row.to(tl.int64) * width + slots, mask=in_width, other=0.0
    )
    logits = tl.where(skip, float("-inf"), logits * scale)
    # The sink is one more logit that only feeds the denominator.
    sink = tl.load(sink_ptr + row % num_heads).to(tl.float32)
    peak = tl.maximum(tl.max(logits, axis=0), sink)
    weights = tldevice.exp(logits - peak)
    denominator = tl.sum(weights, axis=0) + tldevice.exp(sink - peak)
    tl.store(
        probs_ptr + row.to(tl.int64) * width + slots,
        (weights / denominator).to(probs_ptr.dtype.element_ty),
        mask=in_width,
    )


def sparse_attn_prefill_bmm(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    lengths: torch.Tensor,
    scale: float,
    attn_sink: torch.Tensor,
    output: torch.Tensor,
    keys: torch.Tensor,
    scores: torch.Tensor,
    probs: torch.Tensor,
) -> None:
    """q [T, H, D], kv [S, D], indices [T, W] into kv, lengths [T], attn_sink
    [H]. A slot is unused when it holds -1 or lies at or past its token's
    length; past the length it may hold anything, since callers fill the
    indices into a reused workspace. The three buffers come from the workspace
    specs and may be larger than this call needs; they hold
    `MAX_TOKENS_PER_PASS` tokens, so a longer chunk takes several passes. The
    softmax runs in float32 over the keys plus one sink column that only feeds
    the denominator."""
    total_tokens, width = indices.shape
    num_heads = q.shape[1]
    unused = indices < 0
    unused |= torch.arange(width, device=indices.device)[None, :] >= lengths[:, None]

    for start in range(0, total_tokens, MAX_TOKENS_PER_PASS):
        stop = min(start + MAX_TOKENS_PER_PASS, total_tokens)
        num_tokens = stop - start
        pass_keys = _fit(keys, (num_tokens, width, kv.shape[-1]))
        gather_rows(pass_keys, kv, indices[start:stop], unused[start:stop])
        attend_gathered_keys(
            q[start:stop],
            pass_keys,
            unused[start:stop],
            scale,
            attn_sink,
            output[start:stop],
            _fit(scores, (num_tokens, num_heads, width)),
            _fit(probs, (num_tokens, num_heads, width)),
        )


def attend_gathered_keys(
    q: torch.Tensor,
    keys: torch.Tensor,
    unused: torch.Tensor,
    scale: float,
    attn_sink: torch.Tensor,
    output: torch.Tensor,
    scores: torch.Tensor,
    probs: torch.Tensor,
) -> None:
    """Attention of q [T, H, D] over its own gathered keys [T, W, D]; unused
    [T, W] marks slots that hold no key, and those must hold zeros (their
    weight is zero, but 0 * NaN is NaN). Buffers are exactly sized."""
    num_tokens, num_heads, _ = q.shape
    width = keys.shape[1]
    # Preserve the existing FP32 QK score boundary before softmax. Rounding
    # scores to FP16 here can overflow even when the final attention is finite.
    torch.bmm(q, keys.transpose(1, 2), out=scores, out_dtype=torch.float32)
    # Scale, mask, sink column and softmax in one pass over the FP32 scores;
    # as separate torch ops they took longer than both matmuls together.
    block_w = triton.next_power_of_2(width)
    _sink_softmax_kernel[(num_tokens * num_heads,)](
        scores,
        unused,
        attn_sink,
        probs,
        scale,
        num_heads,
        width,
        unused.stride(0),
        BLOCK_W=block_w,
        num_warps=min(8, max(1, block_w // 256)),
    )
    # Existing HMMA attention also converts probabilities to the key dtype
    # before PV; keep that precision boundary rather than narrowing QK.
    torch.bmm(probs, keys, out=output)
