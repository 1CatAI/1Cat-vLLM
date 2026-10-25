# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Small-batch sparse FP16 attention with FP32 scores and tensor-core GEMM."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _gather_sparse_fp16_kernel(
    kv_ptr,
    indices_ptr,
    lengths_ptr,
    gathered_ptr,
    kv_stride,
    indices_stride,
    gathered_stride,
    num_slots,
    WIDTH: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    positions = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    dims = tl.arange(0, DIM)
    length = tl.load(lengths_ptr + row)
    slots = tl.load(
        indices_ptr + row * indices_stride + positions,
        mask=positions < WIDTH,
        other=-1,
    )
    valid = (positions < WIDTH) & (positions < length)
    valid &= (slots >= 0) & (slots < num_slots)
    safe_slots = tl.where(valid, slots, 0)
    values = tl.load(
        kv_ptr + safe_slots[:, None] * kv_stride + dims[None, :],
        mask=valid[:, None],
        other=0.0,
    )
    tl.store(
        gathered_ptr + row * gathered_stride + positions[:, None] * DIM + dims[None, :],
        values,
        mask=(positions < WIDTH)[:, None],
    )


@triton.jit
def _sparse_fp32_scores_softmax_kernel(
    scores_ptr,
    indices_ptr,
    lengths_ptr,
    probs_ptr,
    indices_stride,
    num_slots,
    scale,
    HEADS: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row, head = tl.program_id(0), tl.program_id(1)
    positions = tl.arange(0, BLOCK_K)
    length = tl.load(lengths_ptr + row)
    slots = tl.load(
        indices_ptr + row * indices_stride + positions,
        mask=positions < WIDTH,
        other=-1,
    )
    valid = (positions < WIDTH) & (positions < length)
    valid &= (slots >= 0) & (slots < num_slots)
    base = (row * HEADS + head) * WIDTH + positions
    scores = tl.load(scores_ptr + base, mask=positions < WIDTH, other=0.0)
    scores = tl.where(valid, scores * scale, -3.4028234663852886e38)
    numerator = tl.where(valid, tl.exp(scores - tl.max(scores, 0)), 0.0)
    denominator = tl.sum(numerator, 0)
    probs = numerator / tl.maximum(denominator, 1.0e-30)
    tl.store(probs_ptr + base, probs, mask=positions < WIDTH)


def sparse_attention_fp16_gemm(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    lengths: torch.Tensor,
    scale: float,
    out: torch.Tensor,
    gathered_kv: torch.Tensor,
    scores: torch.Tensor,
    probs: torch.Tensor,
    *,
    use_batched_gemm: bool = False,
) -> None:
    """NoPE sparse attention; caller owns all scratch buffers before capture.

    Default to per-token GEMM geometries so speculative batch size does not
    change the reduction order. Explicit prefill callers may batch the GEMMs.
    QK stays FP32 through softmax to avoid overflowing finite inputs in FP16.
    """
    tokens, heads, dim = q.shape
    kv = kv.reshape(-1, dim)
    indices = indices.reshape(tokens, -1)
    lengths = lengths.reshape(-1)
    width = indices.shape[1]
    assert q.dtype == kv.dtype == out.dtype == torch.float16
    assert dim == 512 and out.shape == q.shape
    assert q.stride(-1) == kv.stride(-1) == out.stride(-1) == 1
    assert indices.dtype == lengths.dtype == torch.int32
    assert indices.stride(-1) == 1 and lengths.is_contiguous()
    assert lengths.shape == (tokens,)
    assert gathered_kv.shape == (tokens, width, dim)
    assert scores.shape == probs.shape == (tokens, heads, width)
    assert gathered_kv.dtype == probs.dtype == torch.float16
    assert scores.dtype == torch.float32
    assert gathered_kv.is_contiguous() and scores.is_contiguous()
    assert probs.is_contiguous()

    _gather_sparse_fp16_kernel[(tokens, triton.cdiv(width, 8))](
        kv,
        indices,
        lengths,
        gathered_kv,
        kv.stride(0),
        indices.stride(0),
        gathered_kv.stride(0),
        kv.shape[0],
        WIDTH=width,
        DIM=dim,
        BLOCK_K=8,
        num_warps=4,
    )
    if use_batched_gemm:
        torch.bmm(
            q,
            gathered_kv.transpose(1, 2),
            out_dtype=torch.float32,
            out=scores,
        )
    else:
        for token in range(tokens):
            torch.mm(
                q[token],
                gathered_kv[token].t(),
                out_dtype=torch.float32,
                out=scores[token],
            )
    _sparse_fp32_scores_softmax_kernel[(tokens, heads)](
        scores,
        indices,
        lengths,
        probs,
        indices.stride(0),
        kv.shape[0],
        float(scale),
        HEADS=heads,
        WIDTH=width,
        BLOCK_K=triton.next_power_of_2(width),
        num_warps=8,
    )
    if use_batched_gemm:
        torch.bmm(probs, gathered_kv, out=out)
    else:
        for token in range(tokens):
            torch.mm(probs[token], gathered_kv[token], out=out[token])


def sparse_attention_fp16_gemm_chunked(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    lengths: torch.Tensor,
    scale: float,
    out: torch.Tensor,
    gathered_kv: torch.Tensor,
    scores: torch.Tensor,
    probs: torch.Tensor,
) -> None:
    """Bound prefill scratch space while batching the per-query GEMMs.

    Keep the default per-token geometry for speculative decode. The caller must
    prove every query is prefill before selecting this batched route.
    """
    capacity = gathered_kv.shape[0]
    assert capacity > 0 and out.shape == q.shape
    for start in range(0, q.shape[0], capacity):
        end = min(start + capacity, q.shape[0])
        count = end - start
        sparse_attention_fp16_gemm(
            q[start:end],
            kv,
            indices[start:end],
            lengths[start:end],
            scale,
            out[start:end],
            gathered_kv[:count],
            scores[:count],
            probs[:count],
            use_batched_gemm=True,
        )
