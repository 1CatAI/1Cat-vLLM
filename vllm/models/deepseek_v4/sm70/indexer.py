# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""DeepSeek V4 C4 indexer fallback using FP16 HMMA on SM70."""

import os

import torch

from vllm.models.deepseek_v4.common.ops.fp8_software import (
    fp8_e4m3fn_bits_to_fp32,
)
from vllm.triton_utils import tl, triton

_INDEX_HEAD_DIM = 128
_INDEX_CACHE_BYTES = _INDEX_HEAD_DIM + 4

# Fuse dequant+scoring into one paged kernel instead of materialising
# [rows, max_seq_len, 128] fp16 and running a dense bmm. Set to 0 to get the
# old gather+bmm path back for an A/B. Only affects decode; prefill scores a
# contiguous chunk whose width is the real chunk length already.
_FUSED_DECODE_LOGITS = os.getenv("VLLM_SM70_INDEXER_FUSED_LOGITS", "1") == "1"

# Score with the model's actual indexer function,
#     I[t, s] = sum_h weights[t, h] * relu(q[t, h] . k[s]),
# instead of the cheaper (sum_h weights[t, h] * q[t, h]) . k[s]. Those are
# equal only without the relu, and the relu is not optional: it is what the
# checkpoint was trained with (reference `Indexer.forward`,
# inference/model.py:427, and the in-repo `fp8_mqa_logits` reference at
# v1/attention/ops/rocm_aiter_mla_sparse.py:510 -- every non-SM70 backend
# applies it inside its logits kernel). Dropping it costs 64x fewer flops and
# ranks the wrong keys, which only shows up once top-k is actually selective.
# Set 0 to restore the factored form for an A/B.
_RELU_LOGITS = os.getenv("VLLM_SM70_INDEXER_RELU", "1") == "1"

# Key-axis splits are chosen so the launch is roughly this many blocks, which
# keeps all 80 SMs busy at long context without paying for thousands of
# no-op programs at short context.
_LOGITS_TARGET_BLOCKS = 512
_LOGITS_BLOCK_N = 32
# The relu path multiplies q against k with an MMA instead of a broadcast
# multiply, so the key tile has to be at least one 16x16x16 HMMA fragment
# wide; 64 keeps the [64 heads, 128] x [128, N] shape efficient.
_RELU_BLOCK_N = 64
_RELU_BLOCK_M = 32

# Route prefill scoring through cuBLAS instead of the Triton MMA. Set 0 for an
# A/B against the Triton kernel above (which stays as the reference shape and
# is what the decode path still uses, since decode gathers from a paged cache).
_PREFILL_CUBLAS = os.getenv("VLLM_SM70_INDEXER_PREFILL_CUBLAS", "1") == "1"
# Cap (MiB) on the [tokens*heads, key_tile] fp16 score tile. At the default
# 2048-token prefill chunk and 64 heads this is 128 KiB per key, so the tile
# is what keeps a long-context chunk from asking for gigabytes in one go.
_PREFILL_TILE_MB = int(os.getenv("VLLM_SM70_INDEXER_PREFILL_TILE_MB", "192"))
_EPILOGUE_BLOCK_K = 256
_EPILOGUE_BLOCK_H = 8


@triton.jit
def _index_cache_ptrs(
    cache_ptr,
    physical_block,
    pos_in_block,
    block_stride,
    cache_block_size,
    head_dim: tl.constexpr,
):
    """Value and scale pointers for one token of the paged indexer cache.

    The layout is block-major, NOT token-interleaved: within a block come all
    `cache_block_size` rows of FP8 values, and only then the FP32 scales, one
    per token (csrc/libtorch_stable/cache_kernels.cu, the
    `indexer_k_quant_and_cache` store and the `cp_gather_indexer_k_quant_cache`
    load both address it this way). Reading a token as
    `[128 values][4 scale]` at `pos * cache.stride(1)` therefore lands on the
    wrong bytes for every token but the first of each block, and the "scale"
    it picks up is really four FP8 value bytes.
    """
    base = cache_ptr + physical_block.to(tl.int64) * block_stride
    value_ptr = base + pos_in_block.to(tl.int64) * head_dim
    scale_ptr = (base + cache_block_size * head_dim + pos_in_block * 4).to(
        tl.pointer_type(tl.float32)
    )
    return value_ptr, scale_ptr


@triton.jit
def _weighted_query_kernel(
    q_ptr,
    weights_ptr,
    out_ptr,
    q_stride0,
    q_stride1,
    weights_stride0,
    out_stride0,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    head_offsets = tl.arange(0, num_heads)
    dim_offsets = block_idx * BLOCK_D + tl.arange(0, BLOCK_D)
    dim_mask = dim_offsets < head_dim
    q = tl.load(
        q_ptr
        + token_idx * q_stride0
        + head_offsets[:, None] * q_stride1
        + dim_offsets[None, :],
        mask=dim_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    weights = tl.load(weights_ptr + token_idx * weights_stride0 + head_offsets).to(
        tl.float32
    )
    combined = tl.sum(q * weights[:, None], axis=0)
    tl.store(
        out_ptr + token_idx * out_stride0 + dim_offsets,
        combined.to(out_ptr.type.element_ty),
        mask=dim_mask,
    )


@triton.jit
def _dequant_contiguous_index_k_kernel(
    k_ptr,
    scale_ptr,
    out_ptr,
    k_stride0,
    out_stride0,
    head_dim: tl.constexpr,
):
    row_idx = tl.program_id(0)
    offsets = tl.arange(0, head_dim)
    packed_ptr = k_ptr.to(tl.pointer_type(tl.uint8))
    values = tl.load(packed_ptr + row_idx * k_stride0 + offsets)
    scale = tl.load(scale_ptr + row_idx).to(tl.float32)
    dequant = fp8_e4m3fn_bits_to_fp32(values) * scale
    tl.store(
        out_ptr + row_idx * out_stride0 + offsets,
        dequant.to(out_ptr.type.element_ty),
    )


@triton.jit
def _dequant_paged_index_k_kernel(
    cache_ptr,
    block_table_ptr,
    seq_lens_ptr,
    out_ptr,
    cache_stride0,
    block_table_stride0,
    out_stride0,
    out_stride1,
    cache_block_size,
    max_seq_len,
    head_dim: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row_idx = tl.program_id(0)
    key_offsets = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    dim_offsets = tl.arange(0, head_dim)
    seq_len = tl.load(seq_lens_ptr + row_idx)
    valid = (key_offsets < seq_len) & (key_offsets < max_seq_len)
    block_in_seq = key_offsets // cache_block_size
    pos_in_block = key_offsets % cache_block_size
    physical_block = tl.load(
        block_table_ptr + row_idx * block_table_stride0 + block_in_seq,
        mask=valid,
        other=0,
    )
    token_ptr, scale_ptr = _index_cache_ptrs(
        cache_ptr,
        physical_block,
        pos_in_block,
        cache_stride0,
        cache_block_size,
        head_dim,
    )
    packed = tl.load(
        token_ptr[:, None] + dim_offsets[None, :],
        mask=valid[:, None],
        other=0,
    )
    fp8 = fp8_e4m3fn_bits_to_fp32(packed)
    scale = tl.load(scale_ptr, mask=valid, other=0.0).to(tl.float32)
    dequant = fp8 * scale[:, None]
    tl.store(
        out_ptr
        + row_idx * out_stride0
        + key_offsets[:, None] * out_stride1
        + dim_offsets[None, :],
        dequant.to(out_ptr.type.element_ty),
        mask=valid[:, None],
    )


@triton.jit
def _paged_index_logits_kernel(
    q_ptr,
    cache_ptr,
    block_table_ptr,
    seq_lens_ptr,
    out_ptr,
    q_stride0,
    cache_stride0,
    block_table_stride0,
    out_stride0,
    cache_block_size,
    head_dim: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
):
    """Index logits straight out of the paged FP8 cache, no gather buffer.

    `_dequant_paged_index_k_kernel` + bmm has a grid and a GEMM shape derived
    from `max_seq_len`, and under a captured decode CUDA Graph that is frozen
    at `max_model_len` forever. Here the grid is static -- (rows, NUM_SPLITS),
    sized from `max_seq_len` on the host -- and only each program's trip count
    follows the live `seq_len`, so one captured graph serves any context while
    a short context costs almost nothing.
    """
    row_idx = tl.program_id(0)
    split_id = tl.program_id(1)
    dim_offsets = tl.arange(0, head_dim)
    q = tl.load(q_ptr + row_idx * q_stride0 + dim_offsets).to(tl.float32)

    seq_len = tl.load(seq_lens_ptr + row_idx)
    chunks = (seq_len + BLOCK_N - 1) // BLOCK_N
    chunks_per_split = (chunks + NUM_SPLITS - 1) // NUM_SPLITS
    lo = split_id * chunks_per_split * BLOCK_N
    hi = tl.minimum(lo + chunks_per_split * BLOCK_N, seq_len)

    for start in range(lo, hi, BLOCK_N):
        key_offsets = start + tl.arange(0, BLOCK_N)
        valid = key_offsets < seq_len
        block_in_seq = key_offsets // cache_block_size
        pos_in_block = key_offsets % cache_block_size
        physical_block = tl.load(
            block_table_ptr + row_idx * block_table_stride0 + block_in_seq,
            mask=valid,
            other=0,
        )
        token_ptr, scale_ptr = _index_cache_ptrs(
            cache_ptr,
            physical_block,
            pos_in_block,
            cache_stride0,
            cache_block_size,
            head_dim,
        )
        packed = tl.load(
            token_ptr[:, None] + dim_offsets[None, :],
            mask=valid[:, None],
            other=0,
        )
        fp8 = fp8_e4m3fn_bits_to_fp32(packed)
        scale = tl.load(scale_ptr, mask=valid, other=0.0).to(tl.float32)
        # Per-token scale factors out of the dot product.
        logits = tl.sum(fp8 * q[None, :], axis=1) * scale
        tl.store(out_ptr + row_idx * out_stride0 + key_offsets, logits, mask=valid)


@triton.jit
def _contiguous_index_logits_relu_kernel(
    q_ptr,
    w_ptr,
    k_ptr,
    k_scale_ptr,
    out_ptr,
    q_stride0,
    q_stride1,
    w_stride0,
    k_stride0,
    out_stride0,
    num_q,
    num_k,
    num_heads,
    head_dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Prefill index scores, per head, with the relu inside the head sum.

    Reference shape for the scoring, and the A/B fallback for the cuBLAS path
    (`VLLM_SM70_INDEXER_PREFILL_CUBLAS=0`). It is not the fast path: Triton's
    `tl.dot` does not reach Volta's tensor cores, so this tops out near 3.3
    TFLOP/s no matter how the tile is shaped -- folding heads into the M
    dimension to get a [256, 128] x [128, 64] MMA instead of a per-head
    [32, 128] x [128, 64] one measured 3.34 vs 3.39 TFLOP/s, i.e. nothing.
    cuBLAS does the same shape at 54.

    The key tile is loaded once and reused by every head (the indexer is MQA).
    FP8 keys are widened to FP16 *unscaled* -- e4m3 maxes out at 448, which
    fp16 holds exactly -- and the per-key scale is applied to the finished
    accumulator. That keeps `k * scale` out of fp16 entirely, where a large
    ue8m0 scale could otherwise overflow.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    dim_offsets = tl.arange(0, head_dim)
    m_valid = offs_m < num_q
    n_valid = offs_n < num_k

    packed = tl.load(
        k_ptr + offs_n[:, None] * k_stride0 + dim_offsets[None, :],
        mask=n_valid[:, None],
        other=0,
    )
    k_t = tl.trans(fp8_e4m3fn_bits_to_fp32(packed).to(tl.float16))
    scale = tl.load(k_scale_ptr + offs_n, mask=n_valid, other=0.0).to(tl.float32)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for head in range(num_heads):
        q = tl.load(
            q_ptr
            + offs_m[:, None] * q_stride0
            + head * q_stride1
            + dim_offsets[None, :],
            mask=m_valid[:, None],
            other=0.0,
        )
        w = tl.load(w_ptr + offs_m * w_stride0 + head, mask=m_valid, other=0.0)
        acc += tl.maximum(tl.dot(q, k_t), 0.0) * w[:, None].to(tl.float32)

    tl.store(
        out_ptr + offs_m[:, None] * out_stride0 + offs_n[None, :],
        acc * scale[None, :],
        mask=m_valid[:, None] & n_valid[None, :],
    )


@triton.jit
def _paged_index_logits_relu_kernel(
    q_ptr,
    w_ptr,
    cache_ptr,
    block_table_ptr,
    seq_lens_ptr,
    out_ptr,
    q_stride0,
    q_stride1,
    w_stride0,
    cache_stride0,
    block_table_stride0,
    out_stride0,
    cache_block_size,
    head_dim: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
):
    """Decode counterpart of `_contiguous_index_logits_relu_kernel`.

    Same static-grid property as `_paged_index_logits_kernel`: the grid is
    (rows, NUM_SPLITS) and only the trip count follows the live `seq_len`, so
    one captured CUDA graph serves any context length.
    """
    row_idx = tl.program_id(0)
    split_id = tl.program_id(1)
    head_offsets = tl.arange(0, NUM_HEADS)
    dim_offsets = tl.arange(0, head_dim)
    q = tl.load(
        q_ptr
        + row_idx * q_stride0
        + head_offsets[:, None] * q_stride1
        + dim_offsets[None, :]
    )
    w = tl.load(w_ptr + row_idx * w_stride0 + head_offsets).to(tl.float32)

    seq_len = tl.load(seq_lens_ptr + row_idx)
    chunks = (seq_len + BLOCK_N - 1) // BLOCK_N
    chunks_per_split = (chunks + NUM_SPLITS - 1) // NUM_SPLITS
    lo = split_id * chunks_per_split * BLOCK_N
    hi = tl.minimum(lo + chunks_per_split * BLOCK_N, seq_len)

    for start in range(lo, hi, BLOCK_N):
        key_offsets = start + tl.arange(0, BLOCK_N)
        valid = key_offsets < seq_len
        block_in_seq = key_offsets // cache_block_size
        pos_in_block = key_offsets % cache_block_size
        physical_block = tl.load(
            block_table_ptr + row_idx * block_table_stride0 + block_in_seq,
            mask=valid,
            other=0,
        )
        token_ptr, scale_ptr = _index_cache_ptrs(
            cache_ptr,
            physical_block,
            pos_in_block,
            cache_stride0,
            cache_block_size,
            head_dim,
        )
        packed = tl.load(
            token_ptr[:, None] + dim_offsets[None, :],
            mask=valid[:, None],
            other=0,
        )
        k_t = tl.trans(fp8_e4m3fn_bits_to_fp32(packed).to(tl.float16))
        scale = tl.load(scale_ptr, mask=valid, other=0.0).to(tl.float32)
        # relu(q_h . k) is per head; only then does the weighted head sum
        # collapse it to one logit. The per-key scale is positive, so pulling
        # it out past the relu and the sum is exact.
        scores = tl.maximum(tl.dot(q, k_t), 0.0) * w[:, None]
        logits = tl.sum(scores, axis=0) * scale
        tl.store(out_ptr + row_idx * out_stride0 + key_offsets, logits, mask=valid)


def _combine_index_queries(q: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    assert q.dtype == torch.float16 and q.ndim == 3
    assert q.shape[-1] == _INDEX_HEAD_DIM
    assert weights.shape == q.shape[:2]
    out = torch.empty((q.shape[0], q.shape[-1]), dtype=torch.float16, device=q.device)
    block_d = 32
    _weighted_query_kernel[(q.shape[0], triton.cdiv(q.shape[-1], block_d))](
        q,
        weights,
        out,
        q.stride(0),
        q.stride(1),
        weights.stride(0),
        out.stride(0),
        num_heads=q.shape[1],
        head_dim=q.shape[2],
        BLOCK_D=block_d,
        num_warps=4,
    )
    return out


@triton.jit
def _relu_weight_headsum_kernel(
    s_ptr,
    w_ptr,
    scale_ptr,
    out_ptr,
    s_stride0,
    s_stride1,
    w_stride0,
    out_stride0,
    num_k,
    num_heads: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """relu -> weight -> head sum -> key scale, in a single pass over scores.

    Epilogue for the cuBLAS prefill path. Doing it as torch ops instead costs
    three extra round trips through the [tokens, heads, keys] score tile, which
    is the largest tensor in the whole indexer; fusing them measured 13 -> 26
    TFLOP/s end to end.
    """
    token = tl.program_id(0)
    k_base = tl.program_id(1) * BLOCK_K
    k_offsets = k_base + tl.arange(0, BLOCK_K)
    k_valid = k_offsets < num_k

    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for head in range(0, num_heads, BLOCK_H):
        h_offsets = head + tl.arange(0, BLOCK_H)
        s = tl.load(
            s_ptr
            + token * s_stride0
            + h_offsets[:, None] * s_stride1
            + k_offsets[None, :],
            mask=k_valid[None, :],
            other=0.0,
        ).to(tl.float32)
        w = tl.load(w_ptr + token * w_stride0 + h_offsets).to(tl.float32)
        acc += tl.sum(tl.maximum(s, 0.0) * w[:, None], axis=0)

    scale = tl.load(scale_ptr + k_offsets, mask=k_valid, other=0.0).to(tl.float32)
    tl.store(out_ptr + token * out_stride0 + k_offsets, acc * scale, mask=k_valid)


@triton.jit
def _dequant_index_k_unscaled_kernel(
    k_ptr,
    out_ptr,
    k_stride0,
    out_stride0,
    head_dim: tl.constexpr,
):
    """Widen FP8 index keys to FP16 without applying the per-key scale.

    e4m3 tops out at 448 and the scale is folded into the epilogue instead, so
    the GEMM never sees `k * scale` -- a large ue8m0 scale would otherwise
    overflow fp16 on the way in.
    """
    row_idx = tl.program_id(0)
    offsets = tl.arange(0, head_dim)
    values = tl.load(
        k_ptr.to(tl.pointer_type(tl.uint8)) + row_idx * k_stride0 + offsets
    )
    tl.store(
        out_ptr + row_idx * out_stride0 + offsets,
        fp8_e4m3fn_bits_to_fp32(values).to(tl.float16),
    )


def _prefill_logits_cublas(q, k_quant, k_scales, weights):
    """Prefill index scores through cuBLAS, with the relu inside the head sum.

    Triton's `tl.dot` does not reach Volta's tensor cores: on the exact shape
    this needs it measures 3.3 TFLOP/s, against 54 TFLOP/s for cuBLAS fp16 and
    13 TFLOP/s for plain fp32 CUDA cores. The relu blocks the one-GEMM
    factorisation, but it does not block cuBLAS -- stacking heads into the M
    dimension turns the scoring into a single [tokens*heads, 128] x [128, keys]
    GEMM whose output is reduced over heads afterwards. Keys are tiled so that
    intermediate stays bounded; it is the largest allocation in the indexer.
    """
    num_q, num_heads, head_dim = q.shape
    num_k = k_quant.shape[0]
    out = torch.empty((num_q, num_k), dtype=torch.float32, device=q.device)

    k_fp16 = torch.empty((num_k, head_dim), dtype=torch.float16, device=q.device)
    _dequant_index_k_unscaled_kernel[(num_k,)](
        k_quant.view(torch.uint8),
        k_fp16,
        k_quant.stride(0),
        k_fp16.stride(0),
        head_dim=head_dim,
        num_warps=4,
    )

    q2 = q.reshape(num_q * num_heads, head_dim)
    if not q2.is_contiguous():
        q2 = q2.contiguous()

    rows = num_q * num_heads
    budget = max(1, _PREFILL_TILE_MB * 2**20 // max(1, rows * 2))
    k_tile = max(_RELU_BLOCK_N, min(num_k, budget // _RELU_BLOCK_N * _RELU_BLOCK_N))

    for start in range(0, num_k, k_tile):
        stop = min(start + k_tile, num_k)
        width = stop - start
        # fp16 out with fp32 accumulate: the epilogue reduces in fp32 and only
        # the top-k *set* is consumed downstream, so the 5e-4 rounding here is
        # far below the gap between neighbouring index scores.
        scores = torch.mm(q2, k_fp16[start:stop].t())
        _relu_weight_headsum_kernel[(num_q, triton.cdiv(width, _EPILOGUE_BLOCK_K))](
            scores,
            weights,
            k_scales[start:stop],
            out[:, start:stop],
            scores.stride(0) * num_heads,
            scores.stride(0),
            weights.stride(0),
            out.stride(0),
            width,
            num_heads=num_heads,
            BLOCK_K=_EPILOGUE_BLOCK_K,
            BLOCK_H=_EPILOGUE_BLOCK_H,
            num_warps=4,
        )
    return out


def sm70_indexer_prefill_logits(
    q: torch.Tensor,
    k_quant: torch.Tensor,
    k_scale_storage: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Compute all prefill index scores; caller supplies causal row bounds."""
    assert k_quant.dtype == torch.float8_e4m3fn
    assert k_quant.ndim == 2 and k_quant.shape[1] == _INDEX_HEAD_DIM
    k_scales = k_scale_storage.view(torch.float32).reshape(-1)
    assert k_scales.shape[0] == k_quant.shape[0]

    if _RELU_LOGITS and _PREFILL_CUBLAS:
        return _prefill_logits_cublas(q, k_quant, k_scales, weights)

    if _RELU_LOGITS:
        num_q, num_heads, _ = q.shape
        num_k = k_quant.shape[0]
        out = torch.empty((num_q, num_k), dtype=torch.float32, device=q.device)
        block_m, block_n = _RELU_BLOCK_M, _RELU_BLOCK_N
        _contiguous_index_logits_relu_kernel[
            (triton.cdiv(num_q, block_m), triton.cdiv(num_k, block_n))
        ](
            q,
            weights,
            k_quant.view(torch.uint8),
            k_scales,
            out,
            q.stride(0),
            q.stride(1),
            weights.stride(0),
            k_quant.stride(0),
            out.stride(0),
            num_q,
            num_k,
            num_heads,
            head_dim=_INDEX_HEAD_DIM,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=8,
        )
        return out

    weighted_q = _combine_index_queries(q, weights)
    k_fp16 = torch.empty(k_quant.shape, dtype=torch.float16, device=k_quant.device)
    _dequant_contiguous_index_k_kernel[(k_quant.shape[0],)](
        k_quant.view(torch.uint8),
        k_scales,
        k_fp16,
        k_quant.stride(0),
        k_fp16.stride(0),
        head_dim=_INDEX_HEAD_DIM,
        num_warps=4,
    )
    return torch.mm(weighted_q, k_fp16.t(), out_dtype=torch.float32)


def sm70_indexer_decode_logits(
    q: torch.Tensor,
    cache: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    max_seq_len: int,
) -> torch.Tensor:
    """Gather paged FP8 index keys and compute batched decode scores."""
    assert cache.dtype == torch.uint8 and cache.ndim == 3
    assert cache.shape[-1] >= _INDEX_CACHE_BYTES
    # The relu path scores per head, so it needs q as [rows, heads, dim]; the
    # factored path collapses the head axis up front.
    weighted_q = q if _RELU_LOGITS else _combine_index_queries(q, weights)

    if seq_lens.ndim == 2:
        next_n = seq_lens.shape[1]
        flat_lens = seq_lens.reshape(-1).to(torch.int32)
        block_table = block_table.repeat_interleave(next_n, dim=0)
    else:
        flat_lens = seq_lens.reshape(-1).to(torch.int32)
    assert flat_lens.shape[0] == weighted_q.shape[0]
    assert block_table.shape[0] == weighted_q.shape[0]

    max_seq_len = max(1, int(max_seq_len))
    total_rows = weighted_q.shape[0]

    if _RELU_LOGITS:
        out = torch.empty(
            (total_rows, max_seq_len), dtype=torch.float32, device=q.device
        )
        max_chunks = triton.cdiv(max_seq_len, _RELU_BLOCK_N)
        splits = max(1, min(max_chunks, _LOGITS_TARGET_BLOCKS // total_rows))
        _paged_index_logits_relu_kernel[(total_rows, splits)](
            q,
            weights,
            cache,
            block_table,
            flat_lens,
            out,
            q.stride(0),
            q.stride(1),
            weights.stride(0),
            cache.stride(0),
            block_table.stride(0),
            out.stride(0),
            cache.shape[1],
            head_dim=_INDEX_HEAD_DIM,
            NUM_HEADS=q.shape[1],
            BLOCK_N=_RELU_BLOCK_N,
            NUM_SPLITS=splits,
            num_warps=8,
        )
        # Positions >= seq_len are left uninitialised; every consumer masks by
        # seq_lens (see the note on the factored path below).
        return out

    if _FUSED_DECODE_LOGITS:
        out = torch.empty(
            (total_rows, max_seq_len), dtype=torch.float32, device=q.device
        )
        max_chunks = triton.cdiv(max_seq_len, _LOGITS_BLOCK_N)
        splits = max(1, min(max_chunks, _LOGITS_TARGET_BLOCKS // total_rows))
        _paged_index_logits_kernel[(total_rows, splits)](
            weighted_q,
            cache,
            block_table,
            flat_lens,
            out,
            weighted_q.stride(0),
            cache.stride(0),
            block_table.stride(0),
            out.stride(0),
            cache.shape[1],
            head_dim=_INDEX_HEAD_DIM,
            BLOCK_N=_LOGITS_BLOCK_N,
            NUM_SPLITS=splits,
            num_warps=4,
        )
        # Positions >= seq_len are left uninitialised, exactly as the
        # single-chunk gather path leaves them; every consumer masks by
        # seq_lens.
        return out

    block_n = 16
    gathered_k = torch.empty(
        (total_rows, max_seq_len, _INDEX_HEAD_DIM),
        dtype=torch.float16,
        device=q.device,
    )
    _dequant_paged_index_k_kernel[(total_rows, triton.cdiv(max_seq_len, block_n))](
        cache,
        block_table,
        flat_lens,
        gathered_k,
        cache.stride(0),
        block_table.stride(0),
        gathered_k.stride(0),
        gathered_k.stride(1),
        cache.shape[1],
        max_seq_len,
        head_dim=_INDEX_HEAD_DIM,
        BLOCK_N=block_n,
        num_warps=4,
    )
    return torch.bmm(
        weighted_q.unsqueeze(1),
        gathered_k.transpose(1, 2),
        out_dtype=torch.float32,
    ).squeeze(1)
