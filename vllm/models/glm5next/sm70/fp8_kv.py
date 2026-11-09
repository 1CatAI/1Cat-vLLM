# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Packed E4M3FN KV-cache kernels for GLM-5.3 NoPE MLA on SM70."""

import torch

from vllm.models.deepseek_v4.common.ops.fp8_software import (
    fp8_e4m3fn_bits_to_fp32_bitcast,
    fp32_to_fp8_e4m3fn_bits,
)
from vllm.triton_utils import tl, triton

GLM5_FP8_KV_DIM = 512
GLM5_FP8_KV_GROUP_SIZE = 64
GLM5_FP8_KV_NUM_SCALES = GLM5_FP8_KV_DIM // GLM5_FP8_KV_GROUP_SIZE
GLM5_FP8_KV_SLOT_BYTES = GLM5_FP8_KV_DIM + GLM5_FP8_KV_NUM_SCALES


@triton.jit
def _sm70_glm5_fp8_kv_insert_kernel(
    kv_ptr,
    slot_mapping_ptr,
    cache_ptr,
    cache_stride_block,
    block_size: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
):
    token_idx = tl.program_id(0)
    slot_idx = tl.load(slot_mapping_ptr + token_idx)
    if slot_idx < 0:
        return

    offsets = tl.arange(0, HEAD_DIM)
    values = tl.load(kv_ptr + token_idx * HEAD_DIM + offsets).to(tl.float32)
    grouped = tl.reshape(values, (NUM_GROUPS, GROUP_SIZE))
    absmax = tl.max(tl.abs(grouped), axis=1)
    absmax = tl.maximum(absmax, 1.0e-4)
    exponent = tl.ceil(tl.log2(absmax / 448.0))
    scales = tl.exp2(exponent)
    quantized = tl.clamp(grouped / scales[:, None], -448.0, 448.0)
    packed = fp32_to_fp8_e4m3fn_bits(quantized)

    block_idx = slot_idx // block_size
    pos_in_block = slot_idx % block_size
    block_base = cache_ptr + block_idx.to(tl.int64) * cache_stride_block
    token_data = block_base + pos_in_block * HEAD_DIM
    tl.store(token_data + offsets, tl.reshape(packed, (HEAD_DIM,)))

    scale_offsets = tl.arange(0, NUM_GROUPS)
    encoded_scales = tl.clamp(exponent + 127.0, 0.0, 255.0).to(tl.uint8)
    token_scales = block_base + block_size * HEAD_DIM + pos_in_block * NUM_GROUPS
    tl.store(token_scales + scale_offsets, encoded_scales)


@triton.jit
def _sm70_glm5_sparse_paged_fp8_kernel(
    q_ptr,
    cache_ptr,
    indices_ptr,
    lengths_ptr,
    out_ptr,
    q_stride_t,
    q_stride_h,
    cache_stride_block,
    indices_stride_t,
    out_stride_t,
    out_stride_h,
    num_heads,
    num_cache_slots,
    block_size,
    scale,
    INDEX_WIDTH: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    query_idx = tl.program_id(0)
    head_offsets = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    head_mask = head_offsets < num_heads
    dim_offsets = tl.arange(0, HEAD_DIM)

    q = tl.load(
        q_ptr
        + query_idx * q_stride_t
        + head_offsets[:, None] * q_stride_h
        + dim_offsets[None, :],
        mask=head_mask[:, None],
        other=0.0,
    )

    neg_large = -3.4028234663852886e38
    running_max = tl.full((BLOCK_H,), neg_large, dtype=tl.float32)
    running_sum = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, HEAD_DIM), dtype=tl.float32)
    # Keep the loop bound runtime-visible so CUDA graph replay only scans the
    # selected keys for the current sequence.  INDEX_WIDTH is the fixed graph
    # buffer capacity (2048 for GLM-5.3), not the amount of useful work.
    valid_len = tl.minimum(tl.load(lengths_ptr + query_idx), INDEX_WIDTH)
    key_offsets = tl.arange(0, BLOCK_K)

    for start in tl.range(0, valid_len, BLOCK_K):
        positions = start + key_offsets
        slots = tl.load(
            indices_ptr + query_idx * indices_stride_t + positions,
            mask=positions < INDEX_WIDTH,
            other=-1,
        )
        valid = (positions < valid_len) & (slots >= 0) & (slots < num_cache_slots)
        safe_slots = tl.where(valid, slots, 0)
        block_idx = safe_slots // block_size
        pos_in_block = safe_slots % block_size
        block_base = cache_ptr + block_idx.to(tl.int64) * cache_stride_block
        token_data = block_base + pos_in_block * HEAD_DIM
        token_scales = (
            block_base + block_size * HEAD_DIM + pos_in_block * (HEAD_DIM // GROUP_SIZE)
        )

        packed = tl.load(
            token_data[:, None] + dim_offsets[None, :],
            mask=valid[:, None],
            other=0,
        )
        fp8 = fp8_e4m3fn_bits_to_fp32_bitcast(packed)
        scale_group_offsets = tl.arange(0, HEAD_DIM // GROUP_SIZE)
        encoded_scale = tl.load(
            token_scales[:, None] + scale_group_offsets[None, :],
            mask=valid[:, None],
            other=127,
        )
        dequant_scale = tl.exp2(encoded_scale.to(tl.float32) - 127.0)
        fp8_grouped = tl.reshape(fp8, (BLOCK_K, HEAD_DIM // GROUP_SIZE, GROUP_SIZE))
        scale_grouped = tl.reshape(
            dequant_scale.to(tl.float16),
            (BLOCK_K, HEAD_DIM // GROUP_SIZE, 1),
        )
        kv = tl.reshape(
            fp8_grouped.to(tl.float16) * scale_grouped,
            (BLOCK_K, HEAD_DIM),
        )
        kv = tl.where(valid[:, None], kv, 0.0)

        scores = tl.dot(q, tl.trans(kv)) * scale
        scores = tl.where(head_mask[:, None] & valid[None, :], scores, neg_large)
        block_max = tl.max(scores, axis=1)
        new_max = tl.maximum(running_max, block_max)
        alpha = tl.exp(running_max - new_max)
        probs = tl.exp(scores - new_max[:, None])
        probs = tl.where(head_mask[:, None] & valid[None, :], probs, 0.0)
        acc = acc * alpha[:, None] + tl.dot(probs.to(tl.float16), kv)
        running_sum = running_sum * alpha + tl.sum(probs, axis=1)
        running_max = new_max

    denom = tl.maximum(running_sum, 1.0e-30)
    result = tl.where(running_sum[:, None] > 0.0, acc / denom[:, None], 0.0)
    tl.store(
        out_ptr
        + query_idx * out_stride_t
        + head_offsets[:, None] * out_stride_h
        + dim_offsets[None, :],
        result,
        mask=head_mask[:, None],
    )


@triton.jit
def _sm70_glm5_fp8_kv_gather_dequant_kernel(
    cache_ptr,
    indices_ptr,
    lengths_ptr,
    gathered_ptr,
    cache_stride_block,
    num_cache_slots,
    block_size,
    INDEX_WIDTH: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    position = tl.program_id(0)
    dim_offsets = tl.arange(0, HEAD_DIM)
    valid_len = tl.minimum(tl.load(lengths_ptr), INDEX_WIDTH)
    slot = tl.load(indices_ptr + position, mask=position < INDEX_WIDTH, other=-1)
    valid = (position < valid_len) & (slot >= 0) & (slot < num_cache_slots)
    safe_slot = tl.where(valid, slot, 0)
    block_idx = safe_slot // block_size
    pos_in_block = safe_slot % block_size
    block_base = cache_ptr + block_idx.to(tl.int64) * cache_stride_block
    token_data = block_base + pos_in_block * HEAD_DIM
    token_scales = (
        block_base + block_size * HEAD_DIM + pos_in_block * (HEAD_DIM // GROUP_SIZE)
    )

    packed = tl.load(token_data + dim_offsets, mask=valid, other=0)
    fp8 = fp8_e4m3fn_bits_to_fp32_bitcast(packed)
    scale_group_offsets = tl.arange(0, HEAD_DIM // GROUP_SIZE)
    encoded_scale = tl.load(
        token_scales + scale_group_offsets,
        mask=valid,
        other=127,
    )
    scales = tl.exp2(encoded_scale.to(tl.float32) - 127.0).to(tl.float16)
    grouped = tl.reshape(fp8, (HEAD_DIM // GROUP_SIZE, GROUP_SIZE))
    dequantized = tl.reshape(
        grouped.to(tl.float16) * scales[:, None],
        (HEAD_DIM,),
    )
    tl.store(
        gathered_ptr + position * HEAD_DIM + dim_offsets,
        tl.where(valid, dequantized, 0.0),
    )


@triton.jit
def _sm70_glm5_fp8_kv_gather_dequant_batched_kernel(
    cache_ptr,
    indices_ptr,
    lengths_ptr,
    gathered_ptr,
    cache_stride_block,
    indices_stride_t,
    gathered_stride_t,
    gathered_stride_k,
    num_cache_slots,
    block_size,
    INDEX_WIDTH: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    query_idx = tl.program_id(0)
    position = tl.program_id(1)
    dim_offsets = tl.arange(0, HEAD_DIM)
    valid_len = tl.minimum(tl.load(lengths_ptr + query_idx), INDEX_WIDTH)
    slot = tl.load(
        indices_ptr + query_idx * indices_stride_t + position,
        mask=position < INDEX_WIDTH,
        other=-1,
    )
    valid = (position < valid_len) & (slot >= 0) & (slot < num_cache_slots)
    safe_slot = tl.where(valid, slot, 0)
    block_idx = safe_slot // block_size
    pos_in_block = safe_slot % block_size
    block_base = cache_ptr + block_idx.to(tl.int64) * cache_stride_block
    token_data = block_base + pos_in_block * HEAD_DIM
    token_scales = (
        block_base + block_size * HEAD_DIM + pos_in_block * (HEAD_DIM // GROUP_SIZE)
    )

    packed = tl.load(token_data + dim_offsets, mask=valid, other=0)
    fp8 = fp8_e4m3fn_bits_to_fp32_bitcast(packed)
    scale_group_offsets = tl.arange(0, HEAD_DIM // GROUP_SIZE)
    encoded_scale = tl.load(
        token_scales + scale_group_offsets,
        mask=valid,
        other=127,
    )
    scales = tl.exp2(encoded_scale.to(tl.float32) - 127.0).to(tl.float16)
    grouped = tl.reshape(fp8, (HEAD_DIM // GROUP_SIZE, GROUP_SIZE))
    dequantized = tl.reshape(
        grouped.to(tl.float16) * scales[:, None],
        (HEAD_DIM,),
    )
    tl.store(
        gathered_ptr
        + query_idx * gathered_stride_t
        + position * gathered_stride_k
        + dim_offsets,
        tl.where(valid, dequantized, 0.0),
    )


@triton.jit
def _sm70_glm5_sparse_scores_softmax_kernel(
    scores_ptr,
    indices_ptr,
    lengths_ptr,
    probs_ptr,
    scores_stride_h,
    num_cache_slots,
    scale,
    INDEX_WIDTH: tl.constexpr,
    WIDTH_BLOCK: tl.constexpr,
):
    head_idx = tl.program_id(0)
    positions = tl.arange(0, WIDTH_BLOCK)
    width_mask = positions < INDEX_WIDTH
    valid_len = tl.minimum(tl.load(lengths_ptr), INDEX_WIDTH)
    slots = tl.load(indices_ptr + positions, mask=width_mask, other=-1)
    valid = (
        width_mask & (positions < valid_len) & (slots >= 0) & (slots < num_cache_slots)
    )
    neg_large = -3.4028234663852886e38
    scores = tl.load(
        scores_ptr + head_idx * scores_stride_h + positions,
        mask=width_mask,
        other=neg_large,
    ).to(tl.float32)
    scores = tl.where(valid, scores * scale, neg_large)
    scores_max = tl.max(scores, axis=0)
    numerators = tl.exp(scores - scores_max)
    numerators = tl.where(valid, numerators, 0.0)
    denominator = tl.sum(numerators, axis=0)
    probs = tl.where(
        denominator > 0.0,
        numerators / tl.maximum(denominator, 1.0e-30),
        0.0,
    )
    tl.store(
        probs_ptr + head_idx * scores_stride_h + positions,
        probs,
        mask=width_mask,
    )


@triton.jit
def _sm70_glm5_sparse_scores_softmax_batched_kernel(
    scores_ptr,
    indices_ptr,
    lengths_ptr,
    probs_ptr,
    scores_stride_t,
    scores_stride_h,
    indices_stride_t,
    probs_stride_t,
    probs_stride_h,
    num_cache_slots,
    scale,
    INDEX_WIDTH: tl.constexpr,
    WIDTH_BLOCK: tl.constexpr,
):
    query_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    positions = tl.arange(0, WIDTH_BLOCK)
    width_mask = positions < INDEX_WIDTH
    valid_len = tl.minimum(tl.load(lengths_ptr + query_idx), INDEX_WIDTH)
    slots = tl.load(
        indices_ptr + query_idx * indices_stride_t + positions,
        mask=width_mask,
        other=-1,
    )
    valid = (
        width_mask & (positions < valid_len) & (slots >= 0) & (slots < num_cache_slots)
    )
    neg_large = -3.4028234663852886e38
    scores = tl.load(
        scores_ptr
        + query_idx * scores_stride_t
        + head_idx * scores_stride_h
        + positions,
        mask=width_mask,
        other=neg_large,
    ).to(tl.float32)
    scores = tl.where(valid, scores * scale, neg_large)
    scores_max = tl.max(scores, axis=0)
    numerators = tl.exp(scores - scores_max)
    numerators = tl.where(valid, numerators, 0.0)
    denominator = tl.sum(numerators, axis=0)
    probs = tl.where(
        denominator > 0.0,
        numerators / tl.maximum(denominator, 1.0e-30),
        0.0,
    )
    tl.store(
        probs_ptr + query_idx * probs_stride_t + head_idx * probs_stride_h + positions,
        probs,
        mask=width_mask,
    )


def sm70_glm5_fp8_kv_insert(
    kv: torch.Tensor,
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Quantize FP16 latent KV into the packed GLM-5.3 paged layout."""
    assert kv.dtype == torch.float16 and kv.ndim == 2
    assert kv.shape[1] == GLM5_FP8_KV_DIM and kv.is_contiguous()
    assert cache.dtype == torch.uint8 and cache.ndim == 3
    assert cache.shape[2] == GLM5_FP8_KV_SLOT_BYTES
    num_tokens = slot_mapping.numel()
    if num_tokens == 0:
        return
    _sm70_glm5_fp8_kv_insert_kernel[(num_tokens,)](
        kv,
        slot_mapping,
        cache,
        cache.stride(0),
        block_size=cache.shape[1],
        HEAD_DIM=GLM5_FP8_KV_DIM,
        GROUP_SIZE=GLM5_FP8_KV_GROUP_SIZE,
        NUM_GROUPS=GLM5_FP8_KV_NUM_SCALES,
        num_warps=8,
    )


def sm70_glm5_sparse_attention_paged_fp8(
    q: torch.Tensor,
    cache: torch.Tensor,
    indices: torch.Tensor,
    lengths: torch.Tensor,
    scale: float,
    out: torch.Tensor,
) -> None:
    """Run sparse NoPE MLA directly over the packed GLM-5.3 FP8 cache."""
    assert q.dtype == out.dtype == torch.float16
    assert q.shape == out.shape and q.shape[-1] == GLM5_FP8_KV_DIM
    assert cache.dtype == torch.uint8 and cache.ndim == 3
    assert cache.shape[2] == GLM5_FP8_KV_SLOT_BYTES
    indices_2d = indices.reshape(q.shape[0], -1)
    lengths_1d = lengths.reshape(-1).to(torch.int32)
    assert indices_2d.shape[0] == q.shape[0] == lengths_1d.shape[0]

    block_h = 8
    _sm70_glm5_sparse_paged_fp8_kernel[(q.shape[0], triton.cdiv(q.shape[1], block_h))](
        q,
        cache,
        indices_2d,
        lengths_1d,
        out,
        q.stride(0),
        q.stride(1),
        cache.stride(0),
        indices_2d.stride(0),
        out.stride(0),
        out.stride(1),
        q.shape[1],
        cache.shape[0] * cache.shape[1],
        cache.shape[1],
        float(scale),
        INDEX_WIDTH=indices_2d.shape[1],
        BLOCK_H=block_h,
        BLOCK_K=16,
        HEAD_DIM=GLM5_FP8_KV_DIM,
        GROUP_SIZE=GLM5_FP8_KV_GROUP_SIZE,
        num_warps=4,
    )


def sm70_glm5_sparse_attention_paged_fp8_gemm(
    q: torch.Tensor,
    cache: torch.Tensor,
    indices: torch.Tensor,
    lengths: torch.Tensor,
    scale: float,
    out: torch.Tensor,
    gathered_kv: torch.Tensor,
    scores: torch.Tensor,
    probs: torch.Tensor,
) -> None:
    """Run B1 sparse MLA with an SM70 dequant + tensor-core GEMM pipeline."""
    assert q.dtype == out.dtype == torch.float16
    assert q.shape == out.shape and q.shape[-1] == GLM5_FP8_KV_DIM
    assert q.shape[0] == 1
    assert cache.dtype == torch.uint8 and cache.ndim == 3
    assert cache.shape[2] == GLM5_FP8_KV_SLOT_BYTES
    indices_2d = indices.reshape(1, -1)
    lengths_1d = lengths.reshape(-1).to(torch.int32)
    index_width = indices_2d.shape[1]
    assert lengths_1d.shape == (1,)
    assert gathered_kv.shape == (index_width, GLM5_FP8_KV_DIM)
    assert gathered_kv.dtype == torch.float16 and gathered_kv.is_contiguous()
    assert scores.shape == probs.shape == (q.shape[1], index_width)
    assert scores.dtype == probs.dtype == torch.float16
    assert scores.is_contiguous() and probs.is_contiguous()

    _sm70_glm5_fp8_kv_gather_dequant_kernel[(index_width,)](
        cache,
        indices_2d,
        lengths_1d,
        gathered_kv,
        cache.stride(0),
        cache.shape[0] * cache.shape[1],
        cache.shape[1],
        INDEX_WIDTH=index_width,
        HEAD_DIM=GLM5_FP8_KV_DIM,
        GROUP_SIZE=GLM5_FP8_KV_GROUP_SIZE,
        num_warps=4,
    )
    torch.mm(q[0], gathered_kv.t(), out=scores)
    _sm70_glm5_sparse_scores_softmax_kernel[(q.shape[1],)](
        scores,
        indices_2d,
        lengths_1d,
        probs,
        scores.stride(0),
        cache.shape[0] * cache.shape[1],
        float(scale),
        INDEX_WIDTH=index_width,
        WIDTH_BLOCK=triton.next_power_of_2(index_width),
        num_warps=8,
    )
    torch.mm(probs, gathered_kv, out=out[0])


def sm70_glm5_sparse_attention_paged_fp8_batched_gemm(
    q: torch.Tensor,
    cache: torch.Tensor,
    indices: torch.Tensor,
    lengths: torch.Tensor,
    scale: float,
    out: torch.Tensor,
    gathered_kv: torch.Tensor,
    scores: torch.Tensor,
    probs: torch.Tensor,
) -> None:
    """Run small-batch sparse MLA with the B1 tensor-core arithmetic path."""
    assert q.dtype == out.dtype == torch.float16
    assert q.shape == out.shape and q.shape[-1] == GLM5_FP8_KV_DIM
    assert cache.dtype == torch.uint8 and cache.ndim == 3
    assert cache.shape[2] == GLM5_FP8_KV_SLOT_BYTES
    num_tokens, num_heads = q.shape[:2]
    indices_2d = indices.reshape(num_tokens, -1)
    lengths_1d = lengths.reshape(-1).to(torch.int32)
    index_width = indices_2d.shape[1]
    assert lengths_1d.shape == (num_tokens,)
    assert gathered_kv.shape == (num_tokens, index_width, GLM5_FP8_KV_DIM)
    assert gathered_kv.dtype == torch.float16 and gathered_kv.is_contiguous()
    assert scores.shape == probs.shape == (num_tokens, num_heads, index_width)
    assert scores.dtype == probs.dtype == torch.float16
    assert scores.is_contiguous() and probs.is_contiguous()

    _sm70_glm5_fp8_kv_gather_dequant_batched_kernel[(num_tokens, index_width)](
        cache,
        indices_2d,
        lengths_1d,
        gathered_kv,
        cache.stride(0),
        indices_2d.stride(0),
        gathered_kv.stride(0),
        gathered_kv.stride(1),
        cache.shape[0] * cache.shape[1],
        cache.shape[1],
        INDEX_WIDTH=index_width,
        HEAD_DIM=GLM5_FP8_KV_DIM,
        GROUP_SIZE=GLM5_FP8_KV_GROUP_SIZE,
        num_warps=4,
    )
    torch.bmm(q, gathered_kv.transpose(1, 2), out=scores)
    _sm70_glm5_sparse_scores_softmax_batched_kernel[(num_tokens, num_heads)](
        scores,
        indices_2d,
        lengths_1d,
        probs,
        scores.stride(0),
        scores.stride(1),
        indices_2d.stride(0),
        probs.stride(0),
        probs.stride(1),
        cache.shape[0] * cache.shape[1],
        float(scale),
        INDEX_WIDTH=index_width,
        WIDTH_BLOCK=triton.next_power_of_2(index_width),
        num_warps=8,
    )
    # cublas uses a different reduction order for strided-batched PV once
    # enough keys are nonzero. Keep each small PV on the same tensor-core GEMM
    # arithmetic as B1 so speculative verification preserves target logits.
    for token_idx in range(num_tokens):
        torch.mm(probs[token_idx], gathered_kv[token_idx], out=out[token_idx])
