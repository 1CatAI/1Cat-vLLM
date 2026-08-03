# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""FP16 sparse MLA kernels for DeepSeek V4 on Volta."""

import torch

from vllm.models.deepseek_v4.common.ops.fp8_software import (
    fp8_e4m3fn_bits_to_fp32,
    fp8_e4m3fn_bits_to_fp32_bitcast,
)
from vllm.triton_utils import tl, triton

_HEAD_DIM = 512
_NOPE_DIM = 448
_ROPE_DIM = 64
_SPLITK_BLOCK_K = 16


@triton.jit
def _sm70_sparse_gathered_kernel(
    q_ptr,
    kv_ptr,
    indices_ptr,
    lengths_ptr,
    sink_ptr,
    out_ptr,
    q_stride_t,
    q_stride_h,
    kv_stride_n,
    indices_stride_t,
    out_stride_t,
    out_stride_h,
    num_heads,
    num_kv,
    scale,
    INDEX_WIDTH: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    query_idx = tl.program_id(0)
    head_offsets = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    head_mask = head_offsets < num_heads
    dim_offsets = tl.arange(0, BLOCK_D)

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
    acc = tl.zeros((BLOCK_H, BLOCK_D), dtype=tl.float32)
    valid_len = tl.load(lengths_ptr + query_idx)
    key_offsets = tl.arange(0, BLOCK_K)

    for start in tl.range(0, INDEX_WIDTH, BLOCK_K):
        positions = start + key_offsets
        in_range = positions < valid_len
        slots = tl.load(
            indices_ptr + query_idx * indices_stride_t + positions,
            mask=positions < INDEX_WIDTH,
            other=-1,
        )
        valid = in_range & (slots >= 0) & (slots < num_kv)
        safe_slots = tl.where(valid, slots, 0)
        kv = tl.load(
            kv_ptr + safe_slots[:, None] * kv_stride_n + dim_offsets[None, :],
            mask=valid[:, None],
            other=0.0,
        )

        scores = tl.dot(q, tl.trans(kv)) * scale
        scores = tl.where(head_mask[:, None] & valid[None, :], scores, neg_large)
        block_max = tl.max(scores, axis=1)
        new_max = tl.maximum(running_max, block_max)
        alpha = tl.exp(running_max - new_max)
        probs = tl.exp(scores - new_max[:, None])
        probs = tl.where(head_mask[:, None] & valid[None, :], probs, 0.0)
        acc = acc * alpha[:, None] + tl.dot(probs.to(kv.dtype), kv)
        running_sum = running_sum * alpha + tl.sum(probs, axis=1)
        running_max = new_max

    sink = tl.load(sink_ptr + head_offsets, mask=head_mask, other=neg_large).to(
        tl.float32
    )
    final_max = tl.maximum(running_max, sink)
    alpha = tl.exp(running_max - final_max)
    final_sum = running_sum * alpha + tl.exp(sink - final_max)
    denom = tl.maximum(final_sum, 1.0e-30)
    result = tl.where(
        final_sum[:, None] > 0.0,
        acc * alpha[:, None] / denom[:, None],
        0.0,
    )
    tl.store(
        out_ptr
        + query_idx * out_stride_t
        + head_offsets[:, None] * out_stride_h
        + dim_offsets[None, :],
        result,
        mask=head_mask[:, None],
    )


@triton.jit
def _sm70_sparse_paged_fp8_kernel(
    q_ptr,
    main_cache_ptr,
    main_indices_ptr,
    main_lengths_ptr,
    extra_cache_ptr,
    extra_indices_ptr,
    extra_lengths_ptr,
    sink_ptr,
    out_ptr,
    q_stride_t,
    q_stride_h,
    out_stride_t,
    out_stride_h,
    main_cache_stride0,
    extra_cache_stride0,
    main_indices_stride0,
    extra_indices_stride0,
    main_num_rows,
    extra_num_rows,
    main_block_size,
    extra_block_size,
    scale,
    num_heads,
    HAS_EXTRA: tl.constexpr,
    MAIN_WIDTH: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    NOPE_BLOCK: tl.constexpr,
    ROPE_DIM: tl.constexpr,
):
    query_idx = tl.program_id(0)
    head_offsets = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    head_mask = head_offsets < num_heads
    nope_offsets = tl.arange(0, NOPE_BLOCK)
    nope_mask = nope_offsets < NOPE_DIM
    rope_offsets = tl.arange(0, ROPE_DIM)

    q_row = q_ptr + query_idx * q_stride_t + head_offsets[:, None] * q_stride_h
    q_nope = tl.load(
        q_row + nope_offsets[None, :],
        mask=head_mask[:, None] & nope_mask[None, :],
        other=0.0,
    )
    q_rope = tl.load(
        q_row + NOPE_DIM + rope_offsets[None, :],
        mask=head_mask[:, None],
        other=0.0,
    )

    neg_large = -3.4028234663852886e38
    running_max = tl.full((BLOCK_H,), neg_large, dtype=tl.float32)
    running_sum = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc_nope = tl.zeros((BLOCK_H, NOPE_BLOCK), dtype=tl.float32)
    acc_rope = tl.zeros((BLOCK_H, ROPE_DIM), dtype=tl.float32)
    key_offsets = tl.arange(0, BLOCK_K)
    main_len = tl.load(main_lengths_ptr + query_idx)

    for start in tl.range(0, MAIN_WIDTH, BLOCK_K):
        positions = start + key_offsets
        in_range = positions < main_len
        slots = tl.load(
            main_indices_ptr + query_idx * main_indices_stride0 + positions,
            mask=positions < MAIN_WIDTH,
            other=-1,
        )
        valid = in_range & (slots >= 0) & (slots < main_num_rows)
        safe_slots = tl.where(valid, slots, 0)
        block_idx = safe_slots // main_block_size
        pos_in_block = safe_slots % main_block_size
        cache_block = main_cache_ptr + block_idx.to(tl.int64) * main_cache_stride0
        token_data = cache_block + pos_in_block * 576
        token_scales = cache_block + main_block_size * 576 + pos_in_block * 8

        packed = tl.load(
            token_data[:, None] + nope_offsets[None, :],
            mask=valid[:, None] & nope_mask[None, :],
            other=0,
        )
        fp8 = fp8_e4m3fn_bits_to_fp32(packed)
        encoded_scale = tl.load(
            token_scales[:, None] + nope_offsets[None, :] // 64,
            mask=valid[:, None] & nope_mask[None, :],
            other=127,
        )
        dequant_scale = tl.exp2(encoded_scale.to(tl.float32) - 127.0)
        k_nope = fp8.to(tl.float16) * dequant_scale.to(tl.float16)
        k_nope = tl.where(valid[:, None] & nope_mask[None, :], k_nope, 0.0)

        rope_ptr = (token_data + NOPE_DIM).to(tl.pointer_type(tl.bfloat16))
        k_rope = tl.load(
            rope_ptr[:, None] + rope_offsets[None, :],
            mask=valid[:, None],
            other=0.0,
        ).to(tl.float16)

        scores = tl.dot(q_nope, tl.trans(k_nope))
        scores += tl.dot(q_rope, tl.trans(k_rope))
        scores *= scale
        scores = tl.where(head_mask[:, None] & valid[None, :], scores, neg_large)
        block_max = tl.max(scores, axis=1)
        new_max = tl.maximum(running_max, block_max)
        alpha = tl.exp(running_max - new_max)
        probs = tl.exp(scores - new_max[:, None])
        probs = tl.where(head_mask[:, None] & valid[None, :], probs, 0.0)
        acc_nope = acc_nope * alpha[:, None] + tl.dot(probs.to(k_nope.dtype), k_nope)
        acc_rope = acc_rope * alpha[:, None] + tl.dot(probs.to(k_rope.dtype), k_rope)
        running_sum = running_sum * alpha + tl.sum(probs, axis=1)
        running_max = new_max

    if HAS_EXTRA:
        extra_len = tl.load(extra_lengths_ptr + query_idx)
        # The C128 logical width changes with context length. Drive this loop
        # from device metadata so one FULL CUDA Graph remains valid as context
        # grows, while the underlying row stride stays fixed.
        for start in range(0, extra_len, BLOCK_K):
            positions = start + key_offsets
            in_range = positions < extra_len
            slots = tl.load(
                extra_indices_ptr + query_idx * extra_indices_stride0 + positions,
                mask=in_range,
                other=-1,
            )
            valid = in_range & (slots >= 0) & (slots < extra_num_rows)
            safe_slots = tl.where(valid, slots, 0)
            block_idx = safe_slots // extra_block_size
            pos_in_block = safe_slots % extra_block_size
            cache_block = extra_cache_ptr + block_idx.to(tl.int64) * extra_cache_stride0
            token_data = cache_block + pos_in_block * 576
            token_scales = cache_block + extra_block_size * 576 + pos_in_block * 8

            packed = tl.load(
                token_data[:, None] + nope_offsets[None, :],
                mask=valid[:, None] & nope_mask[None, :],
                other=0,
            )
            fp8 = fp8_e4m3fn_bits_to_fp32(packed)
            encoded_scale = tl.load(
                token_scales[:, None] + nope_offsets[None, :] // 64,
                mask=valid[:, None] & nope_mask[None, :],
                other=127,
            )
            dequant_scale = tl.exp2(encoded_scale.to(tl.float32) - 127.0)
            k_nope = fp8.to(tl.float16) * dequant_scale.to(tl.float16)
            k_nope = tl.where(valid[:, None] & nope_mask[None, :], k_nope, 0.0)

            rope_ptr = (token_data + NOPE_DIM).to(tl.pointer_type(tl.bfloat16))
            k_rope = tl.load(
                rope_ptr[:, None] + rope_offsets[None, :],
                mask=valid[:, None],
                other=0.0,
            ).to(tl.float16)

            scores = tl.dot(q_nope, tl.trans(k_nope))
            scores += tl.dot(q_rope, tl.trans(k_rope))
            scores *= scale
            scores = tl.where(head_mask[:, None] & valid[None, :], scores, neg_large)
            block_max = tl.max(scores, axis=1)
            new_max = tl.maximum(running_max, block_max)
            alpha = tl.exp(running_max - new_max)
            probs = tl.exp(scores - new_max[:, None])
            probs = tl.where(head_mask[:, None] & valid[None, :], probs, 0.0)
            acc_nope = acc_nope * alpha[:, None] + tl.dot(
                probs.to(k_nope.dtype), k_nope
            )
            acc_rope = acc_rope * alpha[:, None] + tl.dot(
                probs.to(k_rope.dtype), k_rope
            )
            running_sum = running_sum * alpha + tl.sum(probs, axis=1)
            running_max = new_max

    sink = tl.load(sink_ptr + head_offsets, mask=head_mask, other=neg_large).to(
        tl.float32
    )
    final_max = tl.maximum(running_max, sink)
    alpha = tl.exp(running_max - final_max)
    final_sum = running_sum * alpha + tl.exp(sink - final_max)
    denom = tl.maximum(final_sum, 1.0e-30)
    out_nope = tl.where(
        final_sum[:, None] > 0.0,
        acc_nope * alpha[:, None] / denom[:, None],
        0.0,
    )
    out_rope = tl.where(
        final_sum[:, None] > 0.0,
        acc_rope * alpha[:, None] / denom[:, None],
        0.0,
    )
    out_row = out_ptr + query_idx * out_stride_t + head_offsets[:, None] * out_stride_h
    tl.store(
        out_row + nope_offsets[None, :],
        out_nope,
        mask=head_mask[:, None] & nope_mask[None, :],
    )
    tl.store(
        out_row + NOPE_DIM + rope_offsets[None, :],
        out_rope,
        mask=head_mask[:, None],
    )


@triton.jit
def _sm70_sparse_paged_fp8_splitk_kernel(
    q_ptr,
    main_cache_ptr,
    main_indices_ptr,
    main_lengths_ptr,
    extra_cache_ptr,
    extra_indices_ptr,
    extra_lengths_ptr,
    partial_max_ptr,
    partial_sum_ptr,
    partial_acc_ptr,
    q_stride_t,
    q_stride_h,
    main_cache_stride0,
    extra_cache_stride0,
    main_indices_stride0,
    extra_indices_stride0,
    main_num_rows,
    extra_num_rows,
    main_block_size,
    extra_block_size,
    scale,
    num_heads,
    MAIN_WIDTH: tl.constexpr,
    EXTRA_WIDTH: tl.constexpr,
    NUM_PARTIALS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    NOPE_BLOCK: tl.constexpr,
    ROPE_DIM: tl.constexpr,
):
    query_idx = tl.program_id(0)
    head_offsets = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    partial_idx = tl.program_id(2)
    head_mask = head_offsets < num_heads
    nope_offsets = tl.arange(0, NOPE_BLOCK)
    nope_mask = nope_offsets < NOPE_DIM
    rope_offsets = tl.arange(0, ROPE_DIM)
    key_offsets = tl.arange(0, BLOCK_K)

    q_row = q_ptr + query_idx * q_stride_t + head_offsets[:, None] * q_stride_h
    q_nope = tl.load(
        q_row + nope_offsets[None, :],
        mask=head_mask[:, None] & nope_mask[None, :],
        other=0.0,
    )
    q_rope = tl.load(
        q_row + NOPE_DIM + rope_offsets[None, :],
        mask=head_mask[:, None],
        other=0.0,
    )

    main_partial_count: tl.constexpr = tl.cdiv(MAIN_WIDTH, BLOCK_K)
    source_is_main = partial_idx < main_partial_count
    source_block = tl.where(
        source_is_main, partial_idx, partial_idx - main_partial_count
    )
    positions = source_block * BLOCK_K + key_offsets

    if source_is_main:
        source_len = tl.load(main_lengths_ptr + query_idx)
        slots = tl.load(
            main_indices_ptr + query_idx * main_indices_stride0 + positions,
            mask=positions < MAIN_WIDTH,
            other=-1,
        )
        valid = (positions < source_len) & (slots >= 0) & (slots < main_num_rows)
        safe_slots = tl.where(valid, slots, 0)
        block_idx = safe_slots // main_block_size
        pos_in_block = safe_slots % main_block_size
        cache_block = main_cache_ptr + block_idx.to(tl.int64) * main_cache_stride0
        token_data = cache_block + pos_in_block * 576
        token_scales = cache_block + main_block_size * 576 + pos_in_block * 8
    else:
        source_len = tl.load(extra_lengths_ptr + query_idx)
        slots = tl.load(
            extra_indices_ptr + query_idx * extra_indices_stride0 + positions,
            mask=positions < EXTRA_WIDTH,
            other=-1,
        )
        valid = (positions < source_len) & (slots >= 0) & (slots < extra_num_rows)
        safe_slots = tl.where(valid, slots, 0)
        block_idx = safe_slots // extra_block_size
        pos_in_block = safe_slots % extra_block_size
        cache_block = extra_cache_ptr + block_idx.to(tl.int64) * extra_cache_stride0
        token_data = cache_block + pos_in_block * 576
        token_scales = cache_block + extra_block_size * 576 + pos_in_block * 8

    packed = tl.load(
        token_data[:, None] + nope_offsets[None, :],
        mask=valid[:, None] & nope_mask[None, :],
        other=0,
    )
    fp8 = fp8_e4m3fn_bits_to_fp32_bitcast(packed)
    scale_group_offsets = tl.arange(0, NOPE_BLOCK // 64)
    scale_group_mask = scale_group_offsets * 64 < NOPE_DIM
    encoded_scale = tl.load(
        token_scales[:, None] + scale_group_offsets[None, :],
        mask=valid[:, None] & scale_group_mask[None, :],
        other=127,
    )
    dequant_scale = tl.exp2(encoded_scale.to(tl.float32) - 127.0)
    fp8_grouped = tl.reshape(fp8, (BLOCK_K, NOPE_BLOCK // 64, 64))
    scale_grouped = tl.reshape(
        dequant_scale.to(tl.float16), (BLOCK_K, NOPE_BLOCK // 64, 1)
    )
    k_nope = tl.reshape(
        fp8_grouped.to(tl.float16) * scale_grouped, (BLOCK_K, NOPE_BLOCK)
    )
    k_nope = tl.where(valid[:, None] & nope_mask[None, :], k_nope, 0.0)

    rope_ptr = (token_data + NOPE_DIM).to(tl.pointer_type(tl.bfloat16))
    k_rope = tl.load(
        rope_ptr[:, None] + rope_offsets[None, :],
        mask=valid[:, None],
        other=0.0,
    ).to(tl.float16)

    neg_large = -3.4028234663852886e38
    scores = tl.dot(q_nope, tl.trans(k_nope))
    scores += tl.dot(q_rope, tl.trans(k_rope))
    scores *= scale
    scores = tl.where(head_mask[:, None] & valid[None, :], scores, neg_large)
    block_max = tl.max(scores, axis=1)
    probs = tl.exp(scores - block_max[:, None])
    probs = tl.where(head_mask[:, None] & valid[None, :], probs, 0.0)
    block_sum = tl.sum(probs, axis=1)
    acc_nope = tl.dot(probs.to(k_nope.dtype), k_nope)
    acc_rope = tl.dot(probs.to(k_rope.dtype), k_rope)

    state_offsets = (query_idx * num_heads + head_offsets) * NUM_PARTIALS + partial_idx
    tl.store(partial_max_ptr + state_offsets, block_max, mask=head_mask)
    tl.store(partial_sum_ptr + state_offsets, block_sum, mask=head_mask)
    acc_row = partial_acc_ptr + state_offsets[:, None] * (NOPE_DIM + ROPE_DIM)
    tl.store(
        acc_row + nope_offsets[None, :],
        acc_nope,
        mask=head_mask[:, None] & nope_mask[None, :],
    )
    tl.store(
        acc_row + NOPE_DIM + rope_offsets[None, :],
        acc_rope,
        mask=head_mask[:, None],
    )


@triton.jit
def _sm70_sparse_paged_fp8_splitk_qk_tiles_kernel(
    q_ptr,
    main_cache_ptr,
    main_indices_ptr,
    main_lengths_ptr,
    extra_cache_ptr,
    extra_indices_ptr,
    extra_lengths_ptr,
    partial_qk_ptr,
    q_stride_t,
    q_stride_h,
    main_cache_stride0,
    extra_cache_stride0,
    main_indices_stride0,
    extra_indices_stride0,
    main_num_rows,
    extra_num_rows,
    main_block_size,
    extra_block_size,
    num_heads,
    MAIN_WIDTH: tl.constexpr,
    EXTRA_WIDTH: tl.constexpr,
    NUM_PARTIALS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    D_TILES: tl.constexpr,
):
    query_idx = tl.program_id(0)
    head_offsets = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    work_idx = tl.program_id(2)
    partial_idx = work_idx // D_TILES
    d_tile = work_idx % D_TILES
    head_mask = head_offsets < num_heads
    key_offsets = tl.arange(0, BLOCK_K)
    dim_offsets = d_tile * BLOCK_D + tl.arange(0, BLOCK_D)
    dim_mask = dim_offsets < HEAD_DIM
    nope_mask = dim_offsets < NOPE_DIM

    main_partial_count: tl.constexpr = tl.cdiv(MAIN_WIDTH, BLOCK_K)
    source_is_main = partial_idx < main_partial_count
    source_block = tl.where(
        source_is_main, partial_idx, partial_idx - main_partial_count
    )
    positions = source_block * BLOCK_K + key_offsets

    if source_is_main:
        source_len = tl.load(main_lengths_ptr + query_idx)
        slots = tl.load(
            main_indices_ptr + query_idx * main_indices_stride0 + positions,
            mask=positions < MAIN_WIDTH,
            other=-1,
        )
        valid = (positions < source_len) & (slots >= 0) & (slots < main_num_rows)
        safe_slots = tl.where(valid, slots, 0)
        block_idx = safe_slots // main_block_size
        pos_in_block = safe_slots % main_block_size
        cache_block = main_cache_ptr + block_idx.to(tl.int64) * main_cache_stride0
        token_data = cache_block + pos_in_block * 576
        token_scales = cache_block + main_block_size * 576 + pos_in_block * 8
    else:
        source_len = tl.load(extra_lengths_ptr + query_idx)
        slots = tl.load(
            extra_indices_ptr + query_idx * extra_indices_stride0 + positions,
            mask=positions < EXTRA_WIDTH,
            other=-1,
        )
        valid = (positions < source_len) & (slots >= 0) & (slots < extra_num_rows)
        safe_slots = tl.where(valid, slots, 0)
        block_idx = safe_slots // extra_block_size
        pos_in_block = safe_slots % extra_block_size
        cache_block = extra_cache_ptr + block_idx.to(tl.int64) * extra_cache_stride0
        token_data = cache_block + pos_in_block * 576
        token_scales = cache_block + extra_block_size * 576 + pos_in_block * 8

    q_row = q_ptr + query_idx * q_stride_t + head_offsets[:, None] * q_stride_h
    q_tile = tl.load(
        q_row + dim_offsets[None, :],
        mask=head_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )
    packed = tl.load(
        token_data[:, None] + dim_offsets[None, :],
        mask=valid[:, None] & nope_mask[None, :],
        other=0,
    )
    fp8 = fp8_e4m3fn_bits_to_fp32_bitcast(packed)
    encoded_scale = tl.load(
        token_scales + d_tile,
        mask=valid & (d_tile * BLOCK_D < NOPE_DIM),
        other=127,
    )
    dequant_scale = tl.exp2(encoded_scale.to(tl.float32) - 127.0)
    k_nope = fp8.to(tl.float16) * dequant_scale[:, None].to(tl.float16)

    rope_offsets = dim_offsets - NOPE_DIM
    rope_ptr = (token_data + NOPE_DIM).to(tl.pointer_type(tl.bfloat16))
    k_rope = tl.load(
        rope_ptr[:, None] + rope_offsets[None, :],
        mask=valid[:, None] & (~nope_mask[None, :]) & dim_mask[None, :],
        other=0.0,
    ).to(tl.float16)
    k_tile = tl.where(nope_mask[None, :], k_nope, k_rope)
    k_tile = tl.where(valid[:, None] & dim_mask[None, :], k_tile, 0.0)
    partial_scores = tl.dot(q_tile, tl.trans(k_tile))

    score_base = (
        ((query_idx * num_heads + head_offsets) * NUM_PARTIALS + partial_idx) * D_TILES
        + d_tile
    ) * BLOCK_K
    tl.store(
        partial_qk_ptr + score_base[:, None] + key_offsets[None, :],
        partial_scores,
        mask=head_mask[:, None],
    )


@triton.jit
def _sm70_sparse_paged_fp8_splitk_qk_reduce_kernel(
    main_indices_ptr,
    main_lengths_ptr,
    extra_indices_ptr,
    extra_lengths_ptr,
    partial_qk_ptr,
    partial_max_ptr,
    partial_sum_ptr,
    partial_probs_ptr,
    main_indices_stride0,
    extra_indices_stride0,
    main_num_rows,
    extra_num_rows,
    scale,
    num_heads,
    MAIN_WIDTH: tl.constexpr,
    EXTRA_WIDTH: tl.constexpr,
    NUM_PARTIALS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    D_TILES: tl.constexpr,
):
    query_idx = tl.program_id(0)
    head_offsets = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    partial_idx = tl.program_id(2)
    head_mask = head_offsets < num_heads
    key_offsets = tl.arange(0, BLOCK_K)

    main_partial_count: tl.constexpr = tl.cdiv(MAIN_WIDTH, BLOCK_K)
    source_is_main = partial_idx < main_partial_count
    source_block = tl.where(
        source_is_main, partial_idx, partial_idx - main_partial_count
    )
    positions = source_block * BLOCK_K + key_offsets
    if source_is_main:
        source_len = tl.load(main_lengths_ptr + query_idx)
        slots = tl.load(
            main_indices_ptr + query_idx * main_indices_stride0 + positions,
            mask=positions < MAIN_WIDTH,
            other=-1,
        )
        valid = (positions < source_len) & (slots >= 0) & (slots < main_num_rows)
    else:
        source_len = tl.load(extra_lengths_ptr + query_idx)
        slots = tl.load(
            extra_indices_ptr + query_idx * extra_indices_stride0 + positions,
            mask=positions < EXTRA_WIDTH,
            other=-1,
        )
        valid = (positions < source_len) & (slots >= 0) & (slots < extra_num_rows)

    state_offsets = (query_idx * num_heads + head_offsets) * NUM_PARTIALS + partial_idx
    score_base = state_offsets * D_TILES * BLOCK_K
    scores = tl.zeros((BLOCK_H, BLOCK_K), dtype=tl.float32)
    for d_tile in range(D_TILES):
        scores += tl.load(
            partial_qk_ptr
            + score_base[:, None]
            + d_tile * BLOCK_K
            + key_offsets[None, :],
            mask=head_mask[:, None],
            other=0.0,
        )
    scores *= scale
    neg_large = -3.4028234663852886e38
    scores = tl.where(head_mask[:, None] & valid[None, :], scores, neg_large)
    block_max = tl.max(scores, axis=1)
    probs = tl.exp(scores - block_max[:, None])
    probs = tl.where(head_mask[:, None] & valid[None, :], probs, 0.0)
    block_sum = tl.sum(probs, axis=1)

    tl.store(partial_max_ptr + state_offsets, block_max, mask=head_mask)
    tl.store(partial_sum_ptr + state_offsets, block_sum, mask=head_mask)
    prob_row = partial_probs_ptr + state_offsets[:, None] * BLOCK_K
    tl.store(
        prob_row + key_offsets[None, :], probs.to(tl.float16), mask=head_mask[:, None]
    )


@triton.jit
def _sm70_sparse_paged_fp8_splitk_values_kernel(
    main_cache_ptr,
    main_indices_ptr,
    main_lengths_ptr,
    extra_cache_ptr,
    extra_indices_ptr,
    extra_lengths_ptr,
    partial_probs_ptr,
    partial_acc_ptr,
    main_cache_stride0,
    extra_cache_stride0,
    main_indices_stride0,
    extra_indices_stride0,
    main_num_rows,
    extra_num_rows,
    main_block_size,
    extra_block_size,
    num_heads,
    MAIN_WIDTH: tl.constexpr,
    EXTRA_WIDTH: tl.constexpr,
    NUM_PARTIALS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    D_TILES: tl.constexpr,
):
    query_idx = tl.program_id(0)
    head_offsets = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    work_idx = tl.program_id(2)
    partial_idx = work_idx // D_TILES
    d_tile = work_idx % D_TILES
    head_mask = head_offsets < num_heads
    key_offsets = tl.arange(0, BLOCK_K)
    dim_offsets = d_tile * BLOCK_D + tl.arange(0, BLOCK_D)
    dim_mask = dim_offsets < HEAD_DIM
    nope_mask = dim_offsets < NOPE_DIM

    main_partial_count: tl.constexpr = tl.cdiv(MAIN_WIDTH, BLOCK_K)
    source_is_main = partial_idx < main_partial_count
    source_block = tl.where(
        source_is_main, partial_idx, partial_idx - main_partial_count
    )
    positions = source_block * BLOCK_K + key_offsets

    if source_is_main:
        source_len = tl.load(main_lengths_ptr + query_idx)
        slots = tl.load(
            main_indices_ptr + query_idx * main_indices_stride0 + positions,
            mask=positions < MAIN_WIDTH,
            other=-1,
        )
        valid = (positions < source_len) & (slots >= 0) & (slots < main_num_rows)
        safe_slots = tl.where(valid, slots, 0)
        block_idx = safe_slots // main_block_size
        pos_in_block = safe_slots % main_block_size
        cache_block = main_cache_ptr + block_idx.to(tl.int64) * main_cache_stride0
        token_data = cache_block + pos_in_block * 576
        token_scales = cache_block + main_block_size * 576 + pos_in_block * 8
    else:
        source_len = tl.load(extra_lengths_ptr + query_idx)
        slots = tl.load(
            extra_indices_ptr + query_idx * extra_indices_stride0 + positions,
            mask=positions < EXTRA_WIDTH,
            other=-1,
        )
        valid = (positions < source_len) & (slots >= 0) & (slots < extra_num_rows)
        safe_slots = tl.where(valid, slots, 0)
        block_idx = safe_slots // extra_block_size
        pos_in_block = safe_slots % extra_block_size
        cache_block = extra_cache_ptr + block_idx.to(tl.int64) * extra_cache_stride0
        token_data = cache_block + pos_in_block * 576
        token_scales = cache_block + extra_block_size * 576 + pos_in_block * 8

    state_offsets = (query_idx * num_heads + head_offsets) * NUM_PARTIALS + partial_idx
    prob_row = partial_probs_ptr + state_offsets[:, None] * BLOCK_K
    probs = tl.load(prob_row + key_offsets[None, :], mask=head_mask[:, None], other=0.0)

    packed = tl.load(
        token_data[:, None] + dim_offsets[None, :],
        mask=valid[:, None] & nope_mask[None, :],
        other=0,
    )
    fp8 = fp8_e4m3fn_bits_to_fp32_bitcast(packed)
    encoded_scale = tl.load(
        token_scales + d_tile,
        mask=valid & (d_tile * BLOCK_D < NOPE_DIM),
        other=127,
    )
    dequant_scale = tl.exp2(encoded_scale.to(tl.float32) - 127.0)
    k_nope = fp8.to(tl.float16) * dequant_scale[:, None].to(tl.float16)

    rope_offsets = dim_offsets - NOPE_DIM
    rope_ptr = (token_data + NOPE_DIM).to(tl.pointer_type(tl.bfloat16))
    k_rope = tl.load(
        rope_ptr[:, None] + rope_offsets[None, :],
        mask=valid[:, None] & (~nope_mask[None, :]) & dim_mask[None, :],
        other=0.0,
    ).to(tl.float16)
    values = tl.where(nope_mask[None, :], k_nope, k_rope)
    values = tl.where(valid[:, None] & dim_mask[None, :], values, 0.0)
    partial_acc = tl.dot(probs, values)

    acc_row = partial_acc_ptr + state_offsets[:, None] * HEAD_DIM
    tl.store(
        acc_row + dim_offsets[None, :],
        partial_acc,
        mask=head_mask[:, None] & dim_mask[None, :],
    )


@triton.jit
def _sm70_sparse_paged_fp8_splitk_reduce_kernel(
    partial_max_ptr,
    partial_sum_ptr,
    partial_acc_ptr,
    sink_ptr,
    out_ptr,
    out_stride_t,
    out_stride_h,
    num_heads,
    NUM_PARTIALS: tl.constexpr,
    PARTIAL_BLOCK: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    query_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    dim_offsets = tl.program_id(2) * BLOCK_D + tl.arange(0, BLOCK_D)
    dim_mask = dim_offsets < HEAD_DIM
    partial_offsets = tl.arange(0, PARTIAL_BLOCK)
    partial_mask = partial_offsets < NUM_PARTIALS
    state_base = (query_idx * num_heads + head_idx) * NUM_PARTIALS

    neg_large = -3.4028234663852886e38
    partial_max = tl.load(
        partial_max_ptr + state_base + partial_offsets,
        mask=partial_mask,
        other=neg_large,
    )
    partial_sum = tl.load(
        partial_sum_ptr + state_base + partial_offsets,
        mask=partial_mask,
        other=0.0,
    )
    sink = tl.load(sink_ptr + head_idx).to(tl.float32)
    final_max = tl.maximum(tl.max(partial_max, axis=0), sink)
    weights = tl.exp(partial_max - final_max)
    weights = tl.where(partial_mask, weights, 0.0)
    final_sum = tl.sum(partial_sum * weights, axis=0) + tl.exp(sink - final_max)

    acc_offsets = (state_base + partial_offsets[:, None]) * HEAD_DIM + dim_offsets[
        None, :
    ]
    partial_acc = tl.load(
        partial_acc_ptr + acc_offsets,
        mask=partial_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )
    numerator = tl.sum(partial_acc * weights[:, None], axis=0)
    result = numerator / tl.maximum(final_sum, 1.0e-30)
    tl.store(
        out_ptr + query_idx * out_stride_t + head_idx * out_stride_h + dim_offsets,
        result,
        mask=dim_mask,
    )


def sm70_sparse_attention_gathered(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    lengths: torch.Tensor,
    scale: float,
    attn_sink: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """Sparse FP16 attention over a contiguous gathered KV workspace."""
    assert q.dtype == kv.dtype == out.dtype == torch.float16
    assert q.shape[-1] == kv.shape[-1] == out.shape[-1] == _HEAD_DIM
    kv_2d = kv.reshape(-1, _HEAD_DIM)
    indices_2d = indices.reshape(indices.shape[0], -1)
    lengths = lengths.reshape(-1).to(torch.int32)
    assert indices_2d.shape[0] == q.shape[0] == lengths.shape[0]

    block_h = 8
    _sm70_sparse_gathered_kernel[(q.shape[0], triton.cdiv(q.shape[1], block_h))](
        q,
        kv_2d,
        indices_2d,
        lengths,
        attn_sink.contiguous(),
        out,
        q.stride(0),
        q.stride(1),
        kv_2d.stride(0),
        indices_2d.stride(0),
        out.stride(0),
        out.stride(1),
        q.shape[1],
        kv_2d.shape[0],
        float(scale),
        INDEX_WIDTH=indices_2d.shape[1],
        BLOCK_H=block_h,
        BLOCK_K=16,
        BLOCK_D=_HEAD_DIM,
        num_warps=4,
    )


def sm70_sparse_attention_paged_fp8(
    q: torch.Tensor,
    main_cache: torch.Tensor,
    main_indices: torch.Tensor,
    main_lengths: torch.Tensor,
    scale: float,
    attn_sink: torch.Tensor,
    out: torch.Tensor,
    extra_cache: torch.Tensor | None = None,
    extra_indices: torch.Tensor | None = None,
    extra_lengths: torch.Tensor | None = None,
) -> None:
    """Decode directly from the packed DeepSeek FP8 paged-cache layout."""
    assert q.dtype == out.dtype == torch.float16
    assert q.shape == out.shape and q.shape[-1] == _HEAD_DIM
    assert main_cache.dtype == torch.uint8 and main_cache.ndim == 3

    main_indices_2d = main_indices.reshape(q.shape[0], -1)
    main_lengths = main_lengths.reshape(-1).to(torch.int32)
    has_extra = (
        extra_cache is not None
        and extra_indices is not None
        and extra_lengths is not None
    )
    if has_extra:
        assert extra_cache is not None
        assert extra_indices is not None
        assert extra_lengths is not None
        assert extra_cache.dtype == torch.uint8 and extra_cache.ndim == 3
        extra_indices_2d = extra_indices.reshape(q.shape[0], -1)
        extra_lengths_1d = extra_lengths.reshape(-1).to(torch.int32)
    else:
        extra_cache = main_cache
        extra_indices_2d = main_indices_2d[:, :1]
        extra_lengths_1d = torch.zeros_like(main_lengths)

    block_h = 8
    _sm70_sparse_paged_fp8_kernel[(q.shape[0], triton.cdiv(q.shape[1], block_h))](
        q,
        main_cache,
        main_indices_2d,
        main_lengths,
        extra_cache,
        extra_indices_2d,
        extra_lengths_1d,
        attn_sink.contiguous(),
        out,
        q.stride(0),
        q.stride(1),
        out.stride(0),
        out.stride(1),
        main_cache.stride(0),
        extra_cache.stride(0),
        main_indices_2d.stride(0),
        extra_indices_2d.stride(0),
        main_cache.shape[0] * main_cache.shape[1],
        extra_cache.shape[0] * extra_cache.shape[1],
        main_cache.shape[1],
        extra_cache.shape[1],
        float(scale),
        q.shape[1],
        HAS_EXTRA=has_extra,
        MAIN_WIDTH=main_indices_2d.shape[1],
        BLOCK_H=block_h,
        BLOCK_K=16,
        NOPE_DIM=_NOPE_DIM,
        NOPE_BLOCK=_HEAD_DIM,
        ROPE_DIM=_ROPE_DIM,
        num_warps=4,
    )


def sm70_sparse_attention_paged_fp8_splitk(
    q: torch.Tensor,
    main_cache: torch.Tensor,
    main_indices: torch.Tensor,
    main_lengths: torch.Tensor,
    scale: float,
    attn_sink: torch.Tensor,
    out: torch.Tensor,
    extra_cache: torch.Tensor | None,
    extra_indices: torch.Tensor | None,
    extra_lengths: torch.Tensor | None,
    partial_max: torch.Tensor,
    partial_sum: torch.Tensor,
    partial_acc: torch.Tensor,
    stage1_num_warps: int = 4,
    stage1_block_h: int = 8,
) -> None:
    """Split sparse KV blocks across CTAs, then reduce online-softmax states."""
    assert q.dtype == out.dtype == torch.float16
    assert q.shape == out.shape and q.shape[-1] == _HEAD_DIM
    assert main_cache.dtype == torch.uint8 and main_cache.ndim == 3

    main_indices_2d = main_indices.reshape(q.shape[0], -1)
    main_lengths_1d = main_lengths.reshape(-1).to(torch.int32)
    has_extra = all(
        value is not None for value in (extra_cache, extra_indices, extra_lengths)
    )
    assert has_extra or all(
        value is None for value in (extra_cache, extra_indices, extra_lengths)
    )
    if has_extra:
        assert extra_cache is not None
        assert extra_indices is not None
        assert extra_lengths is not None
        assert extra_cache.dtype == torch.uint8 and extra_cache.ndim == 3
        extra_indices_2d = extra_indices.reshape(q.shape[0], -1)
        extra_lengths_1d = extra_lengths.reshape(-1).to(torch.int32)
    else:
        extra_cache = main_cache
        extra_indices_2d = main_indices_2d[:, :0]
        extra_lengths_1d = main_lengths_1d
    main_partials = triton.cdiv(main_indices_2d.shape[1], _SPLITK_BLOCK_K)
    extra_partials = triton.cdiv(extra_indices_2d.shape[1], _SPLITK_BLOCK_K)
    num_partials = main_partials + extra_partials
    expected_acc_shape = (*q.shape[:2], num_partials, _HEAD_DIM)
    expected_state_shape = (*q.shape[:2], num_partials)
    assert partial_max.dtype == partial_sum.dtype == torch.float32
    assert partial_acc.dtype == torch.float32
    assert partial_max.shape == partial_sum.shape == expected_state_shape
    assert partial_acc.shape == expected_acc_shape
    assert partial_max.is_contiguous() and partial_sum.is_contiguous()
    assert partial_acc.is_contiguous()

    block_h = stage1_block_h
    _sm70_sparse_paged_fp8_splitk_kernel[
        (
            q.shape[0],
            triton.cdiv(q.shape[1], block_h),
            num_partials,
        )
    ](
        q,
        main_cache,
        main_indices_2d,
        main_lengths_1d,
        extra_cache,
        extra_indices_2d,
        extra_lengths_1d,
        partial_max,
        partial_sum,
        partial_acc,
        q.stride(0),
        q.stride(1),
        main_cache.stride(0),
        extra_cache.stride(0),
        main_indices_2d.stride(0),
        extra_indices_2d.stride(0),
        main_cache.shape[0] * main_cache.shape[1],
        extra_cache.shape[0] * extra_cache.shape[1],
        main_cache.shape[1],
        extra_cache.shape[1],
        float(scale),
        q.shape[1],
        MAIN_WIDTH=main_indices_2d.shape[1],
        EXTRA_WIDTH=extra_indices_2d.shape[1],
        NUM_PARTIALS=num_partials,
        BLOCK_H=block_h,
        BLOCK_K=_SPLITK_BLOCK_K,
        NOPE_DIM=_NOPE_DIM,
        NOPE_BLOCK=_HEAD_DIM,
        ROPE_DIM=_ROPE_DIM,
        num_warps=stage1_num_warps,
    )
    _sm70_sparse_paged_fp8_splitk_reduce_kernel[
        (
            q.shape[0],
            q.shape[1],
            triton.cdiv(_HEAD_DIM, 64),
        )
    ](
        partial_max,
        partial_sum,
        partial_acc,
        attn_sink.contiguous(),
        out,
        out.stride(0),
        out.stride(1),
        q.shape[1],
        NUM_PARTIALS=num_partials,
        PARTIAL_BLOCK=triton.next_power_of_2(num_partials),
        HEAD_DIM=_HEAD_DIM,
        BLOCK_D=64,
        num_warps=4,
    )


def sm70_sparse_attention_paged_fp8_splitk_qk_dsplit(
    q: torch.Tensor,
    main_cache: torch.Tensor,
    main_indices: torch.Tensor,
    main_lengths: torch.Tensor,
    scale: float,
    attn_sink: torch.Tensor,
    out: torch.Tensor,
    extra_cache: torch.Tensor | None,
    extra_indices: torch.Tensor | None,
    extra_lengths: torch.Tensor | None,
    partial_qk: torch.Tensor,
    partial_max: torch.Tensor,
    partial_sum: torch.Tensor,
    partial_probs: torch.Tensor,
    partial_acc: torch.Tensor,
    stage1_num_warps: int = 4,
    stage1_block_h: int = 8,
    block_d: int = 64,
) -> None:
    """Partition both QK and PV into 64-column CTA tiles."""
    assert q.dtype == out.dtype == torch.float16
    assert q.shape == out.shape and q.shape[-1] == _HEAD_DIM
    assert main_cache.dtype == torch.uint8 and main_cache.ndim == 3

    main_indices_2d = main_indices.reshape(q.shape[0], -1)
    main_lengths_1d = main_lengths.reshape(-1).to(torch.int32)
    has_extra = all(
        value is not None for value in (extra_cache, extra_indices, extra_lengths)
    )
    assert has_extra or all(
        value is None for value in (extra_cache, extra_indices, extra_lengths)
    )
    if has_extra:
        assert extra_cache is not None
        assert extra_indices is not None
        assert extra_lengths is not None
        assert extra_cache.dtype == torch.uint8 and extra_cache.ndim == 3
        extra_indices_2d = extra_indices.reshape(q.shape[0], -1)
        extra_lengths_1d = extra_lengths.reshape(-1).to(torch.int32)
    else:
        extra_cache = main_cache
        extra_indices_2d = main_indices_2d[:, :0]
        extra_lengths_1d = main_lengths_1d

    main_partials = triton.cdiv(main_indices_2d.shape[1], _SPLITK_BLOCK_K)
    extra_partials = triton.cdiv(extra_indices_2d.shape[1], _SPLITK_BLOCK_K)
    num_partials = main_partials + extra_partials
    assert block_d in (32, 64, 128)
    assert _HEAD_DIM % block_d == 0
    d_tiles = triton.cdiv(_HEAD_DIM, block_d)
    expected_qk_shape = (
        *q.shape[:2],
        num_partials,
        d_tiles,
        _SPLITK_BLOCK_K,
    )
    expected_acc_shape = (*q.shape[:2], num_partials, _HEAD_DIM)
    expected_state_shape = (*q.shape[:2], num_partials)
    expected_probs_shape = (*q.shape[:2], num_partials, _SPLITK_BLOCK_K)
    assert partial_qk.dtype == torch.float32
    assert partial_max.dtype == partial_sum.dtype == torch.float32
    assert partial_probs.dtype == torch.float16
    assert partial_acc.dtype == torch.float32
    assert partial_qk.shape == expected_qk_shape
    assert partial_max.shape == partial_sum.shape == expected_state_shape
    assert partial_probs.shape == expected_probs_shape
    assert partial_acc.shape == expected_acc_shape
    assert partial_qk.is_contiguous()
    assert partial_max.is_contiguous() and partial_sum.is_contiguous()
    assert partial_probs.is_contiguous() and partial_acc.is_contiguous()

    block_h = stage1_block_h
    head_groups = triton.cdiv(q.shape[1], block_h)
    common_kwargs = dict(
        MAIN_WIDTH=main_indices_2d.shape[1],
        EXTRA_WIDTH=extra_indices_2d.shape[1],
        NUM_PARTIALS=num_partials,
        BLOCK_H=block_h,
        BLOCK_K=_SPLITK_BLOCK_K,
    )
    _sm70_sparse_paged_fp8_splitk_qk_tiles_kernel[
        (q.shape[0], head_groups, num_partials * d_tiles)
    ](
        q,
        main_cache,
        main_indices_2d,
        main_lengths_1d,
        extra_cache,
        extra_indices_2d,
        extra_lengths_1d,
        partial_qk,
        q.stride(0),
        q.stride(1),
        main_cache.stride(0),
        extra_cache.stride(0),
        main_indices_2d.stride(0),
        extra_indices_2d.stride(0),
        main_cache.shape[0] * main_cache.shape[1],
        extra_cache.shape[0] * extra_cache.shape[1],
        main_cache.shape[1],
        extra_cache.shape[1],
        q.shape[1],
        HEAD_DIM=_HEAD_DIM,
        NOPE_DIM=_NOPE_DIM,
        BLOCK_D=block_d,
        D_TILES=d_tiles,
        num_warps=stage1_num_warps,
        **common_kwargs,
    )
    _sm70_sparse_paged_fp8_splitk_qk_reduce_kernel[
        (q.shape[0], head_groups, num_partials)
    ](
        main_indices_2d,
        main_lengths_1d,
        extra_indices_2d,
        extra_lengths_1d,
        partial_qk,
        partial_max,
        partial_sum,
        partial_probs,
        main_indices_2d.stride(0),
        extra_indices_2d.stride(0),
        main_cache.shape[0] * main_cache.shape[1],
        extra_cache.shape[0] * extra_cache.shape[1],
        float(scale),
        q.shape[1],
        D_TILES=d_tiles,
        num_warps=4,
        **common_kwargs,
    )
    _sm70_sparse_paged_fp8_splitk_values_kernel[
        (q.shape[0], head_groups, num_partials * d_tiles)
    ](
        main_cache,
        main_indices_2d,
        main_lengths_1d,
        extra_cache,
        extra_indices_2d,
        extra_lengths_1d,
        partial_probs,
        partial_acc,
        main_cache.stride(0),
        extra_cache.stride(0),
        main_indices_2d.stride(0),
        extra_indices_2d.stride(0),
        main_cache.shape[0] * main_cache.shape[1],
        extra_cache.shape[0] * extra_cache.shape[1],
        main_cache.shape[1],
        extra_cache.shape[1],
        q.shape[1],
        HEAD_DIM=_HEAD_DIM,
        NOPE_DIM=_NOPE_DIM,
        BLOCK_D=block_d,
        D_TILES=d_tiles,
        num_warps=4,
        **common_kwargs,
    )
    _sm70_sparse_paged_fp8_splitk_reduce_kernel[
        (q.shape[0], q.shape[1], triton.cdiv(_HEAD_DIM, 64))
    ](
        partial_max,
        partial_sum,
        partial_acc,
        attn_sink.contiguous(),
        out,
        out.stride(0),
        out.stride(1),
        q.shape[1],
        NUM_PARTIALS=num_partials,
        PARTIAL_BLOCK=triton.next_power_of_2(num_partials),
        HEAD_DIM=_HEAD_DIM,
        BLOCK_D=64,
        num_warps=4,
    )
