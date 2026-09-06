# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark-only grid-stride scorer, derived from the NVIDIA QSA kernel.

Only tile ownership changes; each tile retains the same dot/head reduction.
A bounded grid still visits ALL live tiles, including graph replay growth.
No production dispatcher references this experiment.
"""

from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["num_requests"])
def strided_qsa_mqa_paged_kernel(
    q_ptr,
    k_cache_ptr,
    page_table_ptr,
    token_to_req_ptr,
    query_positions_ptr,
    sequence_lengths_ptr,
    visible_blocks_ptr,
    logits_ptr,
    stride_q_row,
    stride_q_head,
    stride_q_dim,
    stride_cache_block,
    stride_cache_token,
    stride_cache_dim,
    stride_table_req,
    stride_table_page,
    stride_logits_row,
    num_rows,
    num_columns,
    num_pages,
    num_requests,
    score_divisor,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    TILES_PER_PROG: tl.constexpr,
    STAGES: tl.constexpr,
    MAX_N: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    dims = tl.arange(0, BLOCK_D)
    heads = tl.arange(0, MAX_N)
    request = tl.load(token_to_req_ptr + row)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    query_position = tl.load(query_positions_ptr + row)
    sequence_length = tl.load(
        sequence_lengths_ptr + safe_request,
        mask=(request >= 0) & (request < num_requests),
        other=0,
    )
    visible = tl.minimum(
        (query_position + 1) // COMPRESS_RATIO,
        sequence_length // COMPRESS_RATIO,
    )
    if tl.program_id(1) == 0:
        tl.store(visible_blocks_ptr + row, visible)
    tile_start = tl.program_id(1)
    # Top-k is bounded by visible_blocks, so columns beyond it need no value.
    if tile_start * BLOCK_N >= visible:
        return
    tile_end = tl.cdiv(visible, BLOCK_N)
    tile_end = tl.minimum(tile_end, tl.cdiv(num_columns, BLOCK_N))

    # Pad the small head axis to a tensor-core-compatible N dimension.
    query = tl.load(
        q_ptr
        + row * stride_q_row
        + heads[None, :] * stride_q_head
        + dims[:, None] * stride_q_dim,
        mask=(heads[None, :] < NUM_HEADS) & (dims[:, None] < HEAD_DIM),
        other=0.0,
    )
    column_offsets = tl.arange(0, BLOCK_N)
    for tile in tl.range(tile_start, tile_end, tl.num_programs(1), num_stages=STAGES):
        columns = tile * BLOCK_N + column_offsets
        live = columns < visible
        logical_page = tl.minimum(columns // PAGE_SIZE, PAGE_TABLE_WIDTH - 1)
        page_offset = columns % PAGE_SIZE
        physical_page = tl.load(
            page_table_ptr
            + safe_request * stride_table_req
            + logical_page * stride_table_page,
            mask=live,
            other=-1,
        )
        page_valid = live & (physical_page >= 0) & (physical_page < num_pages)
        # physical_page * block stride can overflow int32 for large caches.
        safe_physical_page = tl.maximum(physical_page, 0).to(tl.int64)
        keys = tl.load(
            k_cache_ptr
            + safe_physical_page[:, None] * stride_cache_block
            + page_offset[:, None] * stride_cache_token
            + dims[None, :] * stride_cache_dim,
            mask=page_valid[:, None] & (dims[None, :] < HEAD_DIM),
            other=0.0,
            eviction_policy="evict_first",
        )
        scores = tl.dot(keys, query, out_dtype=tl.float32)
        scores = tl.where(heads[None, :] < NUM_HEADS, tl.maximum(scores, 0.0), 0.0)
        score = tl.sum(scores, axis=1) / score_divisor
        tl.store(
            logits_ptr + row * stride_logits_row + columns,
            tl.where(page_valid, score, -float("inf")),
            mask=live & (columns < num_columns),
        )
