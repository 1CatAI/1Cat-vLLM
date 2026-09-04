# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton kernels for the Qwen4Exp weight-free QSA path."""

from __future__ import annotations

import math
import os

import regex as re
import torch

from vllm.logger import init_logger
from vllm.models.deepseek_v4.common.ops.fp8_software import (
    fp8_e4m3fn_bits_to_fp32_bitcast as fp8_e4m3fn_bits_to_fp32,
)
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, tl, triton

logger = init_logger(__name__)

_LOGITS_WORKSPACE_BYTES = 128 * 1024 * 1024
_TOPK_WORKSPACE_BYTES = 1024 * 1024
_SM70_INDEXER_CUBLAS = os.getenv("VLLM_SM70_QSA_INDEXER_CUBLAS", "1") == "1"
_SM70_INDEXER_SCORE_TILE_BYTES = (
    int(os.getenv("VLLM_SM70_QSA_INDEXER_SCORE_TILE_MB", "64")) * 1024 * 1024
)
_SM70_INDEXER_CUBLAS_MIN_ROWS = int(
    os.getenv("VLLM_SM70_QSA_INDEXER_CUBLAS_MIN_ROWS", "512")
)
_SM70_INDEXER_CUBLAS_MIN_SCORE_ELEMENTS = int(
    os.getenv("VLLM_SM70_QSA_INDEXER_CUBLAS_MIN_SCORE_ELEMENTS", str(1024**2))
)
_SM70_QSA_XQA_PAGE4 = os.getenv("VLLM_SM70_QSA_XQA_PAGE4", "1") == "1"
_SM70_QSA_XQA_PAGE4_MIN_ROWS = int(
    # Operator crossover on SM70 is around 48 rows for the fixed QSA width.
    # Use a conservative 64-row workload gate rather than coupling the route
    # to a particular server's max_num_batched_tokens setting.
    os.getenv("VLLM_SM70_QSA_XQA_PAGE4_MIN_ROWS", "64")
)
_SM70_QSA_XQA_PAGE4_PARTITION = 1024
_SM70_QSA_XQA_PAGE4_PAGES = 513
_SM70_QSA_XQA_PAGE4_MARKER = 1 << 30
_SM70_QSA_GROUPED_PAGE4 = os.getenv("VLLM_SM70_QSA_GROUPED_PAGE4", "1") == "1"
_SM70_QSA_GROUPED_PAGE4_QUERIES = 8
_SM70_QSA_GROUPED_PAGE4_OUTPUT_PAGES = (
    _SM70_QSA_XQA_PAGE4_PAGES * _SM70_QSA_GROUPED_PAGE4_QUERIES + 56
)
_SM70_QSA_XQA_PAGE4_WORKSPACES: dict[
    tuple[int, int, int, int, int],
    tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
] = {}
_SM70_QSA_GROUPED_PAGE4_WORKSPACES: dict[
    tuple[int, int],
    tuple[
        int,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
] = {}
_SM70_QSA_GROUPED_PAGE4_ABI_CACHE: tuple[object, int] | None = None


def _qsa_grouped_page4_abi_version(flash_attn_v100_cuda) -> int:
    """Return the grouped-page4 ABI without probing it on the hot path.

    New Flash-V100 builds expose an explicit version. Wheels predating that
    capability query are recognized conservatively from pybind's generated
    signature: ABI v1 has arguments 0..8, while ABI v2 has arguments 0..11.
    An unknown binding is treated as unsupported instead of risking a server
    crash on the first large prefill.
    """
    global _SM70_QSA_GROUPED_PAGE4_ABI_CACHE
    cached = _SM70_QSA_GROUPED_PAGE4_ABI_CACHE
    if cached is not None and cached[0] is flash_attn_v100_cuda:
        return cached[1]

    version = 0
    capability = getattr(flash_attn_v100_cuda, "grouped_sparse_page4_abi_version", None)
    if callable(capability):
        try:
            version = int(capability())
        except (RuntimeError, TypeError, ValueError):
            version = 0
    else:
        binding = getattr(flash_attn_v100_cuda, "grouped_sparse_page4_fwd", None)
        doc = getattr(binding, "__doc__", "") or ""
        argument_ids = [int(match) for match in re.findall(r"\barg(\d+):", doc)]
        if argument_ids:
            highest_argument = max(argument_ids)
            if highest_argument >= 11:
                version = 2
            elif highest_argument >= 8:
                version = 1

    _SM70_QSA_GROUPED_PAGE4_ABI_CACHE = (flash_attn_v100_cuda, version)
    return version


def _qsa_grouped_page4_supported(
    flash_attn_v100_cuda,
    kv_cache_dtype: str,
) -> bool:
    forward = getattr(flash_attn_v100_cuda, "grouped_sparse_page4_fwd", None)
    planner = getattr(flash_attn_v100_cuda, "grouped_sparse_page4_plan_fwd", None)
    if not callable(forward) or not callable(planner):
        return False
    abi_version = _qsa_grouped_page4_abi_version(flash_attn_v100_cuda)
    return abi_version >= 2 or (
        abi_version == 1 and kv_cache_dtype in ("auto", "float16")
    )


@triton.jit(do_not_specialize=["num_requests"])
def _qsa_mqa_paged_kernel(
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
    tile_start = tl.program_id(1) * TILES_PER_PROG
    # Top-k is bounded by visible_blocks, so columns beyond it need no value.
    if tile_start * BLOCK_N >= visible:
        return
    tile_end = tl.minimum(tile_start + TILES_PER_PROG, tl.cdiv(visible, BLOCK_N))
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
    for tile in tl.range(tile_start, tile_end, num_stages=STAGES):
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


@triton.jit
def _qsa_visible_blocks_kernel(
    token_to_req_ptr,
    query_positions_ptr,
    sequence_lengths_ptr,
    visible_blocks_ptr,
    rows,
    num_requests,
    COMPRESS_RATIO: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    request = tl.load(token_to_req_ptr + row, mask=row < rows, other=-1)
    valid_request = (request >= 0) & (request < num_requests)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    query_position = tl.load(query_positions_ptr + row, mask=row < rows, other=-1)
    sequence_length = tl.load(
        sequence_lengths_ptr + safe_request,
        mask=(row < rows) & valid_request,
        other=0,
    )
    visible = tl.minimum(
        (query_position + 1) // COMPRESS_RATIO,
        sequence_length // COMPRESS_RATIO,
    )
    tl.store(
        visible_blocks_ptr + row,
        tl.where(valid_request, tl.maximum(visible, 0), 0),
        mask=row < rows,
    )


@triton.jit
def _qsa_gather_single_request_keys_kernel(
    cache_ptr,
    page_table_ptr,
    keys_ptr,
    valid_ptr,
    stride_cache_page,
    stride_cache_token,
    stride_cache_dim,
    stride_table_page,
    stride_keys_row,
    columns,
    num_pages,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    column = tl.program_id(0)
    dims = tl.arange(0, BLOCK_D)
    logical_page = column // PAGE_SIZE
    page_offset = column % PAGE_SIZE
    physical_page = tl.load(
        page_table_ptr + logical_page * stride_table_page,
        mask=(column < columns) & (logical_page < PAGE_TABLE_WIDTH),
        other=-1,
    )
    valid = (column < columns) & (physical_page >= 0) & (physical_page < num_pages)
    safe_page = tl.maximum(physical_page, 0).to(tl.int64)
    key = tl.load(
        cache_ptr
        + safe_page * stride_cache_page
        + page_offset * stride_cache_token
        + dims * stride_cache_dim,
        mask=valid & (dims < HEAD_DIM),
        other=0.0,
    )
    tl.store(
        keys_ptr + column * stride_keys_row + dims,
        key,
        mask=(column < columns) & (dims < HEAD_DIM),
    )
    tl.store(valid_ptr + column, valid, mask=column < columns)


@triton.jit
def _qsa_relu_headsum_visible_kernel(
    score_ptr,
    visible_ptr,
    key_valid_ptr,
    logits_ptr,
    stride_score_row,
    stride_score_column,
    stride_logits_row,
    width,
    column_start,
    score_divisor,
    NUM_HEADS: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    columns = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    absolute_columns = column_start + columns
    visible = tl.load(visible_ptr + row)
    key_valid = tl.load(
        key_valid_ptr + absolute_columns,
        mask=columns < width,
        other=0,
    ).to(tl.int1)
    live = (columns < width) & (absolute_columns < visible) & key_valid
    score = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for head in range(NUM_HEADS):
        values = tl.load(
            score_ptr
            + (row * NUM_HEADS + head) * stride_score_row
            + columns * stride_score_column,
            mask=columns < width,
            other=0.0,
        ).to(tl.float32)
        score += tl.maximum(values, 0.0)
    tl.store(
        logits_ptr + row * stride_logits_row + absolute_columns,
        tl.where(live, score / score_divisor, -float("inf")),
        mask=columns < width,
    )


@triton.jit
def _expand_qsa_indices_kernel(
    block_indices_ptr,
    query_positions_ptr,
    sequence_lengths_ptr,
    token_to_req_ptr,
    output_ptr,
    stride_blocks_row,
    stride_blocks_column,
    stride_output_row,
    stride_output_column,
    rows,
    num_requests,
    BLOCK_TOPK: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    TOKEN_TOPK: tl.constexpr,
    OUTPUT_WIDTH: tl.constexpr,
    COLUMN_BLOCK: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    columns = tl.program_id(1) * COLUMN_BLOCK + tl.arange(0, COLUMN_BLOCK)
    query_position = tl.load(query_positions_ptr + row)
    request = tl.load(token_to_req_ptr + row)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    sequence_length = tl.load(
        sequence_lengths_ptr + safe_request,
        mask=(request >= 0) & (request < num_requests),
        other=0,
    )
    complete_blocks = tl.minimum(
        tl.minimum(
            (query_position + 1) // COMPRESS_RATIO,
            sequence_length // COMPRESS_RATIO,
        ),
        BLOCK_TOPK,
    )
    expanded_count = complete_blocks * COMPRESS_RATIO
    tail_start = ((query_position + 1) // COMPRESS_RATIO) * COMPRESS_RATIO
    tail_count = (query_position + 1) - tail_start

    is_expanded = columns < expanded_count
    block_rank = columns // COMPRESS_RATIO
    offset = columns % COMPRESS_RATIO
    safe_rank = tl.minimum(block_rank, BLOCK_TOPK - 1)
    block = tl.load(
        block_indices_ptr + row * stride_blocks_row + safe_rank * stride_blocks_column,
        mask=(row < rows) & is_expanded,
        other=-1,
    )
    expanded = block * COMPRESS_RATIO + offset
    tail_offset = columns - expanded_count
    is_tail = (
        (columns >= expanded_count)
        & (tail_offset < tail_count)
        & (tail_offset < COMPRESS_RATIO - 1)
    )
    token = tl.where(is_expanded, expanded, tail_start + tail_offset)
    valid = (
        (row < rows)
        & (columns < OUTPUT_WIDTH)
        & (is_expanded | is_tail)
        & (token >= 0)
        & (token < sequence_length)
    )
    tl.store(
        output_ptr + row * stride_output_row + columns * stride_output_column,
        tl.where(valid, token, -1),
        mask=(row < rows) & (columns < OUTPUT_WIDTH),
    )


@triton.jit
def _qsa_xqa_page4_table_kernel(
    indices_ptr,
    block_table_ptr,
    token_to_req_ptr,
    query_positions_ptr,
    sequence_lengths_ptr,
    encoded_pages_ptr,
    xqa_sequence_lengths_ptr,
    stride_indices_row,
    stride_table_req,
    stride_encoded_row,
    rows,
    num_cache_blocks,
    num_requests,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    COMPLETE_PAGES: tl.constexpr,
    OUTPUT_PAGES: tl.constexpr,
    BLOCK_PAGES: tl.constexpr,
    PHYSICAL_PAGE_STRIDE: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    slots = tl.arange(0, BLOCK_PAGES)
    request = tl.load(token_to_req_ptr + row)
    request_is_valid = (request >= 0) & (request < num_requests)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    query_position = tl.load(query_positions_ptr + row)
    sequence_length = tl.load(
        sequence_lengths_ptr + safe_request,
        mask=request_is_valid,
        other=0,
    )
    # Padded graph rows use position -1. Clamp malformed or stale positions to
    # the request's live sequence so they cannot expose a synthetic tail page.
    visible_tokens = tl.minimum(
        tl.maximum(query_position + 1, 0),
        sequence_length,
    )
    complete_pages = tl.minimum(
        tl.minimum(visible_tokens // 4, sequence_length // 4),
        COMPLETE_PAGES,
    )
    tail_count = visible_tokens - (visible_tokens // 4) * 4
    is_complete = slots < complete_pages
    selected_token = tl.load(
        indices_ptr + row * stride_indices_row + slots * 4,
        mask=(row < rows) & is_complete,
        other=-1,
    )
    tail_token = (visible_tokens // 4) * 4
    selected_tail_token = tl.load(
        indices_ptr + row * stride_indices_row + complete_pages * 4,
        mask=(row < rows) & (tail_count > 0),
        other=-1,
    )
    tail_is_valid = (
        (tail_count > 0)
        & (selected_tail_token == tail_token)
        & (selected_tail_token < sequence_length)
    )
    is_tail = (slots == complete_pages) & tail_is_valid
    logical_token = tl.where(is_tail, selected_tail_token, selected_token)
    safe_token = tl.maximum(logical_token, 0)
    logical_page = safe_token // PAGE_SIZE
    page_offset = safe_token - logical_page * PAGE_SIZE
    valid = (
        (row < rows)
        & request_is_valid
        & (logical_token >= 0)
        & (logical_token < sequence_length)
        & (logical_page < PAGE_TABLE_WIDTH)
        & (is_complete | is_tail)
    )
    physical_page = tl.load(
        block_table_ptr
        + safe_request * stride_table_req
        + tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1),
        mask=valid,
        other=-1,
    )
    valid &= (physical_page >= 0) & (physical_page < num_cache_blocks)
    physical_microblock = (
        tl.maximum(physical_page, 0) * PHYSICAL_PAGE_STRIDE + page_offset // 4
    )
    # Sort by logical token, not allocator-dependent physical page ID. Keep
    # the partial causal page after all complete pages and invalid slots last.
    logical_key = safe_token.to(tl.int64) << 31
    encoded = tl.where(
        valid & is_complete,
        logical_key | physical_microblock.to(tl.int64),
        tl.where(
            valid & is_tail,
            (1 << 62) | logical_key | physical_microblock.to(tl.int64),
            9223372036854775807,
        ),
    )
    tl.store(
        encoded_pages_ptr + row * stride_encoded_row + slots,
        encoded,
        mask=(row < rows) & (slots < OUTPUT_PAGES),
    )
    tl.store(
        xqa_sequence_lengths_ptr + row,
        complete_pages * 4 + tl.where(tail_is_valid, tail_count, 0),
        mask=row < rows,
    )


@triton.jit(do_not_specialize=["num_requests"])
def _qsa_sparse_paged_gqa_splitk_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    indices_ptr,
    block_table_ptr,
    token_to_req_ptr,
    partial_output_ptr,
    partial_lse_ptr,
    output_ptr,
    stride_q_row,
    stride_q_head,
    stride_k_block,
    stride_k_token,
    stride_k_head,
    stride_v_block,
    stride_v_token,
    stride_v_head,
    stride_indices_row,
    stride_table_req,
    stride_output_row,
    stride_output_head,
    num_rows,
    num_cache_blocks,
    num_requests,
    k_scale,
    v_scale,
    TOPK: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    NUM_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    KV_E4M3: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    split_id = tl.program_id(2)
    request = tl.load(token_to_req_ptr + row)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)

    head_offsets = tl.arange(0, BLOCK_M)
    dim_offsets = tl.arange(0, HEAD_DIM)
    column_offsets = tl.arange(0, BLOCK_N)
    first_head = kv_head * GROUP_SIZE
    query = tl.load(
        q_ptr
        + row * stride_q_row
        + (first_head + head_offsets[:, None]) * stride_q_head
        + dim_offsets[None, :],
        mask=head_offsets[:, None] < GROUP_SIZE,
        other=0.0,
    )

    max_value = tl.full((BLOCK_M,), -1.0e20, dtype=tl.float32)
    normalizer = tl.zeros((BLOCK_M,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    softmax_scale_log2: tl.constexpr = (HEAD_DIM**-0.5) * 1.4426950408889634
    qk_scale_log2 = softmax_scale_log2
    if KV_E4M3:
        # Keep the exactly decoded E4M3 values in the dot inputs and apply the
        # scalar K dequantization factor once to the accumulated QK scores.
        qk_scale_log2 *= k_scale

    # Dynamic bounds avoid padded main-loop iterations for uneven splits.
    split_tile_start = split_id * NUM_TILES // NUM_SPLITS
    split_tile_end = (split_id + 1) * NUM_TILES // NUM_SPLITS
    for tile in range(split_tile_start, split_tile_end):
        columns = tile * BLOCK_N + column_offsets
        logical_token = tl.load(
            indices_ptr + row * stride_indices_row + columns,
            mask=columns < TOPK,
            other=-1,
        )
        safe_token = tl.maximum(logical_token, 0)
        logical_page = safe_token // PAGE_SIZE
        page_offset = safe_token % PAGE_SIZE
        valid = (
            (request >= 0)
            & (request < num_requests)
            & (logical_token >= 0)
            & (logical_page < PAGE_TABLE_WIDTH)
        )
        physical_page = tl.load(
            block_table_ptr
            + safe_request * stride_table_req
            + tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1),
            mask=valid,
            other=-1,
        )
        valid &= (physical_page >= 0) & (physical_page < num_cache_blocks)
        # physical_page * block stride can overflow int32 for large caches.
        safe_page = tl.maximum(physical_page, 0).to(tl.int64)
        keys = tl.load(
            k_cache_ptr
            + safe_page[None, :] * stride_k_block
            + page_offset[None, :] * stride_k_token
            + kv_head * stride_k_head
            + dim_offsets[:, None],
            mask=valid[None, :],
            other=0.0,
        )
        values = tl.load(
            v_cache_ptr
            + safe_page[:, None] * stride_v_block
            + page_offset[:, None] * stride_v_token
            + kv_head * stride_v_head
            + dim_offsets[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        if KV_E4M3:
            keys = fp8_e4m3fn_bits_to_fp32(keys).to(query.dtype)
            values = fp8_e4m3fn_bits_to_fp32(values).to(query.dtype)
        scores = tl.dot(query, keys)
        # Scaling scores avoids re-quantizing a scaled query to BF16.
        scores *= qk_scale_log2
        scores = tl.where(valid[None, :], scores, -1.0e20)
        next_max = tl.maximum(max_value, tl.max(scores, axis=1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.where(
            valid[None, :], tl.math.exp2(scores - next_max[:, None]), 0.0
        )
        accumulator = tl.dot(
            probabilities.to(values.dtype),
            values,
            acc=accumulator * alpha[:, None],
        )
        normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)
        max_value = next_max

    has_values = normalizer > 0
    normalized_output = tl.where(
        has_values[:, None],
        accumulator / tl.maximum(normalizer[:, None], 1.0e-20),
        0.0,
    )
    output_mask = head_offsets[:, None] < GROUP_SIZE
    if NUM_SPLITS == 1:
        if KV_E4M3:
            # V dequantization is linear, so apply its scalar after the
            # normalized FP32 accumulation instead of to every loaded value.
            normalized_output *= v_scale
        tl.store(
            output_ptr
            + row * stride_output_row
            + (first_head + head_offsets[:, None]) * stride_output_head
            + dim_offsets[None, :],
            normalized_output,
            mask=output_mask,
        )
    else:
        partial_lse = tl.where(
            has_values,
            max_value + tl.math.log2(tl.maximum(normalizer, 1.0e-20)),
            -float("inf"),
        )
        tl.store(
            partial_output_ptr
            + (
                (split_id * num_rows + row) * NUM_QUERY_HEADS
                + first_head
                + head_offsets[:, None]
            )
            * HEAD_DIM
            + dim_offsets[None, :],
            normalized_output,
            mask=output_mask,
        )
        tl.store(
            partial_lse_ptr
            + (split_id * num_rows + row) * NUM_QUERY_HEADS
            + first_head
            + head_offsets,
            partial_lse,
            mask=head_offsets < GROUP_SIZE,
        )


@triton.jit
def _qsa_merge_splitk_kernel(
    partial_output_ptr,
    partial_lse_ptr,
    output_ptr,
    stride_output_row,
    stride_output_head,
    num_rows,
    v_scale,
    HEAD_DIM: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_SPLITS: tl.constexpr,
    KV_E4M3: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    head = tl.program_id(1)
    split_offsets = tl.arange(0, BLOCK_SPLITS)
    dim_offsets = tl.arange(0, HEAD_DIM)
    split_mask = split_offsets < NUM_SPLITS
    lse = tl.load(
        partial_lse_ptr + (split_offsets * num_rows + row) * NUM_QUERY_HEADS + head,
        mask=split_mask,
        other=-float("inf"),
    )
    lse_max = tl.max(lse, axis=0)
    has_values = lse_max > -float("inf")
    shifted = tl.where(split_mask & has_values, lse - lse_max, -float("inf"))
    weights = tl.math.exp2(shifted)
    denominator = tl.sum(weights, axis=0)
    partial_output = tl.load(
        partial_output_ptr
        + ((split_offsets[:, None] * num_rows + row) * NUM_QUERY_HEADS + head)
        * HEAD_DIM
        + dim_offsets[None, :],
        mask=split_mask[:, None],
        other=0.0,
    )
    merged = tl.sum(partial_output * weights[:, None], axis=0)
    merged = tl.where(denominator > 0, merged / denominator, 0.0)
    if KV_E4M3:
        # Apply the V scale once after combining all independently normalized
        # splits. Scaling partials earlier would repeat this work per split.
        merged *= v_scale
    tl.store(
        output_ptr + row * stride_output_row + head * stride_output_head + dim_offsets,
        merged,
    )


@triton.jit
def _store_qsa_rows_kernel(
    cache_ptr,
    slots_ptr,
    rows_ptr,
    stride_cache_block,
    stride_cache_token,
    stride_cache_dim,
    stride_rows_row,
    stride_rows_dim,
    num_rows,
    num_blocks,
    PAGE_SIZE: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    dims = tl.arange(0, BLOCK_D)
    slot = tl.load(slots_ptr + row)
    valid = (row < num_rows) & (slot >= 0) & (slot < num_blocks * PAGE_SIZE)
    block = tl.maximum(slot, 0) // PAGE_SIZE
    token = tl.maximum(slot, 0) % PAGE_SIZE
    values = tl.load(
        rows_ptr + row * stride_rows_row + dims * stride_rows_dim,
        mask=valid & (dims < WIDTH),
        other=0,
    )
    tl.store(
        cache_ptr
        + block * stride_cache_block
        + token * stride_cache_token
        + dims * stride_cache_dim,
        values,
        mask=valid & (dims < WIDTH),
    )


@triton.jit
def _compress_qsa_groups_kernel(
    raw_keys_ptr,  # this step's raw key rows, straight from activations
    raw_positions_ptr,  # this step's per-token positions
    compressor_state_cache_ptr,  # per-request ring of previous raw keys
    rope_cache_ptr,  # packed RoPE position tail of the ring
    compressor_state_table_ptr,
    token_to_req_ptr,
    query_start_loc_ptr,
    logical_positions_ptr,
    compressed_slots_ptr,
    pooled_ptr,
    first_positions_ptr,
    stride_raw_row,
    stride_raw_dim,
    stride_raw_positions_row,
    stride_raw_positions_dim,
    stride_compressor_state_block,
    stride_compressor_state_token,
    stride_compressor_state_dim,
    stride_rope_block,
    stride_rope_token,
    stride_rope_dim,
    stride_compressor_state_table_req,
    stride_pooled_row,
    stride_pooled_dim,
    stride_positions_row,
    stride_positions_dim,
    num_rows,
    num_compressor_state_blocks,
    num_requests,
    COMPRESSOR_STATE_SIZE: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    LOAD_ROPE_POSITIONS: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    dims = tl.arange(0, BLOCK_D)
    request = tl.load(token_to_req_ptr + row)
    end_position = tl.load(logical_positions_ptr + row)
    compressed_slot = tl.load(compressed_slots_ptr + row)
    valid_request = (request >= 0) & (request < num_requests)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    query_row_start = tl.load(
        query_start_loc_ptr + safe_request, mask=valid_request, other=0
    )
    query_row_end = tl.load(
        query_start_loc_ptr + safe_request + 1, mask=valid_request, other=0
    )
    chunk_start_position = end_position - (row - query_row_start)
    compressor_state_block = tl.load(
        compressor_state_table_ptr + safe_request * stride_compressor_state_table_req,
        mask=valid_request,
        other=-1,
    )
    valid_compressor_state_block = (compressor_state_block >= 0) & (
        compressor_state_block < num_compressor_state_blocks
    )
    valid_row = (
        (row < num_rows)
        & valid_request
        & (row >= query_row_start)
        & (row < query_row_end)
        & (end_position >= COMPRESS_RATIO - 1)
        & (compressed_slot >= 0)
    )
    accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)

    # A group can span the compressor-state ring (older members) and this
    # step's raw rows (members at positions >= chunk_start_position).
    for group_offset in tl.range(0, COMPRESS_RATIO):
        position = end_position - (COMPRESS_RATIO - 1 - group_offset)
        use_raw = position >= chunk_start_position
        raw_row = query_row_start + position - chunk_start_position
        raw_values = tl.load(
            raw_keys_ptr + raw_row * stride_raw_row + dims * stride_raw_dim,
            mask=valid_row
            & use_raw
            & (raw_row >= query_row_start)
            & (raw_row < query_row_end)
            & (raw_row < num_rows)
            & (dims < HEAD_DIM),
            other=0.0,
        ).to(tl.float32)
        compressor_state_values = tl.load(
            compressor_state_cache_ptr
            + tl.maximum(compressor_state_block, 0).to(tl.int64)
            * stride_compressor_state_block
            + (position % COMPRESSOR_STATE_SIZE) * stride_compressor_state_token
            + dims * stride_compressor_state_dim,
            mask=valid_row
            & ~use_raw
            & valid_compressor_state_block
            & (dims < HEAD_DIM),
            other=0.0,
        ).to(tl.float32)
        accumulator += tl.where(use_raw, raw_values, compressor_state_values)

    tl.store(
        pooled_ptr + row * stride_pooled_row + dims * stride_pooled_dim,
        accumulator / COMPRESS_RATIO,
        mask=(row < num_rows) & (dims < HEAD_DIM),
    )

    position_dims = tl.arange(0, 4)
    first_position = end_position - COMPRESS_RATIO + 1
    if LOAD_ROPE_POSITIONS:
        first_from_raw = first_position >= chunk_start_position
        raw_first_row = query_row_start + first_position - chunk_start_position
        raw_position_values = tl.load(
            raw_positions_ptr
            + raw_first_row * stride_raw_positions_row
            + position_dims * stride_raw_positions_dim,
            mask=valid_row
            & first_from_raw
            & (raw_first_row >= query_row_start)
            & (raw_first_row < query_row_end)
            & (raw_first_row < num_rows)
            & (position_dims < 3),
            other=0,
        )
        compressor_state_position_values = tl.load(
            rope_cache_ptr
            + tl.maximum(compressor_state_block, 0).to(tl.int64) * stride_rope_block
            + (first_position % COMPRESSOR_STATE_SIZE) * stride_rope_token
            + position_dims * stride_rope_dim,
            mask=valid_row
            & ~first_from_raw
            & valid_compressor_state_block
            & (position_dims < 3),
            other=0,
        )
        position_values = tl.where(
            first_from_raw,
            raw_position_values,
            compressor_state_position_values,
        )
    else:
        position_values = tl.where(valid_row, first_position, 0)
    tl.store(
        first_positions_ptr
        + row * stride_positions_row
        + position_dims * stride_positions_dim,
        position_values,
        mask=(row < num_rows) & (position_dims < 3),
    )


def _validate_mqa(q: torch.Tensor) -> None:
    if q.ndim != 3 or q.shape[1] <= 0 or q.shape[2] <= 0:
        raise ValueError("QSA query must be [rows, heads, head_dim]")


def qsa_mqa_paged(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    compress_ratio: int,
    num_columns: int | None = None,
    score_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute QSA scores directly from a paged compressed-key cache."""

    _validate_mqa(q)
    if not q.is_cuda or not HAS_TRITON:
        raise RuntimeError("paged QSA scoring requires CUDA and Triton")
    if k_cache.ndim != 4 or k_cache.shape[2] != 1:
        raise ValueError("QSA cache must be [pages, page_size, 1, head_dim]")
    if k_cache.shape[3] != q.shape[2]:
        raise ValueError("QSA query and cache dimensions must match")
    if page_table.ndim != 2:
        raise ValueError("QSA page table must be two-dimensional")
    if q.shape[0] and (not all(k_cache.shape[:2]) or not all(page_table.shape)):
        raise ValueError("QSA paged scoring cache and page table must be nonempty")
    if token_to_req.shape != (q.shape[0],):
        raise ValueError("QSA request mapping must match query rows")
    if query_positions.shape != (q.shape[0],):
        raise ValueError("QSA query positions must match query rows")
    if sequence_lengths.shape != (page_table.shape[0],):
        raise ValueError("QSA sequence lengths must match page-table requests")
    if compress_ratio <= 0:
        raise ValueError("QSA compression ratio must be positive")
    score_divisor = math.sqrt(q.shape[2]) if score_scale is None else score_scale
    if score_divisor <= 0:
        raise ValueError("QSA score scale must be positive")

    capacity = page_table.shape[1] * k_cache.shape[1]
    columns = capacity if num_columns is None else num_columns
    if columns < 0:
        raise ValueError("QSA score width must be non-negative")
    logits = torch.empty((q.shape[0], columns), dtype=torch.float32, device=q.device)
    visible_blocks = torch.empty(q.shape[0], dtype=torch.int32, device=q.device)
    if not q.shape[0] or not columns:
        return logits, visible_blocks
    sm70_single_token = q.shape[0] == 1 and current_platform.is_device_capability(70)
    # On V100 the GB300 decode tile leaves the 128-d scorer badly
    # under-occupied. A 32-column, two-warp tile preserves the selected QSA
    # blocks while exposing enough independent CTAs for the single-row path.
    BLOCK_N = 32 if sm70_single_token else 64
    BLOCK_D = max(16, triton.next_power_of_2(q.shape[2]))
    MAX_N = max(16, triton.next_power_of_2(q.shape[1]))
    # Tuned on GB300: larger row batches provide enough parallelism to reuse Q.
    tiles_per_program = 1 if q.shape[0] <= 32 else 8
    _qsa_mqa_paged_kernel[
        (q.shape[0], triton.cdiv(columns, BLOCK_N * tiles_per_program))
    ](
        q,
        k_cache,
        page_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        visible_blocks,
        logits,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(3),
        page_table.stride(0),
        page_table.stride(1),
        logits.stride(0),
        q.shape[0],
        columns,
        k_cache.shape[0],
        page_table.shape[0],
        float(score_divisor),
        PAGE_SIZE=k_cache.shape[1],
        PAGE_TABLE_WIDTH=page_table.shape[1],
        NUM_HEADS=q.shape[1],
        HEAD_DIM=q.shape[2],
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        TILES_PER_PROG=tiles_per_program,
        STAGES=2,
        MAX_N=MAX_N,
        COMPRESS_RATIO=compress_ratio,
        num_warps=2,
    )
    return logits, visible_blocks


def _qsa_indexer_cublas_shape_supported(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
) -> bool:
    """Whether the exact Qwen3.8 single-request index shape can use cuBLAS."""

    return (
        q.dtype == torch.float16
        and k_cache.dtype == torch.float16
        and q.ndim == 3
        and q.shape[1:] == (4, 128)
        and k_cache.ndim == 4
        and k_cache.shape[2:] == (1, 128)
        and page_table.ndim == 2
        and page_table.shape[0] == 1
    )


def _use_sm70_qsa_indexer_cublas(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
) -> bool:
    return (
        _SM70_INDEXER_CUBLAS
        and current_platform.is_device_capability(70)
        and q.shape[0] >= _SM70_INDEXER_CUBLAS_MIN_ROWS
        and _qsa_indexer_cublas_shape_supported(q, k_cache, page_table)
    )


def _qsa_indexer_cublas_work_supported(rows: int, columns: int) -> bool:
    return rows * columns >= _SM70_INDEXER_CUBLAS_MIN_SCORE_ELEMENTS


def _use_sm70_qsa_lexicographic_topk(topk: int) -> bool:
    """Use the exact, deterministic selector for Volta QSA."""

    return topk == 512 and current_platform.is_device_capability(70)


def _qsa_visible_blocks(
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    compress_ratio: int,
) -> torch.Tensor:
    rows = query_positions.shape[0]
    visible = torch.empty(rows, dtype=torch.int32, device=query_positions.device)
    if rows:
        _qsa_visible_blocks_kernel[(rows,)](
            token_to_req,
            query_positions,
            sequence_lengths,
            visible,
            rows,
            sequence_lengths.shape[0],
            COMPRESS_RATIO=compress_ratio,
            num_warps=1,
        )
    return visible


def _qsa_gather_single_request_keys(
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    columns: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    keys = torch.empty(
        (columns, k_cache.shape[3]), dtype=k_cache.dtype, device=k_cache.device
    )
    valid = torch.empty(columns, dtype=torch.uint8, device=k_cache.device)
    if columns:
        _qsa_gather_single_request_keys_kernel[(columns,)](
            k_cache,
            page_table[0],
            keys,
            valid,
            k_cache.stride(0),
            k_cache.stride(1),
            k_cache.stride(3),
            page_table.stride(1),
            keys.stride(0),
            columns,
            k_cache.shape[0],
            PAGE_SIZE=k_cache.shape[1],
            PAGE_TABLE_WIDTH=page_table.shape[1],
            HEAD_DIM=k_cache.shape[3],
            BLOCK_D=triton.next_power_of_2(k_cache.shape[3]),
            num_warps=4,
        )
    return keys, valid


def _qsa_mqa_cublas(
    q: torch.Tensor,
    keys: torch.Tensor,
    key_valid: torch.Tensor,
    visible: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score one request with Volta Tensor Cores and a fused QSA epilogue."""

    rows, num_heads, head_dim = q.shape
    columns = keys.shape[0]
    logits = torch.empty((rows, columns), dtype=torch.float32, device=q.device)
    if not rows or not columns:
        return logits, visible

    q2 = q.reshape(rows * num_heads, head_dim)
    bytes_per_column = max(1, q2.shape[0] * torch.float32.itemsize)
    tile_columns = max(
        256,
        min(columns, _SM70_INDEXER_SCORE_TILE_BYTES // bytes_per_column),
    )
    tile_columns = max(256, tile_columns // 256 * 256)
    for column_start in range(0, columns, tile_columns):
        column_stop = min(column_start + tile_columns, columns)
        width = column_stop - column_start
        # FP16 inputs, FP32 accumulation/output. This still selects Volta HMMA
        # through cuBLAS while avoiding an FP16 round trip before top-k.
        scores = torch.mm(
            q2,
            keys[column_start:column_stop].t(),
            out_dtype=torch.float32,
        )
        _qsa_relu_headsum_visible_kernel[(rows, triton.cdiv(width, 256))](
            scores,
            visible,
            key_valid,
            logits,
            scores.stride(0),
            scores.stride(1),
            logits.stride(0),
            width,
            column_start,
            math.sqrt(head_dim),
            NUM_HEADS=num_heads,
            BLOCK_N=256,
            num_warps=4,
        )
    return logits, visible


def expand_qsa_block_indices_cuda(
    block_indices: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    token_to_req: torch.Tensor,
    compress_ratio: int,
    token_topk: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Expand compressed blocks and compact the causal tail of the open group."""

    if not block_indices.is_cuda or not HAS_TRITON:
        raise RuntimeError("QSA CUDA expansion requires Triton")
    if token_topk % compress_ratio:
        raise ValueError("QSA token top-k must be divisible by compression ratio")
    block_topk = token_topk // compress_ratio
    output_width = token_topk + compress_ratio - 1
    if block_indices.shape != (query_positions.numel(), block_topk):
        raise ValueError("QSA compressed top-k has an invalid shape")
    if token_to_req.shape != query_positions.shape:
        raise ValueError("QSA request mapping must match query positions")
    if sequence_lengths.ndim != 1 or not sequence_lengths.shape[0]:
        raise ValueError("QSA request sequence lengths must be nonempty")
    if out is None:
        out = torch.empty(
            (block_indices.shape[0], output_width),
            dtype=torch.int32,
            device=block_indices.device,
        )
    elif out.shape != (block_indices.shape[0], output_width):
        raise ValueError("QSA expansion output has an invalid shape")
    if not block_indices.shape[0]:
        return out
    column_block = 256
    _expand_qsa_indices_kernel[
        (block_indices.shape[0], triton.cdiv(output_width, column_block))
    ](
        block_indices,
        query_positions,
        sequence_lengths,
        token_to_req,
        out,
        block_indices.stride(0),
        block_indices.stride(1),
        out.stride(0),
        out.stride(1),
        block_indices.shape[0],
        sequence_lengths.shape[0],
        BLOCK_TOPK=block_topk,
        COMPRESS_RATIO=compress_ratio,
        TOKEN_TOPK=token_topk,
        OUTPUT_WIDTH=output_width,
        COLUMN_BLOCK=column_block,
        num_warps=4,
    )
    return out


def qsa_select_paged_tokens(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    token_topk: int,
    compress_ratio: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Score, select, and expand QSA indices without host synchronization."""

    rows = q.shape[0]
    output_width = token_topk + compress_ratio - 1
    if out is None:
        out = torch.empty((rows, output_width), dtype=torch.int32, device=q.device)
    if out.shape != (rows, output_width):
        raise ValueError("QSA selection output has an invalid shape")
    if not rows:
        return out

    capacity_columns = page_table.shape[1] * k_cache.shape[1]
    score_columns = capacity_columns
    block_topk = token_topk // compress_ratio
    contiguous_keys: torch.Tensor | None = None
    key_valid: torch.Tensor | None = None
    all_visible: torch.Tensor | None = None
    if _use_sm70_qsa_indexer_cublas(q, k_cache, page_table):
        all_visible = _qsa_visible_blocks(
            token_to_req,
            query_positions,
            sequence_lengths,
            compress_ratio,
        )
        # cuBLAS has a rectangular host-side launch shape, unlike the paged
        # Triton kernel's device-side early exit. Bound it to this chunk's live
        # prefix so early-context prefill does not multiply the unused 140K
        # model-capacity tail. This is one scalar sync per QSA layer.
        score_columns = min(
            capacity_columns,
            max(block_topk, int(all_visible.max().item())),
        )
        if _qsa_indexer_cublas_work_supported(rows, score_columns):
            logger.info_once(
                "Using SM70 QSA indexer prefill cuBLAS path "
                "(single-request FP16, rows=%d, score_tile_mib=%d).",
                rows,
                _SM70_INDEXER_SCORE_TILE_BYTES // (1024 * 1024),
            )
            # Gather this request's paged MQA keys once, then reuse them across
            # every bounded logits chunk below. Generic, short-work and
            # multi-request batches keep the paged Triton fallback.
            contiguous_keys, key_valid = _qsa_gather_single_request_keys(
                k_cache, page_table, score_columns
            )
        else:
            score_columns = capacity_columns
            all_visible = None
    rows_per_chunk = max(1, _LOGITS_WORKSPACE_BYTES // max(score_columns * 4, 1))
    chunk_rows = min(rows, rows_per_chunk)
    blocks_buffer = torch.empty(
        (chunk_rows, block_topk), dtype=torch.int32, device=q.device
    )
    topk_workspace = torch.empty(
        (_TOPK_WORKSPACE_BYTES,), dtype=torch.uint8, device=q.device
    )
    for row_start in range(0, rows, rows_per_chunk):
        row_end = min(row_start + rows_per_chunk, rows)
        row_slice = slice(row_start, row_end)
        if contiguous_keys is not None and key_valid is not None:
            assert all_visible is not None
            logits, visible_blocks = _qsa_mqa_cublas(
                q[row_slice],
                contiguous_keys,
                key_valid,
                all_visible[row_slice],
            )
        else:
            logits, visible_blocks = qsa_mqa_paged(
                q[row_slice],
                k_cache,
                page_table,
                token_to_req[row_slice],
                query_positions[row_slice],
                sequence_lengths,
                compress_ratio,
            )
        blocks = blocks_buffer[: row_end - row_start]
        use_cooperative_topk = (
            blocks.shape[0] <= 32
            and logits.stride(0) % 4 == 0
            and current_platform.has_device_capability(90)
            and not current_platform.is_device_capability_family(120)
        )
        if _use_sm70_qsa_lexicographic_topk(block_topk):
            logger.info_once(
                "Using exact SM70 QSA lexicographic top-k "
                "(score descending, block index ascending)."
            )
            torch.ops._C.qsa_lexicographic_topk(
                logits,
                visible_blocks,
                blocks,
                block_topk,
            )
        else:
            topk_op = (
                torch.ops._C.cooperative_topk
                if use_cooperative_topk
                else torch.ops._C.persistent_topk
            )
            topk_op(
                logits,
                visible_blocks,
                blocks,
                topk_workspace,
                block_topk,
                score_columns,
            )
        expand_qsa_block_indices_cuda(
            blocks,
            query_positions[row_slice],
            sequence_lengths,
            token_to_req[row_slice],
            compress_ratio,
            token_topk,
            out[row_slice],
        )
    return out


def _qsa_xqa_page4_shape_supported(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor | None,
    sequence_lengths: torch.Tensor | None,
) -> bool:
    return (
        query_positions is not None
        and sequence_lengths is not None
        and q.dtype == torch.float16
        and k_cache.dtype == v_cache.dtype
        and k_cache.dtype in (torch.float16, torch.uint8)
        and q.device
        == k_cache.device
        == v_cache.device
        == logical_indices.device
        == block_table.device
        == token_to_req.device
        == query_positions.device
        == sequence_lengths.device
        and q.ndim == 3
        and q.shape[1:] == (6, 256)
        and q.stride(2) == 1
        and k_cache.ndim == 4
        and v_cache.shape == k_cache.shape
        and k_cache.shape[2:] == (1, 256)
        and k_cache.shape[1] % 4 == 0
        and k_cache.stride(3) == v_cache.stride(3) == 1
        and k_cache.stride(1) == v_cache.stride(1) == 256
        and k_cache.stride(0) == v_cache.stride(0)
        and k_cache.stride(0) in (k_cache.shape[1] * 256, 2 * k_cache.shape[1] * 256)
        and logical_indices.shape == (q.shape[0], 2051)
        and logical_indices.dtype == torch.int32
        and logical_indices.stride(1) == 1
        and block_table.ndim == 2
        and block_table.dtype == torch.int32
        and block_table.stride(1) == 1
        and token_to_req.shape == (q.shape[0],)
        and token_to_req.dtype == torch.int32
        and token_to_req.stride(0) == 1
        and query_positions.shape == (q.shape[0],)
        and query_positions.dtype == torch.int64
        and query_positions.stride(0) == 1
        and sequence_lengths.shape == (block_table.shape[0],)
        and sequence_lengths.dtype == torch.int32
        and sequence_lengths.stride(0) == 1
        and k_cache.shape[0] * (k_cache.stride(0) // (4 * 256))
        < _SM70_QSA_XQA_PAGE4_MARKER
    )


def _use_sm70_qsa_xqa_page4(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor | None,
    sequence_lengths: torch.Tensor | None,
) -> bool:
    return (
        _SM70_QSA_XQA_PAGE4
        and current_platform.is_device_capability(70)
        and (
            q.shape[0] >= _SM70_QSA_XQA_PAGE4_MIN_ROWS
            or (k_cache.dtype == torch.uint8 and q.shape[0] > 16)
        )
        and _qsa_xqa_page4_shape_supported(
            q,
            k_cache,
            v_cache,
            logical_indices,
            block_table,
            token_to_req,
            query_positions,
            sequence_lengths,
        )
    )


def _qsa_xqa_page4_block_table(
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    num_cache_blocks: int,
    page_size: int,
    physical_page_stride: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if physical_page_stride is None:
        physical_page_stride = page_size // 4
    rows = logical_indices.shape[0]
    encoded_pages = torch.empty(
        (rows, _SM70_QSA_XQA_PAGE4_PAGES),
        dtype=torch.int64,
        device=logical_indices.device,
    )
    xqa_sequence_lengths = torch.empty(
        (rows,), dtype=torch.int32, device=logical_indices.device
    )
    _qsa_xqa_page4_table_kernel[(rows,)](
        logical_indices,
        block_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        encoded_pages,
        xqa_sequence_lengths,
        logical_indices.stride(0),
        block_table.stride(0),
        encoded_pages.stride(0),
        rows,
        num_cache_blocks,
        block_table.shape[0],
        PAGE_SIZE=page_size,
        PAGE_TABLE_WIDTH=block_table.shape[1],
        COMPLETE_PAGES=2048 // 4,
        OUTPUT_PAGES=_SM70_QSA_XQA_PAGE4_PAGES,
        BLOCK_PAGES=1024,
        PHYSICAL_PAGE_STRIDE=physical_page_stride,
        num_warps=4,
    )
    sorted_pages = torch.sort(encoded_pages, dim=1).values
    physical_pages = torch.bitwise_and(
        sorted_pages,
        _SM70_QSA_XQA_PAGE4_MARKER - 1,
    ).to(torch.int32)
    return physical_pages, xqa_sequence_lengths


def _qsa_xqa_page4_workspace(
    q: torch.Tensor,
    num_partitions: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device_index = q.device.index if q.device.index is not None else -1
    stream_id = int(torch.cuda.current_stream(q.device).cuda_stream)
    key = (device_index, stream_id, q.shape[1], q.shape[2], num_partitions)
    workspace = _SM70_QSA_XQA_PAGE4_WORKSPACES.get(key)
    rows = q.shape[0]
    if workspace is None or workspace[0] < rows:
        capacity = 1 << (rows - 1).bit_length()
        temporary_output = torch.empty(
            (capacity, q.shape[1], num_partitions, q.shape[2]),
            dtype=torch.float16,
            device=q.device,
        )
        max_logits = torch.empty(
            (capacity, q.shape[1], num_partitions),
            dtype=torch.float32,
            device=q.device,
        )
        exp_sums = torch.empty_like(max_logits)
        active_num_partitions = torch.tensor(
            [num_partitions], dtype=torch.int32, device=q.device
        )
        workspace = (
            capacity,
            temporary_output,
            max_logits,
            exp_sums,
            active_num_partitions,
        )
        _SM70_QSA_XQA_PAGE4_WORKSPACES[key] = workspace
    _, temporary_output, max_logits, exp_sums, active_num_partitions = workspace
    return (
        temporary_output[:rows],
        max_logits[:rows],
        exp_sums[:rows],
        active_num_partitions,
    )


def _qsa_grouped_page4_workspace(
    q: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    groups = q.shape[0] // _SM70_QSA_GROUPED_PAGE4_QUERIES
    device_index = q.device.index if q.device.index is not None else -1
    stream_id = int(torch.cuda.current_stream(q.device).cuda_stream)
    key = (device_index, stream_id)
    workspace = _SM70_QSA_GROUPED_PAGE4_WORKSPACES.get(key)
    if workspace is None or workspace[0] < groups:
        capacity = 1 << (groups - 1).bit_length()
        grouped_pages = torch.empty(
            (capacity, _SM70_QSA_GROUPED_PAGE4_OUTPUT_PAGES),
            dtype=torch.int32,
            device=q.device,
        )
        token_masks = torch.empty(
            (capacity, _SM70_QSA_GROUPED_PAGE4_OUTPUT_PAGES),
            dtype=torch.uint32,
            device=q.device,
        )
        grouped_sequence_lengths = torch.empty(
            (capacity,), dtype=torch.int32, device=q.device
        )
        lse = torch.empty(
            (capacity * _SM70_QSA_GROUPED_PAGE4_QUERIES, q.shape[1]),
            dtype=torch.float32,
            device=q.device,
        )
        workspace = (
            capacity,
            grouped_pages,
            token_masks,
            grouped_sequence_lengths,
            lse,
        )
        _SM70_QSA_GROUPED_PAGE4_WORKSPACES[key] = workspace
    _, grouped_pages, token_masks, grouped_sequence_lengths, lse = workspace
    return (
        grouped_pages[:groups],
        token_masks[:groups],
        grouped_sequence_lengths[:groups],
        lse[: q.shape[0]],
    )


def _qsa_xqa_page4_physical_kv(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    microblock_stride = 4 * q.shape[2]
    if k_cache.stride(0) == k_cache.shape[1] * q.shape[2]:
        microblocks_per_cache_block = k_cache.shape[1] // 4
        physical_k_cache = k_cache.view(
            k_cache.shape[0] * microblocks_per_cache_block,
            4,
            1,
            q.shape[2],
        )
        physical_v_cache = v_cache.view_as(physical_k_cache)
    else:
        # The local FlashAttention ABI interleaves K and V inside every
        # physical cache block. The virtual page IDs carry that doubled block
        # stride, while this narrow view exposes a four-token page stride.
        physical_shape = (k_cache.shape[0], 4, 1, q.shape[2])
        physical_strides = (
            microblock_stride,
            q.shape[2],
            q.shape[2],
            1,
        )
        physical_k_cache = k_cache.as_strided(physical_shape, physical_strides)
        physical_v_cache = v_cache.as_strided(physical_shape, physical_strides)
    return physical_k_cache, physical_v_cache


def _qsa_grouped_page4_forward(
    flash_attn_v100_cuda,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    out: torch.Tensor,
    grouped_pages: torch.Tensor,
    token_masks: torch.Tensor,
    grouped_sequence_lengths: torch.Tensor,
    lse: torch.Tensor,
    softmax_scale: float,
    kv_cache_dtype: str,
    k_scale: float,
    v_scale: float,
) -> None:
    forward_args = (
        q,
        k_cache,
        v_cache,
        out,
        grouped_pages,
        token_masks,
        grouped_sequence_lengths,
        lse,
        softmax_scale,
    )
    abi_version = _qsa_grouped_page4_abi_version(flash_attn_v100_cuda)
    if abi_version >= 2:
        flash_attn_v100_cuda.grouped_sparse_page4_fwd(
            *forward_args,
            kv_cache_dtype,
            k_scale,
            v_scale,
        )
        return

    # ABI v1 only supports the original FP16 K/V contract. The route
    # eligibility check rejects quantized K/V before the planner runs.
    assert abi_version == 1 and kv_cache_dtype in ("auto", "float16")
    flash_attn_v100_cuda.grouped_sparse_page4_fwd(*forward_args)


def _qsa_sparse_paged_attention_sm70_grouped_page4(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    out: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: float,
    v_scale: float,
    flash_attn_v100_cuda,
) -> torch.Tensor:
    grouped_pages, token_masks, grouped_sequence_lengths, lse = (
        _qsa_grouped_page4_workspace(q)
    )
    physical_page_stride = k_cache.stride(0) // (4 * q.shape[2])
    flash_attn_v100_cuda.grouped_sparse_page4_plan_fwd(
        logical_indices,
        block_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        grouped_pages,
        token_masks,
        grouped_sequence_lengths,
        k_cache.shape[1],
        physical_page_stride,
        k_cache.shape[0],
    )
    physical_k_cache, physical_v_cache = _qsa_xqa_page4_physical_kv(q, k_cache, v_cache)
    _qsa_grouped_page4_forward(
        flash_attn_v100_cuda,
        q,
        physical_k_cache,
        physical_v_cache,
        out,
        grouped_pages,
        token_masks,
        grouped_sequence_lengths,
        lse,
        q.shape[2] ** -0.5,
        kv_cache_dtype,
        k_scale,
        v_scale,
    )
    logger.info_once(
        "Using SM70 grouped QSA Flash-V100 page4 prefill route (rows=%d, groups=%d).",
        q.shape[0],
        q.shape[0] // _SM70_QSA_GROUPED_PAGE4_QUERIES,
    )
    return out


def _qsa_sparse_paged_attention_sm70_xqa_page4_batch(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    out: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: float,
    v_scale: float,
    flash_attn_v100_cuda,
) -> torch.Tensor:
    virtual_block_table, xqa_sequence_lengths = _qsa_xqa_page4_block_table(
        logical_indices,
        block_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        k_cache.shape[0],
        k_cache.shape[1],
        k_cache.stride(0) // (4 * q.shape[2]),
    )
    # Generic E4M3 G6 XQA supports P256 for virtual page4 caches. Its P1024
    # specialization is restricted to the page1568 layout.
    partition_size = (
        256 if kv_cache_dtype == "fp8_e4m3" else _SM70_QSA_XQA_PAGE4_PARTITION
    )
    num_partitions = math.ceil(logical_indices.shape[1] / partition_size)
    temporary_output, max_logits, exp_sums, active_num_partitions = (
        _qsa_xqa_page4_workspace(q, num_partitions)
    )
    physical_k_cache, physical_v_cache = _qsa_xqa_page4_physical_kv(q, k_cache, v_cache)
    flash_attn_v100_cuda.decode_paged_xqa_fwd(
        q,
        physical_k_cache,
        physical_v_cache,
        out,
        virtual_block_table,
        xqa_sequence_lengths,
        temporary_output,
        max_logits,
        exp_sums,
        active_num_partitions,
        q.shape[2] ** -0.5,
        partition_size,
        num_partitions,
        kv_cache_dtype,
        k_scale,
        v_scale,
        -1,
        -1,
        0,
    )
    logger.info_once(
        "Using SM70 QSA Flash-V100 XQA page4 prefill route (rows=%d, partitions=%d).",
        q.shape[0],
        num_partitions,
    )
    return out


def _qsa_sparse_paged_attention_sm70_xqa_page4(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    out: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: float,
    v_scale: float,
) -> torch.Tensor | None:
    try:
        from flash_attn_v100.flash_attn_interface import flash_attn_v100_cuda
    except ImportError:
        logger.warning_once(
            "SM70 QSA page4 XQA route is unavailable because Flash-V100 "
            "could not be imported; using Triton sparse attention."
        )
        return None
    if not hasattr(flash_attn_v100_cuda, "decode_paged_xqa_fwd"):
        logger.warning_once(
            "SM70 QSA page4 XQA route is unavailable in this Flash-V100 build; "
            "using Triton sparse attention."
        )
        return None

    grouped_enabled = _SM70_QSA_GROUPED_PAGE4 and _qsa_grouped_page4_supported(
        flash_attn_v100_cuda, kv_cache_dtype
    )
    if grouped_enabled:
        grouped_rows = (
            q.shape[0] // _SM70_QSA_GROUPED_PAGE4_QUERIES
        ) * _SM70_QSA_GROUPED_PAGE4_QUERIES
        if grouped_rows:
            _qsa_sparse_paged_attention_sm70_grouped_page4(
                q[:grouped_rows],
                k_cache,
                v_cache,
                logical_indices[:grouped_rows],
                block_table,
                token_to_req[:grouped_rows],
                query_positions[:grouped_rows],
                sequence_lengths,
                out[:grouped_rows],
                kv_cache_dtype,
                k_scale,
                v_scale,
                flash_attn_v100_cuda,
            )
        if grouped_rows == q.shape[0]:
            return out

        logger.info_once(
            "Splitting a non-grouped page4 batch across grouped/XQA routes "
            "(rows=%d, grouped_rows=%d, kv_cache_dtype=%s).",
            q.shape[0],
            grouped_rows,
            kv_cache_dtype,
        )
        _qsa_sparse_paged_attention_sm70_xqa_page4_batch(
            q[grouped_rows:],
            k_cache,
            v_cache,
            logical_indices[grouped_rows:],
            block_table,
            token_to_req[grouped_rows:],
            query_positions[grouped_rows:],
            sequence_lengths,
            out[grouped_rows:],
            kv_cache_dtype,
            k_scale,
            v_scale,
            flash_attn_v100_cuda,
        )
        return out

    if kv_cache_dtype == "fp8_e4m3" and q.shape[0] > 16:
        # The generic E4M3 XQA kernel accepts at most 16 query rows. Scheduler
        # iterations can mix a large prefill (or catch-up chunk) with decode
        # rows. If an older Flash-V100 build lacks the quantized grouped ABI,
        # retain correctness by slicing the work into supported XQA batches.
        # This avoids both an invalid B>16 launch and the larger Triton split-K
        # fallback workspace.
        logger.info_once(
            "Splitting an E4M3 page4 batch across supported XQA launches "
            "because this Flash-V100 build lacks the quantized grouped ABI "
            "(rows=%d).",
            q.shape[0],
        )
        for row_start in range(0, q.shape[0], 16):
            row_end = min(row_start + 16, q.shape[0])
            _qsa_sparse_paged_attention_sm70_xqa_page4_batch(
                q[row_start:row_end],
                k_cache,
                v_cache,
                logical_indices[row_start:row_end],
                block_table,
                token_to_req[row_start:row_end],
                query_positions[row_start:row_end],
                sequence_lengths,
                out[row_start:row_end],
                kv_cache_dtype,
                k_scale,
                v_scale,
                flash_attn_v100_cuda,
            )
        return out

    return _qsa_sparse_paged_attention_sm70_xqa_page4_batch(
        q,
        k_cache,
        v_cache,
        logical_indices,
        block_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        out,
        kv_cache_dtype,
        k_scale,
        v_scale,
        flash_attn_v100_cuda,
    )


def qsa_sparse_paged_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    out: torch.Tensor | None = None,
    query_positions: torch.Tensor | None = None,
    sequence_lengths: torch.Tensor | None = None,
    kv_cache_dtype: str = "auto",
    k_scale: float = 1.0,
    v_scale: float = 1.0,
) -> torch.Tensor:
    """Run sparse GQA over paged FP16/BF16 or calibrated E4M3 K/V."""

    if not q.is_cuda or not HAS_TRITON:
        raise RuntimeError("paged QSA sparse attention requires CUDA and Triton")
    if q.ndim != 3 or k_cache.ndim != 4 or v_cache.shape != k_cache.shape:
        raise ValueError("QSA sparse attention received invalid Q/K/V shapes")
    if logical_indices.ndim != 2 or logical_indices.shape[0] != q.shape[0]:
        raise ValueError("QSA indices must have one row per query")
    if token_to_req.shape != (q.shape[0],) or block_table.ndim != 2:
        raise ValueError("QSA sparse attention metadata has invalid shapes")
    if not all(k_cache.shape[:3]) or not all(block_table.shape):
        raise ValueError("QSA sparse attention cache and block table must be nonempty")
    if logical_indices.shape[1] <= 0:
        raise ValueError("QSA sparse attention requires a positive selection width")
    if q.shape[2] != k_cache.shape[3] or q.shape[1] % k_cache.shape[2]:
        raise ValueError("QSA sparse attention requires valid grouped-query heads")
    head_dim = q.shape[2]
    assert head_dim >= 16 and (head_dim & (head_dim - 1)) == 0
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("QSA sparse attention requires FP16 or BF16 queries")
    kv_e4m3 = kv_cache_dtype in ("fp8", "fp8_e4m3")
    if kv_e4m3:
        if k_cache.dtype != torch.uint8 or v_cache.dtype != torch.uint8:
            raise ValueError("QSA E4M3 K/V caches must use uint8 storage")
        if not math.isfinite(k_scale) or not math.isfinite(v_scale):
            raise ValueError("QSA E4M3 K/V scales must be finite")
        if k_scale <= 0.0 or v_scale <= 0.0:
            raise ValueError("QSA E4M3 K/V scales must be positive")
    else:
        if kv_cache_dtype not in ("auto", "float16", "bfloat16"):
            raise ValueError(f"Unsupported QSA K/V cache dtype: {kv_cache_dtype}")
        if q.dtype != k_cache.dtype or q.dtype != v_cache.dtype:
            raise ValueError("QSA unquantized K/V caches must match query dtype")
    assert logical_indices.dtype == block_table.dtype == torch.int32
    assert token_to_req.dtype == torch.int32
    assert q.device == k_cache.device == v_cache.device
    assert q.device == logical_indices.device == block_table.device
    assert q.device == token_to_req.device
    assert q.stride(2) == k_cache.stride(3) == v_cache.stride(3) == 1
    assert logical_indices.stride(1) == block_table.stride(1) == 1
    assert token_to_req.stride(0) == 1
    if out is None:
        out = torch.empty_like(q)
    if out.shape != q.shape:
        raise ValueError("QSA sparse output must match its query")
    assert out.dtype == q.dtype and out.device == q.device
    assert out.stride(2) == 1
    if not q.shape[0]:
        return out

    if _use_sm70_qsa_xqa_page4(
        q,
        k_cache,
        v_cache,
        logical_indices,
        block_table,
        token_to_req,
        query_positions,
        sequence_lengths,
    ):
        assert query_positions is not None and sequence_lengths is not None
        xqa_output = _qsa_sparse_paged_attention_sm70_xqa_page4(
            q,
            k_cache,
            v_cache,
            logical_indices,
            block_table,
            token_to_req,
            query_positions,
            sequence_lengths,
            out,
            "fp8_e4m3" if kv_e4m3 else "auto",
            k_scale,
            v_scale,
        )
        if xqa_output is not None:
            return xqa_output

    group_size = q.shape[1] // k_cache.shape[2]
    block_m = triton.next_power_of_2(group_size)
    base_programs = q.shape[0] * k_cache.shape[2]
    block_n, target_splits, partial_warps = _qsa_sparse_launch_profile(
        base_programs,
        block_m,
        not current_platform.has_device_capability(80),
    )

    if (
        q.shape[0] == 1
        and group_size == 6
        and head_dim == 256
        and current_platform.is_device_capability(70)
    ):
        # Exact Qwen4Exp TP4 decode shape. Two warps preserve the existing
        # split/merge arithmetic and cut the partial-kernel time on V100.
        partial_warps = 2

    num_tiles = triton.cdiv(logical_indices.shape[1], block_n)
    # Avoid empty splits when the selection width is smaller than the profile.
    max_useful_splits = 1 << (num_tiles.bit_length() - 1)
    num_splits = min(max_useful_splits, target_splits)

    # Split=1 writes output directly and compiles out all workspace accesses.
    if num_splits == 1:
        partial_output = out
        partial_lse = out
    else:
        # FP32 partials preserve accuracy when merging independently normalized
        # splits.
        partial_output = torch.empty(
            (num_splits, *q.shape), dtype=torch.float32, device=q.device
        )
        partial_lse = torch.empty(
            (num_splits, q.shape[0], q.shape[1]),
            dtype=torch.float32,
            device=q.device,
        )

    partial_grid = (q.shape[0], k_cache.shape[2], num_splits)
    _qsa_sparse_paged_gqa_splitk_kernel[partial_grid](
        q,
        k_cache,
        v_cache,
        logical_indices,
        block_table,
        token_to_req,
        partial_output,
        partial_lse,
        out,
        q.stride(0),
        q.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        logical_indices.stride(0),
        block_table.stride(0),
        out.stride(0),
        out.stride(1),
        q.shape[0],
        k_cache.shape[0],
        block_table.shape[0],
        k_scale,
        v_scale,
        TOPK=logical_indices.shape[1],
        PAGE_SIZE=k_cache.shape[1],
        PAGE_TABLE_WIDTH=block_table.shape[1],
        GROUP_SIZE=group_size,
        HEAD_DIM=q.shape[2],
        NUM_QUERY_HEADS=q.shape[1],
        NUM_SPLITS=num_splits,
        NUM_TILES=num_tiles,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        KV_E4M3=kv_e4m3,
        num_warps=partial_warps,
        num_stages=2,
    )
    if num_splits == 1:
        return out

    _qsa_merge_splitk_kernel[(q.shape[0], q.shape[1])](
        partial_output,
        partial_lse,
        out,
        out.stride(0),
        out.stride(1),
        q.shape[0],
        v_scale,
        HEAD_DIM=q.shape[2],
        NUM_QUERY_HEADS=q.shape[1],
        NUM_SPLITS=num_splits,
        BLOCK_SPLITS=triton.next_power_of_2(num_splits),
        KV_E4M3=kv_e4m3,
        num_warps=2,
        num_stages=1,
    )
    return out


def _qsa_sparse_launch_profile(
    base_programs: int,
    block_m: int,
    is_pre_ampere: bool,
) -> tuple[int, int, int]:
    """Return BLOCK_N, target splits, and warps for sparse QSA."""
    small_profile_limit = 8 if block_m <= 8 else 4

    # Tuned on GB300 for the Qwen-Air TP1, TP2, and TP4 attention shapes.
    # Narrow tiles favor decode; wide tiles improve throughput for prefill.
    if base_programs <= small_profile_limit:
        block_n, target_splits, partial_warps = 16, 64, 4
    elif base_programs < 32:
        block_n, target_splits, partial_warps = 16, 32, 4
    elif base_programs <= 256:
        block_n, target_splits, partial_warps = 64, 8, 2
    elif base_programs <= 512:
        block_n, target_splits, partial_warps = 64, 4, 2
    else:
        block_n, target_splits, partial_warps = 64, 1, 2
    if is_pre_ampere and block_n == 64:
        # Pre-Ampere: the 64-column tile at D=256 does not fit Turing's
        # 64 KiB shared-memory limit (Triton OutOfResources -- the kernel
        # cannot launch on SM75 at all), and two warps serialize the D=256
        # tensor-core work on V100. A 16-column tile with four warps
        # launches on both and measured 1.16-2.6x faster than the best
        # previously runnable profile across the 64..2048-row prefill
        # regimes (V100-PCIE-32GB and Quadro RTX 8000, see #441).
        partial_warps = 4
        block_n = 16
    return block_n, target_splits, partial_warps


def qsa_store_cache_rows(
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    rows: torch.Tensor,
) -> None:
    """Store fixed-width rows in a QSA cache without boolean indexing."""

    if not cache.is_cuda or not HAS_TRITON:
        raise RuntimeError("QSA CUDA cache stores require Triton")
    if cache.ndim != 4 or cache.shape[2] != 1:
        raise ValueError("QSA cache must be [pages, page_size, 1, width]")
    if not all(cache.shape):
        raise ValueError("QSA cache dimensions must be nonzero")
    if rows.ndim == 3:
        if rows.shape[1] != 1:
            raise ValueError("QSA cache rows must have one head")
        rows = rows[:, 0]
    if rows.shape != (slot_mapping.numel(), cache.shape[3]):
        raise ValueError("QSA cache rows and slots have incompatible shapes")
    if not rows.shape[0]:
        return
    _store_qsa_rows_kernel[(rows.shape[0],)](
        cache,
        slot_mapping,
        rows,
        cache.stride(0),
        cache.stride(1),
        cache.stride(3),
        rows.stride(0),
        rows.stride(1),
        rows.shape[0],
        cache.shape[0],
        PAGE_SIZE=cache.shape[1],
        WIDTH=cache.shape[3],
        BLOCK_D=triton.next_power_of_2(cache.shape[3]),
        num_warps=4,
    )


def qsa_compress_groups_with_ratio(
    raw_keys: torch.Tensor,  # this step's raw key rows [rows, 1, head_size]
    raw_positions: torch.Tensor,  # this step's positions [rows, 1, 3] int64
    compressor_state_cache: torch.Tensor,
    compressor_state_block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_start_loc: torch.Tensor,
    logical_positions: torch.Tensor,
    compressed_slots: torch.Tensor,
    compress_ratio: int,
    rope_cache: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool completed groups from the compressor-state ring and raw token rows."""

    if not raw_keys.is_cuda or not HAS_TRITON:
        raise RuntimeError("QSA CUDA compression requires Triton")
    rows = token_to_req.numel()
    if compress_ratio <= 0:
        raise ValueError("QSA compression ratio must be positive")
    if raw_keys.ndim != 3 or raw_keys.shape[:2] != (rows, 1):
        raise ValueError("QSA raw keys must be [rows, 1, head_size]")
    if raw_positions.shape != (rows, 1, 3) or raw_positions.dtype != torch.int64:
        raise ValueError("QSA raw positions must be [rows, 1, 3] int64")
    if logical_positions.shape != (rows,) or compressed_slots.shape != (rows,):
        raise ValueError("QSA compression metadata must match token rows")
    if compressor_state_cache.ndim != 4 or compressor_state_cache.shape[2] != 1:
        raise ValueError("QSA compressor-state cache has an invalid shape")
    if (
        # The ring is wider than one group so speculative rows cannot alias
        # onto the committed keys of the group still being collected.
        compressor_state_cache.shape[1] < compress_ratio
        or compressor_state_cache.shape[3] != raw_keys.shape[2]
        or compressor_state_cache.dtype != raw_keys.dtype
    ):
        raise ValueError(
            "QSA compressor-state cache does not match the compression layout"
        )
    if (
        compressor_state_block_table.ndim != 2
        or compressor_state_block_table.shape[1] < 1
    ):
        raise ValueError(
            "QSA compressor-state block table must contain one block per request"
        )
    if query_start_loc.ndim != 1 or query_start_loc.shape[0] < 2:
        raise ValueError("QSA query starts must contain a terminal offset")
    num_requests = query_start_loc.shape[0] - 1
    if compressor_state_block_table.shape[0] < num_requests:
        raise ValueError("QSA compressor-state block table has too few request rows")
    if rope_cache is not None and (
        rope_cache.ndim != 4
        or rope_cache.shape[:3] != compressor_state_cache.shape[:3]
        or rope_cache.shape[3] != 3
        or rope_cache.dtype != torch.int64
    ):
        raise ValueError("QSA packed position view has an invalid shape or dtype")
    if rows and (
        not all(compressor_state_cache.shape)
        or not all(compressor_state_block_table.shape)
    ):
        raise ValueError("QSA compressor-state cache and block table must be nonempty")
    pooled = torch.empty(
        (rows, 1, raw_keys.shape[2]),
        dtype=raw_keys.dtype,
        device=raw_keys.device,
    )
    first_positions = torch.empty((rows, 3), dtype=torch.int64, device=raw_keys.device)
    if not rows:
        return pooled, first_positions
    if rope_cache is None:
        rope_cache = compressor_state_cache
        load_rope_positions = False
    else:
        load_rope_positions = True
    _compress_qsa_groups_kernel[(rows,)](
        raw_keys,
        raw_positions,
        compressor_state_cache,
        rope_cache,
        compressor_state_block_table,
        token_to_req,
        query_start_loc,
        logical_positions,
        compressed_slots,
        pooled,
        first_positions,
        raw_keys.stride(0),
        raw_keys.stride(2),
        raw_positions.stride(0),
        raw_positions.stride(2),
        compressor_state_cache.stride(0),
        compressor_state_cache.stride(1),
        compressor_state_cache.stride(3),
        rope_cache.stride(0),
        rope_cache.stride(1),
        rope_cache.stride(3),
        compressor_state_block_table.stride(0),
        pooled.stride(0),
        pooled.stride(2),
        first_positions.stride(0),
        first_positions.stride(1),
        rows,
        compressor_state_cache.shape[0],
        num_requests,
        COMPRESSOR_STATE_SIZE=compressor_state_cache.shape[1],
        COMPRESS_RATIO=compress_ratio,
        HEAD_DIM=raw_keys.shape[2],
        LOAD_ROPE_POSITIONS=load_rope_positions,
        BLOCK_D=triton.next_power_of_2(raw_keys.shape[2]),
        num_warps=4,
    )
    return pooled, first_positions


__all__ = [
    "expand_qsa_block_indices_cuda",
    "qsa_compress_groups_with_ratio",
    "qsa_mqa_paged",
    "qsa_select_paged_tokens",
    "qsa_sparse_paged_attention",
    "qsa_store_cache_rows",
]
