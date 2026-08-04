# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.triton_utils import tl, triton


def save_partial_states(
    kv: torch.Tensor,
    score: torch.Tensor,
    ape: torch.Tensor,
    positions: torch.Tensor,
    state_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
    state_width: int,
    compress_ratio: int,
    pdl_kwargs: dict | None = None,
) -> None:
    """Write packed [kv, score+ape] partial states into the compressor cache.

    One program per token; pads (slot_id == -1) are skipped.
    """
    num_actual = slot_mapping.shape[0]
    head_size = kv.shape[-1]
    _save_partial_states_kernel[(num_actual,)](
        kv,
        kv.stride(0),
        score,
        score.stride(0),
        ape,
        ape.stride(0),
        positions,
        state_cache,
        state_cache.stride(0),
        state_cache.stride(1),
        slot_mapping,
        block_size,
        HEAD_SIZE=head_size,
        TRITON_BLOCK_SIZE=triton.next_power_of_2(head_size),
        STATE_WIDTH=state_width,
        COMPRESS_RATIO=compress_ratio,
        **(pdl_kwargs or {}),
    )


def save_partial_states_to_ring(
    kv: torch.Tensor,
    score: torch.Tensor,
    ape: torch.Tensor,
    positions: torch.Tensor,
    state_ring: torch.Tensor,
    slot_mapping: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    state_width: int,
    compress_ratio: int,
    pdl_kwargs: dict | None = None,
) -> None:
    """Keep only the compressor history that can cross a forward boundary."""
    num_slots = slot_mapping.shape[0]
    head_size = kv.shape[-1]
    _save_partial_states_to_ring_kernel[(num_slots,)](
        kv,
        kv.stride(0),
        score,
        score.stride(0),
        ape,
        ape.stride(0),
        positions,
        state_ring,
        state_ring.stride(0),
        state_ring.stride(1),
        slot_mapping,
        token_to_req_indices,
        seq_lens,
        HEAD_SIZE=head_size,
        TRITON_BLOCK_SIZE=triton.next_power_of_2(head_size),
        STATE_WIDTH=state_width,
        RING_SIZE=state_ring.shape[1],
        COMPRESS_RATIO=compress_ratio,
        **(pdl_kwargs or {}),
    )


def stage_partial_states_from_ring(
    kv: torch.Tensor,
    score: torch.Tensor,
    ape: torch.Tensor,
    positions: torch.Tensor,
    state_ring: torch.Tensor,
    dense_state: torch.Tensor,
    slot_mapping: torch.Tensor,
    state_width: int,
    history_size: int,
    compress_ratio: int,
    pdl_kwargs: dict | None = None,
) -> None:
    """Stage private history and the current chunk into one dense state view."""
    num_slots = slot_mapping.shape[0]
    copy_block_size = triton.next_power_of_2(2 * state_width)
    _copy_ring_history_to_dense_kernel[(history_size,)](
        positions,
        state_ring,
        state_ring.stride(1),
        dense_state,
        dense_state.stride(0),
        STATE_DIM=2 * state_width,
        HISTORY_SIZE=history_size,
        RING_SIZE=state_ring.shape[1],
        BLOCK_SIZE=copy_block_size,
        **(pdl_kwargs or {}),
    )
    _save_current_states_to_dense_kernel[(num_slots,)](
        kv,
        kv.stride(0),
        score,
        score.stride(0),
        ape,
        ape.stride(0),
        positions,
        dense_state,
        dense_state.stride(0),
        slot_mapping,
        HEAD_SIZE=kv.shape[-1],
        BLOCK_SIZE=triton.next_power_of_2(kv.shape[-1]),
        STATE_WIDTH=state_width,
        HISTORY_SIZE=history_size,
        COMPRESS_RATIO=compress_ratio,
        **(pdl_kwargs or {}),
    )


@triton.jit
def _save_partial_states_kernel(
    kv_ptr,
    kv_stride,
    score_ptr,
    score_stride,
    ape_ptr,
    ape_stride,
    positions_ptr,
    state_cache_ptr,
    state_cache_stride0,
    state_cache_stride1,
    slot_mapping_ptr,
    block_size,
    HEAD_SIZE: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
    # state_cache last dim packs [kv_state, score_state], each STATE_WIDTH wide.
    STATE_WIDTH: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
):
    token_idx = tl.program_id(0)
    slot_id = tl.load(slot_mapping_ptr + token_idx)

    # Skip padded / invalid tokens (slot_id == -1 is the PAD sentinel used
    # by vLLM).  During CUDA graph replay the batch may contain padding
    # tokens whose slot_mapping is -1; writing to kv_state[-1] would be an
    # illegal memory access.
    if slot_id < 0:
        return

    block_idx = slot_id // block_size
    pos_in_block = slot_id % block_size
    base_ptr = (
        state_cache_ptr
        + block_idx * state_cache_stride0
        + pos_in_block * state_cache_stride1
    )

    block = tl.arange(0, TRITON_BLOCK_SIZE)
    mask = block < HEAD_SIZE

    kv = tl.load(kv_ptr + token_idx * kv_stride + block, mask=mask)
    tl.store(base_ptr + block, kv, mask=mask)

    # Fused: score += ape[position % compress_ratio]
    position = tl.load(positions_ptr + token_idx)
    ape_row = position % COMPRESS_RATIO
    ape = tl.load(ape_ptr + ape_row * ape_stride + block, mask=mask)
    score = tl.load(score_ptr + token_idx * score_stride + block, mask=mask)
    tl.store(
        base_ptr + STATE_WIDTH + block,
        score + ape,
        mask=mask,
    )


@triton.jit
def _save_partial_states_to_ring_kernel(
    kv_ptr,
    kv_stride,
    score_ptr,
    score_stride,
    ape_ptr,
    ape_stride,
    positions_ptr,
    state_ring_ptr,
    state_ring_stride0,
    state_ring_stride1,
    slot_mapping_ptr,
    token_to_req_indices_ptr,
    seq_lens_ptr,
    HEAD_SIZE: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
    STATE_WIDTH: tl.constexpr,
    RING_SIZE: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
):
    token_idx = tl.program_id(0)
    if tl.load(slot_mapping_ptr + token_idx) < 0:
        return

    req_idx = tl.load(token_to_req_indices_ptr + token_idx)
    position = tl.load(positions_ptr + token_idx)
    seq_len = tl.load(seq_lens_ptr + req_idx)
    if position < seq_len - RING_SIZE:
        return

    ring_row = position % RING_SIZE
    base_ptr = (
        state_ring_ptr
        + req_idx.to(tl.int64) * state_ring_stride0
        + ring_row.to(tl.int64) * state_ring_stride1
    )
    block = tl.arange(0, TRITON_BLOCK_SIZE)
    mask = block < HEAD_SIZE

    kv = tl.load(kv_ptr + token_idx * kv_stride + block, mask=mask).to(tl.float32)
    tl.store(base_ptr + block, kv, mask=mask)

    ape_row = position % COMPRESS_RATIO
    ape = tl.load(ape_ptr + ape_row * ape_stride + block, mask=mask).to(tl.float32)
    score = tl.load(score_ptr + token_idx * score_stride + block, mask=mask).to(
        tl.float32
    )
    tl.store(base_ptr + STATE_WIDTH + block, score + ape, mask=mask)


@triton.jit
def _copy_ring_history_to_dense_kernel(
    positions_ptr,
    state_ring_ptr,
    state_ring_stride1,
    dense_state_ptr,
    dense_state_stride0,
    STATE_DIM: tl.constexpr,
    HISTORY_SIZE: tl.constexpr,
    RING_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    history_idx = tl.program_id(0)
    current_start = tl.load(positions_ptr)
    position = current_start - HISTORY_SIZE + history_idx
    if position < 0:
        return

    block = tl.arange(0, BLOCK_SIZE)
    mask = block < STATE_DIM
    ring_row = position % RING_SIZE
    values = tl.load(
        state_ring_ptr + ring_row.to(tl.int64) * state_ring_stride1 + block,
        mask=mask,
    )
    tl.store(
        dense_state_ptr + history_idx * dense_state_stride0 + block,
        values,
        mask=mask,
    )


@triton.jit
def _save_current_states_to_dense_kernel(
    kv_ptr,
    kv_stride,
    score_ptr,
    score_stride,
    ape_ptr,
    ape_stride,
    positions_ptr,
    dense_state_ptr,
    dense_state_stride0,
    slot_mapping_ptr,
    HEAD_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    STATE_WIDTH: tl.constexpr,
    HISTORY_SIZE: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
):
    token_idx = tl.program_id(0)
    if tl.load(slot_mapping_ptr + token_idx) < 0:
        return

    block = tl.arange(0, BLOCK_SIZE)
    mask = block < HEAD_SIZE
    row_ptr = dense_state_ptr + (HISTORY_SIZE + token_idx) * dense_state_stride0
    kv = tl.load(kv_ptr + token_idx * kv_stride + block, mask=mask).to(tl.float32)
    tl.store(row_ptr + block, kv, mask=mask)

    position = tl.load(positions_ptr + token_idx)
    ape_row = position % COMPRESS_RATIO
    ape = tl.load(ape_ptr + ape_row * ape_stride + block, mask=mask).to(tl.float32)
    score = tl.load(score_ptr + token_idx * score_stride + block, mask=mask).to(
        tl.float32
    )
    tl.store(row_ptr + STATE_WIDTH + block, score + ape, mask=mask)
