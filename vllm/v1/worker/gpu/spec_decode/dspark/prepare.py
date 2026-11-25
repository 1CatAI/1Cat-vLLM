# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inputs of one Inkling2 DSpark step after a decode step, as ONE kernel that
runs INSIDE the captured draft graph.

`prepare_dflash_inputs` (dflash/speculator.py) is launched eagerly once per
draft KV group before every draft graph; on the model runner V2 a Triton
launch from Python costs 0.1-0.2 ms of host time, and at draft width 0 the
host is what bounds the step. This kernel does the same work for the fill-in
layout for all draft KV groups at once, reads only buffers with a fixed
address (so a graph can hold it), and also writes what lets the context K/V
and the recurrent head's feature row run in the same graph:

- context rows are padded to the fixed row count of the graph (padding and
  rejected rows get PAD slots: they write no K/V);
- `f_rnn_rows`: each request's last committed target row.

Per-step values that live in fresh tensors (batch row -> request state row,
sampled and rejected counts) are staged by the caller into `stage`
`(max_reqs, 3)`. Requests hold at most `SPAN` target tokens (a decode step:
1 + drafts). While `live[0] == 0` (graph capture and warm-up) every slot is
PAD and every sampling row inert, whatever the stale inputs say.
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.block_table import _load_ptr
from vllm.v1.worker.gpu.input_batch import InputBuffers

SPAN = 16
MAX_GROUPS = 4


@triton.jit
def _prepare_dspark_inputs_kernel(
    # Draft query batch
    out_input_ids_ptr,
    out_positions_ptr,
    out_query_start_loc_ptr,
    out_seq_lens_ptr,
    out_query_slots_ptr,
    query_slots_stride,
    # Context rows (the target tokens of this step)
    out_ctx_positions_ptr,
    out_ctx_slots_ptr,
    ctx_slots_stride,
    # Sampling rows
    out_sample_indices_ptr,
    out_sample_pos_ptr,
    out_sample_idx_mapping_ptr,
    out_f_rnn_rows_ptr,
    # Target batch
    target_positions_ptr,
    target_query_start_loc_ptr,
    stage_ptr,
    last_sampled_ptr,
    next_prefill_tokens_ptr,
    # Draft KV groups: block table pointers, strides, block sizes, rows of
    # the query slot buffer.
    table_ptrs,
    table_strides,
    block_sizes,
    query_rows,
    live_ptr,
    # Scalars
    mask_token_id,
    num_query_per_req,
    num_speculative_steps,
    max_num_reqs,
    max_num_tokens,
    max_model_len,
    ctx_rows,
    NUM_GROUPS: tl.constexpr,
    PAD: tl.constexpr,
    SPAN: tl.constexpr,
):
    req_idx = tl.program_id(0)
    num_reqs = tl.num_programs(0)
    live = tl.load(live_ptr) != 0
    req_state_idx = tl.load(stage_ptr + req_idx * 3)
    num_sampled = tl.load(stage_ptr + req_idx * 3 + 1)
    num_rejected = tl.load(stage_ptr + req_idx * 3 + 2)

    ctx_start = tl.load(target_query_start_loc_ptr + req_idx)
    ctx_end = tl.load(target_query_start_loc_ptr + req_idx + 1)
    num_ctx = ctx_end - ctx_start
    valid_ctx_end = ctx_end - num_rejected
    num_valid_ctx = valid_ctx_end - ctx_start

    if num_sampled > 0:
        bonus_token = tl.load(last_sampled_ptr + req_state_idx).to(tl.int32)
    else:
        # Chunked prefill: the next prompt token.
        bonus_token = tl.load(next_prefill_tokens_ptr + req_state_idx).to(tl.int32)
    last_row = tl.maximum(valid_ctx_end - 1, 0)
    last_valid_pos = tl.load(target_positions_ptr + last_row)

    j = tl.arange(0, SPAN)
    # --- Context rows: committed ones get their slot, rejected ones PAD.
    ctx_idx = ctx_start + j
    is_ctx = (j < num_ctx) & (ctx_idx < max_num_tokens)
    is_valid_ctx = is_ctx & (j < num_valid_ctx)
    ctx_pos = tl.load(target_positions_ptr + ctx_idx, mask=is_valid_ctx, other=0)
    tl.store(out_ctx_positions_ptr + ctx_idx, ctx_pos, mask=is_ctx)

    # --- Query block: the bonus token, then the masks.
    is_query = j < num_query_per_req
    query_pos = last_valid_pos + 1 + j
    query_base = req_idx * num_query_per_req
    query_idx = query_base + j
    tl.store(
        out_input_ids_ptr + query_idx,
        tl.where(j == 0, bonus_token, mask_token_id),
        mask=is_query,
    )
    tl.store(
        out_positions_ptr + query_idx,
        tl.minimum(query_pos, max_model_len - 1),
        mask=is_query,
    )

    for g in tl.static_range(NUM_GROUPS):
        table_stride = tl.load(table_strides + g)
        table = _load_ptr(table_ptrs + g, tl.int32) + req_idx * table_stride
        block_size = tl.load(block_sizes + g)
        # Block 0 is the null block (evicted sliding-window positions,
        # padding): never a writable slot.
        ctx_block = tl.load(
            table + tl.minimum(ctx_pos // block_size, table_stride - 1),
            mask=is_valid_ctx,
            other=0,
        ).to(tl.int64)
        tl.store(
            out_ctx_slots_ptr + g * ctx_slots_stride + ctx_idx,
            tl.where(
                live & is_valid_ctx & (ctx_block != 0),
                ctx_block * block_size + ctx_pos % block_size,
                PAD,
            ),
            mask=is_ctx,
        )
        query_block = tl.load(
            table + tl.minimum(query_pos // block_size, table_stride - 1),
            mask=is_query,
            other=0,
        ).to(tl.int64)
        query_row = tl.load(query_rows + g)
        tl.store(
            out_query_slots_ptr + query_row * query_slots_stride + query_idx,
            tl.where(
                live & (query_block != 0),
                query_block * block_size + query_pos % block_size,
                PAD,
            ),
            mask=is_query,
        )

    # --- Sampling rows: every mask predicts the token at its own position.
    is_sample = is_query & (j >= 1)
    sample_idx = req_idx * num_speculative_steps + j - 1
    tl.store(out_sample_indices_ptr + sample_idx, query_idx, mask=is_sample)
    tl.store(out_sample_pos_ptr + sample_idx, query_pos, mask=is_sample)
    tl.store(
        out_sample_idx_mapping_ptr + sample_idx,
        tl.where(live, req_state_idx, -1),
        mask=is_sample,
    )

    tl.store(out_query_start_loc_ptr + req_idx, query_base)
    tl.store(
        out_seq_lens_ptr + req_idx,
        tl.where(
            live, tl.minimum(last_valid_pos + 1 + num_query_per_req, max_model_len), 0
        ),
    )
    tl.store(out_f_rnn_rows_ptr + req_idx, last_row)

    if req_idx == num_reqs - 1:
        # Requests past the batch are inert in a graph captured for more.
        for i in range(num_reqs, max_num_reqs + 1, SPAN):
            block = i + j
            tl.store(
                out_query_start_loc_ptr + block,
                num_reqs * num_query_per_req,
                mask=block < max_num_reqs + 1,
            )
        for i in range(num_reqs, max_num_reqs, SPAN):
            block = i + j
            mask = block < max_num_reqs
            tl.store(out_seq_lens_ptr + block, 0, mask=mask)
            tl.store(out_f_rnn_rows_ptr + block, 0, mask=mask)
        pad_end = max_num_reqs * num_speculative_steps
        for i in range(num_reqs * num_speculative_steps, pad_end, SPAN):
            block = i + j
            mask = block < pad_end
            tl.store(out_sample_indices_ptr + block, 0, mask=mask)
            tl.store(out_sample_pos_ptr + block, 0, mask=mask)
            tl.store(out_sample_idx_mapping_ptr + block, -1, mask=mask)
        q_pad_end = max_num_reqs * num_query_per_req
        for i in range(num_reqs * num_query_per_req, q_pad_end, SPAN):
            block = i + j
            mask = block < q_pad_end
            for g in tl.static_range(NUM_GROUPS):
                query_row = tl.load(query_rows + g)
                tl.store(
                    out_query_slots_ptr + query_row * query_slots_stride + block,
                    PAD,
                    mask=mask,
                )
        # Context rows the graph reads past this step's target tokens.
        for i in range(tl.maximum(ctx_end, 0), ctx_rows, SPAN):
            block = i + j
            mask = block < ctx_rows
            tl.store(out_ctx_positions_ptr + block, 0, mask=mask)
            for g in tl.static_range(NUM_GROUPS):
                tl.store(
                    out_ctx_slots_ptr + g * ctx_slots_stride + block, PAD, mask=mask
                )


def prepare_dspark_inputs(
    num_reqs: int,
    input_buffers: InputBuffers,
    # [num_kv_groups, max_num_tokens]
    query_slot_mappings: torch.Tensor,
    context_positions: torch.Tensor,
    # [num_draft_groups, max_num_tokens]
    context_slot_mappings: torch.Tensor,
    sample_indices: torch.Tensor,
    sample_pos: torch.Tensor,
    sample_idx_mapping: torch.Tensor,
    f_rnn_rows: torch.Tensor,
    target_buffers: InputBuffers,
    # [max_num_reqs, 3] int32: request state row, sampled, rejected
    stage: torch.Tensor,
    last_sampled: torch.Tensor,
    next_prefill_tokens: torch.Tensor,
    # Per draft KV group (uint64 / int64 / int32 / int32, [num_draft_groups])
    table_ptrs: torch.Tensor,
    table_strides: torch.Tensor,
    block_sizes: torch.Tensor,
    query_rows: torch.Tensor,
    live: torch.Tensor,
    mask_token_id: int,
    num_query_per_req: int,
    num_speculative_steps: int,
    max_num_reqs: int,
    max_model_len: int,
    ctx_rows: int,
) -> None:
    """See the module docstring. `ctx_rows`: the context rows the graph
    processes (the step's target tokens must fit)."""
    num_groups = table_ptrs.shape[0]
    assert (
        1 <= num_groups <= MAX_GROUPS and context_slot_mappings.shape[0] >= num_groups
    )
    assert num_query_per_req <= SPAN and stage.shape[1] == 3 and stage.is_contiguous()
    assert stage.dtype == torch.int32 and table_ptrs.dtype == torch.uint64
    max_num_tokens = min(context_positions.shape[0], context_slot_mappings.shape[1])
    assert ctx_rows <= max_num_tokens
    _prepare_dspark_inputs_kernel[(num_reqs,)](
        input_buffers.input_ids,
        input_buffers.positions,
        input_buffers.query_start_loc,
        input_buffers.seq_lens,
        query_slot_mappings,
        query_slot_mappings.stride(0),
        context_positions,
        context_slot_mappings,
        context_slot_mappings.stride(0),
        sample_indices,
        sample_pos,
        sample_idx_mapping,
        f_rnn_rows,
        target_buffers.positions,
        target_buffers.query_start_loc,
        stage,
        last_sampled,
        next_prefill_tokens,
        table_ptrs,
        table_strides,
        block_sizes,
        query_rows,
        live,
        mask_token_id,
        num_query_per_req,
        num_speculative_steps,
        max_num_reqs,
        max_num_tokens,
        max_model_len,
        ctx_rows,
        NUM_GROUPS=num_groups,
        PAD=PAD_SLOT_ID,
        SPAN=SPAN,
        num_warps=1,
    )
