# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V4.1 TP-local greedy head and exact single-request verification."""

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.sample.ops.fast_argmax import _argmax_parts


@triton.jit
def _local_pair(
    values, indices, output, parts, START: tl.constexpr, PARTS: tl.constexpr
):
    row = tl.program_id(0)
    part = tl.arange(0, PARTS)
    value = tl.load(values + row * PARTS + part, part < parts, float("-inf"))
    index = tl.load(indices + row * PARTS + part, part < parts, 16777215)
    maximum = tl.max(value, 0)
    winner = tl.min(tl.where(value == maximum, index, 16777215), 0)
    global_idx = tl.where(winner < 16777215, winner + START, 16777215)
    column = tl.arange(0, 4)
    pair = tl.where(column == 0, maximum, tl.where(column == 1, global_idx, 0))
    tl.store(output + row * 4 + column, pair)


@triton.jit
def _global_top1(pairs, output, RANKS: tl.constexpr):
    row = tl.program_id(0)
    rank = tl.arange(0, RANKS)
    value = tl.load(pairs + row * RANKS * 4 + rank * 4)
    index = tl.load(pairs + row * RANKS * 4 + rank * 4 + 1).to(tl.int64)
    maximum = tl.max(value, 0)
    winner = tl.min(tl.where(value == maximum, index, 16777215), 0)
    tl.store(output + row, winner)


@triton.jit
def _verify(
    top_tokens,
    target_indices,
    bonus_indices,
    drafts,
    cu_drafts,
    output,
    VOCAB: tl.constexpr,
    STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    req = tl.program_id(0)
    start = tl.where(req == 0, 0, tl.load(cu_drafts + req - 1, req > 0, 0))
    end = tl.load(cu_drafts + req)
    num_drafts = end - start
    position = tl.arange(0, BLOCK)
    in_drafts = position < num_drafts
    row = tl.load(target_indices + start + position, in_drafts, 0)
    target = tl.load(top_tokens + row, in_drafts, 0).to(tl.int32)
    valid_t = (target >= 0) & (target < VOCAB)
    proposed = tl.load(drafts + start + position, in_drafts, -1)
    first_rejected = tl.min(
        tl.where(in_drafts & (~valid_t | (target != proposed)), position, num_drafts),
        0,
    )
    bonus_row = tl.load(bonus_indices + req)
    bonus = tl.load(top_tokens + bonus_row).to(tl.int32)
    token = tl.where(position < num_drafts, target, bonus)
    token = tl.where((token >= 0) & (token < VOCAB), token, -1)
    token = tl.where(position <= first_rejected, token, -1)
    tl.store(output + req * STRIDE + position, token, position < STRIDE)


def verify_top_tokens(top_tokens, metadata, vocab_size: int = 129280):
    """Match the ordinary greedy rejection sampler, including its -1 padding."""
    num_reqs = len(metadata.num_draft_tokens)
    stride = metadata.max_spec_len + 1
    output = torch.empty(
        (num_reqs, stride), dtype=torch.int32, device=top_tokens.device
    )
    _verify[(num_reqs,)](
        top_tokens,
        metadata.target_logits_indices,
        metadata.bonus_logits_indices,
        metadata.draft_token_ids,
        metadata.cu_num_draft_tokens,
        output,
        VOCAB=int(vocab_size),
        STRIDE=stride,
        BLOCK=triton.next_power_of_2(stride),
        num_warps=1,
    )
    return output


def greedy_top_tokens(head, hidden):
    """Compute exact TP4 top-1 candidates with a small pair exchange."""
    hidden = hidden.contiguous()
    from .native import tp_all_gather

    logits = head.quant_method.apply(head, hidden, bias=None)
    rows, columns = logits.shape
    valid_columns = columns - head.shard_indices.num_org_vocab_padding
    parts = triton.cdiv(valid_columns, 4096)
    width = triton.next_power_of_2(parts)
    values = torch.empty((rows, width), dtype=torch.float32, device=hidden.device)
    indices = torch.empty((rows, width), dtype=torch.int32, device=hidden.device)
    _argmax_parts[(rows, parts)](
        logits,
        logits.stride(0),
        valid_columns,
        values,
        indices,
        PARTS=width,
        CHUNK=4096,
        num_warps=8,
    )
    pairs = torch.empty((rows, 4), dtype=torch.float32, device=hidden.device)
    _local_pair[(rows,)](
        values,
        indices,
        pairs,
        parts,
        START=head.shard_indices.org_vocab_start_index,
        PARTS=width,
        num_warps=1,
    )
    gathered = tp_all_gather(pairs)
    output = torch.empty(rows, dtype=torch.int64, device=hidden.device)
    _global_top1[(rows,)](gathered, output, RANKS=4, num_warps=1)
    return output


def can_verify_top_tokens(runner, scheduler_output, spec_decode_metadata):
    """Keep the native DSpark eligibility contract with its implementation."""
    config = runner.speculative_config
    if not (
        config is not None
        and config.method == "dspark"
        and not config.use_dflash_ddtree()
        and spec_decode_metadata is not None
        and len(spec_decode_metadata.num_draft_tokens) == runner.input_batch.num_reqs
        and spec_decode_metadata.max_spec_len <= 5
        and runner.input_batch.num_reqs >= 1
        and runner.parallel_config.tensor_parallel_size == 4
        and not runner.broadcast_pp_output
        and not runner.is_pooling_model
        and not scheduler_output.has_structured_output_requests
        and not runner.rejection_sampler.synthetic_mode
        and runner.rejection_sampler._sampling_policy.dynamic_tail_size == 0
        and not any(
            runner.rejection_sampler._trace.value(flag)
            for flag in (
                "alignment",
                "corruption",
                "draft_logits",
                "rejection_profile",
            )
        )
        and hasattr(runner.model, "get_v41_top_tokens")
    ):
        return False
    return runner._ddtree_greedy_sampling_supported(
        runner.input_batch.sampling_metadata
    )
