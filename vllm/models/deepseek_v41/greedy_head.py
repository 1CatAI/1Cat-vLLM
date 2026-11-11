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
    index = tl.load(indices + row * PARTS + part, part < parts, 2147483647)
    maximum = tl.max(value, 0)
    winner = tl.min(tl.where(value == maximum, index, 2147483647), 0)
    column = tl.arange(0, 4)
    pair = tl.where(column == 0, maximum, tl.where(column == 1, winner + START, 0))
    tl.store(output + row * 4 + column, pair)


@triton.jit
def _global_top1(pairs, output, RANKS: tl.constexpr):
    row = tl.program_id(0)
    rank = tl.arange(0, RANKS)
    value = tl.load(pairs + row * RANKS * 4 + rank * 4)
    index = tl.load(pairs + row * RANKS * 4 + rank * 4 + 1).to(tl.int64)
    maximum = tl.max(value, 0)
    winner = tl.min(tl.where(value == maximum, index, 2147483647), 0)
    tl.store(output + row, winner)


@triton.jit
def _verify(
    top_tokens,
    target_indices,
    bonus_indices,
    drafts,
    output,
    DRAFTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    position = tl.arange(0, BLOCK)
    row = tl.load(target_indices + position, position < DRAFTS, 0)
    target = tl.load(top_tokens + row).to(tl.int32)
    proposed = tl.load(drafts + position, position < DRAFTS, -1)
    first_rejected = tl.min(
        tl.where((position < DRAFTS) & (target != proposed), position, DRAFTS), 0
    )
    bonus_row = tl.load(bonus_indices)
    bonus = tl.load(top_tokens + bonus_row).to(tl.int32)
    token = tl.where(position < DRAFTS, target, bonus)
    token = tl.where(position <= first_rejected, token, -1)
    tl.store(output + position, token, position <= DRAFTS)


def verify_top_tokens(top_tokens, metadata):
    """Match the ordinary greedy rejection sampler, including its -1 padding."""
    drafts = metadata.num_draft_tokens[0]
    output = torch.empty((1, drafts + 1), dtype=torch.int32, device=top_tokens.device)
    _verify[(1,)](
        top_tokens,
        metadata.target_logits_indices,
        metadata.bonus_logits_indices,
        metadata.draft_token_ids,
        output,
        DRAFTS=drafts,
        BLOCK=triton.next_power_of_2(drafts + 1),
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
        and len(spec_decode_metadata.num_draft_tokens) == 1
        and spec_decode_metadata.max_spec_len <= 5
        and runner.input_batch.num_reqs == 1
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
