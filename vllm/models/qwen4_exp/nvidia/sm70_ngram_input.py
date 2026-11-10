# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact PLE context gathering and query-boundary staging in one launch."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _prepare_ngram(
    context,
    query_out,
    query_in,
    mapping,
    computed,
    token_ids,
    num_reqs: tl.constexpr,
    padded_reqs: tl.constexpr,
    context_len: tl.constexpr,
    token_stride: tl.constexpr,
    eos: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, block)
    active = row < num_reqs
    request = tl.load(mapping + row, active, other=0).to(tl.int64)
    end = tl.load(computed + request, active, other=0).to(tl.int64)
    position = end - context_len + col
    valid = active & (col < context_len) & (position >= 0)
    value = tl.load(token_ids + request * token_stride + position, valid, other=eos)
    tl.store(
        context + row * context_len + col,
        value,
        (row < padded_reqs) & (col < context_len),
    )
    query = tl.load(query_in + row)
    tl.store(query_out + row, query)


def prepare_ngram_input(
    context: torch.Tensor,
    query_out: torch.Tensor,
    query_in: torch.Tensor,
    mapping: torch.Tensor,
    computed: torch.Tensor,
    token_ids: torch.Tensor,
    num_reqs: int,
    eos: int,
) -> None:
    _prepare_ngram[(context.shape[0] + 1,)](
        context,
        query_out,
        query_in,
        mapping,
        computed,
        token_ids,
        num_reqs,
        context.shape[0],
        context.shape[1],
        token_ids.stride(0),
        eos,
        triton.next_power_of_2(context.shape[1]),
        num_warps=1,
    )
