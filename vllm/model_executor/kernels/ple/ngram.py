# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact small-batch PLE n-gram IDs without packing or shifted tensors."""

from dataclasses import dataclass

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton


@dataclass(frozen=True)
class Sm70PLENgramCapability:
    operator: str = "sm70_ple_ngram_ids"
    graph_safe: bool = True
    min_m: int = 2
    max_m: int = 32

    def reason(self, ids, starts, context, multipliers, sizes, offsets):
        if not ids.is_cuda or not current_platform.is_device_capability(70):
            return "requires_sm70"
        m, requests = ids.numel(), starts.numel() - 1
        if not self.min_m <= m <= self.max_m or not 1 <= requests <= 32:
            return "outside_small_batch_band"
        if context.ndim != 2 or context.shape[0] < requests or context.shape[1] != 2:
            return "requires_two_history_tokens_per_request"
        if (multipliers.numel(), sizes.numel(), offsets.numel()) != (3, 16, 16):
            return "requires_trigram_eight_heads_per_order"
        for tensor in (ids, starts, context, multipliers, sizes, offsets):
            if tensor.device != ids.device or not tensor.is_contiguous():
                return "requires_contiguous_colocated_inputs"
            if tensor.dtype not in (torch.int32, torch.int64):
                return "requires_integer_inputs"
        return None


SM70_PLE_NGRAM = Sm70PLENgramCapability()


@triton.jit
def _ngram_ids(
    IDS,
    STARTS,
    CONTEXT,
    MULTIPLIERS,
    SIZES,
    OFFSETS,
    OUT,
    REQUESTS: tl.constexpr,
    BLOCK_R: tl.constexpr,
    EOS: tl.constexpr,
):
    token = tl.program_id(0)
    request_indices = tl.arange(0, BLOCK_R)
    starts = tl.load(
        STARTS + request_indices, request_indices <= REQUESTS, other=0x7FFFFFFF
    ).to(tl.int64)
    # Match searchsorted(right=True), including duplicate starts and graph
    # padding assigned to the final request by the existing GPU reference.
    request = tl.minimum(tl.sum((starts <= token).to(tl.int32)) - 1, REQUESTS - 1)
    begin = tl.load(STARTS + request).to(tl.int64)
    column = token - begin
    current = tl.load(IDS + token).to(tl.int64)
    previous = tl.load(IDS + token - 1, column >= 1, other=0).to(tl.int64)
    history1 = tl.load(CONTEXT + request * 2 + 1).to(tl.int64)
    previous = tl.where(column >= 1, previous, history1)
    older = tl.load(IDS + token - 2, column >= 2, other=0).to(tl.int64)
    history0 = tl.load(CONTEXT + request * 2).to(tl.int64)
    older = tl.where(column >= 2, older, tl.where(column == 1, history1, history0))
    older = tl.where(previous == EOS, EOS, older)
    a = tl.load(MULTIPLIERS).to(tl.int64)
    b = tl.load(MULTIPLIERS + 1).to(tl.int64)
    c = tl.load(MULTIPLIERS + 2).to(tl.int64)
    mixed2 = (current * a) ^ (previous * b)
    mixed3 = mixed2 ^ (older * c)
    head = tl.arange(0, 16)
    mixed = tl.where(head < 8, mixed2, mixed3)
    size = tl.load(SIZES + head).to(tl.int64)
    offset = tl.load(OFFSETS + head).to(tl.int64)
    remainder = mixed % size
    remainder = tl.where(remainder < 0, remainder + size, remainder)
    tl.store(OUT + token * 16 + head, remainder + offset)


def sm70_ple_ngram_ids(ids, starts, context, multipliers, sizes, offsets, eos):
    output = torch.empty((ids.numel(), 16), device=ids.device, dtype=torch.int64)
    requests = starts.numel() - 1
    _ngram_ids[(ids.numel(),)](
        ids,
        starts,
        context,
        multipliers,
        sizes,
        offsets,
        output,
        requests,
        triton.next_power_of_2(requests + 1),
        eos,
        num_warps=1,
    )
    return output
