# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Greedy speculative verification from target argmax ids.

Equivalent to ``rejection_sample`` when every request is greedy: the output
row holds the target argmax for each position up to and including the first
draft mismatch (or the bonus position), and ``num_sampled`` counts the leading
matches plus one.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _greedy_verify_kernel(
    sampled_ptr,
    sampled_stride,
    num_sampled_ptr,
    target_ptr,
    draft_ptr,
    cu_num_logits_ptr,
):
    req = tl.program_id(0)
    start = tl.load(cu_num_logits_ptr + req)
    end = tl.load(cu_num_logits_ptr + req + 1)
    accepted = True
    count = 0
    for i in range(end - start):
        target = tl.load(target_ptr + start + i)
        tl.store(sampled_ptr + req * sampled_stride + i, target)
        if i < end - start - 1:
            draft = tl.load(draft_ptr + start + i + 1)
            accepted &= target == draft
            count += accepted.to(tl.int32)
    tl.store(num_sampled_ptr + req, count + 1)


def greedy_verify(
    target_ids: torch.Tensor,
    draft_sampled: torch.Tensor,
    cu_num_logits: torch.Tensor,
    num_speculative_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_reqs = cu_num_logits.shape[0] - 1
    sampled = target_ids.new_zeros(
        (num_reqs, num_speculative_steps + 1), dtype=torch.int64
    )
    num_sampled = sampled.new_empty(num_reqs, dtype=torch.int32)
    _greedy_verify_kernel[(num_reqs,)](
        sampled,
        sampled.stride(0),
        num_sampled,
        target_ids.to(torch.int64),
        draft_sampled.to(torch.int64),
        cu_num_logits,
        num_warps=1,
    )
    return sampled, num_sampled
