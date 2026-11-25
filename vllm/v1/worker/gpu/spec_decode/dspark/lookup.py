# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prompt-lookup drafts for Inkling2 DSpark (exact, device only).

A request that copies a span of its own context (a file it was asked to edit,
a tool result it quotes, a repeated list) gets the continuation of the longest
earlier occurrence of its last tokens as draft instead of the drafter's
sample. The verifier decides as always, so the output distribution is the
target's either way; only how many drafted tokens it accepts changes.

Whole vector, decided from the context alone: if the suffix of the committed
tokens (the bonus token included) matches an earlier span of `nuse` or more
tokens, ALL `width` drafts of the request come from the lookup (positions after
the span's end are filler tokens the verifier rejects); otherwise all stay the
drafter's. A mixed vector would pair the drafter's samples, drawn after its
own prefix, with a prefix it did not draw; and a decision that looked at the
drafter's samples would bias the accept test. Sampling requests (T > 0) verify
the lookup's tokens against a point-mass proposal (logit 0 at the token, -inf
elsewhere in this rank's shard of `dense_local`: probability 1), which is a
valid proposal like any other; greedy ones just compare tokens.

The search is `dflash2/lookup.py::suffix_lookup` (longest match, then the most
recent).
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.spec_decode.dflash2.lookup import suffix_lookup


@triton.jit
def _choose_kernel(
    draft_tokens_ptr,
    draft_stride,
    lookup_tokens_ptr,
    lookup_stride,
    match_len_ptr,
    valid_ptr,
    use_ptr,
    hits_ptr,
    nuse,
    width,
    BLOCK_K: tl.constexpr,
    FILLER: tl.constexpr,
):
    req = tl.program_id(0)
    match_len = tl.load(match_len_ptr + req)
    valid = tl.load(valid_ptr + req)
    take = (match_len >= nuse) & (valid >= 1)
    tl.store(use_ptr + req, take.to(tl.int32))
    if take:
        offsets = tl.arange(0, BLOCK_K)
        looked = tl.load(
            lookup_tokens_ptr + req * lookup_stride + offsets,
            mask=offsets < valid,
            other=FILLER,
        )
        tl.store(
            draft_tokens_ptr + req * draft_stride + offsets,
            looked.to(draft_tokens_ptr.dtype.element_ty),
            mask=offsets < width,
        )
        tl.atomic_add(hits_ptr, 1)


@triton.jit
def _point_mass_kernel(
    dense_ptr,
    dense_stride,
    draft_tokens_ptr,
    draft_stride,
    use_ptr,
    stage_ptr,
    stage_stride,
    num_spec,
    width,
    shard,
    id_base,
    BLOCK: tl.constexpr,
):
    flat = tl.program_id(0)
    chunk = tl.program_id(1)
    req = flat // width
    step = flat % width
    if tl.load(use_ptr + req) == 0:
        return
    state = tl.load(stage_ptr + req * stage_stride).to(tl.int64)
    token = tl.load(draft_tokens_ptr + req * draft_stride + step).to(tl.int64)
    cols = chunk * BLOCK + tl.arange(0, BLOCK)
    row = dense_ptr + (state * num_spec + step) * dense_stride
    mine = (cols == (token - id_base)) & (cols < shard)
    tl.store(row + cols, tl.where(mine, 0.0, float("-inf")), mask=cols < shard)


class DSparkLookup:
    """Lookup state and launches for one speculator (all device tensors are
    allocated once; the kernels are launched eagerly after the draft graph)."""

    def __init__(
        self,
        max_reqs: int,
        num_spec: int,
        device: torch.device,
        nmin: int,
        nmax: int,
        nuse: int,
        search_max: int,
    ) -> None:
        self.num_spec = num_spec
        self.nmin, self.nmax, self.nuse, self.search_max = nmin, nmax, nuse, search_max
        self.eligible = torch.ones(max_reqs, dtype=torch.int32, device=device)
        self.tokens = torch.zeros(max_reqs, num_spec, dtype=torch.int32, device=device)
        self.match_len = torch.zeros(max_reqs, dtype=torch.int32, device=device)
        self.valid = torch.zeros(max_reqs, dtype=torch.int32, device=device)
        self.use = torch.zeros(max_reqs, dtype=torch.int32, device=device)
        self.hits = torch.zeros((), dtype=torch.int64, device=device)
        self.steps = 0

    def apply(
        self,
        all_token_ids: torch.Tensor,
        total_len: torch.Tensor,
        stage: torch.Tensor,
        draft_tokens: torch.Tensor,
        num_reqs: int,
        width: int,
        dense_local: torch.Tensor | None,
        id_base: int,
    ) -> None:
        """Replace the first `width` drafts of requests with a long enough match.
        `stage[:, 0]` is the request-state row of each batch row;
        `dense_local` (probabilistic drafts) gets point-mass rows for them."""
        suffix_lookup(
            all_token_ids,
            total_len,
            stage,
            self.eligible,
            num_reqs,
            self.num_spec,
            idx_mapping_stride=stage.stride(0),
            nmax=self.nmax,
            nmin=self.nmin,
            search_max=self.search_max,
            out_tokens=self.tokens,
            out_len=self.match_len,
            out_valid=self.valid,
        )
        _choose_kernel[(num_reqs,)](
            draft_tokens,
            draft_tokens.stride(0),
            self.tokens,
            self.tokens.stride(0),
            self.match_len,
            self.valid,
            self.use,
            self.hits,
            self.nuse,
            width,
            BLOCK_K=triton.next_power_of_2(self.num_spec),
            FILLER=0,
            num_warps=1,
        )
        if dense_local is not None:
            shard = dense_local.shape[1]
            block = 4096
            _point_mass_kernel[(num_reqs * width, triton.cdiv(shard, block))](
                dense_local,
                dense_local.stride(0),
                draft_tokens,
                draft_tokens.stride(0),
                self.use,
                stage,
                stage.stride(0),
                self.num_spec,
                width,
                shard,
                id_base,
                BLOCK=block,
                num_warps=4,
            )
        self.steps += 1
