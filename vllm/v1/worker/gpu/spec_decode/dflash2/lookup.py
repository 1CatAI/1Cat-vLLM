# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU lookup-augmented block drafting for DFlash2.

Adapted for MRV2/SM70 from syv-ai/qwen38-27b-rtx3090's Apache-2.0
``dflash2-lookup-drafting.patch`` (fixed source revision is documented in the
design note).
"""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton

_SCORE_STRIDE = tl.constexpr(1 << 32)


@triton.jit
def _suffix_lookup_kernel(
    all_token_ids_ptr,
    all_token_ids_stride,
    total_len_ptr,
    idx_mapping_ptr,
    idx_mapping_stride,
    eligible_ptr,
    out_tokens_ptr,
    out_tokens_stride,
    out_len_ptr,
    out_valid_ptr,
    search_max,
    k: tl.constexpr,
    NMAX: tl.constexpr,
    NMIN: tl.constexpr,
    BLOCK: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    req = tl.program_id(0)
    tl.store(out_len_ptr + req, 0)
    tl.store(out_valid_ptr + req, 0)
    if tl.load(eligible_ptr + req) == 0:
        return
    req_state = tl.load(idx_mapping_ptr + req * idx_mapping_stride)
    if req_state < 0:
        return

    total_len = tl.load(total_len_ptr + req_state)
    if total_len < NMIN + 2:
        return
    base = all_token_ids_ptr + req_state.to(tl.int64) * all_token_ids_stride
    suffix_end = total_len - 1
    search_end = suffix_end
    search_start = tl.maximum(NMIN - 1, total_len - search_max)
    best = tl.full([], -1, tl.int64)

    # Select the longest suffix match and then the most recent occurrence.
    # Candidate matches may overlap the current suffix, which preserves
    # periodic patterns such as indentation, list markers, and code fences.
    for start in range(search_start, search_end, BLOCK):
        candidate_end = start + tl.arange(0, BLOCK)
        alive = (candidate_end >= search_start) & (candidate_end < search_end)
        for offset in range(NMIN):
            suffix_token = tl.load(base + suffix_end - offset)
            candidate_token = tl.load(
                base + candidate_end - offset,
                mask=alive & ((candidate_end - offset) >= 0),
                other=-1,
            )
            alive &= candidate_token == suffix_token

        if tl.max(alive.to(tl.int32)) > 0:
            match_len = tl.where(alive, NMIN, 0)
            longer = alive
            for offset in range(NMIN, NMAX):
                suffix_token = tl.load(
                    base + suffix_end - offset,
                    mask=(suffix_end - offset) >= 0,
                    other=-2,
                )
                candidate_token = tl.load(
                    base + candidate_end - offset,
                    mask=longer & ((candidate_end - offset) >= 0),
                    other=-1,
                )
                longer &= candidate_token == suffix_token
                match_len = tl.where(longer, match_len + 1, match_len)
            score = tl.where(
                alive,
                match_len.to(tl.int64) * _SCORE_STRIDE
                + candidate_end.to(tl.int64),
                -1,
            )
            best = tl.maximum(best, tl.max(score, axis=0))

    if best < 0:
        return
    match_len = (best // _SCORE_STRIDE).to(tl.int32)
    candidate_end = (best % _SCORE_STRIDE).to(tl.int32)
    valid = tl.minimum(k, suffix_end - candidate_end)
    offsets = tl.arange(0, BLOCK_K)
    tokens = tl.load(
        base + candidate_end + 1 + offsets,
        mask=offsets < valid,
        other=0,
    )
    tl.store(
        out_tokens_ptr + req * out_tokens_stride + offsets,
        tokens,
        mask=offsets < k,
    )
    tl.store(out_len_ptr + req, match_len)
    tl.store(out_valid_ptr + req, valid)


@triton.jit
def _fuse_draft_kernel(
    draft_tokens_ptr,
    draft_stride,
    lookup_tokens_ptr,
    lookup_stride,
    match_len_ptr,
    valid_ptr,
    use_ptr,
    idx_mapping_ptr,
    idx_mapping_stride,
    hits_ptr,
    take_flags_ptr,
    nmin,
    nstrong,
    agree_min,
    nmin_tail,
    long_min,
    draft_block,
    k: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    req = tl.program_id(0)
    tl.store(take_flags_ptr + req, 0)
    if tl.load(idx_mapping_ptr + req * idx_mapping_stride) < 0:
        return

    offsets = tl.arange(0, BLOCK_K)
    mask = offsets < k
    match_len = tl.load(match_len_ptr + req)
    valid = tl.load(valid_ptr + req)
    drafted = tl.load(
        draft_tokens_ptr + req * draft_stride + offsets,
        mask=mask,
        other=0,
    )
    looked_up = tl.load(
        lookup_tokens_ptr + req * lookup_stride + offsets,
        mask=mask,
        other=0,
    ).to(tl.int64)

    disagreement = (
        (drafted != looked_up)
        & (offsets < valid)
        & (offsets < draft_block)
        & mask
    )
    agreement = tl.minimum(
        tl.min(tl.where(disagreement, offsets, draft_block)), valid
    )
    take_head = (match_len >= nstrong) | (
        (match_len >= nmin) & (agreement >= agree_min)
    )
    tail = offsets >= draft_block
    take_tail = (match_len >= nmin_tail) & (
        take_head | (agreement >= draft_block)
    )
    from_lookup = (
        tl.where(tail, take_tail, take_head) & (offsets < valid) & mask
    )
    tl.store(
        draft_tokens_ptr + req * draft_stride + offsets,
        looked_up,
        mask=from_lookup,
    )

    # Every tail row needs an explicit point-mass proposal whenever a long
    # block is replayed, including sticky transition steps without a match.
    use = (from_lookup | tail) & mask
    tl.store(use_ptr + req * k + offsets, use.to(tl.int32), mask=mask)
    has_tail = (
        (match_len >= long_min) & (valid > draft_block) & take_tail
    )
    tl.store(take_flags_ptr + req, has_tail.to(tl.int32))
    tl.atomic_add(hits_ptr, take_head.to(tl.int64))


@triton.jit
def _point_mass_draft_logits_kernel(
    draft_logits_ptr,
    cached_candidate_ptr,
    cached_score_ptr,
    draft_tokens_ptr,
    draft_tokens_stride,
    use_ptr,
    idx_mapping_ptr,
    idx_mapping_stride,
    cache_stride_0,
    cache_stride_1,
    draft_logits_stride_0,
    draft_logits_stride_1,
    num_steps: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK_K: tl.constexpr,
    CACHE_SCORES: tl.constexpr,
):
    flat = tl.program_id(0)
    req = flat // num_steps
    step = flat % num_steps
    req_state = tl.load(idx_mapping_ptr + req * idx_mapping_stride)
    if req_state < 0 or tl.load(use_ptr + flat) == 0:
        return

    offsets = tl.arange(0, BLOCK_K)
    mask = offsets < top_k
    cache_base = req_state * cache_stride_0 + step * cache_stride_1
    old_ids = tl.load(cached_candidate_ptr + cache_base + offsets, mask=mask)
    logits_base = (
        draft_logits_ptr
        + req_state * draft_logits_stride_0
        + step * draft_logits_stride_1
    )
    tl.store(logits_base + old_ids, -float("inf"), mask=mask)

    token = tl.load(draft_tokens_ptr + req * draft_tokens_stride + step)
    is_proposal = offsets == 0
    new_ids = tl.where(is_proposal, token, 0)
    new_scores = tl.where(is_proposal, 0.0, -float("inf"))
    tl.store(cached_candidate_ptr + cache_base + offsets, new_ids, mask=mask)
    if CACHE_SCORES:
        tl.store(cached_score_ptr + cache_base + offsets, new_scores, mask=mask)
    tl.store(logits_base + token, 0.0)


def suffix_lookup(
    all_token_ids: torch.Tensor,
    total_len: torch.Tensor,
    idx_mapping: torch.Tensor,
    eligible: torch.Tensor,
    num_reqs: int,
    num_draft_tokens: int,
    *,
    idx_mapping_stride: int = 1,
    nmax: int = 12,
    nmin: int = 6,
    search_max: int = 1 << 30,
    out_tokens: torch.Tensor | None = None,
    out_len: torch.Tensor | None = None,
    out_valid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Find recent continuations of each request's current suffix."""
    if not 1 <= nmin <= nmax:
        raise ValueError("lookup ngram bounds must satisfy 1 <= nmin <= nmax")
    if num_draft_tokens <= 0:
        raise ValueError("num_draft_tokens must be positive")
    device = idx_mapping.device
    if out_tokens is None:
        out_tokens = torch.zeros(
            (num_reqs, num_draft_tokens), dtype=torch.int32, device=device
        )
    if out_len is None:
        out_len = torch.zeros(num_reqs, dtype=torch.int32, device=device)
    if out_valid is None:
        out_valid = torch.zeros(num_reqs, dtype=torch.int32, device=device)
    _suffix_lookup_kernel[(num_reqs,)](
        all_token_ids,
        all_token_ids.stride(0),
        total_len,
        idx_mapping,
        idx_mapping_stride,
        eligible,
        out_tokens,
        out_tokens.stride(0),
        out_len,
        out_valid,
        search_max,
        k=num_draft_tokens,
        NMAX=nmax,
        NMIN=nmin,
        BLOCK=1024,
        BLOCK_K=triton.next_power_of_2(num_draft_tokens),
        num_warps=4,
    )
    return out_tokens, out_len, out_valid


def fuse_draft(
    draft_tokens: torch.Tensor,
    lookup_tokens: torch.Tensor,
    match_len: torch.Tensor,
    valid: torch.Tensor,
    use: torch.Tensor,
    idx_mapping: torch.Tensor,
    hits: torch.Tensor,
    num_reqs: int,
    num_draft_tokens: int,
    *,
    draft_block: int,
    idx_mapping_stride: int = 1,
    nmin: int = 6,
    nstrong: int = 6,
    agree_min: int = 0,
    nmin_tail: int = 4,
    long_min: int = 6,
    take_flags: torch.Tensor | None = None,
) -> None:
    """Fuse lookup continuations into a model-drafted prefix in place."""
    if take_flags is None:
        take_flags = torch.zeros(
            num_reqs, dtype=torch.int32, device=draft_tokens.device
        )
    _fuse_draft_kernel[(num_reqs,)](
        draft_tokens,
        draft_tokens.stride(0),
        lookup_tokens,
        lookup_tokens.stride(0),
        match_len,
        valid,
        use,
        idx_mapping,
        idx_mapping_stride,
        hits,
        take_flags,
        nmin,
        nstrong,
        agree_min,
        nmin_tail,
        long_min,
        draft_block,
        k=num_draft_tokens,
        BLOCK_K=triton.next_power_of_2(num_draft_tokens),
        num_warps=1,
    )
