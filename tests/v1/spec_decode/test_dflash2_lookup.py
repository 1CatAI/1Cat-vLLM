# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.triton_utils import triton
from vllm.v1.worker.gpu.spec_decode.dflash2.lookup import (
    _point_mass_draft_logits_kernel,
    fuse_draft,
    suffix_lookup,
)
from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import (
    _prepare_lookup_controller_flags_kernel,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for LABD kernels"
)


def _reference_lookup(
    tokens: list[int], k: int, nmin: int, nmax: int
) -> tuple[list[int], int, int]:
    suffix_end = len(tokens) - 1
    best = (-1, -1)
    for candidate_end in range(nmin - 1, suffix_end):
        match_len = 0
        while (
            match_len < nmax
            and candidate_end - match_len >= 0
            and suffix_end - match_len >= 0
            and tokens[candidate_end - match_len] == tokens[suffix_end - match_len]
        ):
            match_len += 1
        if match_len >= nmin and (match_len, candidate_end) > best:
            best = (match_len, candidate_end)

    if best[0] < 0:
        return [0] * k, 0, 0
    match_len, candidate_end = best
    valid = min(k, suffix_end - candidate_end)
    continuation = tokens[candidate_end + 1 : candidate_end + 1 + valid]
    return continuation + [0] * (k - valid), match_len, valid


@pytest.mark.parametrize("k", [7, 15])
def test_suffix_lookup_matches_reference_and_supports_overlap(k: int) -> None:
    generator = torch.Generator().manual_seed(7)
    random_tokens = torch.randint(1, 8, (1500,), generator=generator).tolist()
    cases = [
        random_tokens,
        [5, 6, 7, 8] * 100,
        list(range(1, 300)),
        random_tokens[:500] + random_tokens[80:120],
    ]
    device = torch.device("cuda")

    for sequence in cases:
        all_ids = torch.zeros((3, len(sequence) + 8), dtype=torch.int32, device=device)
        all_ids[2, : len(sequence)] = torch.tensor(
            sequence, dtype=torch.int32, device=device
        )
        total_len = torch.zeros(3, dtype=torch.int32, device=device)
        total_len[2] = len(sequence)
        idx_mapping = torch.full((k,), 2, dtype=torch.int32, device=device)
        eligible = torch.ones(1, dtype=torch.int32, device=device)

        actual_tokens, actual_len, actual_valid = suffix_lookup(
            all_ids,
            total_len,
            idx_mapping,
            eligible,
            1,
            k,
            idx_mapping_stride=k,
            nmin=4,
            nmax=12,
        )
        expected_tokens, expected_len, expected_valid = _reference_lookup(
            sequence, k, 4, 12
        )
        valid = expected_valid
        assert int(actual_len[0]) == expected_len
        assert int(actual_valid[0]) == expected_valid
        assert actual_tokens[0, :valid].tolist() == expected_tokens[:valid]


def test_suffix_lookup_honors_per_request_eligibility() -> None:
    device = torch.device("cuda")
    all_ids = torch.tensor([[1, 2, 3, 4, 1, 2, 3, 0]], dtype=torch.int32, device=device)
    total_len = torch.tensor([7], dtype=torch.int32, device=device)
    idx_mapping = torch.zeros(7, dtype=torch.int32, device=device)
    eligible = torch.zeros(1, dtype=torch.int32, device=device)

    tokens, match_len, valid = suffix_lookup(
        all_ids,
        total_len,
        idx_mapping,
        eligible,
        1,
        7,
        idx_mapping_stride=7,
        nmin=2,
        nmax=4,
    )

    assert torch.count_nonzero(tokens) == 0
    assert int(match_len[0]) == 0
    assert int(valid[0]) == 0


def test_fuse_decouples_seven_drafts_from_fifteen_verify_slots() -> None:
    device = torch.device("cuda")
    k, draft_block = 15, 7
    draft = torch.arange(100, 100 + k, dtype=torch.int64, device=device).view(1, k)
    lookup = torch.arange(200, 200 + k, dtype=torch.int32, device=device).view(1, k)
    use = torch.zeros((1, k), dtype=torch.int32, device=device)
    hits = torch.zeros((), dtype=torch.int64, device=device)
    take_flags = torch.zeros(1, dtype=torch.int32, device=device)
    idx_mapping = torch.zeros(draft_block, dtype=torch.int32, device=device)

    fuse_draft(
        draft,
        lookup,
        torch.tensor([12], dtype=torch.int32, device=device),
        torch.tensor([k], dtype=torch.int32, device=device),
        use,
        idx_mapping,
        hits,
        1,
        k,
        draft_block=draft_block,
        idx_mapping_stride=draft_block,
        nmin=4,
        nstrong=6,
        agree_min=0,
        nmin_tail=4,
        long_min=6,
        take_flags=take_flags,
    )

    assert torch.equal(draft, lookup.to(torch.int64))
    assert torch.all(use == 1)
    assert int(take_flags[0]) == 1
    assert int(hits) == 1


def test_lookup_controller_flag_requires_lookup_and_full_q8_emission() -> None:
    take = torch.tensor([1, 1, 0], dtype=torch.int32, device="cuda")
    emitted = torch.tensor([8, 7, 16], dtype=torch.int32, device="cuda")
    output = torch.full((3,), -1, dtype=torch.int32, device="cuda")

    _prepare_lookup_controller_flags_kernel[(1,)](
        take,
        emitted,
        output,
        8,
        3,
        BLOCK=4,
        num_warps=1,
    )

    assert output.tolist() == [1, 0, 0]


@pytest.mark.parametrize("probabilistic", [False, True])
@pytest.mark.parametrize("match_length", [4, 6])
def test_agreement_prefix_retains_its_proposal_distribution(
    probabilistic: bool, match_length: int
) -> None:
    """Only positions after a random lookup decision may become point masses."""
    k, draft_block, agree_min = 15, 7, 3
    lookup = torch.arange(k, device="cuda", dtype=torch.int64).view(1, k)
    draft = lookup.clone()
    draft[:, agree_min:] += 100
    use = torch.zeros_like(draft, dtype=torch.int32)
    hits = torch.zeros((), device="cuda", dtype=torch.int64)
    fuse_draft(
        draft,
        lookup,
        torch.tensor([match_length], device="cuda", dtype=torch.int32),
        torch.tensor([k], device="cuda", dtype=torch.int32),
        use,
        torch.zeros(1, device="cuda", dtype=torch.int32),
        hits,
        1,
        k,
        draft_block=draft_block,
        nmin=4,
        nstrong=6,
        agree_min=agree_min,
        probabilistic=probabilistic,
    )
    # The repair changes q bookkeeping, not the proposed token sequence.
    assert torch.equal(draft, lookup)
    assert torch.all(use[:, agree_min:] == 1)
    expected_prefix_use = int(not probabilistic or match_length >= 6)
    assert torch.all(use[:, :agree_min] == expected_prefix_use)


@pytest.mark.parametrize("sparse", [False, True])
def test_agreement_lookup_rejection_preserves_target_distribution(sparse: bool) -> None:
    """The old point-mass rewrite turned p(A)=0.8 into p(A)=0.7."""
    from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
        dflash2_sparse_topk_rejection_sample,
        rejection_sample,
    )

    num_reqs, steps, vocab_size = 100_000, 2, 32
    idx = torch.arange(num_reqs, device="cuda", dtype=torch.int32)
    seeds = idx.to(torch.int64)
    temperature = torch.ones(num_reqs, device="cuda")
    positions = torch.arange(steps + 1, device="cuda").repeat(num_reqs) + 100
    dense_q = torch.full((num_reqs, steps, vocab_size), -float("inf"), device="cuda")
    dense_q[:, :, :2] = 0.0
    draft = torch.zeros((num_reqs, steps), device="cuda", dtype=torch.int64)
    draft[:, 0] = gumbel_sample(
        dense_q[:, 0],
        idx,
        temperature,
        seeds,
        positions.view(num_reqs, -1)[:, 0].contiguous(),
        apply_temperature=False,
        is_drafting=True,
    )
    cached_ids = torch.arange(2, device="cuda").repeat(num_reqs, steps, 1)
    cached_scores = torch.zeros_like(cached_ids, dtype=torch.float32)
    use = torch.zeros_like(draft, dtype=torch.int32)
    fuse_draft(
        draft,
        torch.zeros_like(draft),
        torch.full_like(idx, 4),
        torch.full_like(idx, steps),
        use,
        idx,
        torch.zeros((), device="cuda", dtype=torch.int64),
        num_reqs,
        steps,
        draft_block=1,
        nmin=4,
        nstrong=6,
        agree_min=1,
        probabilistic=True,
    )
    _point_mass_draft_logits_kernel[(num_reqs * steps,)](
        dense_q,
        cached_ids,
        cached_scores,
        draft,
        draft.stride(0),
        use,
        idx,
        1,
        cached_ids.stride(0),
        cached_ids.stride(1),
        dense_q.stride(0),
        dense_q.stride(1),
        num_steps=steps,
        top_k=2,
        BLOCK_K=2,
        CACHE_SCORES=True,
        num_warps=1,
    )
    sampled_input = torch.zeros((num_reqs, steps + 1), device="cuda", dtype=torch.int64)
    sampled_input[:, 1:] = draft
    cu = torch.arange(num_reqs + 1, device="cuda", dtype=torch.int32) * (steps + 1)
    target_values = torch.tensor([0.8, 0.2], device="cuda").log()
    target_values = target_values.repeat(num_reqs * (steps + 1), 1)
    target_ids = torch.arange(2, device="cuda").expand_as(target_values).contiguous()
    if sparse:
        sampled, _ = dflash2_sparse_topk_rejection_sample(
            target_ids,
            target_values,
            cached_ids,
            cached_scores,
            sampled_input.flatten(),
            cu,
            positions,
            idx,
            temperature,
            torch.ones_like(temperature),
            seeds,
            steps,
        )
    else:
        target_dense = torch.full(
            (num_reqs * (steps + 1), vocab_size), -float("inf"), device="cuda"
        )
        target_dense[:, :2] = target_values
        sampled, _ = rejection_sample(
            target_dense,
            dense_q,
            sampled_input.flatten(),
            cu,
            positions,
            idx,
            idx.repeat_interleave(steps + 1),
            torch.arange(steps + 1, device="cuda", dtype=torch.int32).repeat(num_reqs),
            temperature,
            seeds,
            steps,
        )
    observed = (sampled[:, 0] == 0).float().mean().item()
    # Eight standard deviations; the old approximately 0.10 bias is far larger.
    tolerance = 8 * (0.8 * 0.2 / num_reqs) ** 0.5
    assert abs(observed - 0.8) < tolerance, observed


def test_point_mass_rewrite_preserves_sparse_cache_invariant() -> None:
    device = torch.device("cuda")
    num_steps, top_k, vocab_size = 15, 16, 64
    draft_logits = torch.full(
        (2, num_steps, vocab_size),
        -float("inf"),
        dtype=torch.float32,
        device=device,
    )
    old_ids = (
        torch.arange(num_steps * top_k, dtype=torch.int64, device=device)
        .view(num_steps, top_k)
        .remainder(32)
    )
    cached_ids = old_ids.repeat(2, 1, 1)
    cached_scores = torch.zeros(
        (2, num_steps, top_k), dtype=torch.float32, device=device
    )
    draft_logits[1].scatter_(1, old_ids, cached_scores[1])
    proposed = torch.arange(40, 40 + num_steps, dtype=torch.int64, device=device)
    draft_tokens = proposed.view(1, num_steps)
    use = torch.ones((1, num_steps), dtype=torch.int32, device=device)
    idx_mapping = torch.ones(7, dtype=torch.int32, device=device)

    _point_mass_draft_logits_kernel[(num_steps,)](
        draft_logits,
        cached_ids,
        cached_scores,
        draft_tokens,
        draft_tokens.stride(0),
        use,
        idx_mapping,
        7,
        cached_ids.stride(0),
        cached_ids.stride(1),
        draft_logits.stride(0),
        draft_logits.stride(1),
        num_steps=num_steps,
        top_k=top_k,
        BLOCK_K=triton.next_power_of_2(top_k),
        CACHE_SCORES=True,
        num_warps=1,
    )

    for step, token in enumerate(proposed.tolist()):
        finite_ids = torch.where(torch.isfinite(draft_logits[1, step]))[0].tolist()
        assert finite_ids == [token]
        assert float(draft_logits[1, step, token]) == 0.0
        assert int(cached_ids[1, step, 0]) == token
        assert float(cached_scores[1, step, 0]) == 0.0
        assert torch.isneginf(cached_scores[1, step, 1:]).all()
