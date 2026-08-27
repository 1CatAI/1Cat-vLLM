# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.config.speculative import SpeculativeConfig
from vllm.v1.spec_decode.ngram_proposer import (
    _find_longest_matched_ngram_and_propose_tokens,
)
from vllm.v1.worker.gpu.spec_decode.dflash2.ngram_assist import (
    DFlash2NgramAssist,
    find_split_ngram_proposal,
)
from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import (
    DFlash2Speculator,
    _advance_lookup_controller,
    _apply_ngram_draft_kernel,
)
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
    dflash2_sparse_topk_rejection_sample,
    rejection_sample,
)


@pytest.mark.parametrize("split", [0, 1, 4, 7, 10])
@pytest.mark.parametrize(("min_ngram", "max_ngram"), [(1, 1), (2, 3), (3, 5)])
def test_split_lookup_matches_standalone_ngram(
    split: int, min_ngram: int, max_ngram: int
) -> None:
    tokens = np.array([1, 2, 3, 4, 1, 2, 3, 5, 1, 2], dtype=np.int32)
    expected = _find_longest_matched_ngram_and_propose_tokens(
        tokens,
        min_ngram,
        max_ngram,
        max_model_len=64,
        k=3,
    )
    prefix = tokens[:split]
    suffix = tokens[split:].astype(np.int64)
    actual = find_split_ngram_proposal(
        prefix,
        len(prefix),
        suffix,
        len(suffix),
        min_ngram,
        max_ngram,
        max_model_len=64,
        num_draft_tokens=3,
    )
    np.testing.assert_array_equal(actual, expected)


def test_batch_assist_reports_only_full_width_hits() -> None:
    assist = DFlash2NgramAssist(2, 3, num_draft_tokens=2, max_model_len=64)
    token_ids = np.zeros((3, 16), dtype=np.int32)
    token_ids[2, :6] = [1, 2, 3, 4, 1, 2]
    token_ids[0, :4] = [8, 9, 10, 11]
    sampled = np.array([[3, -1], [12, -1]], dtype=np.int64)
    sampled_lens = np.array([1, 1], dtype=np.int32)
    output = np.zeros((2, 2), dtype=np.int64)
    output_lens = np.zeros(2, dtype=np.int32)

    hits = assist.propose(
        token_ids,
        req_state_indices=np.array([2, 0], dtype=np.int32),
        prior_lengths=np.array([6, 4], dtype=np.int32),
        sampled_token_ids=sampled,
        num_sampled_tokens=sampled_lens,
        eligible=np.array([True, True]),
        output_tokens=output,
        output_lengths=output_lens,
    )

    assert hits == 1
    np.testing.assert_array_equal(output[0], [4, 1])
    np.testing.assert_array_equal(output_lens, [2, 0])
    assert assist.num_eligible == 2
    assert assist.num_full_hits == 1


class _CopyEvent:
    def __init__(self) -> None:
        self.synchronized = False

    def synchronize(self) -> None:
        self.synchronized = True


def _host_only_speculator() -> DFlash2Speculator:
    speculator = object.__new__(DFlash2Speculator)
    speculator._ngram_assist = DFlash2NgramAssist(
        2, 3, num_draft_tokens=2, max_model_len=64
    )
    speculator._ngram_num_hits = 0
    speculator._ngram_rounds = 0
    speculator._ngram_skipped_rounds = 0
    speculator.num_speculative_steps = 2
    speculator._ngram_tokens_cpu_tensor = torch.zeros((2, 2), dtype=torch.int64)
    speculator._ngram_lengths_cpu_tensor = torch.zeros(2, dtype=torch.int32)
    speculator._ngram_tokens_cpu = speculator._ngram_tokens_cpu_tensor.numpy()
    speculator._ngram_lengths_cpu = speculator._ngram_lengths_cpu_tensor.numpy()
    speculator._ngram_tokens = torch.zeros((2, 2), dtype=torch.int64)
    speculator._ngram_lengths = torch.zeros(2, dtype=torch.int32)
    return speculator


def test_host_token_state_is_requested_only_for_enabled_assist() -> None:
    speculator = object.__new__(DFlash2Speculator)
    speculator._ngram_assist = None
    assert not speculator.requires_host_token_state

    speculator._ngram_assist = object()  # type: ignore[assignment]
    assert speculator.requires_host_token_state


def test_prepare_assist_uses_request_slots_and_skips_only_all_hit() -> None:
    speculator = _host_only_speculator()
    token_ids = np.zeros((3, 16), dtype=np.int32)
    token_ids[2, :6] = [1, 2, 3, 4, 1, 2]
    token_ids[0, :6] = [7, 8, 9, 10, 7, 8]
    batch = SimpleNamespace(
        has_structured_output_reqs=False,
        num_reqs=2,
        num_draft_tokens_per_req=np.array([2, 2], dtype=np.int32),
        seq_lens_cpu_upper_bound=torch.tensor([8, 8], dtype=torch.int32),
        is_prefilling_np=np.array([False, False]),
        idx_mapping_np=np.array([2, 0], dtype=np.int32),
    )
    event = _CopyEvent()

    skip = speculator._prepare_ngram_assist(
        batch,
        event,
        sampled_token_ids_cpu=np.array([[3], [9]], dtype=np.int64),
        num_sampled_tokens_cpu=np.array([1, 1], dtype=np.int32),
        all_token_ids_cpu=token_ids,
    )

    assert event.synchronized
    assert skip
    assert speculator._ngram_num_hits == 2
    assert torch.equal(
        speculator._ngram_tokens,
        torch.tensor([[4, 1], [10, 7]], dtype=torch.int64),
    )


def test_prepare_assist_keeps_dflash_draft_for_mixed_hits() -> None:
    speculator = _host_only_speculator()
    token_ids = np.zeros((3, 16), dtype=np.int32)
    token_ids[2, :6] = [1, 2, 3, 4, 1, 2]
    token_ids[0, :6] = [7, 8, 9, 10, 11, 12]
    batch = SimpleNamespace(
        has_structured_output_reqs=False,
        num_reqs=2,
        num_draft_tokens_per_req=np.array([2, 2], dtype=np.int32),
        seq_lens_cpu_upper_bound=torch.tensor([8, 8], dtype=torch.int32),
        is_prefilling_np=np.array([False, False]),
        idx_mapping_np=np.array([2, 0], dtype=np.int32),
    )
    event = _CopyEvent()

    skip = speculator._prepare_ngram_assist(
        batch,
        event,
        sampled_token_ids_cpu=np.array([[3], [13]], dtype=np.int64),
        num_sampled_tokens_cpu=np.array([1, 1], dtype=np.int32),
        all_token_ids_cpu=token_ids,
    )

    assert event.synchronized
    assert not skip
    assert speculator._ngram_assist is not None
    assert speculator._ngram_assist.num_full_hits == 1
    assert speculator._ngram_num_hits == 0
    assert torch.count_nonzero(speculator._ngram_tokens) == 0


def test_prepare_assist_bypasses_structured_output_without_sync() -> None:
    speculator = _host_only_speculator()
    batch = SimpleNamespace(has_structured_output_reqs=True)
    event = _CopyEvent()

    assert not speculator._prepare_ngram_assist(
        batch,
        event,
        sampled_token_ids_cpu=np.zeros((1, 1), dtype=np.int64),
        num_sampled_tokens_cpu=np.ones(1, dtype=np.int32),
        all_token_ids_cpu=np.zeros((1, 8), dtype=np.int32),
    )
    assert not event.synchronized


def test_ngram_assist_rejects_non_dflash_method() -> None:
    with pytest.raises(ValueError, match="only supported with method='dflash'"):
        SpeculativeConfig(
            method="ngram",
            num_speculative_tokens=2,
            ngram_assist=True,
        )


def test_lookup_controller_requires_two_hits_and_preserves_pending_state() -> None:
    state = dict(
        last_want=False,
        want_streak=0,
        sticky_remaining=0,
        long_active=False,
    )

    values = _advance_lookup_controller(
        want=True,
        num_reqs=1,
        entry_streak=2,
        sticky_steps=3,
        **state,
    )
    state = dict(zip(state, values))
    assert state == {
        "last_want": True,
        "want_streak": 1,
        "sticky_remaining": 0,
        "long_active": False,
    }

    values = _advance_lookup_controller(
        want=True,
        num_reqs=1,
        entry_streak=2,
        sticky_steps=3,
        **state,
    )
    state = dict(zip(state, values))
    assert state["long_active"] is True
    assert state["sticky_remaining"] == 3

    assert _advance_lookup_controller(
        want=None,
        num_reqs=1,
        entry_streak=2,
        sticky_steps=3,
        **state,
    ) == tuple(state.values())


def test_lookup_controller_never_coasts_a_multi_request_batch() -> None:
    values = _advance_lookup_controller(
        want=False,
        num_reqs=2,
        entry_streak=2,
        sticky_steps=3,
        last_want=True,
        want_streak=2,
        sticky_remaining=3,
        long_active=True,
    )

    assert values == (False, 0, 0, False)


def test_lookup_controller_selects_q8_before_two_strong_hits(monkeypatch) -> None:
    speculator = object.__new__(DFlash2Speculator)
    speculator._lookup_enabled = True
    speculator._lookup_current_req_key = (0,)
    speculator._lookup_current_eligible = True
    speculator._lookup_adaptive = True
    speculator._lookup_cheap_context = 0
    speculator._lookup_entry_streak = 2
    speculator._lookup_sticky_steps = 3
    speculator._lookup_last_want = False
    speculator._lookup_want_streak = 0
    speculator._lookup_sticky_remaining = 0
    speculator._lookup_long_active = False
    speculator._lookup_last_verify_tokens = 0
    speculator._lookup_q8_rounds = 0
    speculator._lookup_q16_rounds = 0
    speculator.draft_block = 7
    speculator.num_speculative_steps = 15
    speculator.draft_max_seq_len = 32768
    wants = iter((None, True, True))
    monkeypatch.setattr(speculator, "_consume_lookup_flags", lambda: next(wants))
    monkeypatch.setattr(speculator, "_queue_lookup_flags", lambda _num_reqs: None)

    assert speculator.next_num_draft_tokens() == 7
    assert speculator.next_num_draft_tokens() == 7
    assert speculator.next_num_draft_tokens() == 15
    assert speculator._lookup_q8_rounds == 2
    assert speculator._lookup_q16_rounds == 1


def test_ngram_one_hot_cache_overrides_only_hit_rows() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the Triton cache kernel")

    device = torch.device("cuda")
    num_reqs, num_steps, top_k, vocab_size = 2, 3, 4, 32
    ngram_tokens = torch.tensor(
        [[11, 12, 13], [21, 22, 23]], dtype=torch.int64, device=device
    )
    ngram_lengths = torch.tensor([3, 0], dtype=torch.int32, device=device)
    # Batch row 0 belongs to request-state slot 1.
    sample_req_state = torch.tensor(
        [1, 1, 1, 0, 0, 0], dtype=torch.int32, device=device
    )
    draft_tokens = torch.full(
        (num_reqs, num_steps), 7, dtype=torch.int64, device=device
    )
    cached_ids = (
        torch.arange(
            num_reqs * num_steps * top_k, dtype=torch.int64, device=device
        ).view(num_reqs, num_steps, top_k)
        % vocab_size
    )
    cached_scores = torch.randn(
        num_reqs, num_steps, top_k, dtype=torch.float32, device=device
    )
    draft_logits = torch.full(
        (num_reqs, num_steps, vocab_size),
        -float("inf"),
        dtype=torch.float32,
        device=device,
    )
    draft_logits.scatter_(2, cached_ids, cached_scores)
    miss_ids = cached_ids[0].clone()
    miss_scores = cached_scores[0].clone()
    miss_logits = draft_logits[0].clone()

    _apply_ngram_draft_kernel[(num_reqs * num_steps,)](
        ngram_tokens,
        ngram_lengths,
        sample_req_state,
        draft_tokens,
        draft_tokens.stride(0),
        cached_ids,
        cached_scores,
        cached_ids.stride(0),
        cached_ids.stride(1),
        draft_logits,
        draft_logits.stride(0),
        draft_logits.stride(1),
        num_steps=num_steps,
        top_k=top_k,
        BLOCK_K=top_k,
        CACHE_DRAFT_LOGITS=True,
        CACHE_SCORES=True,
        num_warps=1,
    )

    assert torch.equal(draft_tokens[0], ngram_tokens[0])
    assert torch.equal(draft_tokens[1], torch.full((num_steps,), 7, device=device))
    assert torch.equal(cached_ids[0], miss_ids)
    assert torch.equal(cached_scores[0], miss_scores)
    assert torch.equal(draft_logits[0], miss_logits)
    assert torch.equal(cached_ids[1, :, 0], ngram_tokens[0])
    assert torch.equal(cached_scores[1, :, 0], torch.zeros(num_steps, device=device))
    assert torch.isneginf(cached_scores[1, :, 1:]).all()
    assert torch.equal(
        draft_logits[1].gather(1, ngram_tokens[0, :, None]).squeeze(1),
        torch.zeros(num_steps, device=device),
    )


@pytest.mark.parametrize("top_p", [1.0, 0.95])
def test_ngram_one_hot_sparse_rejection_matches_dense(top_p: float) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for rejection sampling")

    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p

    torch.manual_seed(20260825)
    device = torch.device("cuda")
    num_reqs, num_steps = 16, 3
    target_top_k, draft_top_k, vocab_size = 20, 16, 97
    rows_per_req = num_steps + 1
    num_logits = num_reqs * rows_per_req

    target_ids = torch.stack(
        [
            torch.randperm(vocab_size, device=device)[:target_top_k]
            for _ in range(num_logits)
        ]
    )
    target_logits = (
        torch.randn(num_logits, target_top_k, dtype=torch.float32, device=device)
        .sort(dim=-1, descending=True)
        .values
    )
    target_dense = torch.full(
        (num_logits, vocab_size),
        -float("inf"),
        dtype=torch.float32,
        device=device,
    )
    target_dense.scatter_(1, target_ids, target_logits)
    target_dense = apply_top_k_top_p(
        target_dense,
        torch.full((num_logits,), target_top_k, dtype=torch.int32, device=device),
        torch.full((num_logits,), top_p, dtype=torch.float32, device=device),
    )

    proposals = torch.randint(
        0, vocab_size, (num_reqs, num_steps), dtype=torch.int64, device=device
    )
    draft_ids = torch.zeros(
        num_reqs, num_steps, draft_top_k, dtype=torch.int64, device=device
    )
    draft_ids[:, :, 0] = proposals
    draft_logits = torch.full(
        (num_reqs, num_steps, draft_top_k),
        -float("inf"),
        dtype=torch.float32,
        device=device,
    )
    draft_logits[:, :, 0] = 0.0
    draft_dense = torch.full(
        (num_reqs, num_steps, vocab_size),
        -float("inf"),
        dtype=torch.float32,
        device=device,
    )
    draft_dense.scatter_(2, proposals[:, :, None], 0.0)

    draft_sampled_2d = torch.zeros(
        num_reqs, rows_per_req, dtype=torch.int64, device=device
    )
    draft_sampled_2d[:, 1:] = proposals
    draft_sampled = draft_sampled_2d.flatten()
    cu_num_logits = (
        torch.arange(num_reqs + 1, dtype=torch.int32, device=device) * rows_per_req
    )
    idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=device)
    expanded_idx_mapping = idx_mapping.repeat_interleave(rows_per_req)
    expanded_local_pos = torch.arange(
        rows_per_req, dtype=torch.int32, device=device
    ).repeat(num_reqs)
    positions = torch.arange(num_logits, dtype=torch.int64, device=device) + 4096
    temperature = torch.ones(num_reqs, dtype=torch.float32, device=device)
    seeds = torch.arange(100, 100 + num_reqs, dtype=torch.int64, device=device)

    dense_sampled, dense_lengths = rejection_sample(
        target_dense,
        draft_dense,
        draft_sampled,
        cu_num_logits,
        positions,
        idx_mapping,
        expanded_idx_mapping,
        expanded_local_pos,
        temperature,
        seeds,
        num_steps,
    )
    sparse_sampled, sparse_lengths = dflash2_sparse_topk_rejection_sample(
        target_ids,
        target_logits,
        draft_ids,
        draft_logits,
        draft_sampled,
        cu_num_logits,
        positions,
        idx_mapping,
        temperature,
        torch.full((num_reqs,), top_p, dtype=torch.float32, device=device),
        seeds,
        num_steps,
    )

    assert torch.equal(sparse_lengths, dense_lengths)
    steps = torch.arange(rows_per_req, device=device).unsqueeze(0)
    valid = steps < dense_lengths[:, None]
    assert torch.equal(sparse_sampled[valid], dense_sampled[valid])
