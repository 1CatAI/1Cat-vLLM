# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import OrderedDict
from types import SimpleNamespace

import pytest
import torch

from vllm.v1.spec_decode.static_draft_vocab import (
    DynamicDraftVocabPrefillBootstrapState,
    build_static_draft_vocab_tp_plan,
    compact_dynamic_draft_vocab_tail_state,
    initialize_dynamic_draft_vocab,
    prepare_dynamic_draft_vocab_prefill_candidates,
    remap_dynamic_draft_output,
    remap_reduced_draft_output,
    resolve_mtp_draft_vocab_config,
    select_dynamic_draft_vocab_shard_seed,
    update_dynamic_draft_vocab_lru_state,
    validate_dynamic_draft_vocab_prefill_topk,
)


def test_dynamic_draft_vocab_is_the_default_mtp_route(monkeypatch) -> None:
    for name in (
        "VLLM_SM70_MTP_STATIC_DRAFT_VOCAB_RANKING",
        "VLLM_SM70_MTP_STATIC_DRAFT_VOCAB_SIZE",
        "VLLM_SM70_MTP_DYNAMIC_DRAFT_VOCAB_TAIL_SIZE",
        "VLLM_SM70_MTP_DYNAMIC_DRAFT_VOCAB_FULL_REFRESH_INTERVAL",
        "VLLM_SM70_MTP_DYNAMIC_DRAFT_VOCAB_FUSED_PROPOSAL",
        "VLLM_SM70_MTP_DYNAMIC_DRAFT_VOCAB_GPU_LRU",
        "VLLM_SM70_MTP_DYNAMIC_DRAFT_VOCAB_PREFILL_TOPK",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("VLLM_SM70_MTP_DYNAMIC_DRAFT_VOCAB_DEFAULT", "1")

    config = resolve_mtp_draft_vocab_config("mtp")

    assert config.using_defaults
    assert config.shortlist_size == 98_304
    assert config.dynamic_tail_size == 512
    assert config.full_refresh_interval == 0
    assert config.fused_proposal_enabled
    assert config.gpu_lru_enabled
    assert config.prefill_topk == 2048
    assert config.ranking_path is not None
    assert config.ranking_path.endswith("qwen36_27b_tp2.pt")


def test_dynamic_draft_vocab_default_selects_tp4_asset(monkeypatch) -> None:
    for name in (
        "VLLM_SM70_MTP_STATIC_DRAFT_VOCAB_RANKING",
        "VLLM_SM70_MTP_STATIC_DRAFT_VOCAB_SIZE",
        "VLLM_SM70_MTP_DYNAMIC_DRAFT_VOCAB_TAIL_SIZE",
        "VLLM_SM70_MTP_DYNAMIC_DRAFT_VOCAB_FULL_REFRESH_INTERVAL",
        "VLLM_SM70_MTP_DYNAMIC_DRAFT_VOCAB_FUSED_PROPOSAL",
        "VLLM_SM70_MTP_DYNAMIC_DRAFT_VOCAB_GPU_LRU",
        "VLLM_SM70_MTP_DYNAMIC_DRAFT_VOCAB_PREFILL_TOPK",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("VLLM_SM70_MTP_DYNAMIC_DRAFT_VOCAB_DEFAULT", "1")

    config = resolve_mtp_draft_vocab_config("mtp", tensor_parallel_size=4)

    assert config.using_defaults
    assert config.ranking_path is not None
    assert config.ranking_path.endswith("qwen36_27b_tp4.pt")


def test_dynamic_draft_vocab_default_can_be_disabled(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_SM70_MTP_DYNAMIC_DRAFT_VOCAB_DEFAULT", "0")

    config = resolve_mtp_draft_vocab_config("mtp")

    assert not config.using_defaults
    assert config.ranking_path is None
    assert config.shortlist_size == 0
    assert config.dynamic_tail_size == 0
    assert config.prefill_topk == 0


def test_explicit_static_draft_vocab_overrides_dynamic_default(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_SM70_MTP_DYNAMIC_DRAFT_VOCAB_DEFAULT", "1")
    monkeypatch.setenv("VLLM_SM70_MTP_STATIC_DRAFT_VOCAB_RANKING", "/tmp/r.pt")
    monkeypatch.setenv("VLLM_SM70_MTP_STATIC_DRAFT_VOCAB_SIZE", "131072")

    config = resolve_mtp_draft_vocab_config("mtp")

    assert not config.using_defaults
    assert config.ranking_path == "/tmp/r.pt"
    assert config.shortlist_size == 131_072
    assert config.dynamic_tail_size == 0


def test_build_static_draft_vocab_tp_plan_balances_global_selection() -> None:
    plan = build_static_draft_vocab_tp_plan(
        global_ranking=[0, 4, 1, 5, 2, 3, 6, 7],
        shard_ranges=[(0, 4), (4, 8)],
        shortlist_size=6,
    )

    assert plan.source_token_ids == ((0, 1, 2, 3), (4, 5))
    assert plan.destination_token_ids == ((0, 1, 2), (3, 4, 5))
    assert plan.split_matrix == ((3, 1), (0, 2))
    assert plan.gathered_token_ids == (0, 1, 2, 3, 4, 5)


def test_build_static_draft_vocab_tp_plan_rejects_duplicates() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        build_static_draft_vocab_tp_plan(
            global_ranking=[0, 1, 1, 2],
            shard_ranges=[(0, 2), (2, 4)],
            shortlist_size=3,
        )


def test_build_static_draft_vocab_tp_plan_rejects_token_outside_shards() -> None:
    with pytest.raises(ValueError, match="outside all vocabulary shards"):
        build_static_draft_vocab_tp_plan(
            global_ranking=[0, 1, 8, 2],
            shard_ranges=[(0, 2), (2, 4)],
            shortlist_size=3,
        )


def test_remap_reduced_draft_output() -> None:
    reduced_token_ids = torch.tensor([2, 0], dtype=torch.int64)
    reduced_probs = torch.tensor(
        [[0.1, 0.2, 0.7], [0.4, 0.5, 0.1]], dtype=torch.float32
    )
    token_id_map = torch.tensor([5, 2, 7], dtype=torch.int64)

    token_ids, full_probs = remap_reduced_draft_output(
        reduced_token_ids,
        reduced_probs,
        token_id_map,
        full_vocab_size=8,
    )

    assert torch.equal(token_ids, torch.tensor([7, 5]))
    expected = torch.zeros((2, 8), dtype=torch.float32)
    expected[:, token_id_map] = reduced_probs
    assert torch.equal(full_probs, expected)


def test_remap_dynamic_draft_output_accumulates_inactive_padding() -> None:
    reduced_token_ids = torch.tensor([1], dtype=torch.int64)
    reduced_probs = torch.tensor([[0.25, 0.75, 0.0, 0.0]], dtype=torch.float32)
    token_id_map = torch.tensor([5, 2, 0, 0], dtype=torch.int64)

    token_ids, full_probs = remap_dynamic_draft_output(
        reduced_token_ids,
        reduced_probs,
        token_id_map,
        full_vocab_size=8,
    )

    assert torch.equal(token_ids, torch.tensor([2]))
    expected = torch.zeros((1, 8), dtype=torch.float32)
    expected[0, 5] = 0.25
    expected[0, 2] = 0.75
    assert torch.equal(full_probs, expected)


def test_dynamic_prefill_candidates_are_low_to_high_and_masked() -> None:
    logits = torch.tensor(
        [
            [0.0, 4.0, -torch.inf, 2.0, 3.0],
            [5.0, -torch.inf, 1.0, 4.0, 3.0],
        ],
        dtype=torch.float32,
    )

    candidate_ids = prepare_dynamic_draft_vocab_prefill_candidates(logits, topk=5)

    assert candidate_ids.dtype == torch.int64
    assert candidate_ids.is_contiguous()
    assert torch.equal(
        candidate_ids,
        torch.tensor(
            [
                [-1, 0, 3, 4, 1],
                [-1, 2, 4, 3, 0],
            ],
            dtype=torch.int64,
        ),
    )

    tail_lru: OrderedDict[int, None] = OrderedDict()
    update_dynamic_draft_vocab_lru_state(
        tail_lru,
        torch.empty(0, dtype=torch.int32),
        candidate_ids[:1],
        full_vocab_size=5,
        base_token_ids=frozenset(),
        tail_size=2,
    )
    assert tuple(tail_lru) == (4, 1)


def test_dynamic_prefill_bootstrap_request_lifecycle() -> None:
    state = DynamicDraftVocabPrefillBootstrapState()
    logits = torch.tensor([[0.0, 3.0, 2.0, 1.0]], dtype=torch.float32)
    final_prefill = {
        "topk": 2,
        "num_computed_tokens": 7,
        "num_scheduled_tokens": 1,
        "num_prompt_tokens": 8,
        "spec_decode_active": False,
    }

    assert (
        state.maybe_prepare_candidates(
            "req-a",
            logits,
            **{
                **final_prefill,
                "num_computed_tokens": 0,
                "num_scheduled_tokens": 4,
            },
        )
        is None
    )
    assert state.maybe_prepare_candidates("req-a", None, **final_prefill) is None
    assert (
        state.maybe_prepare_candidates(
            "req-a",
            logits,
            **{**final_prefill, "spec_decode_active": True},
        )
        is None
    )
    candidate_ids = state.maybe_prepare_candidates("req-a", logits, **final_prefill)
    assert candidate_ids is not None
    state.mark_consumed("req-a")
    assert state.maybe_prepare_candidates("req-a", logits, **final_prefill) is None

    assert state.maybe_prepare_candidates("req-b", logits, **final_prefill) is not None
    state.mark_consumed("req-b")
    state.clear_finished_requests({"req-b"})
    assert state.maybe_prepare_candidates("req-b", logits, **final_prefill) is not None


def test_dynamic_prefill_topk_validation() -> None:
    validate_dynamic_draft_vocab_prefill_topk(
        0,
        gpu_lru_enabled=False,
        full_vocab_size=100,
    )
    with pytest.raises(ValueError, match="requires the GPU LRU route"):
        validate_dynamic_draft_vocab_prefill_topk(
            1,
            gpu_lru_enabled=False,
            full_vocab_size=100,
        )
    with pytest.raises(ValueError, match=r"\[1, 4096\]"):
        validate_dynamic_draft_vocab_prefill_topk(
            4097,
            gpu_lru_enabled=True,
            full_vocab_size=10_000,
        )


def test_dynamic_draft_vocab_lru_processes_observed_before_candidates() -> None:
    tail_lru: OrderedDict[int, None] = OrderedDict.fromkeys((5, 6))

    changed = update_dynamic_draft_vocab_lru_state(
        tail_lru,
        torch.tensor([[6]], dtype=torch.int32),
        torch.tensor([[5]], dtype=torch.int64),
        full_vocab_size=10,
        base_token_ids=frozenset(),
        tail_size=2,
    )

    assert changed
    assert tuple(tail_lru) == (6, 5)


def test_dynamic_draft_vocab_lru_filters_moves_and_evicts() -> None:
    tail_lru: OrderedDict[int, None] = OrderedDict.fromkeys((5, 6, 7))

    changed = update_dynamic_draft_vocab_lru_state(
        tail_lru,
        torch.tensor([[6, -1, 2]], dtype=torch.int32),
        torch.tensor([[5, 9, 6, 10, 0]], dtype=torch.int64),
        full_vocab_size=10,
        base_token_ids=frozenset((0, 2, 4)),
        tail_size=3,
    )

    assert changed
    assert tuple(tail_lru) == (5, 9, 6)


def test_dynamic_draft_vocab_shard_seeds_fill_both_tp_ranks() -> None:
    base = tuple(range(512))
    rank1_top = tuple(range(1024, 1536))
    rank0_tail = tuple(range(512, 1024))
    remaining = tuple(range(1536, 2048))
    ranking = base + rank1_top + rank0_tail + remaining
    base_token_ids = frozenset(base)

    rank0_seed = select_dynamic_draft_vocab_shard_seed(
        ranking[512:],
        base_token_ids=base_token_ids,
        local_shard_start=0,
        local_shard_end=1024,
        tail_size=512,
    )
    rank1_seed = select_dynamic_draft_vocab_shard_seed(
        ranking[512:],
        base_token_ids=base_token_ids,
        local_shard_start=1024,
        local_shard_end=2048,
        tail_size=512,
    )

    assert ranking[512:1024] == rank1_top
    assert rank0_seed == rank0_tail
    assert rank1_seed == rank1_top
    gathered_seed = rank0_seed + rank1_seed
    assert len(gathered_seed) == 1024
    assert len(set(gathered_seed)) == 1024
    assert base_token_ids.isdisjoint(gathered_seed)


def test_dynamic_draft_vocab_shard_lrus_update_independently() -> None:
    rank0_lru: OrderedDict[int, None] = OrderedDict.fromkeys((2, 3, 4))
    rank1_lru: OrderedDict[int, None] = OrderedDict.fromkeys((10, 11, 12))
    observed = torch.tensor([3, 11, 5, -1, 0], dtype=torch.int32)
    candidates = torch.tensor([10, 4, 13, 6, 16], dtype=torch.int64)

    for tail_lru, (shard_start, shard_end) in (
        (rank0_lru, (0, 8)),
        (rank1_lru, (8, 16)),
    ):
        changed = update_dynamic_draft_vocab_lru_state(
            tail_lru,
            observed,
            candidates,
            full_vocab_size=16,
            base_token_ids=frozenset((0, 1, 8, 9)),
            tail_size=3,
            local_shard_start=shard_start,
            local_shard_end=shard_end,
        )
        assert changed

    assert tuple(rank0_lru) == (5, 4, 6)
    assert tuple(rank1_lru) == (11, 10, 13)
    gathered_ids = tuple(rank0_lru) + tuple(rank1_lru)
    assert len(set(gathered_ids)) == len(gathered_ids)
    assert frozenset((0, 1, 8, 9)).isdisjoint(gathered_ids)


def test_dynamic_draft_vocab_tail_compaction_keeps_lru_order() -> None:
    local_ids, source_rows = compact_dynamic_draft_vocab_tail_state(
        (7, 13, 9, 15, 11),
        local_shard_start=8,
        local_shard_end=14,
        local_tail_capacity=4,
    )

    assert local_ids == [13, 9, 11]
    assert source_rows == [5, 1, 3]

    with pytest.raises(RuntimeError, match="exceeded local tail capacity"):
        compact_dynamic_draft_vocab_tail_state(
            (9, 10, 11),
            local_shard_start=8,
            local_shard_end=14,
            local_tail_capacity=2,
        )


def test_dynamic_gpu_lru_rejects_periodic_full_refresh() -> None:
    target_lm_head = SimpleNamespace(
        weight=torch.empty(0, dtype=torch.float16),
    )

    with pytest.raises(ValueError, match="full_refresh_interval=0"):
        initialize_dynamic_draft_vocab(
            target_lm_head,
            "unused.pt",
            98_304,
            512,
            1,
            True,
            4,
            torch.device("cpu"),
            gpu_lru_enabled=True,
        )
