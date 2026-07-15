# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.v1.spec_decode.ddtree_payload import build_ddtree_payloads_from_logits
from vllm.v1.spec_decode.dflash import DFlashProposer
from vllm.v1.spec_decode.utils import (
    build_dflash_ddtree_query_depths,
    dflash_query_positions_from_depths,
)


class _GreedySamplingMetadata:
    all_greedy = True


def test_topk_one_payload_matches_flat_dflash_chain() -> None:
    batch_size = 2
    num_speculative_tokens = 4
    vocab_size = 128
    flat_draft_token_ids = torch.tensor(
        [
            [11, 12, 13, 14],
            [21, 22, 23, 24],
        ],
        dtype=torch.int64,
    )
    logits = torch.full(
        (batch_size * num_speculative_tokens, vocab_size),
        -100.0,
        dtype=torch.float32,
    )
    for row, token_id in enumerate(flat_draft_token_ids.flatten()):
        logits[row, int(token_id.item())] = 10.0

    payloads = build_ddtree_payloads_from_logits(
        logits=logits,
        batch_size=batch_size,
        num_speculative_tokens=num_speculative_tokens,
        budget=num_speculative_tokens,
        top_k=1,
        chain_seed=True,
        flat_draft_token_ids=flat_draft_token_ids,
    )

    assert len(payloads) == batch_size
    for payload, flat_chain in zip(
        payloads,
        flat_draft_token_ids.tolist(),
        strict=True,
    ):
        assert payload.tree_token_ids == tuple(flat_chain)
        assert payload.parent_indices == (-1, 0, 1, 2)
        assert payload.node_depths == (1, 2, 3, 4)
        assert payload.top1_chain_token_ids == tuple(flat_chain)
        assert payload.flat_chain_matches_top1()
        assert payload.is_flat_chain()


def test_payload_builds_sibling_branch_from_topk_logits() -> None:
    vocab_size = 128
    logits = torch.full((3, vocab_size), -100.0, dtype=torch.float32)
    logits[0, 11] = 10.0
    logits[0, 12] = 9.8
    logits[1, 21] = 10.0
    logits[1, 22] = 9.7
    logits[2, 31] = 10.0
    logits[2, 32] = 5.0

    payload = build_ddtree_payloads_from_logits(
        logits=logits,
        batch_size=1,
        num_speculative_tokens=3,
        budget=5,
        top_k=2,
        chain_seed=True,
    )[0]

    assert payload.tree_token_ids[:3] == (11, 21, 31)
    assert payload.parent_indices[:3] == (-1, 0, 1)
    assert payload.num_tree_nodes == 5
    assert 12 in payload.tree_token_ids
    assert payload.top1_chain_token_ids == (11, 21, 31)
    assert not payload.is_flat_chain()


def test_query_positions_from_depths_gives_equal_sibling_positions() -> None:
    root_positions = torch.tensor([100], dtype=torch.int64)
    query_depths = torch.tensor([[0, 1, 2, 1]], dtype=torch.int64)

    positions = dflash_query_positions_from_depths(root_positions, query_depths)

    assert positions.tolist() == [[100, 101, 102, 101]]
    assert positions[0, 1].item() == positions[0, 3].item()


def test_build_dflash_ddtree_query_depths_helper_mixes_tree_and_flat_rows() -> None:
    payload = SimpleNamespace(node_depths=(1, 2, 1))

    depths = build_dflash_ddtree_query_depths(
        [payload, None],
        batch_size=2,
        num_query_per_req=4,
        device="cpu",
    )

    assert depths is not None
    assert depths.view(2, 4).tolist() == [
        [0, 1, 2, 1],
        [0, 1, 2, 3],
    ]


def test_build_dflash_ddtree_query_depths_helper_returns_none_without_payloads() -> None:
    assert (
        build_dflash_ddtree_query_depths(
            [None],
            batch_size=1,
            num_query_per_req=4,
            device="cpu",
        )
        is None
    )


def test_payload_top_k_none_matches_reference_budget_width() -> None:
    torch.manual_seed(0)
    logits = torch.randn(6, 64)

    payload = build_ddtree_payloads_from_logits(
        logits=logits,
        batch_size=1,
        num_speculative_tokens=6,
        budget=16,
        top_k=None,
        chain_seed=False,
    )[0]

    assert payload.budget == 16
    assert payload.top_k == 16
    assert all(
        len(depth_tokens) == 16 for depth_tokens in payload.topk_token_ids_by_depth
    )


def test_payload_root_leaf_mode_preserves_spine_and_leaf_alternatives() -> None:
    vocab_size = 128
    logits = torch.full((3, vocab_size), -100.0, dtype=torch.float32)
    logits[0, 11] = 10.0
    logits[0, 12] = 9.8
    logits[1, 21] = 10.0
    logits[1, 22] = 9.9
    logits[2, 31] = 10.0
    logits[2, 32] = 9.7

    payload = build_ddtree_payloads_from_logits(
        logits=logits,
        batch_size=1,
        num_speculative_tokens=3,
        budget=6,
        top_k=2,
        chain_seed=False,
        tree_mode="root_leaf",
    )[0]

    assert payload.tree_token_ids[:3] == (11, 21, 31)
    assert payload.parent_indices[:3] == (-1, 0, 1)
    assert payload.node_depths[:3] == (1, 2, 3)
    assert payload.tree_mode == "root_leaf"
    for verifier_parent in payload.parent_indices[3:]:
        assert verifier_parent in (-1, 0, 1)


def test_dflash_ddtree_sampling_hook_builds_payload_from_logits() -> None:
    proposer = object.__new__(DFlashProposer)
    proposer.use_ddtree = True
    proposer.num_speculative_tokens = 3
    proposer.ddtree_budget = 3
    proposer.ddtree_top_k = 1
    proposer.ddtree_chain_seed = True
    proposer.ddtree_tree_mode = "best_first"
    proposer._enable_probabilistic_draft_probs = False
    proposer._last_ddtree_payloads = None

    logits = torch.full((3, 128), -100.0, dtype=torch.float32)
    logits[0, 11] = 10.0
    logits[1, 21] = 10.0
    logits[2, 31] = 10.0

    draft_token_ids, draft_probs = DFlashProposer._sample_draft_tokens(
        proposer,
        hidden_states=torch.empty((3, 4), dtype=torch.float32),
        sampling_metadata=_GreedySamplingMetadata(),  # type: ignore[arg-type]
        logits=logits,
        spec_step_idx=0,
    )

    assert draft_probs is None
    assert draft_token_ids.tolist() == [11, 21, 31]
    assert proposer._last_ddtree_payloads is not None
    payload = proposer._last_ddtree_payloads[0]
    assert payload.tree_token_ids == (11, 21, 31)
    assert payload.flat_chain_matches_top1()
