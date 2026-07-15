# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import heapq

import torch

from vllm.v1.spec_decode.ddtree_tree import build_ddtree, greedy_tree_walk


def test_budget_one_matches_top1() -> None:
    tree = build_ddtree(
        [
            [(11, -0.1), (12, -0.2)],
            [(21, -0.1), (22, -0.2)],
        ],
        budget=1,
        top_k=2,
        root_token_id=7,
    )

    assert tree.token_ids_for_verifier() == (11,)
    assert tree.parent_indices_for_verifier() == (-1,)
    assert tree.node_depths_for_verifier() == (1,)


def test_chain_seed_preserves_top1_path() -> None:
    tree = build_ddtree(
        [
            [(11, -0.1), (12, -0.2)],
            [(21, -0.1), (22, -0.2)],
            [(31, -0.1), (32, -0.2)],
        ],
        budget=3,
        top_k=2,
        chain_seed=True,
    )

    assert tree.token_ids_for_verifier() == (11, 21, 31)
    assert tree.parent_indices_for_verifier() == (-1, 0, 1)
    assert tree.path_token_ids(3) == (11, 21, 31)


def test_best_first_adds_sibling_branch() -> None:
    tree = build_ddtree(
        [
            [(11, -0.1), (12, -0.15)],
            [(21, -0.1), (22, -0.12)],
            [(31, -0.1), (32, -2.0)],
        ],
        budget=5,
        top_k=2,
    )

    assert tree.token_ids_for_verifier() == (11, 12, 21, 22, 21)
    assert tree.parent_indices_for_verifier() == (-1, -1, 0, 0, 1)
    assert tree.node_depths_for_verifier() == (1, 1, 2, 2, 2)
    assert 12 in tree.child_by_token(0)
    assert 22 in tree.child_by_token(1)


def test_root_leaf_mode_keeps_alternatives_on_top1_spine() -> None:
    tree = build_ddtree(
        [
            [(11, -0.1), (12, -0.2), (13, -0.3)],
            [(21, -0.1), (22, -0.15), (23, -0.4)],
            [(31, -0.1), (32, -0.12), (33, -0.5)],
        ],
        budget=6,
        top_k=3,
        chain_seed=False,
        tree_mode="root_leaf",
    )

    assert tree.token_ids_for_verifier()[:3] == (11, 21, 31)
    assert tree.parent_indices_for_verifier()[:3] == (-1, 0, 1)
    assert tree.path_token_ids(3) == (11, 21, 31)

    top1_spine_parents = {0, 1, 2}
    for node in tree.non_root_nodes[3:]:
        assert node.parent_index in top1_spine_parents
        assert not tree.child_by_token(node.index)


def test_visibility_is_ancestor_only() -> None:
    tree = build_ddtree(
        [
            [(11, -0.1), (12, -0.2)],
            [(21, -0.1), (22, -0.2)],
            [(31, -0.1), (32, -0.2)],
        ],
        budget=5,
        top_k=2,
    )
    visibility = tree.visibility_mask()

    node_11 = tree.child_by_token(0)[11]
    node_12 = tree.child_by_token(0)[12]
    node_21 = tree.child_by_token(node_11)[21]

    assert visibility[node_21][0]
    assert visibility[node_21][node_11]
    assert visibility[node_21][node_21]
    assert not visibility[node_21][node_12]


def test_greedy_tree_walk_accepts_path_and_bonus() -> None:
    tree = build_ddtree(
        [
            [(11, -0.1), (12, -0.2)],
            [(21, -0.1), (22, -0.2)],
            [(31, -0.1), (32, -0.2)],
        ],
        budget=4,
        top_k=2,
    )

    next_tokens = {
        (): 11,
        (11,): 21,
        (11, 21): 99,
    }
    walk = greedy_tree_walk(tree, lambda path: next_tokens[path])

    assert walk.accepted_node_indices == (0, 1, 3)
    assert walk.accepted_token_ids == (11, 21)
    assert walk.bonus_token_id == 99
    assert walk.output_token_ids == (11, 21, 99)


def test_topk_one_budget_sixteen_is_flat_dflash_chain() -> None:
    candidates = [[(1000 + depth, -0.1)] for depth in range(16)]

    tree = build_ddtree(candidates, budget=16, top_k=1, chain_seed=True)

    assert tree.token_ids_for_verifier() == tuple(range(1000, 1016))
    assert tree.parent_indices_for_verifier() == (-1,) + tuple(range(15))
    assert tree.node_depths_for_verifier() == tuple(range(1, 17))


def _reference_ddtree_arrays(
    draft_logits: torch.Tensor,
    budget: int,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Mirror liranringel/ddtree build_ddtree_tree for unit parity."""

    topk = min(budget, draft_logits.shape[-1])
    depth_limit = int(draft_logits.shape[0])
    logits = draft_logits.float()
    top_logits, top_token_ids = torch.topk(logits, k=topk, dim=-1)
    log_z = torch.logsumexp(logits, dim=-1, keepdim=True)
    top_log_probs_np = (top_logits - log_z).cpu().numpy()
    top_token_ids_np = top_token_ids.cpu().numpy()

    first_logw = float(top_log_probs_np[0, 0])
    heap: list[tuple[float, tuple[int, ...], int, int, int, float]] = [
        (-first_logw, (0,), 0, 1, 0, first_logw)
    ]
    node_token_ids: list[int] = []
    node_depths: list[int] = []
    parents = [-1]

    while heap and len(node_token_ids) < budget:
        _, ranks, parent_index, depth, rank, logw = heapq.heappop(heap)

        node_token_ids.append(int(top_token_ids_np[depth - 1, rank]))
        current_index = len(node_token_ids)
        node_depths.append(depth)
        parents.append(parent_index)

        if rank + 1 < topk:
            sibling_ranks = ranks[:-1] + (rank + 1,)
            sibling_logw = (
                logw
                - float(top_log_probs_np[depth - 1, rank])
                + float(top_log_probs_np[depth - 1, rank + 1])
            )
            heapq.heappush(
                heap,
                (
                    -sibling_logw,
                    sibling_ranks,
                    parent_index,
                    depth,
                    rank + 1,
                    sibling_logw,
                ),
            )

        if depth < depth_limit:
            child_ranks = ranks + (0,)
            child_logw = logw + float(top_log_probs_np[depth, 0])
            heapq.heappush(
                heap,
                (
                    -child_logw,
                    child_ranks,
                    current_index,
                    depth + 1,
                    0,
                    child_logw,
                ),
            )

    verifier_parents = tuple(
        -1 if parent == 0 else parent - 1 for parent in parents[1:]
    )
    return tuple(node_token_ids), verifier_parents, tuple(node_depths)


def test_best_first_matches_reference_ddtree_heap() -> None:
    torch.manual_seed(0)
    logits = torch.randn(6, 64)
    budget = 16
    topk = min(budget, logits.shape[-1])
    top_logits, top_token_ids = torch.topk(logits.float(), k=topk, dim=-1)
    log_normalizer = torch.logsumexp(logits.float(), dim=-1, keepdim=True)
    top_logprobs = top_logits - log_normalizer
    candidates = [
        [
            (int(top_token_ids[depth, rank]), float(top_logprobs[depth, rank]))
            for rank in range(topk)
        ]
        for depth in range(logits.shape[0])
    ]

    tree = build_ddtree(candidates, budget=budget)

    assert (
        tree.token_ids_for_verifier(),
        tree.parent_indices_for_verifier(),
        tree.node_depths_for_verifier(),
    ) == _reference_ddtree_arrays(logits, budget)
