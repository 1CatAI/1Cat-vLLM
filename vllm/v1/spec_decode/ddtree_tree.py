# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reference DDTree data structures for DFlash speculative decoding.

This module is intentionally independent from the engine hot path. It provides
the small, testable tree builder and greedy tree walk needed before wiring a
vLLM-native DDTree verifier.
"""

from __future__ import annotations

import heapq
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from itertools import count
from typing import Literal


DDTreeBuildMode = Literal["best_first", "root_leaf", "spine_leaf"]


@dataclass(frozen=True)
class DraftCandidate:
    token_id: int
    logprob: float


@dataclass(frozen=True)
class DDTreeNode:
    index: int
    parent_index: int | None
    token_id: int
    depth: int
    score: float


@dataclass(frozen=True)
class DDTree:
    """A flattened prefix-closed draft tree.

    Node 0 is the root token. The verifier consumes only non-root nodes; parent
    ids for non-root nodes use verifier coordinates where -1 means root.
    """

    nodes: tuple[DDTreeNode, ...]

    @property
    def non_root_nodes(self) -> tuple[DDTreeNode, ...]:
        return self.nodes[1:]

    def token_ids_for_verifier(self) -> tuple[int, ...]:
        return tuple(node.token_id for node in self.non_root_nodes)

    def parent_indices_for_verifier(self) -> tuple[int, ...]:
        parents: list[int] = []
        for node in self.non_root_nodes:
            if node.parent_index is None or node.parent_index == 0:
                parents.append(-1)
            else:
                parents.append(node.parent_index - 1)
        return tuple(parents)

    def node_depths_for_verifier(self) -> tuple[int, ...]:
        return tuple(node.depth for node in self.non_root_nodes)

    def path_token_ids(self, node_index: int) -> tuple[int, ...]:
        path: list[int] = []
        cursor: int | None = node_index
        while cursor is not None and cursor != 0:
            node = self.nodes[cursor]
            path.append(node.token_id)
            cursor = node.parent_index
        path.reverse()
        return tuple(path)

    def ancestor_indices(
        self,
        node_index: int,
        *,
        include_self: bool = True,
    ) -> set[int]:
        ancestors: set[int] = set()
        cursor = node_index if include_self else self.nodes[node_index].parent_index
        while cursor is not None:
            ancestors.add(cursor)
            cursor = self.nodes[cursor].parent_index
        return ancestors

    def visibility_mask(self) -> tuple[tuple[bool, ...], ...]:
        rows: list[tuple[bool, ...]] = []
        for node in self.nodes:
            visible = self.ancestor_indices(node.index)
            rows.append(tuple(col in visible for col in range(len(self.nodes))))
        return tuple(rows)

    def child_by_token(self, parent_index: int) -> dict[int, int]:
        children: dict[int, int] = {}
        for node in self.non_root_nodes:
            if node.parent_index == parent_index:
                children[node.token_id] = node.index
        return children


@dataclass(frozen=True)
class GreedyDDTreeWalk:
    # Compact tree node indices accepted by the verifier walk. Matches the
    # official DDTree helper: includes root index 0, excludes the bonus token.
    accepted_node_indices: tuple[int, ...]
    # Accepted non-root draft token ids. The root/current token is not included
    # because DDTree instances built in tests may use a synthetic root id.
    accepted_token_ids: tuple[int, ...]
    bonus_token_id: int
    visited_node_indices: tuple[int, ...]

    @property
    def output_token_ids(self) -> tuple[int, ...]:
        return self.accepted_token_ids + (self.bonus_token_id,)


def greedy_tree_walk(
    tree: DDTree,
    next_token_for_path: Callable[[tuple[int, ...]], int],
) -> GreedyDDTreeWalk:
    """Walk a draft tree with a target-model next-token oracle."""

    cursor = 0
    accepted_nodes: list[int] = [0]
    accepted_tokens: list[int] = []
    visited_nodes: list[int] = [0]

    while True:
        path = tree.path_token_ids(cursor)
        next_token = int(next_token_for_path(path))
        child_index = tree.child_by_token(cursor).get(next_token)
        if child_index is None:
            return GreedyDDTreeWalk(
                accepted_node_indices=tuple(accepted_nodes),
                accepted_token_ids=tuple(accepted_tokens),
                bonus_token_id=next_token,
                visited_node_indices=tuple(visited_nodes),
            )

        accepted_nodes.append(child_index)
        accepted_tokens.append(next_token)
        visited_nodes.append(child_index)
        cursor = child_index


def _normalize_candidates(
    candidates_by_depth: Iterable[Iterable[DraftCandidate | tuple[int, float]]],
    top_k: int,
) -> list[list[DraftCandidate]]:
    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    normalized: list[list[DraftCandidate]] = []
    for depth, raw_candidates in enumerate(candidates_by_depth, start=1):
        candidates: list[DraftCandidate] = []
        for raw in raw_candidates:
            if isinstance(raw, DraftCandidate):
                candidate = raw
            else:
                token_id, logprob = raw
                candidate = DraftCandidate(int(token_id), float(logprob))
            candidates.append(candidate)

        if not candidates:
            raise ValueError(f"depth {depth} has no draft candidates")
        candidates.sort(key=lambda item: (item.logprob, -item.token_id),
                        reverse=True)
        normalized.append(candidates[:top_k])

    if not normalized:
        raise ValueError("candidates_by_depth must contain at least one depth")
    return normalized


def build_ddtree(
    candidates_by_depth: Iterable[Iterable[DraftCandidate | tuple[int, float]]],
    *,
    budget: int,
    top_k: int | None = None,
    chain_seed: bool = False,
    tree_mode: DDTreeBuildMode = "best_first",
    min_root_branches: int = 0,
    root_token_id: int = -1,
) -> DDTree:
    """Build a best-first DDTree from per-depth draft candidates.

    `budget` counts non-root verifier nodes. Candidate scores are cumulative
    log probabilities under the one-pass DFlash factorized approximation.
    """

    if budget < 1:
        raise ValueError("budget must be >= 1")

    requested_top_k = budget if top_k is None else top_k
    candidates = _normalize_candidates(candidates_by_depth, requested_top_k)
    nodes: list[DDTreeNode] = [
        DDTreeNode(
            index=0,
            parent_index=None,
            token_id=root_token_id,
            depth=0,
            score=0.0,
        )
    ]
    child_edges: set[tuple[int, int, int]] = set()

    def add_child(parent_index: int, candidate: DraftCandidate) -> DDTreeNode:
        parent = nodes[parent_index]
        depth = parent.depth + 1
        if depth > len(candidates):
            raise ValueError("cannot add child beyond available depths")
        edge = (parent_index, depth, candidate.token_id)
        if edge in child_edges:
            raise ValueError("duplicate child edge")
        node = DDTreeNode(
            index=len(nodes),
            parent_index=parent_index,
            token_id=candidate.token_id,
            depth=depth,
            score=parent.score + candidate.logprob,
        )
        child_edges.add(edge)
        nodes.append(node)
        return node

    if tree_mode not in ("best_first", "root_leaf", "spine_leaf"):
        raise ValueError(f"unsupported DDTree build mode: {tree_mode}")

    if tree_mode in ("root_leaf", "spine_leaf"):
        spine_indices = [0]
        cursor = 0
        while len(nodes) - 1 < budget and nodes[cursor].depth < len(candidates):
            cursor = add_child(cursor, candidates[nodes[cursor].depth][0]).index
            spine_indices.append(cursor)

        order = count()
        heap: list[tuple[float, int, int, DraftCandidate]] = []
        for depth_idx, depth_candidates in enumerate(candidates):
            if depth_idx >= len(spine_indices):
                break
            parent_index = spine_indices[depth_idx]
            parent = nodes[parent_index]
            for candidate in depth_candidates[1:]:
                edge = (parent_index, parent.depth + 1, candidate.token_id)
                if edge in child_edges:
                    continue
                score = parent.score + candidate.logprob
                heapq.heappush(heap, (-score, next(order), parent_index, candidate))

        while len(nodes) - 1 < budget and heap:
            _, _, parent_index, candidate = heapq.heappop(heap)
            parent = nodes[parent_index]
            edge = (parent_index, parent.depth + 1, candidate.token_id)
            if edge in child_edges:
                continue
            add_child(parent_index, candidate)

        return DDTree(nodes=tuple(nodes))

    if chain_seed:
        cursor = 0
        while len(nodes) - 1 < budget and nodes[cursor].depth < len(candidates):
            cursor = add_child(cursor, candidates[nodes[cursor].depth][0]).index
    elif min_root_branches > 0:
        for candidate in candidates[0][:min(min_root_branches,
                                            len(candidates[0]))]:
            if len(nodes) - 1 >= budget:
                break
            add_child(0, candidate)

    if chain_seed or min_root_branches > 0:
        order = count()
        seeded_heap: list[tuple[float, int, int, DraftCandidate]] = []

        def push_children(parent_index: int) -> None:
            parent = nodes[parent_index]
            if parent.depth >= len(candidates):
                return
            depth = parent.depth + 1
            for candidate in candidates[parent.depth]:
                edge = (parent_index, depth, candidate.token_id)
                if edge in child_edges:
                    continue
                score = parent.score + candidate.logprob
                heapq.heappush(
                    seeded_heap, (-score, next(order), parent_index, candidate))

        for node in tuple(nodes):
            push_children(node.index)

        while len(nodes) - 1 < budget and seeded_heap:
            _, _, parent_index, candidate = heapq.heappop(seeded_heap)
            parent = nodes[parent_index]
            edge = (parent_index, parent.depth + 1, candidate.token_id)
            if edge in child_edges:
                continue
            node = add_child(parent_index, candidate)
            push_children(node.index)

        return DDTree(nodes=tuple(nodes))

    order = count()
    heap: list[tuple[float, int, int, int, int, float]] = []
    first_candidate = candidates[0][0]
    heapq.heappush(
        heap,
        (-first_candidate.logprob, next(order), 0, 0, 0,
         first_candidate.logprob),
    )

    while len(nodes) - 1 < budget and heap:
        _, _, parent_index, depth_index, rank, _ = heapq.heappop(heap)
        if rank >= len(candidates[depth_index]):
            continue
        candidate = candidates[depth_index][rank]
        parent = nodes[parent_index]
        edge = (parent_index, parent.depth + 1, candidate.token_id)
        if edge in child_edges:
            continue

        node = add_child(parent_index, candidate)

        next_rank = rank + 1
        if next_rank < len(candidates[depth_index]):
            sibling = candidates[depth_index][next_rank]
            sibling_score = parent.score + sibling.logprob
            heapq.heappush(
                heap,
                (-sibling_score, next(order), parent_index, depth_index,
                 next_rank, sibling_score),
            )

        next_depth_index = depth_index + 1
        if next_depth_index < len(candidates):
            child = candidates[next_depth_index][0]
            child_score = node.score + child.logprob
            heapq.heappush(
                heap,
                (-child_score, next(order), node.index, next_depth_index, 0,
                 child_score),
            )

    return DDTree(nodes=tuple(nodes))
