# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DDTree metadata helpers.

These helpers define the compact verifier-row conventions used by the local
DDTree tests. They are not wired into the scheduler or sampler yet.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm.v1.spec_decode.ddtree_tree import (
    DDTree,
    GreedyDDTreeWalk,
    greedy_tree_walk,
)


@dataclass(frozen=True)
class DDTreeVerifierMetadata:
    """Verifier metadata for one flattened DDTree.

    Compact logit rows are:

    - row 0: root logits from the current committed token.
    - rows 1..N: logits after each non-root tree node.

    A non-root node's token is verified by its parent's compact row.
    """

    prompt_len: int
    tree_token_ids: tuple[int, ...]
    parent_indices: tuple[int, ...]
    node_depths: tuple[int, ...]
    tree_position_ids: tuple[int, ...]
    compact_logits_indices: tuple[int, ...]
    edge_parent_compact_indices: tuple[int, ...]
    node_compact_indices: tuple[int, ...]

    @property
    def num_tree_nodes(self) -> int:
        return len(self.tree_token_ids)

    @classmethod
    def from_tree(cls, *, prompt_len: int, tree: DDTree) -> DDTreeVerifierMetadata:
        if prompt_len < 1:
            raise ValueError("prompt_len must be >= 1")

        tree_token_ids = tree.token_ids_for_verifier()
        parent_indices = tree.parent_indices_for_verifier()
        node_depths = tree.node_depths_for_verifier()
        tree_position_ids = tuple(prompt_len + depth - 1
                                  for depth in node_depths)
        compact_logits_indices = (prompt_len - 1,) + tuple(
            prompt_len + offset for offset in range(len(tree_token_ids)))

        edge_parent_compact_indices: list[int] = []
        node_compact_indices: list[int] = []
        for node in tree.non_root_nodes:
            node_compact_indices.append(node.index)
            if node.parent_index is None or node.parent_index == 0:
                edge_parent_compact_indices.append(0)
            else:
                edge_parent_compact_indices.append(node.parent_index)

        return cls(
            prompt_len=prompt_len,
            tree_token_ids=tree_token_ids,
            parent_indices=parent_indices,
            node_depths=node_depths,
            tree_position_ids=tree_position_ids,
            compact_logits_indices=compact_logits_indices,
            edge_parent_compact_indices=tuple(edge_parent_compact_indices),
            node_compact_indices=tuple(node_compact_indices),
        )

    def all_position_ids(self) -> tuple[int, ...]:
        return tuple(range(self.prompt_len)) + self.tree_position_ids


@dataclass(frozen=True)
class BatchedDDTreeVerifierMetadata:
    prompt_lens: tuple[int, ...]
    tree_token_ids: torch.Tensor
    parent_indices: torch.Tensor
    node_depths: torch.Tensor
    position_ids: torch.Tensor
    cu_num_tree_nodes: torch.Tensor
    compact_logits_indices: torch.Tensor
    edge_parent_compact_indices: torch.Tensor
    node_compact_indices: torch.Tensor


def _offset_parent_indices(
    parent_indices: tuple[int, ...],
    tree_start: int,
) -> list[int]:
    return [parent if parent < 0 else parent + tree_start
            for parent in parent_indices]


def make_prefill_tree_attention_mask(
    *,
    prompt_len: int,
    tree: DDTree,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build a dense prompt-plus-tree ancestor mask for correctness tests."""

    total_len = prompt_len + len(tree.non_root_nodes)
    visible = torch.zeros((total_len, total_len), device=device, dtype=torch.bool)

    for row in range(prompt_len):
        visible[row, :row + 1] = True

    for node in tree.non_root_nodes:
        row = prompt_len + node.index - 1
        visible[row, :prompt_len] = True
        for ancestor_index in tree.ancestor_indices(node.index):
            if ancestor_index == 0:
                continue
            col = prompt_len + ancestor_index - 1
            visible[row, col] = True

    mask = torch.full((total_len, total_len),
                      torch.finfo(dtype).min,
                      device=device,
                      dtype=dtype)
    mask.masked_fill_(visible, 0)
    return mask.unsqueeze(0).unsqueeze(0)


def make_batched_metadata(
    *,
    prompt_lens: list[int],
    trees: list[DDTree],
    device: torch.device,
) -> BatchedDDTreeVerifierMetadata:
    if len(prompt_lens) != len(trees):
        raise ValueError("prompt_lens and trees must have the same length")
    if not trees:
        raise ValueError("at least one tree is required")

    per_request = [
        DDTreeVerifierMetadata.from_tree(prompt_len=prompt_len, tree=tree)
        for prompt_len, tree in zip(prompt_lens, trees, strict=True)
    ]
    cu_nodes: list[int] = []
    total = 0
    for metadata in per_request:
        total += metadata.num_tree_nodes
        cu_nodes.append(total)

    max_total_len = max(
        prompt_len + len(tree.non_root_nodes)
        for prompt_len, tree in zip(prompt_lens, trees, strict=True))
    tree_token_ids: list[int] = []
    parent_indices: list[int] = []
    node_depths: list[int] = []
    compact_logits_indices: list[int] = []
    edge_parent_compact_indices: list[int] = []
    node_compact_indices: list[int] = []
    position_rows: list[list[int]] = []

    seq_start = 0
    tree_start = 0
    compact_start = 0
    for metadata in per_request:
        tree_token_ids.extend(metadata.tree_token_ids)
        parent_indices.extend(
            _offset_parent_indices(metadata.parent_indices, tree_start))
        node_depths.extend(metadata.node_depths)
        compact_logits_indices.extend(
            seq_start + index for index in metadata.compact_logits_indices)
        edge_parent_compact_indices.extend(
            compact_start + index
            for index in metadata.edge_parent_compact_indices)
        node_compact_indices.extend(
            compact_start + index for index in metadata.node_compact_indices)

        row = list(metadata.all_position_ids())
        row.extend([0] * (max_total_len - len(row)))
        position_rows.append(row)

        seq_start += metadata.prompt_len + metadata.num_tree_nodes
        tree_start += metadata.num_tree_nodes
        compact_start += metadata.num_tree_nodes + 1

    return BatchedDDTreeVerifierMetadata(
        prompt_lens=tuple(prompt_lens),
        tree_token_ids=torch.tensor(tree_token_ids, device=device,
                                    dtype=torch.int32),
        parent_indices=torch.tensor(parent_indices, device=device,
                                    dtype=torch.int32),
        node_depths=torch.tensor(node_depths, device=device, dtype=torch.int32),
        position_ids=torch.tensor(position_rows, device=device,
                                  dtype=torch.long),
        cu_num_tree_nodes=torch.tensor(cu_nodes, device=device,
                                       dtype=torch.int32),
        compact_logits_indices=torch.tensor(compact_logits_indices,
                                            device=device,
                                            dtype=torch.int32),
        edge_parent_compact_indices=torch.tensor(edge_parent_compact_indices,
                                                 device=device,
                                                 dtype=torch.int32),
        node_compact_indices=torch.tensor(node_compact_indices,
                                          device=device,
                                          dtype=torch.int32),
    )


def greedy_sample_from_compact_logits(
    *,
    tree: DDTree,
    compact_logits: torch.Tensor,
) -> GreedyDDTreeWalk:
    """Greedy tree sampler over compact root-plus-node logits."""

    if compact_logits.ndim != 2:
        raise ValueError("compact_logits must have shape [rows, vocab]")
    expected_rows = len(tree.non_root_nodes) + 1
    if compact_logits.shape[0] != expected_rows:
        raise ValueError(
            f"compact_logits row mismatch: expected {expected_rows}, "
            f"got {compact_logits.shape[0]}")

    target_argmax = compact_logits.argmax(dim=-1)
    path_to_compact_index: dict[tuple[int, ...], int] = {(): 0}
    for node in tree.non_root_nodes:
        path_to_compact_index[tree.path_token_ids(node.index)] = node.index

    def next_token_for_path(path: tuple[int, ...]) -> int:
        return int(target_argmax[path_to_compact_index[path]].item())

    return greedy_tree_walk(tree, next_token_for_path)


def greedy_sample_from_compact_top_tokens(
    *,
    tree: DDTree,
    compact_top_tokens: torch.Tensor,
) -> GreedyDDTreeWalk:
    """Greedy tree sampler over compact root-plus-node argmax tokens."""

    if compact_top_tokens.ndim != 1:
        raise ValueError("compact_top_tokens must have shape [rows]")
    expected_rows = len(tree.non_root_nodes) + 1
    if compact_top_tokens.shape[0] != expected_rows:
        raise ValueError(
            f"compact_top_tokens row mismatch: expected {expected_rows}, "
            f"got {compact_top_tokens.shape[0]}")

    target_tokens = compact_top_tokens.detach().cpu().tolist()
    path_to_compact_index: dict[tuple[int, ...], int] = {(): 0}
    for node in tree.non_root_nodes:
        path_to_compact_index[tree.path_token_ids(node.index)] = node.index

    def next_token_for_path(path: tuple[int, ...]) -> int:
        return int(target_tokens[path_to_compact_index[path]])

    return greedy_tree_walk(tree, next_token_for_path)
