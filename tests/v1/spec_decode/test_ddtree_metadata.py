# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.v1.spec_decode.ddtree_metadata import (
    DDTreeVerifierMetadata,
    greedy_sample_from_compact_logits,
    greedy_sample_from_compact_top_tokens,
    make_batched_metadata,
    make_prefill_tree_attention_mask,
)
from vllm.v1.spec_decode.ddtree_tree import build_ddtree


def make_tree():
    return build_ddtree(
        [
            [(11, -0.1), (12, -0.2)],
            [(21, -0.1), (22, -0.2)],
            [(31, -0.1), (32, -0.2)],
        ],
        budget=5,
        top_k=2,
    )


def test_single_request_metadata_indices() -> None:
    tree = make_tree()
    metadata = DDTreeVerifierMetadata.from_tree(prompt_len=7, tree=tree)

    assert metadata.tree_token_ids[:3] == (11, 21, 31)
    assert metadata.parent_indices[:3] == (-1, 0, 1)
    assert metadata.compact_logits_indices[0] == 6
    assert metadata.compact_logits_indices[1] == 7
    assert metadata.node_compact_indices == tuple(
        range(1, tree.non_root_nodes[-1].index + 1))
    assert metadata.edge_parent_compact_indices[0] == 0
    assert metadata.edge_parent_compact_indices[1] == 1
    assert metadata.tree_position_ids[:3] == (7, 8, 9)


def test_attention_mask_hides_siblings() -> None:
    tree = make_tree()
    prompt_len = 5
    mask = make_prefill_tree_attention_mask(
        prompt_len=prompt_len,
        tree=tree,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )[0, 0]

    node_11 = tree.child_by_token(0)[11]
    node_12 = tree.child_by_token(0)[12]
    node_21 = tree.child_by_token(node_11)[21]
    row_child = prompt_len + node_21 - 1
    col_parent = prompt_len + node_11 - 1
    col_sibling = prompt_len + node_12 - 1

    assert mask[row_child, :prompt_len].eq(0).all()
    assert mask[row_child, col_parent].item() == 0
    assert mask[row_child, col_sibling].item() < 0


def test_batched_metadata_offsets() -> None:
    left = make_tree()
    right = build_ddtree(
        [
            [(41, -0.1), (42, -0.2)],
            [(51, -0.1), (52, -0.2)],
        ],
        budget=3,
        top_k=2,
    )

    batch = make_batched_metadata(
        prompt_lens=[5, 8],
        trees=[left, right],
        device=torch.device("cpu"),
    )

    left_nodes = len(left.non_root_nodes)
    assert batch.cu_num_tree_nodes.tolist() == [left_nodes, left_nodes + 3]
    right_parent_indices = batch.parent_indices[left_nodes:].tolist()
    assert right_parent_indices[0] == -1
    assert right_parent_indices[1] == left_nodes
    assert batch.tree_token_ids[:3].tolist() == [11, 21, 31]


def test_greedy_sample_from_compact_logits() -> None:
    tree = make_tree()
    node_11 = tree.child_by_token(0)[11]
    node_21 = tree.child_by_token(node_11)[21]
    vocab_size = 128
    logits = torch.full((len(tree.non_root_nodes) + 1, vocab_size), -1000.0)
    logits[0, 11] = 1.0
    logits[node_11, 21] = 1.0
    logits[node_21, 99] = 1.0

    walk = greedy_sample_from_compact_logits(tree=tree, compact_logits=logits)

    assert walk.accepted_token_ids == (11, 21)
    assert walk.bonus_token_id == 99
    assert walk.output_token_ids == (11, 21, 99)


def test_greedy_sample_from_compact_top_tokens() -> None:
    tree = make_tree()
    node_11 = tree.child_by_token(0)[11]
    node_21 = tree.child_by_token(node_11)[21]
    top_tokens = torch.zeros(
        len(tree.non_root_nodes) + 1,
        dtype=torch.int64,
    )
    top_tokens[0] = 11
    top_tokens[node_11] = 21
    top_tokens[node_21] = 99

    walk = greedy_sample_from_compact_top_tokens(
        tree=tree,
        compact_top_tokens=top_tokens,
    )

    assert walk.accepted_token_ids == (11, 21)
    assert walk.bonus_token_id == 99
    assert walk.output_token_ids == (11, 21, 99)
