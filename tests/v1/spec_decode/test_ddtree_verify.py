# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.v1.spec_decode.ddtree_payload import (
    build_ddtree_payloads_from_logits,
    tree_from_payload,
)
from vllm.v1.spec_decode.ddtree_verify import (
    greedy_verify_payload_from_compact_logits,
    greedy_verify_payload_from_compact_top_tokens,
    make_attention_verifier_inputs,
    metadata_from_payload,
)


def test_payload_rebuilds_tree_and_metadata() -> None:
    logits = torch.full((3, 128), -100.0, dtype=torch.float32)
    logits[0, 11] = 10.0
    logits[1, 21] = 10.0
    logits[2, 31] = 10.0
    payload = build_ddtree_payloads_from_logits(
        logits=logits,
        batch_size=1,
        num_speculative_tokens=3,
        budget=3,
        top_k=1,
        chain_seed=True,
    )[0]

    tree = tree_from_payload(payload)
    metadata = metadata_from_payload(prompt_len=7, payload=payload)

    assert tree.token_ids_for_verifier() == (11, 21, 31)
    assert tree.parent_indices_for_verifier() == (-1, 0, 1)
    assert metadata.tree_token_ids == payload.tree_token_ids
    assert metadata.compact_logits_indices == (6, 7, 8, 9)


def test_greedy_verify_payload_full_accept() -> None:
    logits = torch.full((3, 128), -100.0, dtype=torch.float32)
    logits[0, 11] = 10.0
    logits[1, 21] = 10.0
    logits[2, 31] = 10.0
    payload = build_ddtree_payloads_from_logits(
        logits=logits,
        batch_size=1,
        num_speculative_tokens=3,
        budget=3,
        top_k=1,
        chain_seed=True,
    )[0]

    compact_logits = torch.full((payload.num_tree_nodes + 1, 128), -100.0)
    compact_logits[0, 11] = 1.0
    compact_logits[1, 21] = 1.0
    compact_logits[2, 31] = 1.0
    compact_logits[3, 99] = 1.0

    result = greedy_verify_payload_from_compact_logits(
        payload=payload,
        compact_logits=compact_logits,
    )

    assert result.accepted_node_indices == (0, 1, 2, 3)
    assert result.accepted_token_ids == (11, 21, 31)
    assert result.bonus_token_id == 99
    assert result.output_token_ids == (11, 21, 31, 99)


def test_greedy_verify_payload_from_top_tokens_full_accept() -> None:
    logits = torch.full((3, 128), -100.0, dtype=torch.float32)
    logits[0, 11] = 10.0
    logits[1, 21] = 10.0
    logits[2, 31] = 10.0
    payload = build_ddtree_payloads_from_logits(
        logits=logits,
        batch_size=1,
        num_speculative_tokens=3,
        budget=3,
        top_k=1,
        chain_seed=True,
    )[0]

    result = greedy_verify_payload_from_compact_top_tokens(
        payload=payload,
        compact_top_tokens=torch.tensor([11, 21, 31, 99], dtype=torch.int64),
    )

    assert result.accepted_node_indices == (0, 1, 2, 3)
    assert result.accepted_token_ids == (11, 21, 31)
    assert result.bonus_token_id == 99
    assert result.output_token_ids == (11, 21, 31, 99)


def test_greedy_verify_payload_accepts_sibling_branch() -> None:
    logits = torch.full((3, 128), -100.0, dtype=torch.float32)
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
    tree = tree_from_payload(payload)
    sibling_node = tree.child_by_token(0)[12]

    compact_logits = torch.full((payload.num_tree_nodes + 1, 128), -100.0)
    compact_logits[0, 12] = 1.0
    compact_logits[sibling_node, 99] = 1.0

    result = greedy_verify_payload_from_compact_logits(
        payload=payload,
        compact_logits=compact_logits,
    )

    assert result.accepted_node_indices == (0, sibling_node)
    assert result.accepted_token_ids == (12,)
    assert result.bonus_token_id == 99
    assert result.output_token_ids == (12, 99)


def test_make_attention_verifier_inputs_hides_siblings() -> None:
    logits = torch.full((3, 128), -100.0, dtype=torch.float32)
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
    prompt_token_ids = torch.tensor([1, 2, 3, 4], dtype=torch.int64)

    inputs = make_attention_verifier_inputs(
        prompt_token_ids=prompt_token_ids,
        payload=payload,
        mask_dtype=torch.float32,
    )
    tree = tree_from_payload(payload)
    node_11 = tree.child_by_token(0)[11]
    node_12 = tree.child_by_token(0)[12]
    node_21 = tree.child_by_token(node_11)[21]
    row_child = prompt_token_ids.numel() + node_21 - 1
    col_sibling = prompt_token_ids.numel() + node_12 - 1

    assert inputs.input_ids.tolist() == [1, 2, 3, 4] + list(payload.tree_token_ids)
    assert inputs.position_ids.shape == inputs.input_ids.shape
    assert inputs.compact_logits_indices.tolist()[0] == 3
    assert inputs.attention_mask.shape == (
        1,
        1,
        prompt_token_ids.numel() + payload.num_tree_nodes,
        prompt_token_ids.numel() + payload.num_tree_nodes,
    )
    assert inputs.attention_mask[0, 0, row_child, col_sibling].item() < 0
