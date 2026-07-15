# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.v1.spec_decode.ddtree_payload import payload_from_tree
from vllm.v1.spec_decode.ddtree_sampler import (
    greedy_sample_ddtree_payloads,
    greedy_sample_ddtree_payloads_from_top_tokens,
    stochastic_sample_ddtree_payloads,
)
from vllm.v1.spec_decode.ddtree_tree import build_ddtree


def _payload() -> object:
    tree = build_ddtree(
        [
            [(11, 0.0), (12, -0.1)],
            [(21, 0.0), (22, -0.2)],
            [(31, 0.0), (32, -0.3)],
        ],
        budget=5,
        top_k=2,
        chain_seed=True,
    )
    return payload_from_tree(
        tree=tree,
        top1_chain_token_ids=(11, 21, 31),
        flat_draft_token_ids=(11, 21, 31),
        budget=5,
        top_k=2,
        chain_seed=True,
    )


def test_greedy_sample_ddtree_payloads_accepts_branch() -> None:
    payload = _payload()
    assert payload.num_tree_nodes == 5
    logits = torch.full((6, 64), -100.0, dtype=torch.float32)
    logits[0, 12] = 1.0
    logits[4, 21] = 1.0
    logits[5, 41] = 1.0

    result = greedy_sample_ddtree_payloads(
        req_ids=["r0"],
        payload_by_req_id={"r0": payload},
        compact_logits=logits,
        num_draft_tokens=[5],
    )

    assert result is not None
    assert result.sampler_output.sampled_token_ids.tolist() == [[12, 21, 41]]
    assert result.sampler_output.ddtree_accepted_node_indices is not None
    assert result.sampler_output.ddtree_accepted_node_indices.tolist() == [[0, 4, 5]]


def test_greedy_sample_ddtree_payloads_from_top_tokens_accepts_branch() -> None:
    payload = _payload()
    assert payload.num_tree_nodes == 5

    result = greedy_sample_ddtree_payloads_from_top_tokens(
        req_ids=["r0"],
        payload_by_req_id={"r0": payload},
        compact_top_tokens=torch.tensor(
            [
                [12, 0],
                [0, 99],
                [0, 99],
                [0, 99],
                [21, 0],
                [41, 0],
            ]
        ),
        num_draft_tokens=[5],
    )

    assert result is not None
    assert result.sampler_output.sampled_token_ids.tolist() == [[12, 21, 41]]
    assert result.sampler_output.ddtree_accepted_node_indices is not None
    assert result.sampler_output.ddtree_accepted_node_indices.tolist() == [[0, 4, 5]]


def test_greedy_sample_ddtree_payloads_pads_variable_outputs() -> None:
    payload = _payload()
    logits = torch.full((12, 64), -100.0, dtype=torch.float32)
    logits[0, 11] = 1.0
    logits[1, 21] = 1.0
    logits[2, 31] = 1.0
    logits[3, 41] = 1.0
    logits[6, 99 % 64] = 1.0

    result = greedy_sample_ddtree_payloads(
        req_ids=["r0", "r1"],
        payload_by_req_id={"r0": payload, "r1": payload},
        compact_logits=logits,
        num_draft_tokens=[5, 5],
    )

    assert result is not None
    assert result.sampler_output.sampled_token_ids.tolist() == [
        [11, 21, 31, 41],
        [35, -1, -1, -1],
    ]
    assert result.sampler_output.ddtree_accepted_node_indices is not None
    assert result.sampler_output.ddtree_accepted_node_indices.tolist() == [
        [0, 1, 2, 3],
        [0, -1, -1, -1],
    ]


def test_greedy_sample_ddtree_payloads_requires_matching_rows() -> None:
    payload = _payload()
    logits = torch.empty((4, 64), dtype=torch.float32)

    result = greedy_sample_ddtree_payloads(
        req_ids=["r0"],
        payload_by_req_id={"r0": payload},
        compact_logits=logits,
        num_draft_tokens=[3],
    )

    assert result is None


def test_stochastic_sample_ddtree_payloads_accepts_branch_with_top_k() -> None:
    payload = _payload()
    assert payload.num_tree_nodes == 5
    logits = torch.full((6, 64), -100.0, dtype=torch.float32)
    logits[0, 12] = 1.0
    logits[4, 21] = 1.0
    logits[5, 41] = 1.0

    result = stochastic_sample_ddtree_payloads(
        req_ids=["r0"],
        payload_by_req_id={"r0": payload},
        compact_logits=logits,
        num_draft_tokens=[5],
        temperature=torch.tensor([1.0], dtype=torch.float32),
        top_k=torch.tensor([1], dtype=torch.int32),
        top_p=None,
        generators={},
        top_k_cpu=(1,),
    )

    assert result is not None
    assert result.sampler_output.sampled_token_ids.tolist() == [[12, 21, 41]]
    assert result.sampler_output.ddtree_accepted_node_indices is not None
    assert result.sampler_output.ddtree_accepted_node_indices.tolist() == [[0, 4, 5]]
