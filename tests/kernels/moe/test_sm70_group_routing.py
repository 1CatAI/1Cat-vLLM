# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The routing-group kernel against the torch reference that defines its result."""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.sm70.routing import (
    group_routing,
    group_routing_reference,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _routing(tokens: int, top_k: int, experts: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    scores = torch.rand(tokens, experts, device="cuda", generator=generator)
    return scores.topk(top_k, dim=-1).indices.to(torch.int32)


def _assert_same(topk_ids: torch.Tensor) -> None:
    got = group_routing(topk_ids)
    expected = group_routing_reference(topk_ids)
    for name, value, reference in zip(("perm", "gids", "goff"), got, expected):
        assert value.dtype == torch.int32
        assert torch.equal(value, reference), name


@pytest.mark.parametrize("tokens", [1, 2, 3, 5, 8, 15, 16])
@pytest.mark.parametrize("experts", [9, 64, 288])
def test_random_routing_matches_the_reference(tokens, experts):
    for seed in range(8):
        _assert_same(_routing(tokens, min(8, experts), experts, seed))


@pytest.mark.parametrize("tokens", [1, 4, 16])
def test_all_slots_on_one_expert(tokens):
    _assert_same(torch.full((tokens, 8), 7, dtype=torch.int32, device="cuda"))


def test_all_slots_on_distinct_experts():
    ids = torch.arange(64, dtype=torch.int32, device="cuda").flip(0).reshape(8, 8)
    _assert_same(ids)


def test_int64_ids_and_a_slot_count_off_the_power_of_two():
    ids = _routing(5, 6, 288, seed=3).to(torch.int64)
    _assert_same(ids)


def test_more_slots_than_the_kernel_takes_use_the_reference():
    ids = _routing(40, 8, 288, seed=5)
    assert ids.numel() > 128
    _assert_same(ids)
