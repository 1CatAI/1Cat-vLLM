# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The routing-group kernel against its torch reference on the CPU (Triton
interpreter, no GPU needed).

The interpreter runs the kernel's logic (sort keys, tie order, group scatter,
padding lanes), not the GPU's sort/gather lowering; that part is covered by
test_sm70_group_routing.py on CUDA and the CUDA-graph micro-benchmark.

    TRITON_INTERPRET=1 pytest tests/kernels/moe/test_sm70_group_routing_cpu.py
"""

import os
import sys

import pytest

if "triton" in sys.modules and os.environ.get("TRITON_INTERPRET") != "1":
    pytest.skip(
        "needs TRITON_INTERPRET=1 in the environment: triton was imported "
        "before this file could turn the interpreter on",
        allow_module_level=True,
    )
os.environ["TRITON_INTERPRET"] = "1"  # before the kernel is decorated

import torch  # noqa: E402

from vllm.model_executor.layers.fused_moe.sm70 import routing as ROUTING  # noqa: E402


def _assert_same(topk_ids: torch.Tensor) -> None:
    got = ROUTING.group_routing(topk_ids)
    expected = ROUTING.group_routing_reference(topk_ids)
    for name, value, reference in zip(("perm", "gids", "goff"), got, expected):
        assert value.dtype == torch.int32
        assert torch.equal(value, reference), (name, value, reference)


def _random_routing(tokens, top_k, experts, seed) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    scores = torch.rand(tokens, experts, generator=generator)
    return scores.topk(top_k, dim=-1).indices.to(torch.int32)


# (experts, top_k): GLM-5.3-Flash, DeepSeek-V4-Flash, a small-expert model.
_MODEL_SHAPES = [(288, 8), (256, 6), (9, 8)]


@pytest.mark.parametrize("experts,top_k", _MODEL_SHAPES)
@pytest.mark.parametrize("tokens", [1, 2, 3, 5, 6, 8, 15, 16])
def test_random_routing_matches_the_reference(tokens, experts, top_k):
    if tokens * top_k > ROUTING._MAX_KERNEL_SLOTS:
        pytest.skip("beyond the kernel's slot limit")
    for seed in range(6):
        _assert_same(_random_routing(tokens, top_k, experts, seed))


@pytest.mark.parametrize("slots", [1, 2, 3, 7, 8, 33, 64, 127, 128])
def test_many_ties_with_few_experts(slots):
    generator = torch.Generator().manual_seed(slots)
    for experts in (1, 2, 3, 5):
        ids = torch.randint(0, experts, (slots,), generator=generator)
        _assert_same(ids.to(torch.int32).reshape(1, -1))


@pytest.mark.parametrize("slots", [1, 8, 64, 128])
def test_all_slots_on_one_expert(slots):
    for expert in (0, 7, 287):
        _assert_same(torch.full((1, slots), expert, dtype=torch.int32))


@pytest.mark.parametrize("slots", [8, 64, 128])
def test_all_slots_on_distinct_experts_in_reverse_order(slots):
    ids = torch.arange(slots, dtype=torch.int32).flip(0) * 2 + 1
    _assert_same(ids.reshape(1, -1))


def test_experts_at_the_ends_of_the_supported_range():
    ids = torch.tensor([[0, 287, 0, 287, 16_000_000, 0, 16_000_000, 1]])
    _assert_same(ids.to(torch.int32))
    _assert_same(-ids.to(torch.int32))


def test_int64_ids_and_a_slot_count_off_the_power_of_two():
    _assert_same(_random_routing(5, 6, 288, seed=3).to(torch.int64))


def test_more_slots_than_the_kernel_takes_use_the_reference():
    ids = _random_routing(40, 8, 288, seed=5)
    assert ids.numel() > ROUTING._MAX_KERNEL_SLOTS
    _assert_same(ids)
