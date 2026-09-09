# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models.minimax_h3 import collectives
from vllm.model_executor.models.minimax_h3.config import H3Config, H3InputError


@pytest.mark.parametrize("tp", [1, 2, 4])
@pytest.mark.parametrize("partition", ["fl2va", "ref2va"])
def test_explicit_peer_option_keeps_tp_and_partition_compatibility(tp, partition):
    config = H3Config(
        tensor_parallel_size=tp,
        partition=partition,
        residual_sequence_parallel=True,
        residual_reduction="peer",
    )
    assert config.residual_reduction == "peer"
    assert H3Config().residual_reduction == "native"


@pytest.mark.parametrize("budget", [0, -1, True, float("nan"), float("inf"), 1e308])
def test_reject_invalid_communication_budget(budget):
    with pytest.raises(H3InputError):
        H3Config(residual_reduction_memory_gib=budget)


def test_peer_option_requires_residual_rows():
    with pytest.raises(H3InputError, match="residual sequence"):
        H3Config(residual_reduction="peer")


def test_reuse_request_accounting_shape_eviction_and_budget_fallback(monkeypatch):
    plans = []

    class Plan:
        raw_ipc_bytes = 64

        @staticmethod
        def required_memory_bytes(shape):
            return shape[0] * shape[1] * 4

        def __init__(self, group, shape, *, memory_budget_bytes):
            self.shape = shape
            self.closed = False
            plans.append(self)

        def reduce(self, value):
            return (value * 4).chunk(4)[1]

        def close(self):
            self.closed = True

    monkeypatch.setattr(collectives, "SM70ExactRowReductionPlan", Plan)
    group = SimpleNamespace(world_size=4, rank_in_group=1, all_reduce=lambda x: x * 4)
    owner = collectives.H3ResidualReduction(group, memory_budget_bytes=100)
    value = torch.arange(12, dtype=torch.float32).view(4, 3)
    expected = (value * 4).chunk(4)[1]
    assert torch.equal(owner.reduce(value), expected)
    owner.begin_request()
    assert owner.snapshot()["peer_calls"] == 0
    assert owner.snapshot()["raw_ipc_peak_bytes"] == 64
    assert torch.equal(owner.reduce(value), expected)
    assert len(plans) == 1
    assert owner.snapshot()["setup_seconds"] == 0
    large = torch.ones(16, 3)
    assert torch.equal(owner.reduce(large), torch.full((4, 3), 4.0))
    assert plans[0].closed
    assert owner.snapshot()["native_calls"] == 1
    assert owner.snapshot()["fallback_reason"]
    owner.begin_request()
    assert owner.snapshot()["raw_ipc_peak_bytes"] == 0
    owner.close()


def test_tp2_uses_ordinary_reduction_without_plan(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("TP2 must not construct a TP4 peer plan")

    monkeypatch.setattr(collectives, "SM70ExactRowReductionPlan", unexpected)
    group = SimpleNamespace(world_size=2, rank_in_group=1, all_reduce=lambda x: x * 2)
    owner = collectives.H3ResidualReduction(group, memory_budget_bytes=100)
    assert torch.equal(owner.reduce(torch.ones(6, 3)), torch.full((3, 3), 2.0))
    assert owner.snapshot()["native_calls"] == 1
