# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.fused_moe import MoEActivation
from vllm.model_executor.layers.fused_moe.sm70_small_routing import (
    _small_route,
    _small_unroute,
)
from vllm.model_executor.layers.quantization.gguf_turbomind_moe import (
    GGUFTurboMindMoEMethod,
)


@pytest.mark.parametrize("mapped", [False, True])
def test_alignment_integration_preserves_weighted_expert_outputs(monkeypatch, mapped):
    torch.manual_seed(20261004)
    x = torch.randn(2, 2560).half()
    ids = torch.tensor(
        [[17, 0, 1, 1, 4, 9, 2, 8, 3, 7], [3, 3, 17, 4, 1, 9, 8, 2, 6, 7]],
        dtype=torch.int32,
    )
    weights = torch.randn(2, 10).softmax(1)
    calls = []

    def route(*args):
        calls.append("route")
        assert args[1].dtype == torch.int32
        return _small_route(*args)

    def unroute(*args):
        calls.append("unroute")
        return _small_unroute(*args)

    monkeypatch.setattr(torch.ops.vllm, "sm70_small_expert_route", route)
    monkeypatch.setattr(torch.ops.vllm, "sm70_small_expert_unroute", unroute)

    def input_bank(rows, offsets, sorted_ids):
        assert sorted_ids.dtype == torch.int64
        assert offsets.dtype == torch.int32
        return rows[:, :160]

    def output_bank(rows, offsets, sorted_ids):
        return rows.repeat(1, 16)

    expert_map = torch.arange(512) if mapped else None
    if expert_map is not None:
        expert_map[17] = -1
    layer = SimpleNamespace(
        apply_router_weight_on_input=False,
        activation=MoEActivation.SILU,
        expert_map=expert_map,
        gguf_expert_banks={"w1": input_bank, "w3": input_bank, "w2": output_bank},
    )
    method = GGUFTurboMindMoEMethod.__new__(GGUFTurboMindMoEMethod)
    method.num_experts, method.hidden_size = 512, 2560
    method.dp4a_batches = []
    method.raw_gate_up = False
    method.small_routing = False
    reference = method.apply(layer, x, weights, ids, None, None)
    method.small_routing = True
    actual = method.apply(layer, x, weights, ids, None, None)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    assert calls == ([] if mapped else ["route", "unroute"])
