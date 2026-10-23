# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.fused_moe import MoEActivation
from vllm.model_executor.layers.quantization.gguf_moe import GGUFNativeMoEMethod
from vllm.model_executor.layers.quantization.gguf_turbomind_moe import (
    GGUFTurboMindMoEMethod,
)


def fixture(kind, limit, monkeypatch):
    x = torch.tensor([[2.0, -3.0], [-1.0, 4.0]])
    ids = torch.tensor([[1, 0], [0, 1]], dtype=torch.int32)
    probabilities = torch.tensor([[0.75, 0.25], [0.375, 0.625]])
    gate = torch.tensor([[[9.0, -8.0], [-12.0, 6.0]], [[-7.0, 8.0], [9.0, 9.0]]])
    up = torch.tensor([[[11.0, -3.0], [-9.0, 7.0]], [[6.0, -9.0], [-8.0, -5.0]]])
    down = torch.tensor([[[1.0, 2.0], [-1.0, 0.5]], [[-2.0, 1.0], [0.5, 3.0]]])
    layer = SimpleNamespace(
        apply_router_weight_on_input=False,
        activation=MoEActivation.SILU,
        expert_map=None,
        swiglu_limit=limit,
        gguf_w1=gate,
        gguf_w3=up,
        gguf_w2=down,
    )
    cls = GGUFNativeMoEMethod if kind == "native" else GGUFTurboMindMoEMethod
    method = cls.__new__(cls)
    method.num_experts = method.hidden_size = method.intermediate_size = 2
    if kind == "native":
        method.expert_trace_tokens = 0
        method.expert_cache = None
        method.dp4a_admitted = limit is not None
        method.weight_types = dict.fromkeys(("w1", "w3", "w2"), 1)
        method.input_padding = {}
        method.projection_capabilities = dict.fromkeys(
            ("w1", "w3", "w2"), SimpleNamespace(operator="test_clamp_projection")
        )

        def projection(rows, weights, expert_ids, quant_type, n, top_k, tokens):
            values = rows.repeat_interleave(top_k, dim=0)
            return torch.bmm(weights[expert_ids.reshape(-1).long()], values[..., None])[
                ..., 0
            ]

        monkeypatch.setattr(
            torch.ops._C_gguf, "test_clamp_projection", projection, raising=False
        )
    else:
        # Clamped layers must bypass fused activation shortcuts, even when
        # their hardware/shape admission flags would otherwise enable them.
        method.mma_planes = limit is not None
        method.dp4a_batches = [1] if limit is not None else []
        method.small_routing = method.raw_gate_up = False

        def bank(weights):
            return lambda rows, offsets, sorted_ids: torch.bmm(
                weights[sorted_ids.long()], rows[..., None]
            )[..., 0]

        layer.gguf_expert_banks = {"w1": bank(gate), "w3": bank(up), "w2": bank(down)}
    return method, layer, x, ids, probabilities


@pytest.mark.parametrize("kind", ["native", "canonical"])
@pytest.mark.parametrize("limit", [None, 10.0, 0.0])
def test_routed_output_respects_model_clamp(monkeypatch, kind, limit):
    method, layer, x, ids, probabilities = fixture(kind, limit, monkeypatch)
    actual = method.apply(layer, x, probabilities, ids, None, None)
    reference = torch.zeros_like(x)
    for token in range(len(x)):
        for slot, expert in enumerate(ids[token]):
            gate = layer.gguf_w1[expert] @ x[token]
            up = layer.gguf_w3[expert] @ x[token]
            if limit is not None:
                gate = torch.minimum(gate, torch.full_like(gate, limit))
                up = torch.maximum(
                    torch.minimum(up, torch.full_like(up, limit)),
                    torch.full_like(up, -limit),
                )
            output = layer.gguf_w2[expert] @ (F.silu(gate) * up)
            reference[token] += output * probabilities[token, slot]
    torch.testing.assert_close(actual, reference)


def test_clamped_turbomind_cache_fails_before_using_unallocated_raw_banks(monkeypatch):
    method, layer, x, ids, probabilities = fixture("native", 10.0, monkeypatch)
    method.expert_cache = SimpleNamespace(tm=object())
    with pytest.raises(ValueError, match="expert_cache_prefill='mmq'"):
        method.apply(layer, x, probabilities, ids, None, None)
