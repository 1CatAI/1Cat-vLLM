# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.models.qwen4_exp.nvidia import sm70_mtp_fc as route


@pytest.mark.parametrize("rows", [1, 4, 5])
def test_local_residual_moves_gather_without_changing_fp16_bytes(monkeypatch, rows):
    torch.manual_seed(32)
    layers = []
    for _ in range(2):
        layer = object.__new__(ColumnParallelLinear)
        nn.Module.__init__(layer)
        layer.weight = nn.Parameter(
            torch.randn(640, 2560, dtype=torch.float16) * 0.03, requires_grad=False
        )
        layer.bias = None
        layer.quant_method = UnquantizedLinearMethod()
        layers.append(layer)
    model = SimpleNamespace(
        fc_embedding=layers[0],
        fc_hidden=layers[1],
        pre_fc_norm_embedding=nn.Identity(),
        pre_fc_norm_hidden=nn.Identity(),
    )
    e = torch.randn(rows, 2560, dtype=torch.float16)
    h = torch.randn(rows, 4, 2560, dtype=torch.float16)
    calls = []

    def gather(x):
        calls.append(x.shape)
        return torch.cat([x] * 4, dim=-1)

    monkeypatch.setattr(route, "_can_combine_fc", lambda *args: True)
    monkeypatch.setattr(route, "tensor_model_parallel_all_gather", gather)
    actual = route.maybe_combine_fc(model, e, h.flatten(-2))
    assert len(calls) == 1
    el = layers[0].quant_method.apply(layers[0], e)
    hl = layers[1].quant_method.apply(layers[1], h)
    expected = (gather(el).unsqueeze(-2) + gather(hl)).flatten(-2)
    assert torch.equal(actual, expected)


def test_cpu_and_large_prefill_retain_normal_projection_route(monkeypatch):
    monkeypatch.setattr(route, "is_sm70_decode_graph_compiling", lambda: False)
    assert (
        route.maybe_combine_fc(
            SimpleNamespace(), torch.empty(1632, 2560), torch.empty(1632, 10240)
        )
        is None
    )
