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
        layer._layer_trace = SimpleNamespace(unquant_debug=False)
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


@pytest.mark.parametrize("rows", [0, 1, 7, 8, 9])
def test_prefill_fc_owns_rows_and_only_gathers_weights(monkeypatch, rows):
    from vllm.models.qwen4_exp.nvidia import sm70_hc_prefill_peer as peer

    torch.manual_seed(41)
    weights = [torch.randn(2560, 2560, dtype=torch.half) * 0.01 for _ in range(2)]
    e = torch.randn(rows, 2560, dtype=torch.half)
    h = torch.randn(rows, 10240, dtype=torch.half)
    before = h.clone()
    expected = (
        torch.nn.functional.linear(e, weights[0]).unsqueeze(-2)
        + torch.nn.functional.linear(h.view(rows, 4, 2560), weights[1])
    ).flatten(-2)
    local_rows = (rows + 3) // 4
    expected = torch.nn.functional.pad(expected, (0, 0, 0, 4 * local_rows - rows))
    for rank in range(4):
        layers = []
        for weight in weights:
            layer = object.__new__(ColumnParallelLinear)
            nn.Module.__init__(layer)
            layer.weight = nn.Parameter(weight[rank * 640 : (rank + 1) * 640])
            layer.bias = None
            layer.tp_size, layer.gather_output, layer.return_bias = 4, True, False
            layer.quant_method = UnquantizedLinearMethod()
            layers.append(layer)
        model = SimpleNamespace(
            fc_embedding=layers[0],
            fc_hidden=layers[1],
            pre_fc_norm_embedding=nn.Identity(),
            pre_fc_norm_hidden=nn.Identity(),
        )
        calls: list[torch.Tensor] = []

        def gather(x, dim, calls=calls):
            assert dim == 0 and x.shape == (640, 2560)
            calls.append(x)
            return weights[len(calls) - 1]

        def shard(x, rank=rank):
            padded = torch.nn.functional.pad(x, (0, 0, 0, 4 * local_rows - rows))
            return padded[rank * local_rows : (rank + 1) * local_rows].clone()

        monkeypatch.setattr(
            route, "get_tp_group", lambda: SimpleNamespace(all_gather=gather)
        )
        monkeypatch.setattr(peer, "hc_sequence_shard_rows", shard)
        actual = route.maybe_sequence_rows_fc(model, e, h)
        assert len(calls) == 2
        torch.testing.assert_close(
            actual,
            expected[rank * local_rows : (rank + 1) * local_rows],
            rtol=0,
            atol=0,
        )
        assert torch.equal(h, before)
        layers[0].gather_output = False
        assert route.maybe_sequence_rows_fc(model, e, h) is None
