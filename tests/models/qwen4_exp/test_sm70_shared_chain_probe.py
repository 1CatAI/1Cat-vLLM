# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import torch
from torch import nn

from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP
from vllm.models.qwen4_exp.nvidia import sm70_mtp_structural as candidate


def test_shared_probe_preserves_fallback_and_does_not_reduce_tp(monkeypatch):
    class MLP(Qwen2MoeMLP):
        def forward(self, x):
            return x + 1

    layer = MLP.__new__(MLP)
    nn.Module.__init__(layer)
    layer.gate_up_proj = SimpleNamespace(
        _sm70_mtp_shared_packed=torch.empty(10, 160, 2, 32, 8), bias=None
    )
    layer.down_proj = SimpleNamespace(
        weight=torch.empty(2560, 160), bias=None, reduce_results=False
    )
    layer.expert_gate = SimpleNamespace(weight=torch.empty(1, 2560), bias=None)
    monkeypatch.setattr(candidate, "use_sm70_decode_graph_semantics", lambda: True)
    monkeypatch.setattr(
        torch.ops._C, "qwen38_shared_chain_sm70_out", lambda *args: None, raising=False
    )
    assert candidate.prepare_shared_chain_probe(layer) == 1
    assert layer._sm70_shared_down_probe.shape == (160, 2560)
    for rows in (4, 5, 20):
        x = torch.zeros(rows, 2560, dtype=torch.float16)
        assert torch.equal(layer(x), x + 1)  # CPU never enters the CUDA probe.
