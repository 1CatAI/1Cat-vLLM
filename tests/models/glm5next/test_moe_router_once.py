# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch import nn

from vllm.models.glm5next.nvidia.model import Glm5NextMoE


class _CountingGate(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(4, 4))
        self.calls = 0

    def forward(self, x: torch.Tensor):
        self.calls += 1
        return x @ self.weight.T, None


class _RunnerLikeExperts(nn.Module):
    """Mimics FusedMoE with an internal router: the runner applies the gate
    itself and ignores the router_logits argument."""

    is_internal_router = True

    def __init__(self, gate: nn.Module) -> None:
        super().__init__()
        self.gate = gate

    def forward(self, hidden_states, router_logits, input_ids=None):
        logits, _ = self.gate(hidden_states)
        return logits


def test_router_gemm_runs_once_per_moe_forward():
    moe = Glm5NextMoE.__new__(Glm5NextMoE)
    nn.Module.__init__(moe)
    moe.is_sequence_parallel = False
    moe.gate = _CountingGate()
    moe.experts = _RunnerLikeExperts(moe.gate)

    hidden = torch.randn(3, 4)
    out = moe(hidden)

    assert moe.gate.calls == 1
    assert torch.equal(out, hidden @ moe.gate.weight.T)
