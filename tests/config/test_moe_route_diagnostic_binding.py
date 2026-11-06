# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Routing diagnostics remain opt-in and require a CUDA gate parameter."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm import sm70_round_cost
from vllm.config.kernel import KernelConfig


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("gate_kind", ["absent", "empty", "cpu", "cuda"])
def test_route_recorder_binding_preserves_admission(monkeypatch, enabled, gate_kind):
    attach = Mock()
    monkeypatch.setattr(sm70_round_cost, "attach_route_recorder", attach)
    device = torch.device("cuda" if gate_kind == "cuda" else "cpu")
    parameters = [] if gate_kind == "empty" else [SimpleNamespace(device=device)]
    gate = (
        None
        if gate_kind == "absent"
        else SimpleNamespace(parameters=Mock(side_effect=lambda: iter(parameters)))
    )
    router = object()
    config = KernelConfig(sm70_round_cost_diagnostics=enabled)
    config.attach_moe_route_recorder(router, gate, "model.layers.0")
    if enabled and gate_kind == "cuda":
        attach.assert_called_once_with(router, "model.layers.0", device)
    else:
        attach.assert_not_called()
    if not enabled and gate is not None:
        gate.parameters.assert_not_called()
