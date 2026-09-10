# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
from torch import nn

from vllm.model_executor.models.minimax_h3.initialization import (
    load_without_random_parameter_init,
)


def test_only_checkpoint_replaced_parameters_skip_random_fills():
    original = nn.init.kaiming_uniform_
    state = {"weight": torch.ones(4, 3), "bias": torch.full((4,), 2.0)}

    def factory():
        model = nn.Linear(3, 4)
        constant = torch.empty(5)
        nn.init.uniform_(constant, 3, 3)
        model.register_buffer("constant", constant, persistent=False)
        model.load_state_dict(state, strict=True)
        return model

    model = load_without_random_parameter_init(factory)
    assert nn.init.kaiming_uniform_ is original
    assert torch.equal(model.weight, state["weight"])
    assert torch.equal(model.constant, torch.full((5,), 3.0))
    assert torch.equal(model(torch.ones(1, 3)), torch.full((1, 4), 5.0))


def test_partial_checkpoint_retries_with_normal_initialization():
    calls = []

    def factory():
        calls.append(True)
        model = nn.Linear(3, 4)
        model.load_state_dict({"weight": torch.ones(4, 3)}, strict=False)
        return model

    model = load_without_random_parameter_init(factory)
    assert len(calls) == 2
    assert torch.isfinite(model.bias).all()
    assert torch.all(model.bias.abs() <= 1 / 3**0.5)


def test_initializer_and_load_hooks_are_restored_on_error():
    initialize, load = nn.init.uniform_, nn.Module.load_state_dict

    def factory():
        nn.Linear(3, 4).load_state_dict({}, strict=True)

    with pytest.raises(RuntimeError):
        load_without_random_parameter_init(factory)
    assert nn.init.uniform_ is initialize
    assert nn.Module.load_state_dict is load


def test_assigning_checkpoint_storage_is_supported():
    def factory():
        model = nn.Linear(3, 4, bias=False)
        model.load_state_dict({"weight": torch.ones(4, 3)}, assign=True)
        return model

    model = load_without_random_parameter_init(factory)
    assert torch.equal(model.weight, torch.ones(4, 3))
