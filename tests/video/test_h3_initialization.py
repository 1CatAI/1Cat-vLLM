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


def test_matching_checkpoint_is_assigned_without_a_second_copy(tmp_path):
    from safetensors.torch import load_file, save_file

    path = tmp_path / "weights.safetensors"
    save_file({"weight": torch.arange(12, dtype=torch.float32).reshape(4, 3)}, path)
    original = path.read_bytes()
    state = load_file(path)

    def factory():
        model = nn.Linear(3, 4, bias=False)
        model.load_state_dict(state)
        return model

    model = load_without_random_parameter_init(factory)
    assert model.weight.data_ptr() == state["weight"].data_ptr()
    with torch.no_grad():
        model.weight.add_(1)
    assert path.read_bytes() == original  # safetensors CPU mappings are private.


def test_dtype_conversion_keeps_original_copy_semantics():
    state = {"weight": torch.arange(12, dtype=torch.float16).reshape(4, 3)}

    def factory():
        model = nn.Linear(3, 4, bias=False, dtype=torch.float32)
        model.load_state_dict(state)
        return model

    model = load_without_random_parameter_init(factory)
    assert model.weight.dtype == torch.float32
    assert model.weight.data_ptr() != state["weight"].data_ptr()
    assert torch.equal(model.weight, state["weight"].float())


def test_tied_parameters_are_not_detached_by_assignment():
    def factory():
        model = nn.Module()
        model.first = nn.Linear(3, 4, bias=False)
        model.second = nn.Linear(3, 4, bias=False)
        model.second.weight = model.first.weight
        model.load_state_dict(
            {"first.weight": torch.ones(4, 3), "second.weight": torch.full((4, 3), 2.0)}
        )
        return model

    model = load_without_random_parameter_init(factory)
    assert model.first.weight is model.second.weight
    assert torch.equal(model.first.weight, torch.full((4, 3), 2.0))


def test_assignment_does_not_introduce_new_parameter_aliases():
    def factory():
        model = nn.Sequential(nn.Linear(3, 4, bias=False), nn.Linear(3, 4, bias=False))
        weight = torch.ones(4, 3)
        model.load_state_dict({"0.weight": weight, "1.weight": weight})
        return model

    model = load_without_random_parameter_init(factory)
    assert model[0].weight.data_ptr() != model[1].weight.data_ptr()
