# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
from torch import nn

from vllm.model_executor.models.minimax_h3.initialization import (
    load_with_checkpoint_storage,
)


def test_checkpoint_values_and_nonpersistent_buffers_are_preserved():
    original = nn.init.kaiming_uniform_
    state = {"weight": torch.ones(4, 3), "bias": torch.full((4,), 2.0)}

    def factory():
        model = nn.Linear(3, 4)
        constant = torch.empty(5)
        nn.init.uniform_(constant, 3, 3)
        model.register_buffer("constant", constant, persistent=False)
        model.load_state_dict(state, strict=True)
        return model

    model = load_with_checkpoint_storage(factory)
    assert nn.init.kaiming_uniform_ is original
    assert torch.equal(model.weight, state["weight"])
    assert torch.equal(model.constant, torch.full((5,), 3.0))
    assert torch.equal(model(torch.ones(1, 3)), torch.full((1, 4), 5.0))


def test_partial_checkpoint_keeps_normal_initialization():
    calls = []

    def factory():
        calls.append(True)
        model = nn.Linear(3, 4)
        model.load_state_dict({"weight": torch.ones(4, 3)}, strict=False)
        return model

    model = load_with_checkpoint_storage(factory)
    assert len(calls) == 1
    assert torch.isfinite(model.bias).all()
    assert torch.all(model.bias.abs() <= 1 / 3**0.5)


def test_initializer_and_load_hooks_are_restored_on_error():
    initialize, load = nn.init.uniform_, nn.Module.load_state_dict

    def factory():
        nn.Linear(3, 4).load_state_dict({}, strict=True)

    with pytest.raises(RuntimeError):
        load_with_checkpoint_storage(factory)
    assert nn.init.uniform_ is initialize
    assert nn.Module.load_state_dict is load


def test_assigning_checkpoint_storage_is_supported():
    def factory():
        model = nn.Linear(3, 4, bias=False)
        model.load_state_dict({"weight": torch.ones(4, 3)}, assign=True)
        return model

    model = load_with_checkpoint_storage(factory)
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

    model = load_with_checkpoint_storage(factory)
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

    model = load_with_checkpoint_storage(factory)
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

    model = load_with_checkpoint_storage(factory)
    assert model.first.weight is model.second.weight
    assert torch.equal(model.first.weight, torch.full((4, 3), 2.0))


def test_assignment_does_not_introduce_new_parameter_aliases():
    def factory():
        model = nn.Sequential(nn.Linear(3, 4, bias=False), nn.Linear(3, 4, bias=False))
        weight = torch.ones(4, 3)
        model.load_state_dict({"0.weight": weight, "1.weight": weight})
        return model

    model = load_with_checkpoint_storage(factory)
    assert model[0].weight.data_ptr() != model[1].weight.data_ptr()


def test_storage_marker_expires_after_parameter_conversion():
    from vllm.model_executor.models.minimax_h3.initialization import (
        uses_assigned_checkpoint_storage,
    )

    def factory():
        model = nn.Linear(3, 4, bias=False)
        model.load_state_dict({"weight": torch.ones(4, 3)})
        return model

    model = load_with_checkpoint_storage(factory)
    assert uses_assigned_checkpoint_storage(model)
    model.double()
    assert not uses_assigned_checkpoint_storage(model)


def test_replica_identity_requires_all_workers_and_rejects_replacement(monkeypatch):
    from vllm.model_executor.models.minimax_h3 import vae

    monkeypatch.setattr(vae.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(vae.dist, "get_world_size", lambda: 2)
    values: list[tuple[str, int] | None] = [("same", 1), ("same", 1)]
    monkeypatch.setattr(
        vae.dist,
        "all_gather_object",
        lambda result, value: result.__setitem__(slice(None), values),
    )
    assert vae._same_checkpoint_replicas(values[0])
    values[1] = None
    assert not vae._same_checkpoint_replicas(values[0])
    values[1] = ("replacement", 2)
    with pytest.raises(ValueError, match="changed between"):
        vae._same_checkpoint_replicas(values[0])


@pytest.mark.parametrize("complete", [False, True])
def test_random_buffers_and_rng_match_ordinary_loading(complete):
    state = {"weight": torch.ones(4, 3)}
    if complete:
        state["bias"] = torch.ones(4)

    def factory():
        model = nn.Linear(3, 4)
        model.register_buffer("random_buffer", torch.rand(4), persistent=False)
        model.load_state_dict(state, strict=complete)
        return model

    torch.manual_seed(19)
    expected = factory()
    expected_rng = torch.get_rng_state()
    torch.manual_seed(19)
    actual = load_with_checkpoint_storage(factory)
    torch.testing.assert_close(
        actual.random_buffer, expected.random_buffer, rtol=0, atol=0
    )
    torch.testing.assert_close(actual.bias, expected.bias, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), expected_rng)
