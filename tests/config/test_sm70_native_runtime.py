# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm._sm70.runtime import (
    BoundNativeCall,
    NativeRuntimeOwner,
    _active_owner,
    bind_native_runtime,
)
from vllm.config import KernelConfig, set_current_vllm_config
from vllm.runtime_resources import release_runtime_resources


@pytest.fixture
def fake_handles(monkeypatch):
    created = []

    class Handle:
        def __init__(self):
            self.policies = {}
            self.depth = 0
            self.closed = False
            created.append(self)

        def bind(self, slot, token):
            assert self.policies.setdefault(slot, token) == token

        def enter(self):
            assert not self.closed
            self.depth += 1

        def exit(self):
            self.depth -= 1
            assert self.depth >= 0

        def close(self):
            assert self.depth == 0
            self.closed = True

    for name in ("_C", "_moe_C"):
        monkeypatch.setattr(
            torch.ops, name, SimpleNamespace(sm70_native_runtime_abi=lambda: 1)
        )
        monkeypatch.setattr(
            torch.classes, name, SimpleNamespace(Sm70NativeRuntime=Handle)
        )
    return created


def test_engine_native_slots_survive_rebinding_and_owner_exception(fake_handles):
    engines, owners, slots = [], [], []
    for trace in (False, True):
        cfg = SimpleNamespace(kernel_config=KernelConfig())
        native = cfg.kernel_config.sm70_fp8.native
        native.tm_gemm_trace = trace
        native.resolve("fp8")
        with set_current_vllm_config(cfg):
            owner = bind_native_runtime()
            assert bind_native_runtime() is owner
            slots.append(owner.bind(native.values, f"diagnostics-{trace}"))
        engines.append(cfg)
        owners.append(owner)
    assert slots[0] == slots[1] == "sm70:slot:kernel_config.sm70_fp8.native"
    assert (
        owners[0].handles[0].policies[slots[0]]
        != owners[1].handles[0].policies[slots[1]]
    )
    assert (
        engines[0].kernel_config.compute_hash()
        == engines[1].kernel_config.compute_hash()
    )
    with owners[0].activate():
        assert BoundNativeCall(lambda: _active_owner.get(), owners[1])() is owners[1]
        assert _active_owner.get() is owners[0]
        with pytest.raises(ValueError, match="launch failed"), owners[1].activate():
            raise ValueError("launch failed")
        assert _active_owner.get() is owners[0]
    assert _active_owner.get() is None
    release_runtime_resources(engines[0])
    assert owners[0].closed and not owners[1].closed
    with pytest.raises(RuntimeError, match="closed"), owners[0].activate():
        pass
    assert BoundNativeCall(lambda: 7, owners[1])() == 7
    assert all(handle.depth == 0 for handle in fake_handles)


def test_native_runtime_requires_both_binary_owners(monkeypatch):
    monkeypatch.setattr(
        torch.ops, "_C", SimpleNamespace(sm70_native_runtime_abi=lambda: 1)
    )
    monkeypatch.setattr(torch.ops, "_moe_C", SimpleNamespace())
    with pytest.raises(RuntimeError, match="runtime ABI 1"):
        NativeRuntimeOwner()


@pytest.mark.parametrize("shared", [False, True])
def test_native_owner_probes_shared_tls_without_assuming_loader_behavior(
    monkeypatch, fake_handles, shared
):
    monkeypatch.setattr(
        torch.ops._C,
        "sm70_native_runtime_context_id",
        lambda: fake_handles[0].depth,
        raising=False,
    )
    monkeypatch.setattr(
        torch.ops._moe_C,
        "sm70_native_runtime_context_id",
        lambda: fake_handles[0].depth if shared else 0,
        raising=False,
    )
    owner = NativeRuntimeOwner()
    assert len(owner.handles) == (1 if shared else 2)
    assert fake_handles[1].closed == shared
    assert all(handle.depth == 0 for handle in fake_handles)
    with owner.activate():
        assert all(handle.depth == 1 for handle in owner.handles)
    owner.close()
    assert all(handle.closed and handle.depth == 0 for handle in fake_handles)
