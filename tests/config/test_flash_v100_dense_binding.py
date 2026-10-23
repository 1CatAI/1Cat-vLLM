# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dense-LSE loading must bind an engine before returning its native callback."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from tests.config.test_flash_v100_lifecycle import prepare
from vllm.config import set_current_vllm_config
from vllm.config.flash_v100 import FlashV100Options
from vllm.runtime_resources import release_runtime_resources, runtime_resources_for
from vllm.v1.attention.backends.flash_v100 import dense_prefill, ops, runtime
from vllm.v1.attention.backends.mla.prefill import flash_attn as mla_prefill

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def native_package(monkeypatch):
    """Use the real Python runtime/binder with a host-only extension fixture."""
    package_dir = (
        Path(__file__).resolve().parents[2] / "flash-attention-v100/flash_attn_v100"
    )
    package = ModuleType("flash_attn_v100")
    package.__path__ = [str(package_dir)]
    interface = ModuleType("flash_attn_v100.flash_attn_interface")
    created_policies = []

    class PreparedPolicy:
        def __init__(self, inputs):
            self.inputs = tuple(inputs)
            self.observations = ()
            created_policies.append(self)

    def forward(*args, _runtime=None, **kwargs):
        return (_runtime, None, None, None)

    interface.__dict__.update(
        _flash_attn_forward=forward,
        flash_attn_v100_cuda=SimpleNamespace(
            policy_abi_version=1,
            policy_fields=len(prepare()[0].options.native_inputs),
            PreparedPolicy=PreparedPolicy,
        ),
    )
    monkeypatch.setitem(sys.modules, "flash_attn_v100", package)
    monkeypatch.setitem(sys.modules, interface.__name__, interface)
    spec = importlib.util.spec_from_file_location(
        "flash_attn_v100.runtime", package_dir / "runtime.py"
    )
    assert spec is not None and spec.loader is not None
    native_runtime = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, native_runtime)
    spec.loader.exec_module(native_runtime)
    monkeypatch.setattr(ops, "_flash_attn_forward_lse", None)
    monkeypatch.setattr(ops, "_flash_attn_forward_lse_checked", False)
    return SimpleNamespace(forward=forward, created_policies=created_policies)


def engine(scalar_fast=False):
    policy = prepare(FlashV100Options(e4m3_scalar_fast=scalar_fast))[0]
    return SimpleNamespace(attention_config=SimpleNamespace(flash_v100=policy))


def test_cold_selector_prepares_dense_only_runtime(monkeypatch, native_package):
    monkeypatch.setattr(mla_prefill, "_is_sm70_flash_v100_platform", lambda: True)
    config = engine()
    try:
        with set_current_vllm_config(config):
            resources = runtime_resources_for(config)
            assert "flash_v100" not in resources
            # Same entry used during MLA backend selection, before any ordinary
            # Flash-V100 attention implementation has been constructed.
            assert mla_prefill.FlashAttnPrefillBackend.is_available()
            owner = resources["flash_v100"]
            operation = ops.get_flash_dense_forward()
            assert operation()[0] is owner
            assert ops.get_flash_dense_forward() is operation
            assert len(native_package.created_policies) == 1
    finally:
        release_runtime_resources(config)


def test_cached_native_operation_retains_separate_engine_bindings(
    monkeypatch, native_package
):
    monkeypatch.setattr(mla_prefill, "_is_sm70_flash_v100_platform", lambda: True)
    engines = [engine(False), engine(True)]
    bindings, owners = [], []
    for config in engines:
        with set_current_vllm_config(config):
            assert mla_prefill.FlashAttnPrefillBackend.is_available()
            mla_prefill.FlashAttnPrefillBackend(
                num_heads=1,
                scale=1.0,
                kv_lora_rank=512,
                qk_nope_head_dim=128,
                qk_rope_head_dim=0,
                v_head_dim=128,
                vllm_config=config,
            )
            owners.append(runtime_resources_for(config)["flash_v100"])
            bindings.append(ops.get_flash_dense_forward())
    assert owners[0] is not owners[1]
    assert [owner.policy.scalar_fast for owner in owners] == [False, True]
    assert len(native_package.created_policies) == 2
    # A bound callback keeps its own runtime when the active engine changes.
    for index in (1, 0, 1, 0):
        with set_current_vllm_config(engines[index]):
            assert ops.get_flash_dense_forward() is bindings[index]
            assert bindings[1 - index]()[0] is owners[1 - index]
    release_runtime_resources(engines[0])
    assert owners[0].native_policy is None
    assert owners[1].native_policy is not None
    release_runtime_resources(engines[1])


def test_missing_dense_entry_does_not_create_owner(monkeypatch, native_package):
    monkeypatch.setattr(ops, "_flash_attn_forward_lse_checked", True)
    config = engine()
    with set_current_vllm_config(config):
        assert not dense_prefill.flash_v100_dense_prefill_lse_available()
        assert "flash_v100" not in runtime_resources_for(config)
        assert not native_package.created_policies


def test_standalone_dense_entry_preserves_unbound_compatibility(native_package):
    with set_current_vllm_config(None):
        assert ops.get_flash_dense_forward() is native_package.forward
    assert not native_package.created_policies


def test_direct_binding_still_rejects_missing_engine_owner(native_package):
    with (
        set_current_vllm_config(engine()),
        pytest.raises(RuntimeError, match="resources must be bound"),
    ):
        runtime.bind_attention_operation(native_package.forward)


def test_dense_selector_in_fresh_interpreter():
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--confcutdir=tests/config",
            (
                f"{Path(__file__).relative_to(root)}::"
                "test_cold_selector_prepares_dense_only_runtime"
            ),
            "-q",
            "--tb=short",
        ],
        cwd=root,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(root)},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
