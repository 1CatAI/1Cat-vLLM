# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The qualified default group retains individual explicit off switches."""

import pytest

from vllm import envs

NAMES = [
    name
    for name, getter in envs.environment_variables.items()
    if "Flash-Next qualified batch" in getter.metadata.acceleration_paths
]


@pytest.mark.parametrize("name", NAMES)
def test_qualified_default_and_explicit_disable(name, monkeypatch):
    envs.disable_envs_cache()
    monkeypatch.delenv(name, raising=False)
    expected = name != "VLLM_SM70_RMSNORM_GATED_EXACT"
    assert getattr(envs, name) is expected
    metadata = envs.environment_variables[name].metadata
    assert metadata.declared_default == str(expected)
    assert "Default on" in metadata.description
    assert "Set 0" in metadata.description
    monkeypatch.setenv(name, "0")
    assert getattr(envs, name) is False
    envs.disable_envs_cache()


def test_norm_policy_is_per_engine_and_hashed(monkeypatch):
    from vllm.config.kernel import KernelConfig

    monkeypatch.delenv("VLLM_SM70_RMSNORM_GATED_EXACT", raising=False)
    ordinary, qualified = KernelConfig(), KernelConfig()
    ordinary.resolve_sm70_rmsnorm_gated(qualified=False)
    qualified.resolve_sm70_rmsnorm_gated(qualified=True)
    assert ordinary.sm70_rmsnorm_gated_exact is False
    assert qualified.sm70_rmsnorm_gated_exact is True
    assert ordinary.compute_hash() != qualified.compute_hash()
    monkeypatch.setenv("VLLM_SM70_RMSNORM_GATED_EXACT", "0")
    qualified.resolve_sm70_rmsnorm_gated(qualified=False)
    assert qualified.sm70_rmsnorm_gated_exact is True
    disabled = KernelConfig()
    disabled.resolve_sm70_rmsnorm_gated(qualified=True)
    assert disabled.sm70_rmsnorm_gated_exact is False
