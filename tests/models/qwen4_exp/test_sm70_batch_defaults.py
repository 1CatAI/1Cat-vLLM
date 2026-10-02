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
    assert getattr(envs, name) is True
    metadata = envs.environment_variables[name].metadata
    assert metadata.declared_default == "True"
    assert "Default on" in metadata.description
    assert "Set 0" in metadata.description
    monkeypatch.setenv(name, "0")
    assert getattr(envs, name) is False
    envs.disable_envs_cache()
