# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real initialization/serialization contracts; no native operation replaced."""

import pickle

import pytest

from vllm import envs
from vllm.config.attention import AttentionConfig, V41IndexTiledPolicy
from vllm.config.utils import hash_factors


def test_inputs_are_frozen_serialized_and_hash_only_when_bound(monkeypatch):
    envs.disable_envs_cache()
    monkeypatch.setenv("VLLM_V41_INDEX_TILED", "1")
    monkeypatch.setenv("VLLM_V41_INDEX_TILED_LIB", "relative/v41idx.so")
    config = AttentionConfig()
    before = config.compute_hash()
    config.v41_index_tiled.resolve()
    assert config.compute_hash() == before
    copy = pickle.loads(pickle.dumps(config.v41_index_tiled))
    monkeypatch.setenv("VLLM_V41_INDEX_TILED", "invalid-after-initialization")
    copy.resolve()
    assert copy.enabled is True and copy.library == "relative/v41idx.so"
    config.v41_index_tiled.active = True
    assert config.compute_hash() != before


def test_explicit_typed_values_override_invalid_legacy(monkeypatch):
    envs.disable_envs_cache()
    monkeypatch.setenv("VLLM_V41_INDEX_TILED", "invalid")
    policy = V41IndexTiledPolicy(enabled=False, library="")
    policy.resolve()
    assert policy.enabled is False
    assert policy.sources["enabled"] == "typed"


@pytest.mark.parametrize("value", ["2", " ", "true"])
def test_invalid_legacy_flag_raises(monkeypatch, value):
    envs.disable_envs_cache()
    monkeypatch.setenv("VLLM_V41_INDEX_TILED", value)
    with pytest.raises(ValueError, match="VLLM_V41_INDEX_TILED"):
        V41IndexTiledPolicy().resolve()


def test_empty_legacy_flag_preserves_disabled_contract(monkeypatch):
    envs.disable_envs_cache()
    monkeypatch.setenv("VLLM_V41_INDEX_TILED", "")
    policy = V41IndexTiledPolicy()
    policy.resolve()
    assert policy.enabled is False


def test_tensor_core_inputs_resolve_from_legacy_names(monkeypatch):
    envs.disable_envs_cache()
    monkeypatch.setenv("VLLM_V41_INDEX_TILED", "1")
    monkeypatch.setenv("VLLM_V41_INDEX_TC", "1")
    monkeypatch.setenv("VLLM_V41_INDEX_TC_LIB", "relative/v41idx3.so")
    policy = V41IndexTiledPolicy()
    policy.resolve()
    assert policy.tensor_core is True
    assert policy.tensor_core_library == "relative/v41idx3.so"


def test_tensor_core_off_keeps_the_tiled_graph_key(monkeypatch):
    """Off, the key is the one #1372 defined: enabled and library only."""
    envs.disable_envs_cache()
    monkeypatch.setenv("VLLM_V41_INDEX_TILED", "1")
    monkeypatch.setenv("VLLM_V41_INDEX_TILED_LIB", "relative/v41idx.so")
    policy = V41IndexTiledPolicy()
    policy.resolve()
    policy.active = True
    tiled = hash_factors({"enabled": True, "library": "relative/v41idx.so"})
    assert policy.compute_hash() == tiled
    monkeypatch.setenv("VLLM_V41_INDEX_TC", "1")
    envs.disable_envs_cache()
    on = V41IndexTiledPolicy()
    on.resolve()
    on.active = True
    assert on.compute_hash() == hash_factors(
        {
            "enabled": True,
            "library": "relative/v41idx.so",
            "tensor_core": True,
            "tensor_core_library": "",
        }
    )
    assert on.compute_hash() != tiled


@pytest.mark.parametrize("value", ["2", " ", "true"])
def test_invalid_tensor_core_flag_raises(monkeypatch, value):
    envs.disable_envs_cache()
    monkeypatch.setenv("VLLM_V41_INDEX_TC", value)
    with pytest.raises(ValueError, match="VLLM_V41_INDEX_TC"):
        V41IndexTiledPolicy().resolve()
