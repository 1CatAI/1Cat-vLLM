# SPDX-License-Identifier: Apache-2.0
"""Real initialization/serialization contracts; no native operation replaced."""

import pickle

import pytest

from vllm import envs
from vllm.config.attention import AttentionConfig, V41IndexTiledPolicy


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
