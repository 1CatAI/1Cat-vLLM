# SPDX-License-Identifier: Apache-2.0
"""Per-engine projection policy, serialization and inactive graph contracts."""

import pickle

import pytest

from vllm import envs
from vllm.config.kernel import KernelConfig, V41DraftProjectionPolicy


def test_projection_inputs_freeze_and_only_bound_drafter_changes_hash(monkeypatch):
    envs.disable_envs_cache()
    monkeypatch.setenv("VLLM_V41_DRAFT_MP_SHARD", "1")
    config = KernelConfig()
    before = config.compute_hash()
    config.v41_draft_projection.resolve()
    assert config.compute_hash() == before
    copy = pickle.loads(pickle.dumps(config.v41_draft_projection))
    monkeypatch.setenv("VLLM_V41_DRAFT_MP_SHARD", "invalid-after-initialization")
    copy.resolve()
    assert copy.enabled is True
    config.v41_draft_projection.active = True
    assert config.compute_hash() != before


def test_typed_projection_override_wins(monkeypatch):
    envs.disable_envs_cache()
    monkeypatch.setenv("VLLM_V41_DRAFT_MP_SHARD", "invalid")
    policy = V41DraftProjectionPolicy(enabled=False)
    policy.resolve()
    assert policy.enabled is False
    assert policy.sources["enabled"] == "typed"


@pytest.mark.parametrize("value", ["2", " ", "true"])
def test_invalid_legacy_flag_refuses(monkeypatch, value):
    envs.disable_envs_cache()
    monkeypatch.setenv("VLLM_V41_DRAFT_MP_SHARD", value)
    with pytest.raises(ValueError, match="VLLM_V41_DRAFT_MP_SHARD"):
        V41DraftProjectionPolicy().resolve()


def test_empty_legacy_flag_preserves_disabled_contract(monkeypatch):
    envs.disable_envs_cache()
    monkeypatch.setenv("VLLM_V41_DRAFT_MP_SHARD", "")
    policy = V41DraftProjectionPolicy()
    policy.resolve()
    assert policy.enabled is False
