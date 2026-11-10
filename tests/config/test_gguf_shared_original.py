# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
from typing import Any

import pytest

from vllm.config.kernel import KernelConfig, Sm70GgufConfig


def shared_config(**overrides):
    values: dict[str, Any] = dict(
        expert_cache_host_storage="shared_original",
        expert_storage="original",
        expert_arena=True,
        expert_cache_host_slack=0,
    )
    values.update(overrides)
    return Sm70GgufConfig(**values)


@pytest.mark.parametrize(
    "overrides",
    [
        {"expert_storage": "canonical"},
        {"expert_arena": False},
        {"expert_cache_fraction": 0.7},
        {"expert_cache_host_slack": 0.1},
        {"expert_cache_profile": "old-profile.pt"},
        {"expert_cache_prefill": "mmq"},
        {"expert_cache_prefill_prepare": "predict"},
        {"expert_cache_offload_fraction": 0},
        {"expert_cache_offload_fraction": 1},
    ],
)
def test_shared_mode_rejects_conflicting_budget_controls(overrides):
    with pytest.raises(ValueError):
        shared_config(**overrides)


def test_shared_mode_worker_copy_and_graph_hash():
    policy = shared_config()
    assert policy.expert_cache_offload_fraction == 0.30
    assert policy.expert_cache_fraction == policy.expert_cache_host_slack == 0
    config = KernelConfig(sm70_gguf=policy)
    config.sm70_gguf.active = True
    restored = copy.deepcopy(config)
    assert restored.compute_hash() == config.compute_hash()
    candidate = copy.deepcopy(config)
    candidate.sm70_gguf.expert_cache_offload_fraction = 0.25
    assert candidate.compute_hash() != config.compute_hash()


def test_default_rank_local_policy_is_preserved():
    policy = Sm70GgufConfig(expert_cache_fraction=0.7)
    assert policy.expert_cache_host_storage == "rank_local"
    assert policy.expert_cache_host_slack == 0.1
