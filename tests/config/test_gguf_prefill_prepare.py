# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Predictive preparation is explicit and hashes only an active cache route."""

import copy
import pickle

import pytest

from vllm.config.kernel import KernelConfig


@pytest.mark.parametrize("mode", ["predict", "predict_resident"])
def test_default_and_unused_preparation_hashes(mode):
    config = KernelConfig()
    config.sm70_gguf.active = True
    before = config.compute_hash()
    candidate = copy.deepcopy(config)
    candidate.sm70_gguf.expert_cache_prefill_prepare = mode
    assert candidate.compute_hash() == before
    config.sm70_gguf.expert_cache_fraction = 0.8
    candidate = copy.deepcopy(config)
    candidate.sm70_gguf.expert_cache_prefill_prepare = mode
    assert candidate.compute_hash() != config.compute_hash()
    config.sm70_gguf.expert_cache_prefill = "mmq"
    candidate.sm70_gguf.expert_cache_prefill = "mmq"
    assert candidate.compute_hash() == config.compute_hash()


@pytest.mark.parametrize("mode", ["predict", "predict_resident"])
def test_preparation_survives_worker_serialization(mode):
    config = KernelConfig()
    config.sm70_gguf.active = True
    config.sm70_gguf.expert_cache_fraction = 0.8
    config.sm70_gguf.expert_cache_prefill_prepare = mode
    restored = pickle.loads(pickle.dumps(config))
    assert restored.sm70_gguf.expert_cache_prefill_prepare == mode
    assert restored.compute_hash() == config.compute_hash()
