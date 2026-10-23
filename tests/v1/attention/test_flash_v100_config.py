# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import FrozenInstanceError, fields

import pytest

import vllm.envs as envs
from tools.sm70.flash_v100_trace import (
    Recorder,
    cpu_cuda,
    install_ops,
    strict_shim,
)
from vllm.v1.attention.backends.flash_v100 import config

pytestmark = pytest.mark.cpu_test


def test_policy_snapshot_is_frozen_and_legacy_updates_replace_it(monkeypatch):
    monkeypatch.setenv("VLLM_FLASH_V100_DECODE_DENSE_REFERENCE", "0")
    envs.disable_envs_cache()
    with cpu_cuda(monkeypatch, False), strict_shim() as legacy:
        install_ops(monkeypatch, Recorder(), legacy, {})
        options = dict(
            num_heads=6,
            num_kv_heads=1,
            head_size=256,
            scale=0.0625,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
        )
        instance = legacy.FlashAttnV100Impl(**options)
        snapshot = instance.config
        assert isinstance(snapshot, config.V100AttnConfig)
        assert not any(field.name in vars(instance) for field in fields(snapshot))
        with pytest.raises(FrozenInstanceError):
            snapshot.use_decode_dense_reference = True
        monkeypatch.setenv("VLLM_FLASH_V100_DECODE_DENSE_REFERENCE", "1")
        assert config.raw("VLLM_FLASH_V100_DECODE_DENSE_REFERENCE") == "1"
        assert not instance.use_decode_dense_reference
        assert legacy.FlashAttnV100Impl(**options).use_decode_dense_reference
        instance.use_decode_dense_reference = True
        assert instance.config is not snapshot
        assert instance.use_decode_dense_reference
        assert not snapshot.use_decode_dense_reference
