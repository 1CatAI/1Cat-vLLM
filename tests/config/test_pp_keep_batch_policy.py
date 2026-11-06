# SPDX-License-Identifier: Apache-2.0
"""Scheduler settings freeze before workers and do not salt model graphs."""

import pickle

import pytest

from vllm import envs
from vllm.config import SchedulerConfig


def test_scheduler_snapshot_is_typed_serialized_and_hash_neutral(monkeypatch):
    envs.disable_envs_cache()
    monkeypatch.setenv("VLLM_PP_KEEP_BATCH", "1")
    config = SchedulerConfig(max_model_len=512, is_encoder_decoder=False)
    assert config.pp_keep_batch is True
    copy = pickle.loads(pickle.dumps(config))
    monkeypatch.setenv("VLLM_PP_KEEP_BATCH", "invalid-after-initialization")
    assert copy.pp_keep_batch is True
    explicit = SchedulerConfig(
        max_model_len=512, is_encoder_decoder=False, pp_keep_batch=False
    )
    assert explicit.pp_keep_batch is False
    assert config.compute_hash() == explicit.compute_hash()


@pytest.mark.parametrize(
    "method, expected",
    [
        ("ngram", True),
        ("dspark", True),
        ("dflash", True),
        ("ngram_gpu", False),
        ("dflash_ddtree", False),
    ],
)
def test_speculative_config_owns_row_retention_capability(method, expected):
    from vllm.config import SpeculativeConfig

    config = object.__new__(SpeculativeConfig)
    config.method = method
    assert config.supports_pp_row_retention() is expected
