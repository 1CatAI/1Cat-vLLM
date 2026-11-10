# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import multiprocessing

import pytest

from tests.config.runtime_policy_utils import make_policy_defaults
from vllm import envs
from vllm.config import set_current_vllm_config
from vllm.config.policy_defaults import runtime_compile_ignored_aliases
from vllm.config.sm70_runtime import Sm70RuntimeConfig
from vllm.config.sm70_triton_attention import (
    Sm70TritonAttentionPolicy,
    capture_triton_attention_policy,
)


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for alias in Sm70TritonAttentionPolicy.aliases.values():
        monkeypatch.delenv(alias, raising=False)


@pytest.mark.parametrize(
    "overrides,warps",
    [
        ({}, (4, 8)),
        ({"safe_defaults": False}, (0, 0)),
        ({"num_warps": 2}, (2, 2)),
        ({"num_warps": 2, "decode_num_warps": 1}, (2, 1)),
        ({"safe_defaults": False, "prefill_num_warps": 8}, (8, 0)),
    ],
)
def test_schedule_retains_precedence(overrides, warps):
    policy = Sm70TritonAttentionPolicy(**overrides)
    policy.resolve()
    assert policy.resolved_schedule() == (0, 0, *warps, "", "")


@pytest.mark.parametrize(
    "field,raw,error",
    [
        ("prefill_tile_size", "", "invalid literal"),
        ("decode_tile_size", "12", "power of 2"),
        ("prefill_tile_size", "256", r"\[16, 128\]"),
        ("num_warps", "3", "1, 2, 4, 8"),
        ("safe_defaults", "true", "invalid literal"),
        ("qk_input_precision", "invalid", "one of"),
    ],
)
def test_malformed_unused_options_fail_at_original_admission(
    monkeypatch, field, raw, error
):
    monkeypatch.setenv(Sm70TritonAttentionPolicy.aliases[field], raw)
    policy = Sm70TritonAttentionPolicy()
    policy.resolve()
    policy.active = False
    inactive = Sm70TritonAttentionPolicy()
    inactive.active = False
    assert policy.compute_hash() == inactive.compute_hash()
    with pytest.raises(ValueError, match=error):
        policy.resolved_schedule()


def test_schedule_hash_uses_effective_warps_not_redundant_overrides():
    first = Sm70TritonAttentionPolicy()
    second = Sm70TritonAttentionPolicy(
        safe_defaults=False, prefill_num_warps=4, decode_num_warps=8
    )
    for policy in (first, second):
        policy.resolve()
    assert first.compute_hash() == second.compute_hash()
    second.decode_num_warps = 4
    second.resolve()
    assert first.compute_hash() != second.compute_hash()


def test_two_engines_and_worker_keep_resolved_schedule(monkeypatch):
    engines = []
    for value in (2, 8):
        defaults = make_policy_defaults()
        defaults.cfg.attention_config.sm70_triton.num_warps = value
        defaults.finish()
        engines.append(defaults.cfg)
    receiver, sender = multiprocessing.Pipe(duplex=False)
    try:
        sender.send(engines[0].attention_config.sm70_triton)
        transferred = receiver.recv()
    finally:
        receiver.close()
        sender.close()
    for alias in Sm70TritonAttentionPolicy.aliases.values():
        monkeypatch.setenv(alias, "changed-after-init")

        def fail():
            raise AssertionError("execution read compatibility input")

        monkeypatch.setitem(envs.environment_variables, alias, fail)
    transferred.resolve()
    assert transferred.resolved_schedule()[2:4] == (2, 2)
    for cfg in engines * 2:
        with set_current_vllm_config(cfg):
            policy = capture_triton_attention_policy()
            assert policy is cfg.attention_config.sm70_triton
            assert policy.resolved_schedule()[2:4] == (policy.num_warps,) * 2
        assert set(policy.aliases.values()) <= runtime_compile_ignored_aliases(cfg)


def test_warmup_input_errors_remain_deferred_and_typed_overrides_win(monkeypatch):
    monkeypatch.setenv("VLLM_SM70_AWQ_WARMUP", "0")
    monkeypatch.setenv("VLLM_SM70_AWQ_WARMUP_MAX_MOE_TOKENS", "")
    legacy = Sm70RuntimeConfig()
    explicit = Sm70RuntimeConfig(awq_warmup_max_moe_tokens=8)
    monkeypatch.setenv("VLLM_SM70_AWQ_WARMUP", "1")
    assert not legacy.value("awq_warmup")
    with pytest.raises(ValueError, match="invalid literal"):
        legacy.value("awq_warmup_max_moe_tokens")
    assert explicit.value("awq_warmup_max_moe_tokens") == 8
