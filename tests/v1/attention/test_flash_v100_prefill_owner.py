# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefill assembly preserves calls while execution receives owned components."""

from types import SimpleNamespace

import pytest

from vllm.v1.attention.backends.flash_v100 import config, impl, prefill, workspace

pytestmark = pytest.mark.cpu_test


def _instance():
    instance = object.__new__(impl.FlashAttnV100Impl)
    instance.workspace = workspace.V100Workspace()
    instance.scale = 0.125
    instance.kv_cache_dtype = "auto"
    return instance


@pytest.mark.parametrize("unbound", [False, True])
def test_profile_call_executes_on_owner_and_returns_falsey_output(monkeypatch, unbound):
    instance = _instance()
    events = []
    original = prefill.PrefillExecutor._run_prefill_paged_call

    def observe(owner, **kwargs):
        assert type(owner) is prefill.PrefillExecutor
        assert owner.workspace is instance.workspace
        events.append("owned")
        return original(owner, **kwargs)

    monkeypatch.setattr(prefill.PrefillExecutor, "_run_prefill_paged_call", observe)
    monkeypatch.setenv("VLLM_FLASH_V100_PREFILL_CHUNK_PROFILE", "0")

    def native():
        events.append("native")
        return 0

    kwargs = dict(
        route="prefill_prefix_paged",
        q_len=8,
        seq_len=31,
        heads_q=6,
        heads_kv=1,
        head_dim=256,
        block_size=832,
        fn=native,
    )
    result = (
        impl.FlashAttnV100Impl._run_prefill_paged_call(instance, **kwargs)
        if unbound
        else instance._run_prefill_paged_call(**kwargs)
    )
    assert result == 0 and events == ["owned", "native"]


def test_instance_override_survives_sequence_executor_injection():
    calls = []

    class Falsey:
        def __bool__(self):
            return False

        def __call__(self, **kwargs):
            calls.append(kwargs)
            return kwargs["fn"]()

    instance = _instance()
    override = Falsey()
    instance._run_prefill_paged_call = override
    owner = instance._new_prefill_executor()
    candidates = prefill.create_prefill_executor(owner)
    marker = object()
    assert candidates.ops.run_paged(fn=lambda: marker) is marker
    assert len(calls) == 1
    assert candidates.workspace is instance.workspace


def test_native_operator_refresh_and_feature_override_are_not_lost():
    instance = _instance()
    instance._flash_v100_small_query_prefill_as_decode = lambda *args: args
    instance.flash_attn_grouped_e4m3_fp32_paged = "first"
    first = instance._new_prefill_executor()
    instance.flash_attn_grouped_e4m3_fp32_paged = "second"
    second = instance._new_prefill_executor()
    # The external grouped contract uses this legacy native attribute.
    assert first.flash_attn_grouped_e4m3_fp32_paged == "first"
    assert second.flash_attn_grouped_e4m3_fp32_paged == "second"
    candidates = prefill.create_prefill_executor(second)
    assert candidates.ops.small_query("q", "kv") == ("q", "kv")


def test_owner_policy_does_not_read_live_implementation_attributes():
    instance = _instance()
    instance.use_flash_v100_prefill_splitkv = False
    owner = instance._new_prefill_executor()
    instance.use_flash_v100_prefill_splitkv = True
    assert (
        owner._should_use_prefill_splitkv(
            q_len=64,
            seq_len=8192,
            head_dim=256,
            key_cache=SimpleNamespace(dtype=None),
            causal=True,
        )
        is False
    )
    assert isinstance(instance.use_flash_v100_prefill_splitkv, bool)
    assert isinstance(
        impl.FlashAttnV100Impl.use_flash_v100_prefill_splitkv, config.ConfigField
    )
