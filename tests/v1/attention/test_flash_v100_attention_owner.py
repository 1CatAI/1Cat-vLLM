# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention feature policy owns state and preserves legacy guard behavior."""

from types import SimpleNamespace

import pytest

from vllm.v1.attention.backends.flash_v100 import impl
from vllm.v1.attention.backends.flash_v100.spec import attention_policy
from vllm.v1.attention.backends.flash_v100.spec.attention_policy import (
    SpecAttentionState,
)

pytestmark = pytest.mark.cpu_test


def test_feature_fields_have_one_owner_and_are_layer_local():
    first = object.__new__(impl.FlashAttnV100Impl)
    second = object.__new__(impl.FlashAttnV100Impl)
    first.dflash2_grouped_verify_max_query_tokens = 8
    second.dflash2_grouped_verify_max_query_tokens = 16
    assert first.spec_attention is not second.spec_attention
    assert "dflash2_grouped_verify_max_query_tokens" not in vars(first)
    assert first.spec_attention.dflash2_grouped_verify_max_query_tokens == 8
    first.spec_attention.dflash2_grouped_verify_max_query_tokens = 7
    assert first.dflash2_grouped_verify_max_query_tokens == 7
    assert second.dflash2_grouped_verify_max_query_tokens == 16


@pytest.mark.parametrize("available", [False, True])
def test_verifier_configuration_preserves_short_circuit_reads(monkeypatch, available):
    reads = []

    # The switches are read from the engine's Flash-V100 policy and options.
    class Policy:
        @property
        def grouped_verify(self):
            reads.append("grouped_verify")
            return True

        @property
        def grouped_verify_min_model_len(self):
            reads.append("grouped_verify_min_model_len")
            return 32768

    class Options:
        def value(self, name):
            reads.append(name)
            return True

    monkeypatch.setattr(attention_policy, "flash_v100_policy", Policy)
    monkeypatch.setattr(attention_policy._config, "options", Options)
    monkeypatch.setattr(
        attention_policy.current_platform, "is_device_capability", lambda _: True
    )
    state = SpecAttentionState(lambda *args: False)
    native = object() if available else None
    state.initialize_verify_abi(16, 2)
    state.configure_verifier(native)
    assert state.flash_attn_grouped_verify_paged is native
    assert state.dflash2_grouped_verify_max_query_tokens == 16
    assert state.dflash2_grouped_verify_request_major_abi_version == 2
    assert state.use_dflash2_grouped_verify is available
    assert state.use_dflash2_batched_grouped_verify is available
    assert state.dflash2_grouped_verify_min_model_len == 32768
    assert reads == (
        [
            "grouped_verify",
            "dflash2_batched_grouped_verify",
            "grouped_verify_min_model_len",
        ]
        if available
        else ["grouped_verify_min_model_len"]
    )


def test_prefill_keyword_probe_and_wrapper_are_injected(monkeypatch):
    calls = []

    def native(*, dflash2_window_split):
        calls.append(dflash2_window_split)

    def accepts(operator, keyword):
        assert operator is native
        calls.append(keyword)
        return True

    monkeypatch.setattr(
        attention_policy,
        "capture_sm70_dflash2_config",
        lambda: SimpleNamespace(draft_window_split=False),
    )
    state = SpecAttentionState(accepts)
    wrapped = state.configure_prefill(native)
    wrapped()
    assert calls == ["dflash2_window_split", False]
    assert state.flash_attn_prefill_paged is wrapped
    assert state._flash_prefill_paged_dflash2_split_pages == ()


def test_ordinary_legacy_guard_does_not_construct_an_executor():
    instance = object.__new__(impl.FlashAttnV100Impl)

    def unexpected(*args, **kwargs):
        raise AssertionError("ordinary guard must not allocate a verifier")

    instance._new_verification_executor = unexpected
    instance._flash_v100_window_size = unexpected
    instance._validate_dflash_attention_contract(SimpleNamespace(), SimpleNamespace())


def test_legacy_delegate_retains_bound_and_unbound_call_arguments():
    calls = []
    result = object()
    instance = object.__new__(impl.FlashAttnV100Impl)
    query, key, value, metadata = (object() for _ in range(4))

    def admit(*args, **kwargs):
        calls.append((args, kwargs))
        return result

    instance._new_verification_executor = lambda: SimpleNamespace(
        grouped_verify_allowed=admit
    )
    assert (
        instance._dflash2_grouped_verify_allowed(
            query, key, value, metadata, num_query_tokens=7
        )
        is result
    )
    assert (
        impl.FlashAttnV100Impl._dflash2_grouped_verify_allowed(
            instance, query, key, value, metadata, num_query_tokens=7
        )
        is result
    )
    assert calls == [
        ((query, key, value, metadata), {"num_query_tokens": 7}),
        ((query, key, value, metadata), {"num_query_tokens": 7}),
    ]
