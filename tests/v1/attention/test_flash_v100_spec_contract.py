# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Feature contracts retain validation order and process-shared log identity."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm.v1.attention.backends import flash_attn_v100 as legacy
from vllm.v1.attention.backends.flash_v100 import impl
from vllm.v1.attention.backends.flash_v100.spec import contracts

pytestmark = pytest.mark.cpu_test


def _instance():
    instance = object.__new__(impl.FlashAttnV100Impl)
    instance.sliding_window = (-1, -1)

    def unexpected():
        raise AssertionError("contract checks must not construct the verifier")

    instance._new_verification_executor = unexpected
    return instance


def test_legacy_observation_set_is_shared_across_layers_and_instances(monkeypatch):
    seen: set[tuple[object, ...]] = set()
    logger = MagicMock()
    monkeypatch.setattr(legacy, "_logged_dflash_attention_contracts", seen)
    monkeypatch.setattr(contracts, "logger", logger)
    layer = SimpleNamespace(
        is_dflash_draft_attn=True,
        layer_name="layer.a",
        dflash_expected_causal=True,
        dflash_expected_sliding_window=None,
        dflash_rope_is_neox_style=True,
    )
    metadata = SimpleNamespace(causal=True)
    first, second = _instance(), _instance()
    first._validate_dflash_attention_contract(layer, metadata)
    second._validate_dflash_attention_contract(layer, metadata)
    assert logger.info.call_count == 1
    layer.layer_name = "layer.b"
    second._validate_dflash_attention_contract(layer, metadata)
    layer.dflash_rope_is_neox_style = False
    second._validate_dflash_attention_contract(layer, metadata)
    assert logger.info.call_count == 3
    assert contracts.seen_contracts is seen
    assert seen == {
        ("layer.a", True, (-1, -1), True),
        ("layer.b", True, (-1, -1), True),
        ("layer.b", True, (-1, -1), False),
    }


def test_legacy_validator_patch_reaches_the_real_feature_call(monkeypatch):
    original = contracts.validate_contract
    calls = []

    def tracked(layer, metadata, window_size):
        calls.append((layer, metadata))
        return original(layer, metadata, window_size)

    monkeypatch.setattr(legacy, "validate_contract", tracked)
    layer, metadata = SimpleNamespace(), SimpleNamespace()
    _instance()._validate_dflash_attention_contract(layer, metadata)
    assert calls == [(layer, metadata)]
    assert contracts.validate_contract is tracked


@pytest.mark.parametrize(
    "violation,message,window_calls",
    [
        ("missing", "missing its declared", 0),
        ("causal", "causality mismatch", 0),
        ("window", "sliding-window mismatch", 1),
    ],
)
def test_invalid_contract_fails_before_observation(
    monkeypatch, violation, message, window_calls
):
    seen: set[tuple[object, ...]] = set()
    monkeypatch.setattr(contracts, "seen_contracts", seen)
    layer = SimpleNamespace(is_dflash_draft_attn=True)
    if violation != "missing":
        layer.dflash_expected_causal = violation != "causal"
    if violation == "window":
        layer.dflash_expected_sliding_window = 64
    window = MagicMock(return_value=(-1, -1))
    with pytest.raises(RuntimeError, match=message):
        contracts.validate_contract(layer, SimpleNamespace(causal=True), window)
    assert window.call_count == window_calls
    assert not seen
