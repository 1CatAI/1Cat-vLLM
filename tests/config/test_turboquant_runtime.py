# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tests.config.runtime_policy_utils import make_policy_defaults
from vllm.config import set_current_vllm_config
from vllm.config.turboquant_runtime import (
    TurboQuantDiagnostics,
    TurboQuantRuntimePolicy,
)
from vllm.runtime_resources import release_runtime_resources
from vllm.v1.attention.backends import turboquant_attn as attention


def engine(monkeypatch, tmp_path, limit):
    defaults = make_policy_defaults()
    cfg = defaults.cfg
    cfg.model_config = SimpleNamespace(max_model_len=512)
    cfg.cache_config.cache_dtype = "turboquant_k3v4_nc"
    trace = cfg.observability_config.runtime_trace.turboquant
    trace.prefill_limit = limit
    trace.dump_directory = str(tmp_path)
    trace.dump_threshold = 0
    trace.dump_limit = 1
    defaults.finish()
    manager = Mock()
    monkeypatch.setattr(attention, "is_workspace_manager_initialized", lambda: True)
    monkeypatch.setattr(attention, "current_workspace_manager", lambda: manager)
    monkeypatch.setattr(
        attention, "_flash_attn_varlen_supported_on_device", lambda: False
    )
    monkeypatch.setattr(attention, "flash_v100_dense_prefill_available", lambda: True)
    monkeypatch.setattr(
        attention, "flash_v100_turboquant_decode_available", lambda: True
    )
    with set_current_vllm_config(cfg):
        layer = attention.TurboQuantAttentionImpl(
            4, 128, 0.5, 1, kv_cache_dtype="turboquant_k3v4_nc"
        )
    return cfg, layer, manager


def test_two_engines_bind_workspace_and_budget_independently(monkeypatch, tmp_path):
    records = [engine(monkeypatch, tmp_path, limit) for limit in (1, 2)]
    for alias in (
        *TurboQuantRuntimePolicy.aliases.values(),
        *TurboQuantDiagnostics.aliases.values(),
    ):
        monkeypatch.setenv(alias, "invalid-after-init")
    q = torch.zeros(2, 4, 128)
    indices = torch.zeros(1, dtype=torch.int32)
    for cfg, layer, manager in records:
        layer._reserve_continuation_prefill_workspace()
        layer._reserve_continuation_prefill_workspace()
        assert manager.get_simultaneous.call_count == 1
        layer._triton_prefill_attention = Mock(return_value=q)
        for _ in range(3):
            layer._maybe_compare_flash_v100_prefill(q, q, q, q, indices, indices, 2, {})
        assert (
            layer._triton_prefill_attention.call_count
            == layer.compare_policy.prefill_limit
        )
    paths = list(tmp_path.glob("prefill_compare_*.pt"))
    assert len(paths) == 2
    assert paths[0] != paths[1]
    release_runtime_resources(records[0][0])
    assert (
        records[1][1].diagnostics.channels["turboquant_prefill"].saves["compare"] == 2
    )


def test_disabled_compare_never_allocates_or_synchronizes(monkeypatch, tmp_path):
    cfg, layer, _ = engine(monkeypatch, tmp_path, 0)

    def fail(*args, **kwargs):
        raise AssertionError("disabled observer touched tensor")

    layer._triton_prefill_attention = fail
    layer._maybe_compare_flash_v100_prefill(None, None, None, None, None, None, 0, {})
    assert not list(tmp_path.iterdir())
    release_runtime_resources(cfg)


@pytest.mark.parametrize(
    "raw,expected", [(None, True), ("0", False), ("", True), ("false", True)]
)
def test_exact_zero_legacy_dialect(monkeypatch, raw, expected):
    alias = TurboQuantRuntimePolicy.aliases["flash_prefill"]
    if raw is None:
        monkeypatch.delenv(alias, raising=False)
    else:
        monkeypatch.setenv(alias, raw)
    legacy = TurboQuantRuntimePolicy()
    legacy.resolve()
    assert legacy.flash_prefill is expected
    typed = TurboQuantRuntimePolicy(flash_prefill=not expected)
    typed.resolve()
    assert typed.flash_prefill is not expected


def test_unrelated_cache_and_resource_choices_do_not_change_hash():
    first, second = make_policy_defaults(), make_policy_defaults()
    second.cfg.attention_config.flash_v100.turboquant.flash_decode = False
    first.finish()
    second.finish()
    assert (
        first.cfg.attention_config.compute_hash()
        == second.cfg.attention_config.compute_hash()
    )
    first.cfg.attention_config.flash_v100.turboquant.active = True
    second.cfg.attention_config.flash_v100.turboquant.active = True
    assert (
        first.cfg.attention_config.compute_hash()
        != second.cfg.attention_config.compute_hash()
    )
    before = first.cfg.attention_config.compute_hash()
    first.cfg.attention_config.flash_v100.turboquant.reserve_workspace = False
    first.cfg.attention_config.flash_v100.turboquant.continuation_workspace_tokens = 32
    assert first.cfg.attention_config.compute_hash() == before


def test_invalid_numeric_preserves_default_warning(monkeypatch, caplog):
    monkeypatch.setenv("VLLM_SM70_TURBOQUANT_COMPARE_DUMP_THRESHOLD", "broken")
    policy = TurboQuantDiagnostics()
    policy.resolve()
    assert policy.dump_threshold == 0
    # Re-resolution in a worker never consults changed compatibility inputs.
    monkeypatch.setenv("VLLM_SM70_TURBOQUANT_COMPARE_DUMP_THRESHOLD", "5")
    policy.resolve()
    assert policy.dump_threshold == 0
