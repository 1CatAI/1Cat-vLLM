# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

import pytest

from tests.config.runtime_policy_utils import make_policy_defaults
from vllm import envs
from vllm.config import set_current_vllm_config
from vllm.config.execution_policy import CommunicationPolicy, PlePlacementPolicy
from vllm.models.qwen4_exp.common.ple import (
    ple_host_budget_bytes,
    ple_host_reserve_bytes,
    ple_vram_reserve_bytes,
)


@pytest.mark.parametrize("order", [(1.0, 2.0), (2.0, 1.0)])
def test_placement_inputs_are_owned_before_workers(monkeypatch, order):
    configs = []
    aliases = [
        "VLLM_QWEN4EXP_PLE_" + suffix
        for suffix in ("HOST_GIB", "HOST_RESERVE_GIB", "VRAM_RESERVE_GIB")
    ]
    for value in order:
        for alias in aliases:
            monkeypatch.setenv(alias, str(value))
        defaults = make_policy_defaults()
        defaults.finish()
        configs.append(defaults.cfg)
    for alias in aliases:
        monkeypatch.setenv(alias, "broken-after-init")
        monkeypatch.setitem(
            envs.environment_variables, alias, Mock(side_effect=AssertionError(alias))
        )
    for cfg, value in list(zip(configs, order)) * 2:
        with set_current_vllm_config(cfg):
            assert ple_host_budget_bytes() == int(value * 1024**3)
            assert ple_host_reserve_bytes(100 * 1024**3) == int(value * 1024**3)
            assert ple_vram_reserve_bytes(32 * 1024**3) == int(value * 1024**3)
        # EngineArgs checks this before entering a worker/current-config context.
        assert ple_host_budget_bytes(policy=cfg.offload_config.ple) == int(
            value * 1024**3
        )


@pytest.mark.parametrize("raw", ["bad", "nan", "-1", "inf"])
def test_placement_errors_remain_at_resource_admission(monkeypatch, raw):
    monkeypatch.setenv("VLLM_QWEN4EXP_PLE_HOST_GIB", raw)
    policy = PlePlacementPolicy()
    policy.resolve()
    with pytest.raises(ValueError):
        policy.gib_bytes("host_gib")
    policy = PlePlacementPolicy(host_gib=0)
    policy.resolve()
    assert policy.gib_bytes("host_gib") == 0


def test_placement_automatic_defaults_keep_dynamic_capacity(monkeypatch):
    for name in ("HOST_GIB", "HOST_RESERVE_GIB", "VRAM_RESERVE_GIB"):
        monkeypatch.delenv("VLLM_QWEN4EXP_PLE_" + name, raising=False)
    policy = PlePlacementPolicy()
    policy.resolve()
    assert ple_host_budget_bytes(policy=policy) is None
    assert ple_host_reserve_bytes(40 * 1024**3, policy=policy) == 10 * 1024**3
    assert ple_vram_reserve_bytes(100 * 1024**3, policy=policy) == 4 * 1024**3
    assert ple_vram_reserve_bytes(10 * 1024**3, policy=policy) == int(0.8 * 1024**3)


def test_gemma_fusion_input_is_deferred_and_typed_override_wins(monkeypatch):
    monkeypatch.setenv("VLLM_SM70_TP2_AR_GEMMA_RMS_FUSION", "bad")
    policy = CommunicationPolicy()
    policy.resolve()
    with pytest.raises(ValueError):
        policy.value("gemma_rms_tp2")
    typed = CommunicationPolicy(gemma_rms_tp2=True)
    typed.resolve()
    assert typed.value("gemma_rms_tp2") is True


@pytest.mark.parametrize("raw,expected", [("", False), ("TRUE", True), ("off", False)])
def test_dense_graph_policy_retains_legacy_dialect(monkeypatch, raw, expected):
    from vllm.config.execution_policy import GraphPolicy

    monkeypatch.setenv("VLLM_SM70_DENSE_CUDAGRAPH_CAPTURE", raw)
    policy = GraphPolicy()
    policy.resolve()
    assert policy.dense_capture is expected


def test_empty_placement_input_keeps_automatic_budget(monkeypatch):
    monkeypatch.setenv("VLLM_QWEN4EXP_PLE_HOST_GIB", "")
    policy = PlePlacementPolicy()
    policy.resolve()
    assert policy.gib_bytes("host_gib") is None


@pytest.mark.parametrize(
    "primary,fallback,expected",
    [(None, "3", 3), ("", "3", 0), ("-2", "3", 0), ("bad", "3", 0), ("5", "3", 5)],
)
def test_target_profiler_preserves_primary_before_tree_alias(
    monkeypatch, primary, fallback, expected
):
    from vllm.config.sm70_runtime import RuntimeTraceConfig

    alias = "VLLM_SM70_SPEC_TARGET_FORWARD_PROFILER_STEP"
    if primary is None:
        monkeypatch.delenv(alias, raising=False)
    else:
        monkeypatch.setenv(alias, primary)
    monkeypatch.setenv("VLLM_DFLASH_DDTREE_TARGET_FORWARD_PROFILER_STEP", fallback)
    policy = RuntimeTraceConfig()
    assert policy.spec_target_profiler_step == expected
    assert policy.sources["spec_target_profiler_step"] == (
        alias
        if primary is not None
        else "VLLM_DFLASH_DDTREE_TARGET_FORWARD_PROFILER_STEP"
    )
    assert (
        RuntimeTraceConfig(spec_target_profiler_step=7).spec_target_profiler_step == 7
    )


@pytest.mark.parametrize(
    "primary,fallback,expected",
    [("0", "1", True), ("1", "0", True), ("", "true", False)],
)
def test_target_nvtx_combines_legacy_aliases_before_typed_override(
    monkeypatch, primary, fallback, expected
):
    from vllm.config.sm70_runtime import RuntimeTraceConfig

    monkeypatch.setenv("VLLM_SM70_SPEC_TARGET_FORWARD_NVTX", primary)
    monkeypatch.setenv("VLLM_DFLASH_DDTREE_TARGET_FORWARD_NVTX", fallback)
    policy = RuntimeTraceConfig()
    assert policy.spec_target_nvtx is expected
    assert (
        "VLLM_DFLASH_DDTREE_TARGET_FORWARD_NVTX" in policy.sources["spec_target_nvtx"]
    )
    assert RuntimeTraceConfig(spec_target_nvtx=False).spec_target_nvtx is False


def test_resolution_report_retains_legacy_input_when_typed_wins(monkeypatch):
    from vllm.config.policy_defaults import PolicyDefaults, runtime_policy_report

    monkeypatch.setenv("VLLM_SM70_DENSE_CUDAGRAPH_CAPTURE", "1")
    cfg = make_policy_defaults().cfg
    cfg.runtime_default_sources.clear()
    cfg.compilation_config.runtime.dense_capture = False
    cfg.compilation_config.runtime.sources["dense_capture"] = "typed"
    defaults = PolicyDefaults(cfg)
    defaults.finish()
    monkeypatch.setenv("VLLM_SM70_DENSE_CUDAGRAPH_CAPTURE", "poison")
    report = runtime_policy_report(cfg)
    entries = report["default_resolution"]["VLLM_SM70_DENSE_CUDAGRAPH_CAPTURE"]
    assert entries[0] == {
        "source": "legacy_environment",
        "raw": "1",
        "overridden_by_typed": True,
    }
    assert entries[-1] == {"source": "typed", "value": False}
