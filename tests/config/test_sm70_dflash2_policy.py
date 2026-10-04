# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace as NS

import pytest
import torch

from vllm import envs
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.config.sm70_dflash2 import (
    SM70_DFLASH2_LEGACY_FIELDS,
    Sm70DFlash2Config,
    capture_sm70_dflash2_config,
    sm70_dflash2_enabled,
)
from vllm.config.speculative import SpeculativeConfig


@pytest.fixture(autouse=True)
def isolated_legacy_values(monkeypatch):
    for name in SM70_DFLASH2_LEGACY_FIELDS:
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delitem(vars(envs), name, raising=False)
    envs.disable_envs_cache()
    yield
    envs.disable_envs_cache()


def test_two_engine_policies_and_draft_capture_do_not_use_global_environment(
    monkeypatch,
):
    qualified = Sm70DFlash2Config()
    qualified.resolve(qualified=True)
    ordinary = Sm70DFlash2Config()
    ordinary.resolve(qualified=False)
    assert sm70_dflash2_enabled("context_pipeline", qualified)
    assert not sm70_dflash2_enabled("context_pipeline", ordinary)
    assert "VLLM_SM70_DFLASH2_CONTEXT_PIPELINE" not in envs.os.environ
    draft_cfg = NS(speculative_config=NS(sm70_dflash2=qualified))
    assert capture_sm70_dflash2_config(draft_cfg) is qualified
    monkeypatch.setenv("VLLM_SM70_DFLASH2_CONTEXT_PIPELINE", "0")
    assert sm70_dflash2_enabled("context_pipeline", qualified)
    assert not sm70_dflash2_enabled("context_pipeline", ordinary)


def test_resolution_bypasses_a_stale_process_getter_cache(monkeypatch):
    envs.enable_envs_cache()
    assert not envs.VLLM_SM70_DFLASH2_CONTEXT_PIPELINE
    monkeypatch.setenv("VLLM_SM70_DFLASH2_CONTEXT_PIPELINE", "1")
    policy = Sm70DFlash2Config()
    policy.resolve(qualified=False)
    assert policy.context_pipeline


@pytest.mark.parametrize(
    "raw, expected", [("0", False), ("1", True), ("2", True), ("-1", True)]
)
def test_legacy_parser_and_configuration_precedence(monkeypatch, raw, expected):
    monkeypatch.setenv("VLLM_SM70_DFLASH2_CONTEXT_PIPELINE", raw)
    legacy = Sm70DFlash2Config()
    legacy.resolve(qualified=True)
    assert legacy.context_pipeline is expected
    configured = Sm70DFlash2Config(context_pipeline=not expected)
    configured.resolve(qualified=True)
    assert configured.context_pipeline is not expected
    configured.resolve(qualified=False)
    assert configured.context_pipeline is not expected


def test_norm_retains_its_policy_after_initialization_context_ends(monkeypatch):
    from vllm.model_executor.layers import layernorm

    monkeypatch.setattr(layernorm, "_sm70_gemma_long_prefill_available", lambda: True)
    monkeypatch.setattr(envs, "VLLM_SM70_FLASH_V100_0DOT3_COMPILE_GRAPH", True)
    cfg = VllmConfig(device_config=DeviceConfig(device="cpu"))
    policy = Sm70DFlash2Config()
    policy.resolve(qualified=True)
    cfg.speculative_config = NS(sm70_dflash2=policy)
    with set_current_vllm_config(cfg):
        norm = layernorm.GemmaRMSNorm(5120)
    assert norm._sm70_dflash2_policy is policy
    x = NS(
        is_cuda=True,
        dtype=torch.float16,
        device=torch.device("cuda:0"),
        ndim=2,
        shape=(8, 5120),
        is_contiguous=lambda: True,
    )
    weight = NS(
        dtype=torch.float16,
        device=x.device,
        ndim=1,
        shape=(5120,),
        is_contiguous=lambda: True,
    )
    assert layernorm._use_sm70_dflash2_fixed_gemma_rms(x, None, weight, policy)
    assert not layernorm._use_sm70_dflash2_fixed_gemma_rms(x, None, weight)


def _hash_subject(method, policy):
    return NS(
        method=method,
        sm70_dflash2=policy,
        mtp_expert_quantization=None,
        draft_model_config=None,
        use_local_argmax_reduction=None,
        use_dflash_family=lambda: method == "dflash",
        use_dspark=lambda: False,
        use_dflash_ddtree=lambda: False,
    )


def test_graph_options_affect_dflash_hash_and_ignore_unused_mtp_policy():
    policy = Sm70DFlash2Config()
    mtp = _hash_subject("mtp", policy)
    before = SpeculativeConfig.compute_hash(mtp)
    policy.resolve(qualified=False)
    assert SpeculativeConfig.compute_hash(mtp) == before
    policy = Sm70DFlash2Config()
    policy.resolve(qualified=True)
    dflash = _hash_subject("dflash", policy)
    before = SpeculativeConfig.compute_hash(dflash)
    policy.context_pipeline = False
    assert SpeculativeConfig.compute_hash(dflash) != before


@pytest.mark.parametrize(
    "rows,dtype,qkv,z,ba,reason",
    (
        (8, torch.float16, 2560, 1536, 12, None),
        (4, torch.float16, 2560, 1536, 12, "query_rows"),
        (8, torch.bfloat16, 2560, 1536, 12, "dtype"),
        (8, torch.float16, 5120, 3072, 24, "local_tail_layout"),
    ),
)
def test_combined_copy_capability_is_local_layout(rows, dtype, qkv, z, ba, reason):
    from vllm.model_executor.models.qwen3_5 import (
        _can_implement_sm70_combined_gdn_split,
    )

    assert _can_implement_sm70_combined_gdn_split(rows, dtype, qkv, z, ba) == (
        reason is None,
        reason,
    )


def test_loaded_report_includes_actual_verifier_flags_and_reason():
    from vllm.sm70_profiles.acceleration import loaded_sm70_preparations

    layer = torch.nn.Module()
    layer.enable_sm70_dflash2_fused_gdn_combined_split = False
    layer._sm70_dflash2_combined_split_reason = "local_tail_layout"
    row = loaded_sm70_preparations(layer)["variants"][""]
    assert row["flags"]["enable_sm70_dflash2_fused_gdn_combined_split"] is False
    assert row["reasons"]["_sm70_dflash2_combined_split_reason"] == "local_tail_layout"


def test_failed_order_switch_is_removed_and_dense_alias_warns(monkeypatch, caplog):
    assert (
        "VLLM_SM70_DFLASH2_QPN8_ALLOW_CANDIDATE_ORDER" not in envs.environment_variables
    )
    policy = Sm70DFlash2Config()
    assert not hasattr(policy, "qpn8_allow_candidate_order")
    assert not hasattr(policy, "qpn8_dense_order")
    monkeypatch.setenv("VLLM_SM70_DFLASH2_QPN8_DENSE_ORDER", "0")
    policy.resolve(qualified=True)
    assert "dense tie ordering is mandatory" in caplog.text
