# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace as NS

import pytest
import torch

from vllm import envs
from vllm.config.compilation import CompilationMode, CUDAGraphMode
from vllm.config.kernel import KernelConfig
from vllm.config.vllm import _SM70_BATCH_GEMM_DEFAULTS, _SM70_DFLASH2_VERIFIER_DEFAULTS
from vllm.sm70_profiles import acceleration as acc


@pytest.fixture
def config(monkeypatch):
    envs.disable_envs_cache()
    # Other operator tests may monkeypatch a dynamic env attribute. Remove
    # their restored concrete values so this fixture exercises actual getters.
    for name in _SM70_DFLASH2_VERIFIER_DEFAULTS:
        monkeypatch.delitem(vars(envs), name, raising=False)
    for name, value in {
        **_SM70_BATCH_GEMM_DEFAULTS,
        **_SM70_DFLASH2_VERIFIER_DEFAULTS,
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("VLLM_FLASH_V100_DECODE_PARTITION_SIZE", raising=False)
    monkeypatch.setenv("VLLM_SM70_REQUIRE_PROFILE_ACCELERATION", "0")
    monkeypatch.setenv("VLLM_FLASH_V100_E4M3_GROUPED_FP32", "1")
    monkeypatch.setenv("VLLM_SM70_DFLASH2_TAIL_CUDAGRAPHS", "1")
    monkeypatch.setenv("VLLM_FLASH_V100_PREFILL_D256_GQA_V37", "0")
    monkeypatch.setattr(acc, "_is_sm70", lambda cfg: True)
    monkeypatch.setattr(
        acc,
        "_native_capabilities",
        lambda page, policy: dict.fromkeys(
            (
                "grouped_fp32",
                "fp16_grouped",
                "long_operator",
                "long_enabled",
                "page_supported",
                "scalar",
                "q8000",
            ),
            True,
        ),
    )
    cfg = NS(
        kernel_config=KernelConfig(),
        model_config=NS(
            architectures=["Qwen3_5ForConditionalGeneration"],
            dtype=torch.float16,
            enforce_eager=False,
            hf_text_config=NS(
                hidden_size=5120,
                num_attention_heads=24,
                num_key_value_heads=4,
                head_dim=256,
            ),
        ),
        speculative_config=NS(
            method="dflash",
            num_speculative_tokens=7,
            draft_model_config=NS(hf_config=NS(dflash_config={"selector_top_k": 16})),
        ),
        parallel_config=NS(
            tensor_parallel_size=4,
            pipeline_parallel_size=1,
            enable_dbo=False,
            ubatch_size=1,
        ),
        cache_config=NS(cache_dtype="fp8_e4m3", block_size=2048),
        scheduler_config=NS(max_num_batched_tokens=8192),
        compilation_config=NS(
            mode=CompilationMode.VLLM_COMPILE,
            cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
            inductor_compile_config={},
        ),
    )
    from vllm.config import AttentionConfig, ObservabilityConfig, OffloadConfig
    from vllm.config.execution_policy import CommunicationPolicy, GraphPolicy
    from vllm.config.policy_defaults import PolicyDefaults

    cfg.attention_config = AttentionConfig()
    cfg.observability_config = ObservabilityConfig()
    cfg.offload_config = OffloadConfig()
    cfg.compilation_config.runtime = GraphPolicy()
    cfg.parallel_config.communication = CommunicationPolicy()
    cfg.runtime_default_sources = {}
    from vllm.config.sm70_dflash2 import Sm70DFlash2Config
    from vllm.config.speculative_sampling import resolve_sampling_policy

    cfg.speculative_config.sm70_dflash2 = Sm70DFlash2Config()
    cfg.speculative_config.sm70_dflash2.resolve(qualified=False)
    cfg.speculative_config.sampling_policy = resolve_sampling_policy()
    cfg.kernel_config.gdn.projection.resolve()
    PolicyDefaults(cfg).finish()
    return cfg


def test_e5m2_reason(config):
    config.cache_config.cache_dtype = "fp8_e5m2"
    paths = acc.build_report(config)["paths"]
    for name in ("e4m3_grouped_fp32", "long_context", "scalar_tail"):
        assert paths[name]["reason"] == "kv_dtype"
    assert paths["dflash2_verifier"]["enabled"]


@pytest.mark.parametrize("page", [16, 784, 816, 896, 1024, 2048])
def test_bm32_report_admits_aligned_pages(config, monkeypatch, page):
    config.cache_config.cache_dtype = "auto"
    config.cache_config.block_size = page
    monkeypatch.setenv("VLLM_FLASH_V100_PREFILL_D256_LOW_SMEM", "1")
    monkeypatch.setenv("VLLM_FLASH_V100_PREFILL_D256_BM32_PHASE", "1")
    row = acc._bm32_paged_prefill_report(config, {"bm32_aligned_pages": True})
    assert row["enabled"]
    assert row["scope"] == "configured_native_capability"
    assert "at least 32 query rows" in row["runtime_guards"]


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"cache_dtype": "fp8_e4m3"}, "kv_dtype"),
        ({"dtype": torch.bfloat16}, "kv_dtype"),
        ({"head_dim": 128}, "head_dim"),
        ({"block_size": 815}, "page_alignment"),
        ({"block_size": 0}, "page_alignment"),
        ({"native": False}, "operator_missing:aligned_bm32_paged_prefill"),
        ({"phase": "0"}, "user_override"),
        ({"low_smem": "0"}, "user_override"),
    ],
)
def test_bm32_report_explains_fallback(config, monkeypatch, change, reason):
    config.cache_config.cache_dtype = change.get("cache_dtype", "auto")
    config.cache_config.block_size = change.get("block_size", 816)
    config.model_config.dtype = change.get("dtype", torch.float16)
    config.model_config.hf_text_config.head_dim = change.get("head_dim", 256)
    from vllm.config.flash_v100 import FlashV100Options

    # Explain a new resolved policy; reports do not sample the live environment.
    options = FlashV100Options(
        prefill_d256_low_smem=change.get("low_smem", "1") != "0",
        prefill_d256_bm32_phase=change.get("phase", "1") != "0",
    )
    options.resolve()
    config.attention_config.flash_v100.options = options
    row = acc._bm32_paged_prefill_report(
        config, {"bm32_aligned_pages": change.get("native", True)}
    )
    assert not row["enabled"]
    assert row["reason"] == reason


def test_tp2_reason(config):
    config.parallel_config.tensor_parallel_size = 2
    assert acc.build_report(config)["paths"]["profile_hardware"]["reason"] == (
        "contract_mismatch:tensor_parallel_size=2≠4"
    )


def test_five_speculative_tokens_reason(config):
    config.speculative_config.num_speculative_tokens = 5
    assert acc.build_report(config)["paths"]["dflash2_verifier"]["reason"] == (
        "contract_mismatch:num_speculative_tokens=5≠7"
    )


def test_budget_reason(config):
    config.scheduler_config.max_num_batched_tokens = 4096
    assert acc.build_report(config)["paths"]["q8000_prefill"]["reason"] == "budget<8000"


def test_missing_operator_reason(config, monkeypatch):
    monkeypatch.setattr(
        acc,
        "_native_capabilities",
        lambda page, policy: dict.fromkeys(
            (
                "grouped_fp32",
                "long_operator",
                "long_enabled",
                "page_supported",
                "scalar",
                "q8000",
            ),
            False,
        ),
    )
    paths = acc.build_report(config)["paths"]
    assert paths["e4m3_grouped_fp32"]["reason"].startswith("operator_missing:")
    assert paths["q8000_prefill"]["reason"].startswith("operator_missing:")


def test_user_override_and_strict_failure(config, monkeypatch):
    config.speculative_config.sm70_dflash2.verify_fastpath = False
    assert (
        acc.build_report(config)["paths"]["dflash2_verifier"]["reason"]
        == "user_override"
    )
    config.observability_config.runtime_trace.require_profile_acceleration = True
    with pytest.raises(ValueError, match="dflash2_verifier: user_override"):
        acc.log_and_validate(config)


def test_strict_target_requirements_do_not_apply_to_internal_draft(config, monkeypatch):
    config.observability_config.runtime_trace.require_profile_acceleration = True
    config.cache_config.cache_dtype = "auto"
    native_capabilities = acc._native_capabilities
    monkeypatch.setattr(
        acc,
        "_native_capabilities",
        lambda page, policy: {
            **native_capabilities(page, policy),
            "fp16_grouped": False,
        },
    )
    with pytest.raises(ValueError, match="fp16_grouped_fp32: operator_missing"):
        acc.log_and_validate(config)
    config.is_speculative_draft = True
    report = acc.log_and_validate(config)
    assert report["scope"] == "internal_draft_config"
    assert report["expected_acceleration"] == []
    assert report["expected_failures"] == []
    assert not report["paths"]["e4m3_grouped_fp32"]["enabled"]


def test_flashnext_report_uses_its_own_required_paths_and_ignores_kv_dtype(
    config, monkeypatch
):
    config.model_config.architectures = ["Qwen4ExpForCausalLM"]
    config.speculative_config = NS(method="mtp", num_speculative_tokens=3)
    config.kernel_config.layer_execution.fp16_gemv = True
    config.kernel_config.layer_execution.fused_gdn_input = True
    config.kernel_config.layer_execution.fused_hc = True
    config.observability_config.runtime_trace.require_profile_acceleration = True
    # MoE stream policy is independent; target FP16 projections do not read KV.
    config.kernel_config.layer_execution.shared_moe_overlap = False
    config.parallel_config.communication.moe_add_allreduce = False
    for dtype in ("auto", "fp8_e4m3"):
        config.cache_config.cache_dtype = dtype
        report = acc.log_and_validate(config)
        assert report["profile"] == "qwen4exp_fp16_decode"
        assert report["expected_acceleration"] == [
            "qwen38_decode",
            "batch_gemm",
            "compile_graph",
        ] + (["fp16_grouped_fp32"] if dtype == "auto" else [])
        assert not report["expected_failures"]


def test_model_less_component_config_does_not_validate_a_target(config, monkeypatch):
    config.model_config = None
    config.observability_config.runtime_trace.require_profile_acceleration = True

    def unexpected_probe(_page):
        raise AssertionError("component configs must not probe target operators")

    monkeypatch.setattr(acc, "_native_capabilities", unexpected_probe)
    report = acc.log_and_validate(config)
    assert report["scope"] == "component_config"
    assert report["expected_acceleration"] == []
    assert report["expected_failures"] == []
    assert report["paths"] == {}


def test_non_sm70_is_not_applicable(config, monkeypatch):
    monkeypatch.setattr(acc, "_is_sm70", lambda cfg: False)
    report = acc.log_and_validate(config)
    assert not report["expected_failures"]
    assert all(row["reason"] == "not_applicable" for row in report["paths"].values())
    config.observability_config.runtime_trace.require_profile_acceleration = True
    with pytest.raises(ValueError, match="not_applicable"):
        acc.log_and_validate(config)


def test_success_does_not_mutate_compile_hash_input(config):
    config.additional_config = {"customer_setting": "preserved"}
    report = acc.log_and_validate(config)
    assert not report["expected_failures"]
    assert config.additional_config == {"customer_setting": "preserved"}
    assert config.sm70_acceleration_report == report


def test_real_config_diagnostic_field_does_not_affect_hash(monkeypatch):
    from vllm import platforms
    from vllm.config import DeviceConfig, VllmConfig
    from vllm.platforms.interface import UnspecifiedPlatform

    monkeypatch.setattr(platforms, "current_platform", UnspecifiedPlatform())
    cfg = VllmConfig(device_config=DeviceConfig(device="cpu"))
    report = acc.log_and_validate(cfg)
    assert report["sm70"] is False
    before = cfg.compute_hash()
    cfg.sm70_acceleration_report["diagnostic_test"] = True
    assert cfg.compute_hash() == before


def test_read_only_endpoint_uses_api_key_authentication(config):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from vllm.entrypoints.openai.server_utils import AuthenticationMiddleware
    from vllm.entrypoints.serve.sm70.api_router import attach_router

    report = acc.log_and_validate(config)
    app = FastAPI()
    app.state.vllm_config = config
    attach_router(app)
    app.add_middleware(AuthenticationMiddleware, tokens=["test-key"])
    with TestClient(app) as client:
        assert client.get("/v1/sm70/acceleration").status_code == 401
        assert (
            client.get(
                "/v1/sm70/acceleration", headers={"Authorization": "Bearer wrong"}
            ).status_code
            == 401
        )
        response = client.get(
            "/v1/sm70/acceleration", headers={"Authorization": "Bearer test-key"}
        )
        assert response.status_code == 200
        assert response.json() == report


def test_larger_batch_tuning_budget_is_enabled(config, monkeypatch):
    monkeypatch.setenv("VLLM_SM70_NVFP4_DENSE_TUNE_MAX_M", "128")
    assert acc.build_report(config)["paths"]["batch_gemm"]["enabled"]


def test_malformed_draft_contract_has_reason(config):
    config.speculative_config.draft_model_config.hf_config.dflash_config = ["invalid"]
    assert acc.build_report(config)["paths"]["dflash2_verifier"]["reason"] == (
        "contract_mismatch:selector_top_k=None≠16"
    )


def test_release_status_counts_only_expected_paths(config, monkeypatch):
    config.runtime_default_sources["VLLM_DISABLE_COMPILE_CACHE"] = [
        {"source": "process_startup", "raw": "1"}
    ]
    report = acc.build_report(config)
    assert report["paths"]["qwen38_decode"]["reason"] == "not_applicable"
    assert report["paths"]["compile_cache"]["reason"] == "compile_cache_disabled"
    assert "qwen38_decode" not in report["expected_acceleration"]
    assert "compile_cache" not in report["expected_acceleration"]
    assert all(
        report["paths"][name]["enabled"] for name in report["expected_acceleration"]
    )


@pytest.mark.parametrize("disabled_by", ["eager", "mode", "config", "torch"])
def test_compile_cache_reports_effective_disable(config, monkeypatch, disabled_by):
    monkeypatch.setenv("VLLM_DISABLE_COMPILE_CACHE", "0")
    monkeypatch.setattr(torch._inductor.config, "force_disable_caches", False)
    assert acc.build_report(config)["paths"]["compile_cache"]["enabled"]
    if disabled_by == "eager":
        config.model_config.enforce_eager = True
    elif disabled_by == "mode":
        config.compilation_config.mode = CompilationMode.NONE
    elif disabled_by == "config":
        config.compilation_config.inductor_compile_config["force_disable_caches"] = True
    else:
        monkeypatch.setattr(torch._inductor.config, "force_disable_caches", True)
    row = acc.build_report(config)["paths"]["compile_cache"]
    assert not row["enabled"]
    assert row["switches"]["torch_force_disable_caches"] == (disabled_by == "torch")
    assert row["reason"] == (
        "compilation_disabled"
        if disabled_by in ("eager", "mode")
        else "inductor_cache_disabled"
    )


def test_resolved_linear_policy_is_reported_without_reparsing_env(config, monkeypatch):
    config.kernel_config.sm70_nvfp4.resolve(qualified=True)
    monkeypatch.setenv("VLLM_SM70_NVFP4_QPN2", "0")
    envs.disable_envs_cache()
    row = acc.build_report(config)["linear_kernel_policy"]
    assert row["status"] == "runtime_guarded"
    assert row["configuration"]["qpn2"] is True
    assert row["configuration"]["qualified"] is True
    assert row["qpn2_reason"] is None
    assert row["default_qualification_reason"] is None


def test_unqualified_linear_default_reports_reason(config):
    config.kernel_config.sm70_nvfp4.resolve(qualified=False)
    row = acc.build_report(config)["linear_kernel_policy"]
    assert not row["configuration"]["qpn2"]
    assert row["qpn2_reason"] == "disabled_by_configuration_or_legacy_override"
    assert row["default_qualification_reason"] == (
        "draft_selector_state_contract_not_quality_qualified"
    )


def test_flash_next_batch_memory_and_explicit_off(config, monkeypatch):
    config.model_config.architectures = ["Qwen4ExpForCausalLM"]
    config.speculative_config = NS(
        method="mtp",
        num_speculative_tokens=4,
        sampling_policy=config.speculative_config.sampling_policy,
    )
    config.model_config.hf_text_config = NS(
        hidden_size=2560,
        hc_count=4,
        hc_lowrank=320,
        num_hidden_layers=48,
        mtp_num_hidden_layers=1,
        layer_types=["linear_attention"] * 36 + ["full_attention"] * 12,
    )
    report = acc.build_report(config)["flash_next_batch"]
    memory = report["packed_weight_memory"]
    assert memory["components"]["gdn_input"] == int(725.625 * 1024**2)
    assert memory["components"]["hc_target"] == 330 * 1024**2
    assert memory["components"]["hc_draft"] == int(6.875 * 1024**2)
    assert memory["components"]["router"] == int(122.5 * 1024**2)
    assert memory["components"]["shared_expert"] == int(76.5625 * 1024**2)
    assert len(report["controls"]) == 14
    for name in report["controls"]:
        monkeypatch.setenv(name, "0")
    # Existing engines retain initialized choices despite later env changes.
    assert acc.build_report(config)["flash_next_batch"]["controls"][
        "VLLM_SM70_QWEN38_BATCH_FASTPATH"
    ]["enabled"]
    from vllm.config.execution_policy import LayerExecutionPolicy

    policy = LayerExecutionPolicy()
    policy.resolve()
    config.kernel_config.layer_execution = policy
    from vllm.config.gdn_projection import GdnProjectionConfig
    from vllm.config.sm70_sparse import Sm70SparseConfig
    from vllm.config.speculative_sampling import resolve_sampling_policy

    config.kernel_config.sm70_sparse = Sm70SparseConfig()
    config.kernel_config.sm70_sparse.resolve()
    config.kernel_config.gdn.projection = GdnProjectionConfig()
    config.kernel_config.gdn.projection.resolve()
    config.speculative_config.sampling_policy = resolve_sampling_policy()
    report = acc.build_report(config)["flash_next_batch"]
    assert report["packed_weight_memory"]["total_bytes"] == 0
    assert all(row["reason"] == "user_override" for row in report["controls"].values())


def test_flash_next_memory_does_not_guess_other_tp_layout(config):
    config.model_config.architectures = ["Qwen4ExpForCausalLM"]
    config.parallel_config.tensor_parallel_size = 2
    memory = acc.build_report(config)["flash_next_batch"]["packed_weight_memory"]
    assert memory["total_bytes"] is None
    assert memory["reason"] == "estimate_requires_qualified_reference_layout"


@pytest.mark.parametrize("format_name", ("sm70_awq", "sm70_fp8"))
def test_quantized_models_report_their_policy_without_claiming_nvfp4_profile(
    config, format_name
):
    policy = getattr(config.kernel_config, format_name)
    policy.resolve()
    config.speculative_config = None
    report = acc.log_and_validate(config)
    assert report["profile"] is None
    assert report["expected_acceleration"] == []
    assert report["linear_kernel_policies"][format_name]["configuration"]["resolved"]


def test_report_never_evaluates_legacy_getters(config, monkeypatch):
    from unittest.mock import Mock

    expected = acc.build_report(config)
    from vllm.envs_metadata import EnvVar

    monkeypatch.setattr(
        EnvVar, "__call__", Mock(side_effect=AssertionError("report getter"))
    )
    monkeypatch.setattr(
        envs, "__getattr__", Mock(side_effect=AssertionError("report env attribute"))
    )
    assert acc.build_report(config) == expected
