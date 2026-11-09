# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from types import SimpleNamespace as NS

import pytest

from vllm import envs
from vllm.config import set_current_vllm_config
from vllm.config.kernel import KernelConfig
from vllm.model_executor.kernels import linear
from vllm.platforms import PlatformEnum
from vllm.sm70_profiles.acceleration import collect_worker_reports, linear_policy_report


@pytest.mark.parametrize("shape", ((5120, 4096), (2560, 1536), (5120, 8704)))
def test_real_selector_records_reasons_without_extra_probes(shape, monkeypatch):
    calls = []

    class MissingNative:
        @classmethod
        def is_supported(cls, cc):
            calls.append("missing")
            return False, "operator_missing"

    class Supported:
        @classmethod
        def is_supported(cls, cc):
            calls.append("supported")
            return True, None

        @classmethod
        def can_implement(cls, cfg):
            return True, None

    class LowerPriority:
        @classmethod
        def is_supported(cls, cc):
            raise AssertionError("Reporting must not probe lower-priority kernels")

    envs.disable_envs_cache()
    monkeypatch.setenv("VLLM_DISABLED_KERNELS", "")
    monkeypatch.setattr(linear, "current_platform", NS(_enum=PlatformEnum.CUDA))
    cfg = NS(kernel_config=KernelConfig())
    before = cfg.kernel_config.compute_hash()
    with set_current_vllm_config(cfg):
        selected = linear.choose_scaled_mm_linear_kernel(
            NS(weight_shape=shape),
            {PlatformEnum.CUDA: [MissingNative, Supported, LowerPriority]},
            compute_capability=70,
        )
    assert selected is Supported
    assert calls == ["missing", "supported"]
    row = next(iter(cfg.kernel_config.linear_kernel_selections.values()))
    assert row["paths"]["MissingNative"]["reason"].endswith("operator_missing.")
    assert row["paths"]["Supported"] == {"enabled": True, "reason": None}
    assert row["paths"]["LowerPriority"]["reason"] == "lower_priority"
    assert cfg.kernel_config.compute_hash() == before
    assert not KernelConfig().linear_kernel_selections


def test_policies_are_discovered_from_existing_configuration():
    policies = linear_policy_report(KernelConfig())
    assert set(policies) == {
        "sm70_nvfp4",
        "sm70_awq",
        "sm70_fp8",
        "sm70_marlin",
        "sm70_gguf",
        "sm70_ring",
        "sm70_sparse",
        "sm70_moe",
        "sm70_mxfp4",
        "sm70_runtime",
    }
    assert all(row["status"] == "runtime_guarded" for row in policies.values())


@pytest.mark.parametrize("family", ["awq", "fp8"])
def test_policy_report_serializes_resolved_moe_diagnostic_filters(family):
    """Loading resolves MoE dump policies whose parsed filters are frozensets."""
    config = KernelConfig()
    getattr(config.sm70_moe, family).resolve(family)
    policies = linear_policy_report(config)
    resolved = policies["sm70_moe"]["configuration"][family]
    compare = resolved["diagnostics"]["compare_policy"]
    assert compare is not None and compare["filters"]


def test_loaded_reports_collected_once_and_cached_for_http():
    calls = []
    expected = [{"rank": 0, "linear_kernel_selections": {"selected": "native"}}]

    class Engine:
        async def collective_rpc(self, method, timeout):
            calls.append(method)
            return expected

    cfg = NS(sm70_acceleration_report={"sm70": True})
    asyncio.run(collect_worker_reports(Engine(), cfg))
    asyncio.run(collect_worker_reports(Engine(), cfg))
    assert calls == ["get_sm70_acceleration_report"]
    assert cfg.sm70_acceleration_report["worker_routes"] == expected


def test_non_sm70_configuration_does_not_rpc():
    class Engine:
        async def collective_rpc(self, *args, **kwargs):
            raise AssertionError("not applicable")

    asyncio.run(
        collect_worker_reports(Engine(), NS(sm70_acceleration_report={"sm70": False}))
    )


def test_reporting_failure_does_not_change_serving_routes():
    class Engine:
        async def collective_rpc(self, *args, **kwargs):
            raise NotImplementedError("custom executor")

    cfg = NS(sm70_acceleration_report={"sm70": True})
    asyncio.run(collect_worker_reports(Engine(), cfg))
    assert "custom executor" in cfg.sm70_acceleration_report["worker_report_reason"]


def test_inspection_reads_final_kernel_and_preparation_flags():
    import torch

    from vllm.model_executor.kernels.linear.nvfp4.base import NvFp4LinearKernel
    from vllm.sm70_profiles.acceleration import (
        loaded_linear_kernels,
        loaded_sm70_preparations,
    )

    class FinalKernel(NvFp4LinearKernel):
        def __init__(self):
            self.config = NS(weight_shape=(128, 64))

        @classmethod
        def is_supported(cls, *args):
            raise AssertionError("inspection must not reprobe")

        @classmethod
        def can_implement(cls, *args):
            raise AssertionError("inspection must not reselect")

        def process_weights_after_loading(self, *args):
            raise AssertionError("inspection must not prepare")

        def apply_weights(self, *args, **kwargs):
            raise AssertionError("inspection must not execute")

    layer = torch.nn.Module()
    layer.scheme = NS(kernel=FinalKernel())
    layer._sm70_qwen38_dense_batch = True
    layer.register_buffer("_sm70_test_packed", torch.empty(2))
    model = torch.nn.Sequential(layer)
    rows = loaded_linear_kernels(model)
    row = next(iter(rows.values()))
    assert row["kernel"] == "FinalKernel" and row["layers"] == ["0"]
    preparations = loaded_sm70_preparations(model)
    assert preparations["variants"]["0"]["flags"] == {"_sm70_qwen38_dense_batch": True}
    assert preparations["variants"]["0"]["prepared_buffers"] == ["_sm70_test_packed"]
    assert preparations["packed_buffer_bytes"] == 0  # CPU buffers are not VRAM.


def test_native_gguf_report_reads_prepared_admission():
    import torch

    from vllm.sm70_profiles.acceleration import loaded_gguf_layers

    method = type("GGUFNativeMoEMethod", (), {})()
    method.weight_types = {"w1": 21, "w3": 21, "w2": 42}
    method.dp4a_admitted = True
    method.native_admission = {
        "projections": {"w1": {"operator": "ggml_moe_mmvq"}},
        "small_m_dp4a": {"enabled": True, "storage": "original"},
    }
    layer = torch.nn.Module()
    layer.quant_method = method
    row = loaded_gguf_layers(layer)[""]
    assert row["weight_types"] == [21, 21, 42]
    assert row["original_expert_projections"] == method.native_admission["projections"]
    assert row["small_m_dp4a"] == method.native_admission["small_m_dp4a"]
    assert row["acceleration_fallback_reason"] is None


def test_cuda_storage_deduplicates_target_draft_and_partial_views():
    import torch

    from vllm.sm70_profiles.acceleration import loaded_cuda_model_storage

    if not torch.cuda.is_available():
        pytest.skip("CUDA storage inspection requires CUDA")
    weight = torch.empty(64, device="cuda", dtype=torch.float16)
    target, draft = torch.nn.Module(), torch.nn.Module()
    target.register_parameter("weight", torch.nn.Parameter(weight))
    target.register_buffer("partial_view", weight[16:32])
    target.register_buffer("cpu_metadata", torch.empty(100))
    draft.register_parameter("shared_weight", target.weight)
    draft.register_buffer(
        "own_weight", torch.empty(32, device="cuda", dtype=torch.uint8)
    )
    report = loaded_cuda_model_storage({"target": target, "draft": draft})
    assert report["bytes"] == 160
    assert len(report["storages"]) == 2
    assert report["storages"][0]["bytes"] == 128
    assert report["storages"][0]["names"] == [
        "target.weight",
        "target.partial_view",
        "draft.shared_weight",
    ]


def test_cache_storage_deduplicates_shared_staging_and_owner_aliases():
    import torch

    from vllm.sm70_profiles.acceleration import loaded_qsa_cache_storage

    shared = torch.empty(64, dtype=torch.uint8)
    owners = []
    for reference in (False, True):
        hot = {
            key: torch.empty(4, dtype=torch.int32)
            for key in (
                "hot_values",
                "tags",
                "stamps",
                "hands",
                "page_slots",
                "epoch",
                "_stats",
            )
        }
        workspace = {
            key: shared[8:24]
            for key in (
                "staging",
                "remapped",
                "requests",
                "positions",
                "lengths",
                "initial",
                "resolved",
            )
        }
        owners.append(
            NS(
                host_kv=NS(
                    **hot,
                    **workspace,
                    device_reference=reference,
                    device_history_workspace=(shared,),
                    history=torch.empty(32, dtype=torch.uint8),
                    scales=torch.empty(2),
                )
            )
        )
    report = loaded_qsa_cache_storage(
        {"target": owners[0], "alias": owners[0], "draft": owners[1]}
    )
    assert report == {
        "hot_bytes": 224,
        "workspace_bytes": 64,
        "device_history_bytes": 40,
    }
