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
    assert set(policies) == {"sm70_nvfp4", "sm70_awq", "sm70_fp8"}
    assert all(row["status"] == "runtime_guarded" for row in policies.values())


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
