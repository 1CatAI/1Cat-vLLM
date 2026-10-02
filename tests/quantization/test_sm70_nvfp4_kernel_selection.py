# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from tools.sm70_route_snapshot import make_config, snapshot, snapshot_layer
from vllm import envs
from vllm.config.kernel import KernelConfig, Sm70NvFp4Config
from vllm.model_executor.kernels.linear import select_sm70_nvfp4_linear_kernel
from vllm.model_executor.kernels.linear.nvfp4 import sm70
from vllm.model_executor.models.config import sm70_dflash2_nvfp4_qualified

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def clean_route_environment(monkeypatch):
    for name in list(os.environ):
        if name.startswith("VLLM_"):
            monkeypatch.delenv(name)
    envs.disable_envs_cache()
    yield
    envs.disable_envs_cache()


def config():
    policy = Sm70NvFp4Config()
    policy.resolve(qualified=True)
    return sm70.Sm70NvFp4LinearLayerConfig(
        5120,
        4120,
        (4120, 2560),
        policy,
        qpn2_qualified=True,
        qpn4_qualified=False,
        gated_silu=False,
    )


def test_retained_category_snapshot():
    golden = Path(__file__).parent / "data/sm70_nvfp4_routes.json"
    assert snapshot() == json.loads(golden.read_text())


def test_qpn2_capability_is_independent_of_workload_and_projection_policy(monkeypatch):
    monkeypatch.setattr(sm70, "_missing_qpn2_ops", lambda: [])
    original = config()
    # A compatible local tensor is implementable even outside the temporary
    # whole-workload whitelist. The selector separately enforces that policy.
    other = replace(original, qpn2_qualified=False, policy=Sm70NvFp4Config(qpn2=False))
    assert sm70.Qpn2NvFp4LinearKernel.can_implement(original) == (True, None)
    assert sm70.Qpn2NvFp4LinearKernel.can_implement(other) == (True, None)
    kernel, reasons = select_sm70_nvfp4_linear_kernel(other, compute_capability=70)
    assert kernel is sm70.TurboMindNvFp4LinearKernel
    assert "configuration" in reasons["Qpn2NvFp4LinearKernel"]


@pytest.mark.parametrize(
    "k,n,shape,gated,reason",
    [
        (5120, 4120, (4120,), False, "rank two"),
        (5120, 4120, (4120, 2559), False, "K % 128"),
        (4096, 4120, (4120, 2560), False, "disagree"),
        (5120, 4120, (4120, 2560), True, "N % 64"),
    ],
)
def test_capability_reasons(k, n, shape, gated, reason):
    cfg = replace(
        config(), input_size=k, output_size=n, weight_shape=shape, gated_silu=gated
    )
    available, failure = sm70.Qpn2NvFp4LinearKernel.can_implement(cfg)
    assert not available and reason in failure


def test_missing_native_and_disable_use_same_selector(monkeypatch):
    monkeypatch.setattr(sm70, "_missing_qpn2_ops", lambda: ["missing_test_op"])
    selected, reasons = select_sm70_nvfp4_linear_kernel(config(), compute_capability=70)
    assert selected is sm70.TurboMindNvFp4LinearKernel
    assert "missing_test_op" in reasons["Qpn2NvFp4LinearKernel"]
    monkeypatch.setenv("VLLM_DISABLED_KERNELS", "Qpn2NvFp4LinearKernel")
    envs.disable_envs_cache()
    selected, reasons = select_sm70_nvfp4_linear_kernel(config(), compute_capability=70)
    assert selected is sm70.TurboMindNvFp4LinearKernel
    assert "VLLM_DISABLED_KERNELS" in reasons["Qpn2NvFp4LinearKernel"]


def test_non_sm70_never_selects_weight_only_kernels():
    selected, reasons = select_sm70_nvfp4_linear_kernel(config(), compute_capability=80)
    assert selected is None
    assert "requires SM70" in reasons["TurboMindNvFp4LinearKernel"]


def test_explicit_config_and_legacy_compatibility_without_environment_writes(
    monkeypatch,
):
    monkeypatch.setenv("VLLM_SM70_NVFP4_QPN2", "0")
    monkeypatch.setenv("VLLM_SM70_NVFP4_QPN2_PREFILL_MIN_M", "9")
    before = dict(os.environ)
    legacy = Sm70NvFp4Config()
    legacy.resolve(qualified=True)
    assert not legacy.qpn2 and legacy.prefill_min_m == 9
    explicit = Sm70NvFp4Config(qpn2=True, prefill_min_m=1024)
    explicit.resolve(qualified=True)
    assert explicit.qpn2 and explicit.prefill_min_m == 1024
    assert dict(os.environ) == before


def test_two_engine_policies_remain_isolated_after_environment_change(monkeypatch):
    first = KernelConfig()
    first.sm70_nvfp4.resolve(qualified=True)
    monkeypatch.setenv("VLLM_SM70_NVFP4_QPN2", "0")
    envs.disable_envs_cache()
    second = KernelConfig()
    second.sm70_nvfp4.resolve(qualified=True)
    first.sm70_nvfp4.resolve(qualified=False)
    assert first.sm70_nvfp4.qpn2 and not second.sm70_nvfp4.qpn2
    assert first.compute_hash() != second.compute_hash()


def test_model_policy_retains_state_contract_but_not_tp_kv_concurrency():
    for tp, kv, concurrency in [(2, "fp8_e5m2", 8), (4, "float16", 1)]:
        cfg = make_config("dflash", tp, concurrency, kv, 4096)
        assert sm70_dflash2_nvfp4_qualified(cfg)
        cfg.speculative_config.num_speculative_tokens = 5
        assert not sm70_dflash2_nvfp4_qualified(cfg)


@pytest.mark.parametrize(
    "shared_native,version,batch",
    [(False, 1, False), (True, 0, False), (True, 1, False), (True, 1, True)],
)
def test_old_native_and_batch_layout_fallbacks(shared_native, version, batch):
    cfg = make_config("dflash", 4, 4, "fp8_e4m3", 8192)
    result = snapshot_layer(
        __import__("tools.sm70_route_snapshot", fromlist=["candidate"]).candidate,
        cfg,
        shared_native=shared_native,
        compact_version=version,
        batch=batch,
    )
    assert result["qpn2"]
    assert result["shared"] == shared_native
    assert result["compact_scales"] == (shared_native and version >= 1 and not batch)
    assert result["batch_prescale"] == (shared_native and batch)
