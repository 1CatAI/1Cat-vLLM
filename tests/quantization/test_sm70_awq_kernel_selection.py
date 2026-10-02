# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools.sm70_awq_route_snapshot import snapshot
from vllm import envs
from vllm.config.kernel import KernelConfig, Sm70AwqConfig
from vllm.model_executor.kernels import linear
from vllm.model_executor.kernels.linear import choose_mp_linear_kernel
from vllm.model_executor.kernels.linear.mixed_precision.sm70_awq import (
    Sm70AwqLinearLayerConfig,
    TurboMindAwqLinearKernel,
)
from vllm.platforms import PlatformEnum
from vllm.scalar_type import scalar_types

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def clean_route_environment(monkeypatch):
    monkeypatch.setattr(
        linear, "current_platform", SimpleNamespace(_enum=PlatformEnum.CUDA)
    )
    for name in list(os.environ):
        if name.startswith("VLLM_"):
            monkeypatch.delenv(name)
    envs.disable_envs_cache()
    yield
    envs.disable_envs_cache()


def config():
    policy = Sm70AwqConfig()
    policy.resolve()
    return Sm70AwqLinearLayerConfig(
        full_weight_shape=(2560, 3584),
        partition_weight_shape=(2560, 3584),
        weight_type=scalar_types.uint4,
        act_type=torch.float16,
        group_size=32,
        zero_points=True,
        has_g_idx=False,
        policy=policy,
    )


def test_retained_category_snapshot():
    golden = Path(__file__).parent / "data/sm70_awq_routes.json"
    assert snapshot() == json.loads(golden.read_text())


def test_shared_selector_and_provider(monkeypatch):
    monkeypatch.setattr(torch.ops, "_C", SimpleNamespace(awq_sm70_prepare=True))
    assert choose_mp_linear_kernel(config(), 70) is TurboMindAwqLinearKernel


@pytest.mark.parametrize("group", [32, 64, 128])
def test_capability_independent_of_user_policy(monkeypatch, group):
    monkeypatch.setattr(torch.ops, "_C", SimpleNamespace(awq_sm70_prepare=True))
    cfg = replace(config(), group_size=group, policy=Sm70AwqConfig(enabled=False))
    assert TurboMindAwqLinearKernel.can_implement(cfg) == (True, None)


def test_capability_failure_reasons(monkeypatch):
    monkeypatch.setattr(torch.ops, "_C", SimpleNamespace())
    supported, reason = TurboMindAwqLinearKernel.can_implement(config())
    assert not supported and "native" in reason
    supported, reason = TurboMindAwqLinearKernel.can_implement(
        replace(config(), group_size=16)
    )
    assert not supported and "group_size" in reason
    supported, reason = TurboMindAwqLinearKernel.can_implement(
        replace(config(), zero_points=False)
    )
    assert not supported and "asymmetric" in reason
    supported, reason = TurboMindAwqLinearKernel.can_implement(
        replace(config(), act_type=torch.bfloat16)
    )
    assert not supported and "float16" in reason


def test_disable_uses_common_selector(monkeypatch):
    monkeypatch.setattr(torch.ops, "_C", SimpleNamespace(awq_sm70_prepare=True))
    monkeypatch.setenv("VLLM_DISABLED_KERNELS", "TurboMindAwqLinearKernel")
    envs.disable_envs_cache()
    with pytest.raises(ValueError, match="disabled"):
        choose_mp_linear_kernel(config(), 70)


def test_alias_precedence_and_no_environment_mutation(monkeypatch):
    monkeypatch.setenv("VLLM_SM70_AWQ_TURBOMIND", "0")
    monkeypatch.setenv("VLLM_SM70_AWQ_PREFILL_EXACT_DENSE", "0")
    monkeypatch.setenv("VLLM_SM70_AWQ_MLP_ENGINE", "1")
    before = dict(os.environ)
    legacy = Sm70AwqConfig()
    legacy.resolve()
    assert not legacy.enabled and not legacy.prefill_exact_dense and legacy.fused_silu
    explicit = Sm70AwqConfig(enabled=True, fused_silu=False)
    explicit.resolve()
    assert explicit.enabled and not explicit.fused_silu
    assert dict(os.environ) == before


def test_two_engines_have_isolated_policy_and_hash(monkeypatch):
    first = KernelConfig()
    first.sm70_awq.resolve()
    monkeypatch.setenv("VLLM_SM70_AWQ_TURBOMIND", "0")
    envs.disable_envs_cache()
    second = KernelConfig()
    second.sm70_awq.resolve()
    first.sm70_awq.resolve()
    assert first.sm70_awq.enabled and not second.sm70_awq.enabled
    assert first.compute_hash() != second.compute_hash()


def test_unused_awq_policy_keeps_the_existing_graph_fingerprint():
    cfg = KernelConfig()
    cfg.sm70_nvfp4.resolve(qualified=True)
    # Independent pre-migration KernelConfig result for this policy. A new
    # format must not salt NVFP4's capture key before it is actually selected.
    assert cfg.compute_hash() == (
        "352727bc1599123198bf6f01ec4a62c0d27ac5195a67ac088006099a88fbdef7"
    )


@pytest.mark.parametrize(
    "backend,enabled", [("auto", True), ("turbomind", True), ("marlin", False)]
)
def test_shared_legacy_backend_precedence(monkeypatch, backend, enabled):
    monkeypatch.setenv("VLLM_SM70_QUANT_BACKEND", backend)
    cfg = Sm70AwqConfig()
    cfg.resolve()
    assert cfg.enabled == enabled
