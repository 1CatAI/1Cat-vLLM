# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from tools.sm70_fp8_route_snapshot import snapshot
from vllm import envs
from vllm.config.kernel import KernelConfig, Sm70Fp8Config
from vllm.model_executor.kernels import linear
from vllm.model_executor.kernels.linear.scaled_mm import sm70_fp8
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8Dynamic128Sym,
    kFp8Static128BlockSym,
)
from vllm.platforms import PlatformEnum

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    for name in list(os.environ):
        if name.startswith("VLLM_"):
            monkeypatch.delenv(name)
    envs.disable_envs_cache()
    monkeypatch.setattr(linear, "current_platform", NS(_enum=PlatformEnum.CUDA))
    monkeypatch.setattr(sm70_fp8, "current_platform", NS(is_cuda=lambda: True))
    yield
    envs.disable_envs_cache()


def config():
    return sm70_fp8.Sm70Fp8LinearLayerConfig(
        weight_quant_key=kFp8Static128BlockSym,
        activation_quant_key=kFp8Dynamic128Sym,
        weight_shape=(5120, 4352),
        input_dtype=torch.float16,
        out_dtype=torch.float16,
    )


def test_legacy_dispatch_snapshot():
    assert snapshot() == json.loads(
        (Path(__file__).parent / "data/sm70_fp8_routes.json").read_text()
    )


def test_common_selector_and_provider(monkeypatch):
    monkeypatch.setattr(torch.ops, "_C", NS(fp8_sm70_prepare=True))
    selected = linear.choose_scaled_mm_linear_kernel(
        config(), linear._POSSIBLE_FP8_BLOCK_KERNELS, compute_capability=70
    )
    assert selected is sm70_fp8.TurboMindFp8LinearKernel
    assert selected in linear._LINEAR_BACKEND_KERNEL_MAP["turbomind"]
    assert sm70_fp8.TurboMindFp8LinearKernel.is_supported(75)[0] is False


def test_native_dtype_and_layout_reasons(monkeypatch):
    monkeypatch.setattr(torch.ops, "_C", NS())
    assert "native" in sm70_fp8.TurboMindFp8LinearKernel.can_implement(config())[1]
    monkeypatch.setattr(torch.ops, "_C", NS(fp8_sm70_prepare=True))
    assert (
        "float16"
        in sm70_fp8.TurboMindFp8LinearKernel.can_implement(
            replace(config(), input_dtype=torch.bfloat16)
        )[1]
    )
    ordinary = linear.FP8ScaledMMLinearLayerConfig(
        weight_quant_key=kFp8Static128BlockSym,
        activation_quant_key=kFp8Dynamic128Sym,
        weight_shape=(5120, 4352),
        input_dtype=torch.float16,
        out_dtype=torch.float16,
    )
    assert "layout" in sm70_fp8.TurboMindFp8LinearKernel.can_implement(ordinary)[1]


def test_shared_disable_reports_reason(monkeypatch):
    monkeypatch.setattr(torch.ops, "_C", NS(fp8_sm70_prepare=True))
    monkeypatch.setenv("VLLM_DISABLED_KERNELS", "TurboMindFp8LinearKernel")
    envs.disable_envs_cache()
    with pytest.raises(ValueError, match="disabled"):
        linear.choose_scaled_mm_linear_kernel(
            config(), linear._POSSIBLE_FP8_BLOCK_KERNELS, compute_capability=70
        )


@pytest.mark.parametrize("generic", [None, "0", "1"])
@pytest.mark.parametrize("specific", [None, "0", "1"])
def test_legacy_qpn8_precedence(monkeypatch, generic, specific):
    for name, value in (
        ("VLLM_SM70_FP8_QPN8", generic),
        ("VLLM_SM70_FP8_QPN8_PP2_TP4", specific),
    ):
        if value is not None:
            monkeypatch.setenv(name, value)
    before = dict(os.environ)
    policy = Sm70Fp8Config()
    policy.resolve()
    expected = (
        False
        if generic == "0"
        else specific == "1"
        if specific is not None
        else generic == "1"
    )
    assert policy.qpn8_pp2_tp4 == expected
    assert os.environ == before


def test_explicit_config_and_two_engine_isolation(monkeypatch):
    first = KernelConfig()
    first.sm70_fp8.resolve()
    before = first.compute_hash()
    monkeypatch.setenv("VLLM_SM70_FP8_QPN8", "1")
    monkeypatch.setenv("VLLM_SM70_FP8_PREFILL_EXACT_DENSE", "0")
    envs.disable_envs_cache()
    second = KernelConfig()
    second.sm70_fp8.resolve()
    first.sm70_fp8.resolve()
    assert not first.sm70_fp8.qpn8 and second.sm70_fp8.qpn8
    assert (
        first.sm70_fp8.prefill_exact_dense and not second.sm70_fp8.prefill_exact_dense
    )
    assert first.compute_hash() == before
    assert second.compute_hash() != before
    explicit = Sm70Fp8Config(qpn8=False, prefill_exact_dense=True)
    explicit.resolve()
    assert not explicit.qpn8 and explicit.prefill_exact_dense


@pytest.mark.parametrize(
    ("env", "explicit", "expected"),
    [(None, None, False), ("1", None, True), ("0", None, False), ("1", False, False)],
)
def test_small_shape_tuning_policy(monkeypatch, env, explicit, expected):
    name = "VLLM_SM70_FP8_TUNE_SMALL_SHAPES"
    if env is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, env)
    envs.disable_envs_cache()
    before = dict(os.environ)
    policy = Sm70Fp8Config(small_shape_tuning=explicit)
    # Unresolved engines (other FP8 formats) reach the same native selector.
    assert policy.native_small_shape_tuning() is expected
    policy.resolve()
    assert policy.small_shape_tuning is expected
    assert os.environ == before


@pytest.mark.skipif(
    not hasattr(torch.ops._C, "sm70_set_fp8_small_shape_tuning"),
    reason="native SM70 FP8 selector is not built",
)
def test_small_shape_tuning_native_setter():
    from vllm import _sm70_ops as sm70_ops

    sm70_ops.sm70_set_fp8_small_shape_tuning(False)
    sm70_ops.sm70_set_fp8_small_shape_tuning(True)
    sm70_ops.sm70_set_fp8_small_shape_tuning(False)


def test_unused_fp8_policy_preserves_nvfp4_fingerprint():
    config = KernelConfig()
    config.sm70_nvfp4.resolve(qualified=True)
    assert (
        config.compute_hash()
        == "5b3bf1fe15bbb8c7bc87b370f0b07e9e35f0fa117840e919d724445189b631fa"
    )
    # The new Turing routing policy participates in compilation identity.
    before = config.compute_hash()
    config.sm70_nvfp4.dense_qpn2 = False
    assert config.compute_hash() != before
