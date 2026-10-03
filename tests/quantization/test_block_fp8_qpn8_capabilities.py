# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from vllm.config.kernel import Sm70Fp8Config
from vllm.model_executor.kernels import linear
from vllm.model_executor.kernels.linear.scaled_mm.qpn8_blk import (
    QPN8Fp8BlockScaledMMLinearKernel as QPN8,
)
from vllm.model_executor.kernels.linear.scaled_mm.sm70_fp8 import (
    Sm70Fp8LinearLayerConfig,
    TurboMindFp8LinearKernel,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8Dynamic128Sym,
    kFp8Static128BlockSym,
    kFp8StaticTensorSym,
)
from vllm.platforms import PlatformEnum
from vllm.platforms.interface import DeviceCapability


@pytest.fixture
def native_contract(monkeypatch):
    monkeypatch.setattr(linear.current_platform, "_enum", PlatformEnum.CUDA)
    monkeypatch.setattr(linear.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        linear.current_platform, "get_device_capability", lambda: DeviceCapability(7, 0)
    )
    ops = SimpleNamespace(
        fp8_qpn8_prepare_sm70=lambda: None,
        fp8_qpn8_gemm_sm70_out=lambda: None,
        fp8_qpn8_prefill_sm70_out=lambda: None,
        fp8_sm70_prepare=lambda: None,
    )
    monkeypatch.setattr(torch.ops, "_C", ops)
    return ops


def config(shape=(128, 256), **kwargs):
    return Sm70Fp8LinearLayerConfig(
        weight_quant_key=kFp8Static128BlockSym,
        activation_quant_key=kFp8Dynamic128Sym,
        weight_shape=shape,
        input_dtype=torch.float16,
        out_dtype=torch.float16,
        policy=Sm70Fp8Config(),
        **kwargs,
    )


@pytest.mark.parametrize(
    "capability,expected",
    [(70, True), (72, True), (75, True), (80, False), (90, False)],
)
def test_hardware_admission(native_contract, capability, expected):
    assert QPN8.is_supported(capability)[0] is expected


@pytest.mark.parametrize(
    "shape", [(128, 128), (256, 512), (768, 1408), (1536, 4096), (8192, 1024)]
)
def test_all_aligned_shapes_default_to_qpn8(native_contract, shape):
    cfg = config(shape)
    selected = linear.choose_scaled_mm_linear_kernel(
        cfg, linear._POSSIBLE_FP8_BLOCK_KERNELS, compute_capability=70
    )
    assert selected is QPN8


@pytest.mark.parametrize("shape", [(0, 128), (128, 0), (127, 128), (128, 129)])
def test_unaligned_and_empty_weights_report_reason(native_contract, shape):
    accepted, reason = QPN8.can_implement(config(shape))
    assert not accepted and "positive N,K divisible by 128" in reason


@pytest.mark.parametrize("field", ["input_dtype", "out_dtype"])
def test_bf16_does_not_enter_fp16_kernel(native_contract, field):
    cfg = replace(config(), **{field: torch.bfloat16})
    accepted, reason = QPN8.can_implement(cfg)
    assert not accepted and "FP16" in reason


def test_other_scale_layouts_report_reason(native_contract):
    cfg = replace(config(), weight_quant_key=kFp8StaticTensorSym)
    accepted, reason = QPN8.can_implement(cfg)
    assert not accepted and "128 x 128" in reason


@pytest.mark.parametrize("grouped", [True, False])
def test_disable_or_grouped_bmm_retains_legacy_kernel(native_contract, grouped):
    cfg = config(is_bmm=grouped)
    cfg.policy.block_qpn8 = grouped
    selected = linear.choose_scaled_mm_linear_kernel(
        cfg, linear._POSSIBLE_FP8_BLOCK_KERNELS, compute_capability=70
    )
    assert selected is TurboMindFp8LinearKernel
    accepted, reason = QPN8.can_implement(cfg)
    assert not accepted
    assert ("grouped BMM" if grouped else "disabled by kernel_config") in reason


def test_missing_operator_falls_back_with_diagnostic(native_contract, monkeypatch):
    del native_contract.fp8_qpn8_prefill_sm70_out
    cfg = config()
    engine = SimpleNamespace(
        sm70_acceleration_report={},
        kernel_config=SimpleNamespace(
            linear_backend="auto", linear_kernel_selections={}
        ),
    )
    monkeypatch.setattr("vllm.config.get_current_vllm_config_or_none", lambda: engine)
    selected = linear.choose_scaled_mm_linear_kernel(
        cfg, linear._POSSIBLE_FP8_BLOCK_KERNELS, compute_capability=70
    )
    assert selected is TurboMindFp8LinearKernel
    rows = engine.kernel_config.linear_kernel_selections
    assert len(rows) == 1
    row = next(iter(rows.values()))["paths"][QPN8.__name__]
    assert not row["enabled"]
    assert "fp8_qpn8_prefill_sm70_out" in row["reason"]


def test_empty_activation_does_not_launch_native_kernel(monkeypatch):
    from vllm.model_executor.kernels.linear.scaled_mm import qpn8_blk

    def unexpected(*args):
        raise AssertionError("empty input must not launch a native kernel")

    monkeypatch.setattr(qpn8_blk.sm70_ops, "fp8_qpn8_gemm_sm70_out", unexpected)
    output = torch.ops.sm70_fp8.qpn8_native_linear(
        torch.empty((0, 256), dtype=torch.float16),
        torch.empty((256, 128), dtype=torch.uint8),
        torch.empty((2, 4), dtype=torch.float16),
        16,
        2,
        False,
    )
    assert output.shape == (0, 128)
    assert output.dtype == torch.float16


@pytest.mark.parametrize(
    "field", ["gated_silu", "prefill_prescaled", "prescaled_decode"]
)
def test_explicit_existing_variant_keeps_priority(native_contract, field):
    cfg = config()
    setattr(cfg.policy, field, True)
    accepted, reason = QPN8.can_implement(cfg)
    assert not accepted
    assert "explicitly configured" in reason
