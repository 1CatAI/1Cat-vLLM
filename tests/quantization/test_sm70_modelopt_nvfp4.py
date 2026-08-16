# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact-SM70 ModelOpt NVFP4 admission: capability, provenance, routing.

These tests exist so PR 114/166 cannot recur:
- the class-wide gate is not blindly lowered to 70
- scale convention is ModelOpt provenance (amax/2688), never max(|scale|) < 1
- TurboMind prepare is exact SM70 only
"""

from unittest.mock import MagicMock, patch

import torch
from torch.nn.parameter import Parameter

from vllm.config import KernelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.quantization import sm70_turbomind as sm70_tm
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptNvFp4Config,
    ModelOptNvFp4LinearMethod,
    ModelOptNvFp4W4A16LinearMethod,
)


def _modelopt_dequant_oracle(
    packed_nibble: float, block_scale: float, weight_global_scale: float
) -> float:
    """Independent ModelOpt NVFP4 dequant: nibble * block * (amax/2688)."""
    return packed_nibble * block_scale * weight_global_scale


def _ct_dequant_oracle(
    packed_nibble: float, block_scale: float, disk_global: float
) -> float:
    """Independent CT NVFP4 dequant: nibble * block / disk_global."""
    return packed_nibble * block_scale / disk_global


def _rejected_magnitude_heuristic_dequant(
    packed_nibble: float, block_scale: float, disk_global: float
) -> float:
    """PR 166 heuristic: max(|block|) < 1 => drop the global factor."""
    global_scale = 1.0 if abs(block_scale) < 1 else 1.0 / disk_global
    return packed_nibble * block_scale * global_scale


def test_modelopt_nvfp4_min_capability_requires_proven_backend(monkeypatch):
    monkeypatch.delenv("VLLM_USE_NVFP4_CT_EMULATIONS", raising=False)
    monkeypatch.delenv("VLLM_NVFP4_GEMM_BACKEND", raising=False)
    monkeypatch.delenv("VLLM_SM70_QUANT_BACKEND", raising=False)

    with set_current_vllm_config(
        VllmConfig(kernel_config=KernelConfig(linear_backend="auto"))
    ):
        assert ModelOptNvFp4Config.get_min_capability() == 70

    monkeypatch.setenv("VLLM_SM70_NVFP4_TURBOMIND", "0")
    with set_current_vllm_config(
        VllmConfig(kernel_config=KernelConfig(linear_backend="auto"))
    ):
        assert ModelOptNvFp4Config.get_min_capability() == 75

    monkeypatch.delenv("VLLM_SM70_NVFP4_TURBOMIND", raising=False)
    monkeypatch.setenv("VLLM_SM70_QUANT_BACKEND", "marlin")
    with set_current_vllm_config(
        VllmConfig(kernel_config=KernelConfig(linear_backend="auto"))
    ):
        assert ModelOptNvFp4Config.get_min_capability() == 70

    monkeypatch.setenv("VLLM_SM70_QUANT_BACKEND", "turbomind")
    with set_current_vllm_config(
        VllmConfig(kernel_config=KernelConfig(linear_backend="auto"))
    ):
        assert ModelOptNvFp4Config.get_min_capability() == 70

    monkeypatch.setenv("VLLM_SM70_NVFP4_TURBOMIND", "0")
    monkeypatch.setenv("VLLM_SM70_QUANT_BACKEND", "auto")
    with set_current_vllm_config(
        VllmConfig(kernel_config=KernelConfig(linear_backend="emulation"))
    ):
        assert ModelOptNvFp4Config.get_min_capability() == 70


def test_modelopt_scale_provenance_is_not_the_pr166_heuristic():
    # yangzhuxinyzx 100x counterexample: valid CT encoding, FP8 block 0.5,
    # disk global 100, nibble 2. Correct CT dequant is 0.01. The rejected
    # magnitude rule rewrites it to 1.0.
    nibble, block, disk_global = 2.0, 0.5, 100.0
    assert _ct_dequant_oracle(nibble, block, disk_global) == 0.01
    assert _rejected_magnitude_heuristic_dequant(nibble, block, disk_global) == 1.0

    # ModelOpt provenance is explicit: global is already amax/2688.
    # Same numeric tensors must keep the on-disk global (no rewrite).
    modelopt_global = 100.0
    assert _modelopt_dequant_oracle(nibble, block, modelopt_global) == 100.0
    assert abs(block) < 1
    assert _modelopt_dequant_oracle(nibble, block, modelopt_global) != 1.0


def _w4a16_layer(global_scale: float, block_scale: float) -> torch.nn.Module:
    layer = torch.nn.Module()
    layer.weight = Parameter(
        torch.zeros((16, 8), dtype=torch.uint8),
        requires_grad=False,
    )
    layer.weight_scale = Parameter(
        torch.full((16, 1), block_scale, dtype=torch.float8_e4m3fn),
        requires_grad=False,
    )
    layer.weight_scale_2 = Parameter(
        torch.tensor([global_scale], dtype=torch.float32),
        requires_grad=False,
    )
    layer.input_scale = Parameter(
        torch.tensor([1.0], dtype=torch.float32),
        requires_grad=False,
    )
    return layer


def test_w4a16_keeps_modelopt_global_scale_without_reciprocation_or_heuristic():
    layer = _w4a16_layer(global_scale=100.0, block_scale=0.5)
    fake_kernel = MagicMock()

    with patch(
        "vllm.model_executor.layers.quantization.modelopt.MarlinNvFp4LinearKernel",
        return_value=fake_kernel,
    ):
        method = ModelOptNvFp4W4A16LinearMethod(
            ModelOptNvFp4Config(
                quant_method="W4A16_NVFP4",
                is_checkpoint_nvfp4_serialized=True,
            )
        )

    with (
        patch.object(sm70_tm, "should_prepare_turbomind", return_value=False),
        patch.object(
            sm70_tm,
            "prepare_nvfp4_linear",
            side_effect=AssertionError("prepare must not run"),
        ),
    ):
        method.process_weights_after_loading(layer)

    # Provenance: keep on-disk ModelOpt global. Do not invert. Do not
    # force 1.0 because max(|weight_scale|) < 1.
    assert torch.equal(layer.weight_global_scale.cpu(), torch.tensor(100.0))
    assert not hasattr(layer, "weight_scale_2")
    fake_kernel.process_weights_after_loading.assert_called_once_with(layer)


def test_w4a16_turbomind_prepare_only_on_exact_sm70():
    layer = _w4a16_layer(global_scale=0.25, block_scale=1.0)
    fake_kernel = MagicMock()

    with patch(
        "vllm.model_executor.layers.quantization.modelopt.MarlinNvFp4LinearKernel",
        return_value=fake_kernel,
    ):
        method = ModelOptNvFp4W4A16LinearMethod(
            ModelOptNvFp4Config(
                quant_method="W4A16_NVFP4",
                is_checkpoint_nvfp4_serialized=True,
            )
        )

    def fake_prepare(prepared: torch.nn.Module) -> None:
        setattr(prepared, sm70_tm.STATE_ATTR, object())

    with (
        patch.object(sm70_tm, "should_prepare_turbomind", return_value=True),
        patch.object(sm70_tm, "prepare_nvfp4_linear", side_effect=fake_prepare),
    ):
        method.process_weights_after_loading(layer)

    assert sm70_tm.has_prepared_linear(layer)
    assert layer.weight.numel() == 0
    assert layer.weight_scale.numel() == 0
    fake_kernel.process_weights_after_loading.assert_not_called()


def test_w4a4_turbomind_prepare_skipped_off_sm70():
    layer = torch.nn.Module()
    layer.input_scale = Parameter(
        torch.ones(1, dtype=torch.float32),
        requires_grad=False,
    )
    layer.weight_scale_2 = Parameter(
        torch.tensor([0.25], dtype=torch.float32),
        requires_grad=False,
    )
    layer.weight = Parameter(
        torch.zeros((16, 8), dtype=torch.uint8),
        requires_grad=False,
    )
    layer.weight_scale = Parameter(
        torch.ones((16, 1), dtype=torch.float8_e4m3fn),
        requires_grad=False,
    )
    fake_kernel = MagicMock()

    with patch(
        "vllm.model_executor.layers.quantization.modelopt.init_nvfp4_linear_kernel",
        return_value=fake_kernel,
    ):
        method = ModelOptNvFp4LinearMethod(
            ModelOptNvFp4Config(
                quant_method="NVFP4",
                is_checkpoint_nvfp4_serialized=True,
            )
        )

    with (
        patch.object(sm70_tm, "should_prepare_turbomind", return_value=False),
        patch.object(
            sm70_tm,
            "prepare_nvfp4_linear",
            side_effect=AssertionError("prepare must not run off SM70"),
        ),
    ):
        method.process_weights_after_loading(layer)

    fake_kernel.process_weights_after_loading.assert_called_once_with(layer)
    assert not sm70_tm.has_prepared_linear(layer)


def test_apply_uses_prepared_turbomind_path():
    layer = torch.nn.Module()
    fake_kernel = MagicMock()
    fake_out = torch.zeros((1, 4), dtype=torch.float16)

    with patch(
        "vllm.model_executor.layers.quantization.modelopt.MarlinNvFp4LinearKernel",
        return_value=fake_kernel,
    ):
        method = ModelOptNvFp4W4A16LinearMethod(
            ModelOptNvFp4Config(
                quant_method="W4A16_NVFP4",
                is_checkpoint_nvfp4_serialized=True,
            )
        )

    x = torch.ones((1, 16), dtype=torch.float16)
    with (
        patch.object(sm70_tm, "has_prepared_linear", return_value=True),
        patch.object(
            sm70_tm, "apply_prepared_linear", return_value=fake_out
        ) as apply_tm,
    ):
        out = method.apply(layer, x)

    assert out is fake_out
    apply_tm.assert_called_once()
    fake_kernel.apply_weights.assert_not_called()
