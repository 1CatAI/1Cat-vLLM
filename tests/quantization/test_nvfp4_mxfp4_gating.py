# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch.nn.parameter import Parameter

import vllm.model_executor.kernels.linear.nvfp4.emulation as nvfp4_emulation
from vllm.config import KernelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.kernels.linear.nvfp4.base import NvFp4LinearLayerConfig
from vllm.model_executor.kernels.linear.nvfp4.emulation import (
    EmulationNvFp4LinearKernel,
)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    compressed_tensors_w4a4_mxfp4 as mxfp4_scheme,
)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a4_mxfp4 import (  # noqa: E501
    CompressedTensorsW4A4Mxfp4,
)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a4_nvfp4 import (  # noqa: E501
    CompressedTensorsW4A4Fp4,
)


def test_nvfp4_min_capability_honors_linear_backend_emulation(monkeypatch):
    monkeypatch.delenv("VLLM_USE_NVFP4_CT_EMULATIONS", raising=False)
    monkeypatch.delenv("VLLM_NVFP4_GEMM_BACKEND", raising=False)

    with set_current_vllm_config(
        VllmConfig(kernel_config=KernelConfig(linear_backend="auto"))
    ):
        assert CompressedTensorsW4A4Fp4.get_min_capability() == 75

    with set_current_vllm_config(
        VllmConfig(kernel_config=KernelConfig(linear_backend="emulation"))
    ):
        assert CompressedTensorsW4A4Fp4.get_min_capability() == 70


def test_nvfp4_min_capability_honors_legacy_emulation_env(monkeypatch):
    monkeypatch.setenv("VLLM_NVFP4_GEMM_BACKEND", "emulation")

    with set_current_vllm_config(
        VllmConfig(kernel_config=KernelConfig(linear_backend="auto"))
    ):
        assert CompressedTensorsW4A4Fp4.get_min_capability() == 70


def _make_mxfp4_layer() -> torch.nn.Module:
    layer = torch.nn.Module()
    layer.weight_packed = Parameter(
        torch.tensor([[0x10, 0x32], [0x54, 0x76]], dtype=torch.uint8),
        requires_grad=False,
    )
    layer.weight_scale = Parameter(
        torch.ones((2, 1), dtype=torch.uint8),
        requires_grad=False,
    )
    return layer


def test_mxfp4_turbomind_branch_hits_only_exact_sm70(monkeypatch):
    monkeypatch.setenv("VLLM_SM70_MXFP4_TURBOMIND", "1")
    layer = _make_mxfp4_layer()
    scheme = CompressedTensorsW4A4Mxfp4()

    monkeypatch.setattr(
        mxfp4_scheme.sm70_tm,
        "is_exact_sm70_cuda",
        lambda tensor, enabled: enabled,
    )

    def fake_prepare(prepared_layer: torch.nn.Module) -> None:
        setattr(prepared_layer, mxfp4_scheme.sm70_tm.STATE_ATTR, object())

    monkeypatch.setattr(mxfp4_scheme.sm70_tm, "prepare_mxfp4_linear", fake_prepare)

    scheme.process_weights_after_loading(layer)

    assert mxfp4_scheme.sm70_tm.has_prepared_linear(layer)
    assert layer.weight_packed.numel() == 0
    assert layer.weight_scale.numel() == 0


def test_mxfp4_turbomind_branch_falls_back_when_not_sm70(monkeypatch):
    monkeypatch.setenv("VLLM_SM70_MXFP4_TURBOMIND", "1")
    layer = _make_mxfp4_layer()
    scheme = CompressedTensorsW4A4Mxfp4()
    fallback_called = False

    monkeypatch.setattr(
        mxfp4_scheme.sm70_tm,
        "is_exact_sm70_cuda",
        lambda tensor, enabled: False,
    )
    monkeypatch.setattr(
        mxfp4_scheme.sm70_tm,
        "prepare_mxfp4_linear",
        lambda layer: (_ for _ in ()).throw(AssertionError("unexpected prepare")),
    )

    class FakeKernel:
        def process_weights_after_loading(self, loaded_layer: torch.nn.Module) -> None:
            nonlocal fallback_called
            fallback_called = True
            assert hasattr(loaded_layer, "weight")

    monkeypatch.setattr(
        CompressedTensorsW4A4Mxfp4,
        "_fallback_kernel",
        lambda self: FakeKernel(),
    )

    scheme.process_weights_after_loading(layer)

    assert fallback_called
    assert not mxfp4_scheme.sm70_tm.has_prepared_linear(layer)


def test_nvfp4_emulation_predecodes_and_uses_weight_dequant(monkeypatch):
    layer = torch.nn.Module()
    layer.weight = Parameter(
        torch.empty((2, 2), dtype=torch.uint8), requires_grad=False
    )
    layer.weight_scale = Parameter(
        torch.empty((2, 1), dtype=torch.float32), requires_grad=False
    )
    layer.weight_global_scale = Parameter(torch.tensor(1.0), requires_grad=False)
    layer.input_global_scale_inv = Parameter(torch.tensor(1.0), requires_grad=False)

    monkeypatch.setattr(
        nvfp4_emulation,
        "_supports_triton_nvfp4_emulation",
        lambda device=None: False,
    )
    monkeypatch.setattr(
        nvfp4_emulation,
        "dequantize_to_dtype",
        lambda *args, **kwargs: torch.ones((2, 4), dtype=torch.float16),
    )

    kernel = EmulationNvFp4LinearKernel(NvFp4LinearLayerConfig())
    kernel.process_weights_after_loading(layer)

    assert hasattr(layer, "weight_dequant")

    seen = {}

    def fake_run_nvfp4_emulations(**kwargs):
        seen["weight_dequant"] = kwargs["weight_dequant"]
        x = kwargs["x"]
        return torch.zeros((*x.shape[:-1], 2), dtype=x.dtype, device=x.device)

    monkeypatch.setattr(
        nvfp4_emulation,
        "run_nvfp4_emulations",
        fake_run_nvfp4_emulations,
    )

    x = torch.ones((1, 4), dtype=torch.float16)
    out = kernel.apply_weights(layer, x)

    assert seen["weight_dequant"] is layer.weight_dequant
    assert out.shape == (1, 2)
