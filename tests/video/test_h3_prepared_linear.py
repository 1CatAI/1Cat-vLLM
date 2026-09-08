# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Independent operand, adapter and output-precision checks for shared GEMM."""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.sm70_diffusion import (
    fp16_gemm_input,
    fp16_linear_prepared,
)
from vllm.model_executor.models.minimax_h3.lora import TurboLinearMethod, lora_scale
from vllm.model_executor.models.minimax_h3.quantization import (
    DiffusionInt8ConvRotConfig,
    FP32OutputLinearMethod,
    Int8ConvRotLayerConfig,
    Int8ConvRotLinearMethod,
)


@pytest.mark.parametrize("value", [100000.0, float("inf"), float("nan")])
def test_original_weight_loader_rejects_fp16_overflow(value):
    from vllm.model_executor.models.minimax_h3.transformer import MiniMaxH3DiTModel

    model = torch.nn.Module()
    model.register_parameter(
        "weight", torch.nn.Parameter(torch.empty(2, 3, dtype=torch.float16), False)
    )
    checkpoint = torch.full((2, 3), value, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="finite FP16"):
        MiniMaxH3DiTModel.load_weights(model, [("weight", checkpoint)])


def test_original_weight_loader_keeps_sensitive_fp32_parameters():
    from vllm.model_executor.models.minimax_h3.transformer import MiniMaxH3DiTModel

    model = torch.nn.Module()
    model.register_parameter("weight", torch.nn.Parameter(torch.empty(2, 3), False))
    checkpoint = torch.full((2, 3), 100000.0, dtype=torch.bfloat16)
    MiniMaxH3DiTModel.load_weights(model, [("weight", checkpoint)])
    torch.testing.assert_close(model.weight, checkpoint.float(), atol=0, rtol=0)


@pytest.mark.parametrize("layout", ["row", "column"])
def test_dense_layout_keeps_logical_weights_and_wide_output(layout):
    torch.manual_seed(7)
    # Deliberately not an H3 projection shape.
    weight = torch.randn(96, 192, dtype=torch.float16)
    layer = SimpleNamespace(weight=weight.clone(), h3_fp16_weight_layout=layout)
    method = FP32OutputLinearMethod()
    for _ in range(2):
        method.process_weights_after_loading(layer)
        torch.testing.assert_close(layer.weight, weight, atol=0, rtol=0)
        assert layer.weight.untyped_storage().nbytes() == weight.numel() * 2
    x = torch.randn(33, 192) * 100000
    values, scale = fp16_gemm_input(x)
    result = method.apply_prepared(layer, values, scale)
    reference = (values.float() @ weight.float().T) * scale
    torch.testing.assert_close(result, reference, atol=0, rtol=0)
    assert torch.isfinite(result).all()
    assert result.abs().max() > torch.finfo(torch.float16).max
    with pytest.raises(ValueError, match="unrotated"):
        method.apply_prepared(layer, values, scale, input_is_rotated=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires SM70")
@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("adapter_scale", [0.0, 0.75, -0.5])
def test_gpu_prepared_adapter_preserves_wide_intermediates(quantized, adapter_scale):
    from vllm.model_executor.models.minimax_h3.cuda_ops import w8a16_extension

    torch.manual_seed(42)
    ops = w8a16_extension()
    if quantized:
        base = Int8ConvRotLinearMethod(
            DiffusionInt8ConvRotConfig(), Int8ConvRotLayerConfig(True), prefix="probe"
        )
        weight = torch.randint(-7, 8, (384, 256), device="cuda", dtype=torch.int8)
    else:
        base = FP32OutputLinearMethod()
        weight = torch.randn(384, 256, device="cuda", dtype=torch.float16) * 0.03
    layer = SimpleNamespace(
        weight=weight,
        weight_scale=torch.full((384,), 0.01, device="cuda"),
        h3_output_fp32=True,
        h3_lora_a_0=torch.randn(8, 256, device="cuda", dtype=torch.float16),
        h3_lora_b_0=torch.randn(384, 8, device="cuda", dtype=torch.float16),
    )
    method = TurboLinearMethod(base, [(0, 0, 384)], 0.125)
    x = torch.randn(33, 256, device="cuda") * 100000
    values, scale = fp16_gemm_input(x)
    token = lora_scale.set(adapter_scale)
    try:
        expected = method.apply(layer, x)
        actual = method.apply_prepared(layer, values, scale)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        assert actual.dtype == torch.float32 and torch.isfinite(actual).all()
        if quantized:
            rotated = ops.rotate(values)
            actual = method.apply_prepared(
                layer, rotated, scale, input_is_rotated=True, original_input=values
            )
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            if adapter_scale:
                with pytest.raises(ValueError, match="original unrotated"):
                    method.apply_prepared(layer, rotated, scale, input_is_rotated=True)
    finally:
        lora_scale.reset(token)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires SM70")
def test_gpu_shared_linear_handles_non_h3_shapes_and_fp32_outputs():
    from vllm.model_executor.models.minimax_h3.cuda_ops import w8a16_extension

    torch.manual_seed(2026)
    for m, n, k in ((33, 96, 192), (257, 1024, 768)):
        x = torch.randn(m, k, device="cuda", dtype=torch.float16)
        w = torch.randn(n, k, device="cuda", dtype=torch.float16)
        expected = w8a16_extension().gemm(x, w, True)
        for weight in (w, w.T.contiguous().T):
            actual = fp16_linear_prepared(x, weight, output_fp32=True)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
