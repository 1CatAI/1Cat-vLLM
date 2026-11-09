# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.sm70_diffusion import sm70_extension

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a leased SM70 GPU"
)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("alpha", [0.0, 0.0625, 0.75, -0.5, 1.0])
@pytest.mark.parametrize("scaled", [False, True])
def test_scaled_add_retains_fp32_rounding_and_untouched_slices(dtype, alpha, scaled):
    torch.manual_seed(610)
    rows, columns, width, offset = 17, 43, 33, 3
    storage = torch.randn(rows * columns + 1, device="cuda", dtype=dtype)
    actual = storage[1:].reshape(rows, columns)
    base = actual.clone()
    delta = torch.randn(rows, width, device="cuda") * 100
    scales = (
        torch.ldexp(
            torch.ones(rows, 1, device="cuda"),
            (torch.arange(rows, device="cuda") - 8)[:, None],
        )
        if scaled
        else None
    )
    restored = delta if scales is None else delta * scales
    expected = base.float().clone()
    expected[:, offset : offset + width].add_(restored, alpha=alpha)
    result = sm70_extension().scaled_add_(actual, delta, scales, alpha, offset)
    assert result.data_ptr() == actual.data_ptr()
    torch.testing.assert_close(actual, expected.to(dtype), rtol=0, atol=0)
    torch.testing.assert_close(actual[:, :offset], base[:, :offset], rtol=0, atol=0)
    torch.testing.assert_close(
        actual[:, offset + width :], base[:, offset + width :], rtol=0, atol=0
    )


def test_scaled_add_rejects_unsafe_memory_aliases_and_bad_bounds():
    data = torch.zeros(65, device="cuda")
    output = data[:-1].reshape(8, 8)
    overlapping = data[1:].reshape(8, 8)
    op = sm70_extension().scaled_add_
    with pytest.raises(RuntimeError):
        op(output, overlapping, None, 1.0, 0)
    with pytest.raises(RuntimeError):
        op(output.view(torch.float16), output, None, 1.0, 0)
    with pytest.raises(RuntimeError):
        op(output, torch.ones_like(output), data[:8], 1.0, 0)
    with pytest.raises(RuntimeError, match="outside output"):
        op(output, torch.ones_like(output), None, 1.0, 1)
    with pytest.raises(RuntimeError, match="finite FP32"):
        op(output, torch.ones_like(output), None, float("nan"), 0)


def test_scaled_add_graph_reads_current_operands():
    output = torch.zeros(16, 192, device="cuda", dtype=torch.float16)
    delta = torch.randn(16, 192, device="cuda")
    scales = torch.ones(16, 1, device="cuda") * 8
    op = sm70_extension().scaled_add_
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        op(output, delta, scales, 0.75, 0)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(output, delta, scales, 0.75, 0)
    for factor in (1.0, -2.0):
        output.fill_(factor)
        delta.mul_(factor)
        expected = output.float().add(delta * scales, alpha=0.75).half()
        graph.replay()
        torch.testing.assert_close(output, expected, rtol=0, atol=0)


@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("output_fp32", [False, True])
@pytest.mark.parametrize("overlapping", [False, True])
def test_adapter_fusion_matches_unfused_rounding(
    quantized, output_fp32, overlapping, monkeypatch
):
    import vllm.model_executor.models.minimax_h3.lora as adapter
    from vllm.model_executor.models.minimax_h3.quantization import (
        DiffusionInt8ConvRotConfig,
        FP16LinearMethod,
        FP32OutputLinearMethod,
        Int8ConvRotLayerConfig,
        Int8ConvRotLinearMethod,
    )

    torch.manual_seed(913)
    if quantized:
        base = Int8ConvRotLinearMethod(
            DiffusionInt8ConvRotConfig(), Int8ConvRotLayerConfig(True), prefix="probe"
        )
        weight = torch.randint(-7, 8, (384, 256), device="cuda", dtype=torch.int8)
    else:
        base = FP32OutputLinearMethod() if output_fp32 else FP16LinearMethod()
        weight = torch.randn(384, 256, device="cuda", dtype=torch.float16) * 0.03
    layer = SimpleNamespace(
        weight=weight,
        weight_scale=torch.full((384,), 0.01, device="cuda"),
        h3_output_fp32=output_fp32,
    )
    parts = []
    for i in range(3):
        setattr(layer, f"h3_lora_a_{i}", torch.randn(8, 256, device="cuda").half())
        setattr(layer, f"h3_lora_b_{i}", torch.randn(96, 8, device="cuda").half())
        parts.append((i, 32 if overlapping else 16 + i * 112, 96))
    method = adapter.TurboLinearMethod(base, parts, 0.125)
    inputs = torch.randn(33, 256, device="cuda", dtype=torch.float16)
    fused_add = adapter.fp16_linear_add
    calls = []

    def record(*args, **kwargs):
        calls.append(kwargs["offset"])
        return fused_add(*args, **kwargs)

    monkeypatch.setattr(adapter, "fp16_linear_add", record)
    token = adapter.lora_scale.set(-0.75)
    try:
        with monkeypatch.context() as context:
            context.setattr(adapter, "supports_fused_scaled_add", lambda _: False)
            expected = method.apply(layer, inputs)
        actual = method.apply(layer, inputs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert actual.dtype == (torch.float32 if output_fp32 else torch.float16)
        assert calls == ([] if overlapping else [16, 128, 240])
        calls.clear()
        actual = method.apply_prepared(layer, inputs, None)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert calls == ([] if overlapping else [16, 128, 240])
    finally:
        adapter.lora_scale.reset(token)
