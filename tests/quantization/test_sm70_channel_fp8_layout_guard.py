# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a16_fp8 import (  # noqa: E501
    CompressedTensorsW8A16Fp8,
    _sm70_channel_fp8_shape_is_validated,
    _sm70_unpack_channel_fp8,
)


@pytest.mark.parametrize(
    "shape,expected",
    [
        ((4096, 2560), True),
        ((3328, 2560), True),
        ((2560, 1536), True),
        ((320, 2560), True),
        ((32, 128), True),
        ((96, 384), True),
        ((2560, 160), False),
        ((8, 256), False),
        ((40, 128), False),
        ((0, 128), False),
    ],
)
def test_layout_gate_is_independent_of_checkpoint_and_tp(shape, expected):
    layer = SimpleNamespace(
        weight=torch.empty(shape, device="meta"),
        prefix="unlisted.projection",
        tp_size=2,
    )
    assert _sm70_channel_fp8_shape_is_validated(layer) is expected


def _layer(raw, scale):
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(
        raw.view(torch.float8_e4m3fn), requires_grad=False
    )
    layer.weight_scale = torch.nn.Parameter(scale, requires_grad=False)
    layer.orig_dtype = torch.float16
    layer.output_size_per_partition = raw.shape[0]
    layer.prefix = "arbitrary.projection"
    return layer


def test_fallback_bounds_fp32_scratch_and_rounds_checkpoint_scales_once(monkeypatch):
    raw = torch.full((4100, 257), 0x38, dtype=torch.uint8)
    scale = torch.linspace(0.001, 0.1, 4100).view(-1, 1)
    layer = _layer(raw, scale)
    expected = (layer.weight.float() * scale).half()
    converted_sizes = []
    original_to = torch.Tensor.to

    def tracked_to(tensor, *args, **kwargs):
        if tensor.dtype == torch.float8_e4m3fn and args == (torch.float32,):
            converted_sizes.append(tensor.numel())
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", tracked_to)
    _sm70_unpack_channel_fp8(layer)
    assert len(converted_sizes) > 1
    assert max(converted_sizes) * 4 <= 4 * 1024**2
    assert torch.equal(layer.weight, expected)
    assert layer.sm70_fp8_fp16_dequant
    assert not hasattr(layer, "weight_scale")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "shape", [(32, 128), (320, 2560), (2560, 160), (8, 256), (40, 128)]
)
def test_layout_dispatch_values_and_changing_input_graph(shape):
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires SM70")
    torch.manual_seed(484)
    n, k = shape
    raw = torch.randint(0, 64, (n, k), dtype=torch.uint8)
    raw[::2].bitwise_or_(128)
    scale = torch.full((n, 1), 0.03125)
    reference_weight = (raw.view(torch.float8_e4m3fn).float() * scale).half().cuda()
    layer = _layer(raw.cuda(), scale.cuda())
    scheme = object.__new__(CompressedTensorsW8A16Fp8)
    scheme.use_sm70_fp8_turbomind = True
    scheme.use_sm70_fp8_qpn8 = False
    scheme.process_weights_after_loading(layer)
    packed = n % 32 == 0 and k % 128 == 0
    assert getattr(layer, "sm70_fp8_turbomind", False) is packed
    assert getattr(layer, "sm70_fp8_fp16_dequant", False) is (not packed)
    for m in (1, 17):
        x = torch.randn(m, k, dtype=torch.float16, device="cuda") * 0.1
        bias = torch.zeros(n, dtype=torch.float16, device="cuda")
        output = scheme.apply_weights(layer, x, bias)
        reference = torch.nn.functional.linear(x, reference_weight, bias)
        torch.testing.assert_close(output, reference, atol=0.002, rtol=0.01)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            scheme.apply_weights(layer, x)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = scheme.apply_weights(layer, x)
    for _ in range(8):
        x.normal_(std=0.1)
        graph.replay()
        reference = torch.nn.functional.linear(x, reference_weight)
        assert torch.isfinite(output).all()
        torch.testing.assert_close(output, reference, atol=0.002, rtol=0.01)
