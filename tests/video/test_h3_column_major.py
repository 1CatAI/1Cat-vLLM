# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact INT8 layout, scale and GEMM checks for the low-memory Volta route."""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models.minimax_h3 import cuda_ops
from vllm.model_executor.models.minimax_h3.quantization import (
    DiffusionInt8ConvRotConfig,
    Int8ConvRotLayerConfig,
    Int8ConvRotLinearMethod,
)

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires SM70")


@pytest.mark.parametrize("layout", ["row", "column"])
def test_weight_repack_preserves_values_scales_and_size(layout):
    config = DiffusionInt8ConvRotConfig(weight_layout=layout)
    method = Int8ConvRotLinearMethod(
        config, Int8ConvRotLayerConfig(True), prefix="blocks.0.attn.qkv_proj"
    )
    layer = torch.nn.Module()
    original = torch.arange(257 * 256).to(torch.int8).reshape(257, 256)
    scales = torch.linspace(0.001, 0.1, 257)
    layer.register_parameter("weight", torch.nn.Parameter(original.clone(), False))
    layer.register_parameter(
        "weight_scale", torch.nn.Parameter(scales[:, None].clone(), False)
    )
    for _ in range(2):  # Post-load processing is idempotent.
        method.process_weights_after_loading(layer)
        assert torch.equal(layer.weight, original)
        assert torch.equal(layer.weight_scale, scales)
        assert layer.weight.untyped_storage().nbytes() == original.numel()
        assert layer.weight.stride() == ((1, 257) if layout == "column" else (256, 1))


@cuda
@pytest.mark.parametrize("shape", [(0, 7), (1, 256), (257, 259), (129, 1536)])
def test_column_decode_signed_codes_scales_and_offset(shape):
    n, k = shape
    storage = (torch.arange(n * k + 1, device="cuda") % 256 - 128).to(torch.int8)
    weight = storage[1:].reshape(k, n).t()
    scales = torch.logspace(-7, 1, n + 1, device="cuda")[1:]
    actual = cuda_ops.w8a16_extension().dequantize(weight, scales)
    expected = (weight.float() * scales[:, None]).half()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert actual.untyped_storage().nbytes() == weight.numel() * 2


@cuda
@pytest.mark.parametrize(
    "n,k,fp32",
    [(5376, 5376, False), (7168, 5376, False), (5376, 1792, True), (5376, 3584, True)],
)
def test_real_tp4_projection_uses_exact_zero_workspace_plan(n, k, fp32):
    torch.manual_seed(42)
    m = 12352  # DiT activation padding; attention has 12323 valid tokens.
    x = torch.randn(m, k, dtype=torch.float16, device="cuda") * 0.1
    weight = torch.randn(n, k, dtype=torch.float16, device="cuda") * 0.01
    packed = weight.t().contiguous().t()
    plan = cuda_ops._column_major_plan(x.device.index, m, n, k, fp32)
    assert plan.supported, "pinned cu128 SM70 build must expose the validated plan"
    expected = cuda_ops.w8a16_extension().gemm(x, weight, fp32)
    actual = cuda_ops.fp16_gemm(x, packed, fp32)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert actual.dtype == (torch.float32 if fp32 else torch.float16)


@cuda
def test_missing_plan_falls_back_without_changing_math(monkeypatch):
    monkeypatch.setattr(
        cuda_ops, "_column_major_plan", lambda *args: SimpleNamespace(supported=False)
    )
    x = torch.randn(17, 256, dtype=torch.float16, device="cuda")
    w = torch.randn(65, 256, dtype=torch.float16, device="cuda")
    expected = cuda_ops.w8a16_extension().gemm(x, w, True)
    actual = cuda_ops.fp16_gemm(x, w.t().contiguous().t(), True)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@cuda
def test_plan_accepts_16_byte_offsets_advertised_to_heuristic():
    torch.manual_seed(42)
    m, n, k = 12352, 5376, 1792
    x = torch.randn(m * k + 8, device="cuda", dtype=torch.float16)[8:].view(m, k)
    weight = torch.randn(n * k + 8, device="cuda", dtype=torch.float16)[8:]
    weight = weight.view(k, n).t()
    assert x.data_ptr() % 256 == weight.data_ptr() % 256 == 16
    plan = cuda_ops._column_major_plan(x.get_device(), m, n, k, True)
    assert plan.supported
    actual = plan.run(x, weight)
    expected = cuda_ops.w8a16_extension().gemm(x, weight.contiguous(), True)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@cuda
def test_column_plan_rejects_shape_dtype_and_device_mismatch():
    ops = cuda_ops.w8a16_extension()
    plan = ops.ColumnMajorGemmPlan(
        torch.accelerator.current_device_index(), 12352, 5376, 1792, True
    )
    assert plan.supported
    for x, w in [
        (
            torch.empty(2, 1792, device="cuda", dtype=torch.float16),
            torch.empty(1792, 5376, device="cuda", dtype=torch.float16).t(),
        ),
        (
            torch.empty(12352, 1792, device="cuda", dtype=torch.float32),
            torch.empty(1792, 5376, device="cuda", dtype=torch.float16).t(),
        ),
        (
            torch.empty(12352, 1792, device="cuda", dtype=torch.float16),
            torch.empty(1792, 5376, dtype=torch.float16).t(),
        ),
    ]:
        with pytest.raises(RuntimeError, match="matching aligned"):
            plan.run(x, w)


@cuda
def test_column_decode_and_gemm_replay_on_current_stream():
    torch.manual_seed(123)
    m, n, k = 12352, 5376, 1792
    x = torch.randn(m, k, dtype=torch.float16, device="cuda") * 0.1
    w = torch.randint(-128, 128, (n, k), dtype=torch.int8, device="cuda")
    scale = torch.rand(n, device="cuda") * 0.001 + 0.0001
    packed = w.t().contiguous().t()
    ops = cuda_ops.w8a16_extension()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        cuda_ops.fp16_gemm(x, ops.dequantize(packed, scale), True)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = cuda_ops.fp16_gemm(x, ops.dequantize(packed, scale), True)
    x.mul_(2)
    graph.replay()
    expected = ops.gemm(x, ops.dequantize(w, scale), True)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
