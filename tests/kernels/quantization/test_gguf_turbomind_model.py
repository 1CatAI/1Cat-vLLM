# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gguf
import numpy as np
import pytest
import torch
from test_gguf_lattice_transcode import source as lattice_source
from test_gguf_lut_transcode import source as lut_source
from test_gguf_transcode import packed

from vllm.model_executor.layers.quantization.gguf import GGUFConfig, GGUFLinearMethod
from vllm.model_executor.layers.quantization.gguf_layout import GGUFHeadTilingLayout
from vllm.model_executor.layers.quantization.gguf_turbomind import (
    GGUFPreparedProjection,
)
from vllm.sm70_profiles.acceleration import loaded_linear_kernels

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def source(weight_type, n=64, k=512):
    if weight_type in (16, 17, 18, 19, 21, 22, 29):
        return lattice_source(weight_type, 0.0009765625, n=n, k=k)
    if weight_type in (20, 23, 39, 40):
        return lut_source(weight_type, n=n, k=k, scale=0.0009765625)
    return packed(weight_type, rows=n, k=k)


def oracle(data, weight_type):
    return (
        torch.from_numpy(
            gguf.quants.dequantize(data, gguf.GGMLQuantizationType(weight_type))
        )
        .half()
        .cuda()
    )


@pytest.mark.parametrize(
    "weight_type", [2, 3, 8, 12, 20, 23, 16, 17, 18, 19, 21, 22, 29]
)
def test_prepared_dense_projection_uses_family_kernel_and_graph(weight_type):
    torch._dynamo.reset()
    data = source(weight_type)
    projection = GGUFPreparedProjection(
        torch.from_numpy(data).cuda(), weight_type, torch.float16, True, 8
    )
    assert projection.kernel is not None, projection.admission()
    assert not hasattr(projection, "weight")
    dense = oracle(data, weight_type)
    for m in (1, 8, 64):
        x = (torch.randn((m, dense.shape[1]), device="cuda") * 0.125).half()
        expected = x.float() @ dense.float().T
        actual = projection(x)
        assert (actual.float() - expected).norm() < expected.norm() * 0.003
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = projection(x)
        graph.replay()
        torch.testing.assert_close(captured, actual, rtol=0.003, atol=0.003)
    compiled = torch.compile(projection, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(x), projection(x), rtol=0.003, atol=0.003)


def linear_method(layout=None, output_sizes=(64, 64, 64, 64)):
    method = GGUFLinearMethod(GGUFConfig(), layout)
    layer = torch.nn.Module()
    layer.quant_method = method
    method.create_weights(
        layer, 512, list(output_sizes), 512, sum(output_sizes), torch.float16
    )
    return layer, method


@pytest.mark.parametrize("weight_type", [3, 20, 18])
def test_compatible_shards_coalesce_without_crossing_mixed_boundaries(weight_type):
    torch._dynamo.reset()
    sizes = (12, 20, 12, 20)
    layout = GGUFHeadTilingLayout(2, 128)
    layer, method = linear_method(layout, sizes)
    types = (weight_type, weight_type, 8, weight_type)
    data = [source(t, n=n) for t, n in zip(types, sizes)]
    for index in (2, 0, 3, 1):
        layer.qweight.shard_id.append(index)
        layer.qweight.shard_id_map[index] = len(layer.qweight.data_container)
        layer.qweight.data_container.append(torch.from_numpy(data[index]).cuda())
        layer.qweight_type.shard_weight_type[index] = types[index]
    layer.qweight.data = torch.empty(0, device="cuda")
    method.process_weights_after_loading(layer)
    projections = layer.gguf_tm_projections
    assert [p.source_type for p in projections] == [weight_type, 8, weight_type]
    assert [p.source_output_sizes for p in projections] == [(12, 20), (12,), (20,)]
    assert projections[0].output_padding == 0
    assert method.native_admission["canonical_projections"][0][
        "source_output_sizes"
    ] == [
        12,
        20,
    ]
    assert layer.qweight.numel() == 0 and not layer.qweight.data_container
    dense = torch.cat([oracle(d, t) for d, t in zip(data, types)], dim=0)
    bias = torch.randn(sum(sizes), device="cuda").half()
    for m in (1, 4, 16, 512):
        x = (torch.randn((m, 512), device="cuda") * 0.125).half()
        expected = (layout.input_to_gguf(x).float() @ dense.float().T).half() + bias
        actual = method.apply(layer, x, bias)
        torch.testing.assert_close(
            actual.float(), expected.float(), rtol=0.003, atol=0.003
        )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = method.apply(layer, x, bias)
    graph.replay()
    torch.testing.assert_close(captured, actual, rtol=0, atol=0)
    compiled = torch.compile(
        lambda x: method.apply(layer, x, bias), backend="eager", fullgraph=True
    )
    torch.testing.assert_close(compiled(x), actual, rtol=0.003, atol=0.003)


def test_mixed_fused_projection_order_layout_bias_and_startup_report():
    layout = GGUFHeadTilingLayout(2, 128)
    layer, method = linear_method(layout)
    types = (3, 20, 18, 1)
    data = [source(t) if t != 1 else np.ones((64, 512), np.float16) / 16 for t in types]
    for index in (2, 0, 3, 1):
        layer.qweight.shard_id.append(index)
        layer.qweight.shard_id_map[index] = len(layer.qweight.data_container)
        layer.qweight.data_container.append(torch.from_numpy(data[index]).cuda())
        layer.qweight_type.shard_weight_type[index] = types[index]
    # Merged loading leaves the main parameter uninitialized until preparation.
    layer.qweight.data = torch.empty(0, device="cuda")
    method.process_weights_after_loading(layer)
    assert not layer.qweight.data_container and layer.qweight.numel() == 0
    assert [p.source_type for p in layer.gguf_tm_projections] == list(types)
    assert layer.gguf_tm_projections[-1].kernel is None
    report = loaded_linear_kernels(layer)
    assert any("AffineKernel" in key for key in report)
    assert any("Lut4Kernel" in key for key in report)
    assert any("LatticeKernel" in key for key in report)
    x = (torch.randn((8, 512), device="cuda") * 0.125).half()
    bias = torch.randn(256, device="cuda").half()
    physical_input = layout.input_to_gguf(x)
    dense = [
        oracle(d, t) if t != 1 else torch.from_numpy(d).cuda()
        for d, t in zip(data, types)
    ]
    expected = torch.cat([physical_input.float() @ d.float().T for d in dense], -1)
    expected = expected.half() + bias
    actual = method.apply(layer, x, bias)
    assert (actual.float() - expected).norm() < expected.norm() * 0.003
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = method.apply(layer, x, bias)
    graph.replay()
    torch.testing.assert_close(captured, actual, rtol=0.003, atol=0.003)


def test_preparation_preserves_shared_source():
    layer, method = linear_method()
    data = torch.from_numpy(source(3)).cuda()
    layer.qweight.materialize(data.shape, device="cuda", dtype=torch.uint8)
    layer.qweight.data.copy_(data)
    shared = layer.qweight
    layer.qweight_type.weight_type = 3
    method.process_weights_after_loading(layer)
    assert layer.qweight is not shared and layer.qweight.numel() == 0
    torch.testing.assert_close(shared, data, rtol=0, atol=0)


@pytest.mark.parametrize("weight_type", [3, 8, 20, 18])
def test_output_tail_padding_preserves_fp16_rows_and_capture(weight_type):
    torch._dynamo.reset()
    data = source(weight_type, n=12)
    tail = GGUFPreparedProjection(
        torch.from_numpy(data).cuda(), weight_type, torch.float16, True, 8
    )
    assert tail.kernel is not None, tail.admission()
    assert tail.output_padding == 20
    assert tail.kernel.config.partition_weight_shape == (512, 32)
    assert tail.admission()["logical_output_size"] == 12
    dense = oracle(data, weight_type)
    x = (torch.randn((8, 512), device="cuda") * 0.125).half()
    expected = x.float() @ dense.float().T
    actual = tail(x)
    assert actual.shape == (8, 12)
    torch.testing.assert_close(actual.float(), expected, rtol=0.003, atol=0.003)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = tail(x)
    graph.replay()
    torch.testing.assert_close(captured, actual, rtol=0, atol=0)
    compiled = torch.compile(tail, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(x), actual, rtol=0.003, atol=0.003)


@pytest.mark.parametrize("n", [12, 24])
def test_measured_small_fp16_cache_and_packed_intervals(monkeypatch, n):
    from vllm.model_executor.kernels.gguf import dense_fp16_cache_capabilities

    monkeypatch.setattr(
        torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction", False
    )
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_fp16_accumulation", False)
    torch._dynamo.reset()
    data = source(8, n=n, k=5120)
    projection = GGUFPreparedProjection(
        torch.from_numpy(data).cuda(), 8, torch.float16, True, 8
    )
    assert projection.fp16_cache is not None
    dense = oracle(data, 8)
    for m in (1, 8, 32):
        x = (torch.randn((m, 5120), device="cuda") * 0.125).half()
        expected = x.float() @ dense.float().T
        actual = projection(x)
        torch.testing.assert_close(actual.float(), expected, rtol=0.003, atol=0.003)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = projection(x)
        graph.replay()
        torch.testing.assert_close(captured, actual, rtol=0, atol=0)
        compiled = torch.compile(projection, backend="eager", fullgraph=True)
        torch.testing.assert_close(compiled(x), actual, rtol=0.003, atol=0.003)
    admitted = lambda m: any(c.supports_m(m) for c in projection.cache_capabilities)
    assert admitted(1) and admitted(32) and admitted(8192)
    assert admitted(4) == (n == 24)
    assert not admitted(8193)
    unknown = dense_fp16_cache_capabilities(8, 1024, 4, torch.float16)
    assert all(
        c.reason == "small_projection_cache_shape_has_no_calibration" for c in unknown
    )
    monkeypatch.setattr(
        torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction", True
    )
    unsafe = dense_fp16_cache_capabilities(8, 5120, 12, torch.float16)
    assert all(c.reason == "requires_fp32_matmul_policy" for c in unsafe)


@pytest.mark.parametrize(
    "weight_type,n,minimum",
    [
        (12, 2560, 2048),
        (12, 4096, 512),
        (12, 8704, 512),
        (13, 2560, 2048),
        (13, 4096, 512),
    ],
)
def test_coalesced_shape_prefill_capability_oracle_and_graph(weight_type, n, minimum):
    block, size = gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType(weight_type)]
    k = 5120
    data = np.random.default_rng(20261004).integers(
        0, 256, (n, k // block, size), dtype=np.uint8
    )
    data[..., :2] = np.frombuffer(np.float16(1 / 4096).tobytes(), dtype=np.uint8)
    data[..., 2:4] = np.frombuffer(np.float16(1 / 16384).tobytes(), dtype=np.uint8)
    data = data.reshape(n, -1)
    projection = GGUFPreparedProjection(
        torch.from_numpy(data).cuda(), weight_type, torch.float16, True, 8
    )
    capability = projection.kernel.prefill_capability
    assert capability.reason is None and capability.min_m == minimum
    assert not capability.supports_m(minimum - 1) and capability.supports_m(minimum)
    dense = oracle(data, weight_type)
    x = (torch.randn((minimum, k), device="cuda") * 0.125).half()
    expected = x.float() @ dense.float().T
    actual = projection(x)
    assert (actual.float() - expected).norm() < expected.norm() * 0.003
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = projection(x)
    graph.replay()
    torch.testing.assert_close(captured, actual, rtol=0, atol=0)
