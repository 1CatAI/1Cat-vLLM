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


def linear_method(layout=None):
    method = GGUFLinearMethod(GGUFConfig(), layout)
    layer = torch.nn.Module()
    layer.quant_method = method
    method.create_weights(layer, 512, [64] * 4, 512, 256, torch.float16)
    return layer, method


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


def test_preparation_preserves_shared_source_and_reports_output_tail():
    layer, method = linear_method()
    data = torch.from_numpy(source(3)).cuda()
    layer.qweight.materialize(data.shape, device="cuda", dtype=torch.uint8)
    layer.qweight.data.copy_(data)
    shared = layer.qweight
    layer.qweight_type.weight_type = 3
    method.process_weights_after_loading(layer)
    assert layer.qweight is not shared and layer.qweight.numel() == 0
    torch.testing.assert_close(shared, data, rtol=0, atol=0)
    tail = GGUFPreparedProjection(data[:12], 3, torch.float16, True, 8)
    assert tail.kernel is None
    assert tail.rejection_reason == "local_shape_cuts_canonical_group_or_output_pack"
