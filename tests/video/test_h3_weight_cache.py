# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.model_executor.models.minimax_h3.cuda_ops import (
    fp16_gemm,
    w8a16_extension,
)
from vllm.model_executor.models.minimax_h3.weight_cache import FP16WeightCache

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")


def test_cached_gemm_fp32_output_survives_fp16_overflow_and_graph_replay():
    x = torch.full((32, 512), 256, device="cuda", dtype=torch.float16)
    weight_t = torch.ones(512, 256, device="cuda", dtype=torch.float16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fp16_gemm(x, weight_t.t(), True)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = fp16_gemm(x, weight_t.t(), True)
    for multiplier in (1, -2):
        x.mul_(multiplier)
        graph.replay()
        expected = x.float() @ weight_t.float()
        assert torch.isfinite(actual).all() and actual.abs().max() > 65504
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("layout", ["row", "column"])
def test_cache_preserves_encoded_weights_and_cleans_partial_preparation(
    monkeypatch, layout
):
    model = torch.nn.ModuleList([torch.nn.Module(), torch.nn.Module()])
    for layer in model:
        layer.register_buffer(
            "weight",
            torch.randint(-128, 128, (256, 256), device="cuda", dtype=torch.int8),
        )
        layer.register_buffer("weight_scale", torch.rand(256, device="cuda") + 0.01)
    if layout == "column":
        for layer in model:
            layer.weight = layer.weight.t().contiguous().t()
    originals = [(layer.weight.clone(), layer.weight_scale.clone()) for layer in model]
    cache = FP16WeightCache(model, budget_gib=1, layers=("0", "1"))
    cache.prepare()
    for layer, (weight, scale) in zip(model, originals):
        assert layer.h3_fp16_weight.stride() == (1, 256)
        torch.testing.assert_close(
            layer.h3_fp16_weight,
            w8a16_extension().dequantize(weight, scale),
            atol=0,
            rtol=0,
        )
        assert torch.equal(layer.weight, weight) and torch.equal(
            layer.weight_scale, scale
        )
    cache.clear()
    responses = iter([(2**40, 2**40), (0, 2**40)])
    monkeypatch.setattr(torch.accelerator, "get_memory_info", lambda _: next(responses))
    with pytest.raises(RuntimeError, match="available GPU memory"):
        cache.prepare()
    assert all(not hasattr(layer, "h3_fp16_weight") for layer in model)
