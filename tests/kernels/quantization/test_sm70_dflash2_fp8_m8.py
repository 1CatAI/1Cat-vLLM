# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

import vllm._custom_ops  # noqa: F401
from vllm.model_executor.layers.quantization.sm70_dflash2_fp8 import (
    apply_dflash2_fp8_m8,
    prepare_dflash2_fp8_m8,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@torch.inference_mode()
def test_dynamic_prefill_and_batch_keep_fp16_graph_route():
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 is required")
    torch.manual_seed(20261005)
    layer = torch.nn.Linear(5120, 1536, bias=False, device="cuda", dtype=torch.float16)
    layer.weight.copy_(layer.weight.bfloat16().half())
    layer.weight[0].zero_()
    layer.weight[1].fill_(2.0**-24)
    original = layer.weight.clone()
    layer._sm70_dflash2_fp8_m8 = True
    assert prepare_dflash2_fp8_m8(layer)
    assert torch.equal(layer.weight.view(torch.int16), original.view(torch.int16))
    assert not hasattr(layer, "_sm70_dflash2_fp16_packed")
    assert "_sm70_dflash2_fp8_codes" not in layer.state_dict()
    assert torch.isfinite(layer._sm70_dflash2_fp8_scales).all()
    assert (layer._sm70_dflash2_fp8_scales > 0).all()
    compilations = []

    def backend(graph, inputs):
        compilations.append(graph)
        return torch._inductor.compile(graph, inputs)

    compiled = torch.compile(
        lambda x: apply_dflash2_fp8_m8(layer, x, None),
        backend=backend,
        fullgraph=True,
        dynamic=True,
    )
    for rows in [12, 8, 7, 32, 8, 128]:
        x = torch.randn(rows, 5120, device="cuda", dtype=torch.float16) * 0.1
        actual = compiled(x)
        assert torch.isfinite(actual).all()
        if rows != 8:
            # The compiled dynamic graph must preserve the original matrix,
            # rather than silently applying FP8 to prefill or concurrency.
            expected = F.linear(x, original)
            assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
        else:
            expected = apply_dflash2_fp8_m8(layer, x, None)
            assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
            assert not torch.equal(actual, F.linear(x, original))
    assert len(compilations) == 1
    x = torch.randn(8, 5120, device="cuda", dtype=torch.float16) * 0.1
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = compiled(x)
    for amplitude in [0.0, -0.1, 0.2]:
        x.copy_(torch.randn_like(x) * amplitude)
        graph.replay()
        assert torch.equal(
            output.view(torch.int16),
            apply_dflash2_fp8_m8(layer, x, None).view(torch.int16),
        )
    assert apply_dflash2_fp8_m8(layer, x.repeat(4, 1), None) is None
    assert apply_dflash2_fp8_m8(layer, x.T.contiguous().T, None) is None
    assert apply_dflash2_fp8_m8(layer, x.unsqueeze(0), None) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@torch.inference_mode()
def test_head_and_convolution_shapes_are_not_quantized():
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 is required")
    for n, k in [(1280, 5120), (62080, 5120)]:
        layer = torch.nn.Linear(k, n, bias=False, device="cuda", dtype=torch.float16)
        layer._sm70_dflash2_fp8_m8 = True
        assert not prepare_dflash2_fp8_m8(layer)
        assert not hasattr(layer, "_sm70_dflash2_fp8_codes")
