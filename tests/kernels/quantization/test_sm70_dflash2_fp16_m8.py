# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

import vllm._custom_ops  # noqa: F401
from vllm.model_executor.layers.quantization.sm70_dflash2_fp16 import (
    apply_dflash2_fp16_m8,
    prepare_dflash2_fp16_m8,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    "n,k,input_scale",
    [
        (1280, 5120, 1.0),
        (1536, 5120, 1.0),
        (5120, 1024, 1 / 256),
        (8704, 5120, 1 / 32),
        (5120, 4352, 0.1),
    ],
)
@torch.inference_mode()
def test_fp16_m8_replays_live_inputs(n: int, k: int, input_scale: float) -> None:
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("DFlash2 M8 is SM70 only")
    if not hasattr(torch.ops._C, "sm70_dflash2_fp16_m8_out"):
        pytest.fail("Build the source-complete SM70 extension")
    torch.manual_seed(20261004)
    layer = torch.nn.Linear(k, n, bias=False, device="cuda", dtype=torch.float16)
    # Use BF16 weight values within FP16 transport's range.
    layer.weight.copy_(layer.weight.bfloat16().half())
    original = layer.weight.clone()
    layer._sm70_dflash2_fp16_m8 = True
    assert prepare_dflash2_fp16_m8(layer)
    assert torch.equal(layer.weight, original)
    assert "_sm70_dflash2_fp16_packed" not in layer.state_dict()
    x = torch.randn(8, k, device="cuda", dtype=torch.float16) * input_scale
    apply_dflash2_fp16_m8(layer, x, None)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = apply_dflash2_fp16_m8(layer, x, None)
    for amplitude in [1.0, -0.5, 0.0, 2.0]:
        x.copy_(torch.randn_like(x) * (input_scale * amplitude))
        graph.replay()
        oracle = F.linear(x.float(), original.float())
        torch.testing.assert_close(output.float(), oracle, atol=2e-3, rtol=2e-3)
        assert torch.isfinite(output).all()
    # Larger batches and strided/3D operands retain the existing GEMM route.
    assert apply_dflash2_fp16_m8(layer, x.repeat(4, 1), None) is None
    assert apply_dflash2_fp16_m8(layer, x.T.contiguous().T, None) is None
    assert apply_dflash2_fp16_m8(layer, x.unsqueeze(0), None) is None
    assert apply_dflash2_fp16_m8(layer, x, torch.zeros(n, device=x.device)) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@torch.inference_mode()
def test_dynamic_prefill_trace_retains_m8_decode_dispatch():
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("DFlash2 M8 is SM70 only")
    layer = torch.nn.Linear(5120, 1280, bias=False, device="cuda", dtype=torch.float16)
    layer.weight.copy_(layer.weight.bfloat16().half())
    layer._sm70_dflash2_fp16_m8 = True
    assert prepare_dflash2_fp16_m8(layer)
    compilations = []

    def backend(graph, inputs):
        compilations.append(graph)
        return torch._inductor.compile(graph, inputs)

    compiled = torch.compile(
        lambda x: apply_dflash2_fp16_m8(layer, x, None),
        backend=backend,
        fullgraph=True,
        dynamic=True,
    )
    for rows in [12, 8, 7, 32, 8, 128]:
        x = torch.randn(rows, 5120, device="cuda", dtype=torch.float16) * 0.1
        actual = compiled(x)
        if rows == 8:
            expected = torch.empty_like(actual)
            tile, warps = layer._sm70_dflash2_fp16_geometry
            torch.ops._C.sm70_dflash2_fp16_m8_out(
                expected,
                x,
                layer._sm70_dflash2_fp16_packed,
                tile,
                warps,
            )
        else:
            expected = F.linear(x, layer.weight)
        assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    assert len(compilations) == 1
    x = torch.randn(8, 5120, device="cuda", dtype=torch.float16) * 0.1
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = compiled(x)
    for amplitude in [0.0, -0.1, 0.2]:
        x.copy_(torch.randn_like(x) * amplitude)
        graph.replay()
        expected = apply_dflash2_fp16_m8(layer, x, None)
        assert torch.equal(output.view(torch.int16), expected.view(torch.int16))
