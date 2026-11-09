# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVFP4 converter/consumer alignment, including Qwen GDN TP2 width."""

import pytest
import torch

from vllm.model_executor.layers.quantization import sm70_turbomind as tm


@pytest.mark.parametrize("n", [48, 8240])
@pytest.mark.parametrize("m", [1, 8, 17])
@torch.inference_mode()
def test_nvfp4_unaligned_output_matches_dense_and_graph(n, m):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires SM70")
    if not hasattr(torch.ops._C, "nvfp4_sm70_prepare"):
        pytest.skip("requires native TurboMind NVFP4")
    gen = torch.Generator(device="cuda").manual_seed(713)
    k = 512
    packed = torch.randint(
        0, 256, (n, k // 2), device="cuda", dtype=torch.uint8, generator=gen
    )
    scales = torch.full((n, k // 16), 0.125, device="cuda", dtype=torch.float16)
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(packed, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(scales, requires_grad=False)
    layer.weight_global_scale = torch.nn.Parameter(
        torch.tensor(0.5, device="cuda"), requires_grad=False
    )
    codes = torch.stack((packed & 15, packed >> 4), -1).flatten(-2)
    lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device="cuda")
    weight = lut[(codes & 7).long()] * torch.where((codes & 8) != 0, -1.0, 1.0) * 0.0625
    x = torch.randn((m, k), device="cuda", generator=gen).half() * 0.1
    tm.prepare_nvfp4_linear(layer)
    state = getattr(layer, tm.STATE_ATTR)
    assert state.padded_output_size == (n + 31) // 32 * 32
    eager = tm.apply_prepared_linear(layer, x, None)
    expected = x.float() @ weight.t()
    torch.testing.assert_close(eager.float(), expected, atol=2e-3, rtol=1e-3)
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        out = tm.apply_prepared_linear(layer, x, None)
    x.mul_(0.5)
    graph.replay()
    torch.testing.assert_close(
        out, tm.apply_prepared_linear(layer, x, None), atol=0, rtol=0
    )
