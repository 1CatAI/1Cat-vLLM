# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused activation preserves both the FP16 and original Q8_1 bytes."""

import pytest
import torch
import torch.nn.functional as F

from vllm import _custom_ops  # noqa: F401
from vllm.models.deepseek_v41.activations import swiglu

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("m", [1, 2, 3, 4, 5, 6])
def test_routed_swiglu_q8_is_byte_exact_after_clamps_and_graph(m):
    torch.manual_seed(41)
    projection = (torch.randn(m, 6, 2, 576, device="cuda") * 12).half()
    boundary = torch.tensor(
        [-65504, -10.01, -10, -9.99, -0.0, 0.0, 0.001, 9.99, 10, 10.01, 65504],
        device="cuda",
        dtype=torch.half,
    )
    projection[..., : boundary.numel()] = boundary
    probabilities = torch.tensor([0.0001, 0.01, 0.0499, 0.14, 0.3, 0.5], device="cuda")
    probabilities = (probabilities[None].expand(m, -1) * 2.5).contiguous()
    actual = torch.empty(m, 6, 18, 36, device="cuda", dtype=torch.uint8)

    def run():
        torch.ops._C.gguf_v41_swiglu_q8_sm70_out(actual, projection, probabilities)

    def check():
        gate, up = projection[:, :, 0].float(), projection[:, :, 1].float()
        hidden = (
            F.silu(gate.clamp(max=10)) * up.clamp(-10, 10) * probabilities[..., None]
        ).half()
        expected = torch.empty_like(actual)
        torch.ops._C.gguf_quantize_q8_1_sm70_out(
            expected.view(-1, 18, 36), hidden.view(-1, 576)
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    run()
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    projection.mul_(-0.75)
    probabilities.copy_(probabilities.flip(1))
    graph.replay()
    check()


@pytest.mark.parametrize("m", [1, 5, 6, 129, 512])
@pytest.mark.parametrize("width", [512, 576])
def test_shared_swiglu_retains_each_fp16_bit_and_replays(m, width):
    gate, up = [(torch.randn(m, width, device="cuda") * 12).half() for _ in range(2)]

    def check(actual):
        expected = (
            F.silu(gate.float().clamp(max=10)) * up.float().clamp(-10, 10)
        ).half()
        torch.testing.assert_close(
            actual.view(torch.int16), expected.view(torch.int16), rtol=0, atol=0
        )

    check(swiglu(gate, up))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = swiglu(gate, up)
    gate.mul_(-0.75)
    up.mul_(0.5)
    graph.replay()
    check(out)
