# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerical tests for original BF16 weights with FP16 activations."""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.is_device_capability(70),
    reason="SM70 required",
)


@pytest.mark.parametrize("m", [1, 2, 3, 4, 5])
def test_weight_values_below_fp16_range_survive(m):
    # A Half checkpoint cast loses every weight. The retained BF16 values have
    # an observable output when multiplied by a large, finite Half activation.
    weight = torch.full((17, 257), 2**-30, dtype=torch.bfloat16, device="cuda")
    x = torch.full((m, 257), 65504, dtype=torch.float16, device="cuda")
    reference = (x.cpu().double() @ weight.cpu().double().T).half()
    assert torch.count_nonzero(weight.half()) == 0
    output = torch.ops._C.sm70_bf16_weight_linear(x, weight)
    assert torch.equal(output.cpu(), reference)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = torch.ops._C.sm70_bf16_weight_linear(x, weight)
    for scale in (0.5, 1.0, 2.0):
        x.fill_(scale)
        reference = (x.cpu().double() @ weight.cpu().double().T).half()
        graph.replay()
        assert torch.equal(captured.cpu(), reference)


@pytest.mark.parametrize("m", [1, 5])
def test_dense_fp64_oracle_with_column_and_k_tails(m):
    torch.manual_seed(417)
    x = torch.randn((m, 257), device="cuda", dtype=torch.float16)
    weight = torch.randn((19, 257), device="cuda", dtype=torch.bfloat16)
    output = torch.ops._C.sm70_bf16_weight_linear(x, weight)
    reference = x.cpu().double() @ weight.cpu().double().T
    # Compare to the independent FP64 dot product, allowing the Half output
    # boundary and a conservative FP32 forward-error bound for the FMA tree.
    actual = output.cpu().double()
    half = reference.half()
    spacing = (
        torch.nextafter(half.abs(), torch.full_like(half, float("inf"))) - half.abs()
    ).double()
    magnitude = x.cpu().double().abs() @ weight.cpu().double().abs().T
    bound = spacing + magnitude * 32 * torch.finfo(torch.float32).eps
    assert bool((torch.abs(actual - reference) <= bound).all())
