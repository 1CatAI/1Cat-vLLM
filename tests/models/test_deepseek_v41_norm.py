# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Short-row fusion preserves FP32 RMS and the FP16 store boundary."""

import pytest
import torch

from vllm.models.deepseek_v41.norm import short_rms_norm

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("hidden", [128, 512, 1280, 5120])
@pytest.mark.parametrize("magnitude", [0.0, 2**-20, 0.03125, 1.37, 100.0, 10000.0])
def test_short_rms_fp32_reference_row_invariance_and_graph(hidden, magnitude):
    generator = torch.Generator(device="cuda").manual_seed(4190)
    x = (torch.randn(6, hidden, generator=generator, device="cuda") * magnitude).half()
    weight = torch.rand(hidden, generator=generator, device="cuda") + 0.5
    eps = 1e-20
    short_rms_norm(x, weight, eps)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = short_rms_norm(x, weight, eps)
    for multiplier in (-0.75, 1.0):
        x.mul_(multiplier)
        weight.mul_(0.9)
        graph.replay()
        values = x.float()
        expected = (
            values * torch.rsqrt(values.square().mean(-1, keepdim=True) + eps) * weight
        ).half()
        torch.testing.assert_close(output, expected, rtol=0.001, atol=2e-6)
        for rows in range(1, 7):
            partial = short_rms_norm(x[:rows], weight, eps)
            torch.testing.assert_close(partial, output[:rows], rtol=0, atol=0)


def test_short_rms_rejects_outside_qualified_geometry():
    weight = torch.ones(512, dtype=torch.float32, device="cuda")
    for x in (
        torch.ones(7, 512, dtype=torch.float16, device="cuda"),
        torch.ones(6, 512, dtype=torch.float32, device="cuda"),
        torch.ones(6, 1024, dtype=torch.float16, device="cuda")[:, ::2],
    ):
        with pytest.raises(ValueError, match="FP16 M1..6"):
            short_rms_norm(x, weight, 1e-20)
