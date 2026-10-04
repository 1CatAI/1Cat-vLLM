# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BF16 storage reference coverage; these cases do not measure throughput."""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.is_device_capability(70),
    reason="SM70 required",
)


@pytest.mark.parametrize("down", [False, True])
@pytest.mark.parametrize("m", [1, 5])
def test_original_bf16_mtp_routes_and_graph(m, down):
    n, k = (2560, 160) if down else (320, 2560)
    w = torch.full((512, n, k), 2**-30, dtype=torch.bfloat16, device="cuda")
    x = torch.full(
        (m * 10 if down else m, k), 65504, dtype=torch.float16, device="cuda"
    )
    ids = torch.arange(m * 10, device="cuda", dtype=torch.int32) % 512
    ids[-1] = -1
    weights = torch.full((m, 10), 0.5, device="cuda", dtype=torch.float32)
    padded = torch.tensor([m * 20], device="cuda", dtype=torch.int32)
    actual = torch.full((m, 10, n), float("nan"), device="cuda", dtype=torch.float16)

    def launch():
        torch.ops._C.sm70_mtp_moe_bf16_out(actual, x, w, ids, weights, padded, down)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    for activation in (65504, 0.5, 2.0):
        x.fill_(activation)
        actual.fill_(float("nan"))
        graph.replay()
        # A FP16 checkpoint cast makes the expected contribution zero. The
        # retained BF16 dot product uses an independent FP64 scalar reference.
        value = activation * (2**-30) * k * (0.5 if down else 1.0)
        expected = torch.full(actual.shape, value, dtype=torch.float64).half()
        expected.reshape(m * 10, n)[-1].zero_()
        assert torch.equal(actual.cpu(), expected)
    assert bool((actual.reshape(m * 10, n)[:-1] != 0).all())
