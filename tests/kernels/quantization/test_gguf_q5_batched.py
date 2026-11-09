# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batched Q5_K scheduling retains every bit of the upstream decoder."""

import numpy as np
import pytest
import torch
from test_gguf_native import load_native

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def packed_q5(n, k, scale):
    blocks = np.random.default_rng(41).integers(
        0, 256, (n * k // 256, 176), dtype=np.uint8
    )
    blocks[:, :2] = np.array([scale], dtype=np.float16).view(np.uint8)
    blocks[:, 2:4] = np.array([scale / 2], dtype=np.float16).view(np.uint8)
    return torch.from_numpy(blocks.reshape(n, -1)).cuda()


@pytest.mark.parametrize("n,k", [(1, 256), (3, 256), (5, 256), (13, 768), (64, 5120)])
@pytest.mark.parametrize("scale", [2**-10, 0.00137, 1e-6])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_batched_q5_decoder_is_bitwise_upstream_with_tail_and_graph(n, k, scale, dtype):
    native = load_native()
    raw = packed_q5(n, k, scale)

    def run(batched):
        return native.ggml_dequantize_upstream(raw, 13, n, k, dtype, batched)

    old, new = run(False), run(True)
    torch.testing.assert_close(new, old, rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        new = run(True)
    raw.copy_(packed_q5(n, k, scale * 2))
    graph.replay()
    torch.testing.assert_close(new, run(False), rtol=0, atol=0)
