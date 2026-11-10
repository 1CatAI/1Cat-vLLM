# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Separate diagnostic kernels must preserve the ordinary projection results."""

import pytest
import torch

from tests.kernels.quantization.test_gguf_dmv import planes
from vllm.model_executor.layers.quantization.gguf_dmv import table

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize(
    "kinds,kw,tn,split,pair",
    [
        ((12, 12), 4, 4, 1, True),
        ((23, 23), 4, 4, 1, True),
        ((21, 21), 4, 4, 1, True),
        ((18, 18), 4, 4, 1, True),
        ((21, 23), 4, 4, 2, True),
        ((22, 21), 8, 2, 1, True),
        ((17, 16), 8, 2, 1, True),
        ((18, 23), 4, 2, 2, False),
        ((12, 23, 21), 4, 2, 2, False),
        ((12, 23, 18), 4, 2, 1, False),
    ],
)
def test_clocked_projection_preserves_output_bits(kinds, kw, tn, split, pair):
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 required")
    segments = [planes(kind, k=5120) for kind in kinds]
    x = torch.randn(8, 5120, device="cuda", dtype=torch.float16)
    result = torch.empty(8, 64 * len(kinds), device="cuda", dtype=torch.float16)
    outputs = list(result.split(64, dim=1))
    pair_out = torch.empty_like(outputs[0]) if pair else None
    tiles = 2 // (tn // 2) if pair else len(kinds) * ((64 + tn * 32 - 1) // (tn * 32))
    scratch = torch.empty(tiles * split * tn * 256, device="cuda", dtype=torch.float32)
    counters = torch.zeros(tiles, device="cuda", dtype=torch.int32)
    timestamps = torch.zeros(
        tiles * split, kw * tn, 2, device="cuda", dtype=torch.int64
    )
    operands = (
        x,
        [s[1][0] for s in segments],
        [s[1][1] for s in segments],
        [s[1][2] for s in segments],
        outputs,
        [s[0] for s in segments],
        [64] * len(kinds),
        5120,
        split,
        kw,
        scratch,
        counters,
        tn,
        None,
        table(x.device),
        None,
        None,
        pair_out,
        False,
    )
    for amplitude in (0.125, 1.0, 4.0):
        x.normal_().mul_(amplitude)
        torch.ops._C.gguf_dmv_sm70_out(*operands)
        actual = result if pair_out is None else pair_out
        expected = actual.clone()
        torch.ops._C.gguf_dmv_sm70_clocked_out(timestamps, *operands)
        assert torch.equal(expected.view(torch.int16), actual.view(torch.int16))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        torch.ops._C.gguf_dmv_sm70_clocked_out(timestamps, *operands)
    for _ in range(3):
        graph.replay()
        assert torch.equal(expected.view(torch.int16), actual.view(torch.int16))
        assert not counters.any()
        assert (timestamps[..., 0] > 0).all()
        assert (timestamps[..., 1] >= timestamps[..., 0]).all()
        assert timestamps[..., 1].max() > timestamps[..., 0].min()
