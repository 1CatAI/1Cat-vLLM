# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Grouped original down retains TP block boundaries and route-sum order."""

import numpy as np
import pytest
import torch
import vllm._C  # noqa: F401

from tests.kernels.quantization.test_gguf_dp4a import quantize_reference
from vllm.model_executor.layers.quantization.gguf_grouped_gate import grouped_down
from vllm.model_executor.layers.quantization.gguf_native import packed_tp_span
from vllm.transformers_utils.gguf_tensor_reader import dequantize, quant_size

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("source", [20, 42])
@pytest.mark.parametrize("rank", [0, 1, 2, 3])
@pytest.mark.parametrize("m", [25, 30, 40])
def test_grouped_down_changed_graph_exact_and_official_oracle(source, rank, m):
    experts, n, full_k, k, top_k = 512, 32, 640, 160, 10
    block, size = quant_size(source)
    rng = np.random.default_rng(20261015 + source)
    raw = rng.integers(0, 256, (experts, n, full_k // block, size), dtype=np.uint8)
    raw[..., :2] = np.array([0.002], dtype="<f2").view(np.uint8)
    raw = raw.reshape(experts, n, -1)
    begin, end, left, _ = packed_tp_span(full_k, source, 4, rank)
    weight = torch.from_numpy(raw[..., begin:end].copy()).cuda()
    decoded = torch.from_numpy(
        dequantize(raw, source)[..., rank * k : (rank + 1) * k]
    ).cuda()
    hidden = torch.randn(m * top_k, k, device="cuda", dtype=torch.float16) * 0.125
    packet = torch.empty(m * top_k, k // 32, 36, device="cuda", dtype=torch.uint8)
    ids = torch.zeros(m, top_k, device="cuda", dtype=torch.int32)
    probabilities = torch.softmax(torch.randn(m, top_k, device="cuda"), -1)
    out = torch.empty(m, n, device="cuda", dtype=torch.float16)
    reference = torch.empty_like(out)

    def run():
        q, d, s = quantize_reference(hidden)
        packet[..., :4].copy_(torch.stack((d, s), -1).view(torch.uint8))
        packet[..., 4:].copy_(q.view(torch.uint8))
        assert grouped_down(
            out,
            packet.view(m, top_k, k // 32, 36),
            ids,
            probabilities,
            weight,
            source,
            left,
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    routes = torch.arange(m * top_k, device="cuda").view_as(ids)
    for pool in (4, 512, 17):
        ids.copy_((routes * 7 + 3) % pool)
        hidden.normal_(std=0.125)
        probabilities.copy_(torch.softmax(torch.randn_like(probabilities), -1))
        graph.replay()
        for start in range(0, m, 20):
            torch.ops._C.gguf_dp4a_raw_down_unroute_sm70_out(
                reference[start : start + 20],
                packet.view(m, top_k, k // 32, 36)[start : start + 20],
                ids[start : start + 20],
                probabilities[start : start + 20],
                weight,
                source,
                left,
            )
        assert torch.equal(out, reference)
        q, d, _ = quantize_reference(hidden)
        x = (q.float() * d[..., None].float()).reshape(m, top_k, k)
        projected = torch.einsum("mtk,mtnk->mtn", x, decoded[ids.long()]).half()
        expected = (projected.float() * probabilities[..., None]).sum(1).half()
        torch.testing.assert_close(out, expected, rtol=0.002, atol=0.002)
        assert torch.isfinite(out).all()


@pytest.mark.parametrize("m", [20, 24, 41])
def test_grouped_down_range_fallback_preserves_output(m):
    out = torch.full((m, 32), 17.0, device="cuda", dtype=torch.float16)
    ids = torch.zeros(m, 10, device="cuda", dtype=torch.int32)
    assert not grouped_down(
        out,
        torch.empty(m, 10, 5, 36, device="cuda", dtype=torch.uint8),
        ids,
        torch.zeros_like(ids, dtype=torch.float32),
        torch.empty(1, 32, 90, device="cuda", dtype=torch.uint8),
        20,
        0,
    )
    assert bool((out == 17.0).all())
