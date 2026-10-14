# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Live grouping changes preserve the retained Q8_1 expert output bytes."""

import numpy as np
import pytest
import torch
import vllm._C  # noqa: F401

from vllm.model_executor.layers.quantization.gguf_grouped_gate import grouped_gate_up
from vllm.model_executor.layers.quantization.gguf_raw import RawGGUFProjection
from vllm.transformers_utils.gguf_tensor_reader import quant_size

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("source", [18, 21, 22])
@pytest.mark.parametrize("m", [9, 10, 20])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_grouped_experts_changed_route_graph_exact(source, m, index_dtype):
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 required")
    experts, n, k, top_k = 256, 32, 256, 10
    block, size = quant_size(source)
    rng = np.random.default_rng(20261015 + source)
    weights = []
    for _ in range(2):
        raw = rng.integers(0, 256, (experts * n, k // block, size), dtype=np.uint8)
        raw[..., :2] = np.array([0.0007], dtype="<f2").view(np.uint8)
        payload = RawGGUFProjection.from_rows(raw.reshape(experts * n, -1), source)
        weights.append(torch.from_numpy(payload.data.reshape(experts, n, -1)).cuda())
    x = torch.randn(m, k, device="cuda", dtype=torch.float16) * 0.125
    q8 = torch.empty(m, k // 32, 36, device="cuda", dtype=torch.uint8)
    ids = torch.zeros(m, top_k, device="cuda", dtype=index_dtype)
    out = torch.empty(m, top_k, n // 32, 36, device="cuda", dtype=torch.uint8)
    reference = torch.empty_like(out)

    def run():
        torch.ops._C.gguf_quantize_q8_1_sm70_out(q8, x)
        assert grouped_gate_up(out, q8, ids, *weights, source, source == 21)

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    # The captured device admission changes from grouped to sparse and back.
    # Unique routes also exercise the ordinary dot branch and original order.
    routes = torch.arange(m * top_k, device="cuda", dtype=index_dtype).view_as(ids)
    for pool in (4, experts, 17):
        ids.copy_(routes % pool)
        x.normal_(std=0.125)
        graph.replay()
        torch.ops._C.gguf_dp4a_gate_up_sm70_out(
            reference, q8, ids, *weights, source, True, 16, source == 21
        )
        assert torch.equal(out, reference)


@pytest.mark.parametrize("m", [8, 21])
def test_grouped_experts_shape_fallback_does_not_touch_output(m):
    ids = torch.zeros(m, 10, device="cuda", dtype=torch.int32)
    out = torch.full((m, 10, 1, 36), 77, device="cuda", dtype=torch.uint8)
    activation = torch.empty(m, 8, 36, device="cuda", dtype=torch.uint8)
    weight = torch.empty(1, 32, 104, device="cuda", dtype=torch.uint8)
    assert not grouped_gate_up(out, activation, ids, weight, weight, 18, False)
    assert bool((out == 77).all())
