# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.fused_moe.sm70_small_routing import (
    SM70_SMALL_ROUTING,
    _small_route,
    _small_unroute,
)
from vllm.platforms import current_platform


def test_cpu_route_handles_repeated_experts_and_empty_intervals():
    x = torch.arange(12).reshape(3, 4).half()
    ids = torch.tensor([[2, 0], [1, 2], [0, 2]])
    routed, offsets, sorted_ids, inverse = _small_route(x, ids, 4)
    assert offsets.tolist() == [0, 2, 3, 6, 6]
    assert sorted_ids.tolist() == [0, 0, 1, 2, 2, 2]
    torch.testing.assert_close(routed[inverse.long()], x.repeat_interleave(2, 0))


def test_unroute_cpu_fallback_preserves_fp32_weighted_sum():
    torch.manual_seed(20261004)
    down = torch.randn(6, 4).half()
    inverse = torch.tensor([3, 0, 2, 4, 1, 5], dtype=torch.int32)
    weights = torch.randn(3, 2)
    reference = (
        (down[inverse.long()].view(3, 2, 4).float() * weights[:, :, None]).sum(1).half()
    )
    torch.testing.assert_close(
        _small_unroute(down, inverse, weights), reference, rtol=0, atol=0
    )


def test_route_compilation_keeps_dynamic_m_opaque():
    import vllm.model_executor.layers.fused_moe.sm70_small_routing  # noqa: F401

    graphs = []

    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward

    def f(x, ids):
        return torch.ops.vllm.sm70_small_expert_route(x, ids, 512)

    compiled = torch.compile(f, dynamic=True, fullgraph=True, backend=backend)
    for m in (5, 20, 128):
        x = torch.empty(m, 2560, dtype=torch.float16, device="meta")
        ids = torch.empty(m, 10, dtype=torch.int64, device="meta")
        torch._dynamo.mark_dynamic(x, 0, min=2, max=8192)
        torch._dynamo.mark_dynamic(ids, 0, min=2, max=8192)
        routed, offsets, sorted_ids, inverse = compiled(x, ids)
        assert routed.shape == (m * 10, 2560)
        assert offsets.shape == (513,)
        assert sorted_ids.shape == inverse.shape == (m * 10,)
    assert len(graphs) == 1
    assert any(
        "sm70_small_expert_route" in str(n.target) for n in graphs[0].graph.nodes
    )
    torch._dynamo.reset()


@pytest.mark.skipif(
    not current_platform.is_device_capability(70), reason="CUDA SM70 required"
)
@pytest.mark.parametrize("m", [1, 2, 5, 10, 20, 32])
@pytest.mark.parametrize("pattern", ["random", "one_expert", "sparse"])
def test_fused_routing_and_fp32_unroute(m, pattern, monkeypatch):
    torch.manual_seed(20261004 + m)
    x = torch.randn(m, 2560, device="cuda", dtype=torch.float16)
    ids = torch.randn(m, 512, device="cuda").topk(10, dim=1).indices
    if pattern == "one_expert":
        ids.fill_(511)
    elif pattern == "sparse":
        ids = torch.where(ids % 2 == 0, 0, 511)
    if m % 2 == 0:
        ids = ids.int()
    assert SM70_SMALL_ROUTING.reason(x, ids, 512) is None
    reference_ids, order = ids.flatten().sort(stable=True)
    reference_offsets = torch.searchsorted(
        reference_ids, torch.arange(513, device="cuda")
    ).int()
    reference_routed = x[order // 10]

    def reject_sort_pipeline(*args, **kwargs):
        raise AssertionError("admitted routing called searchsorted")

    monkeypatch.setattr(torch, "searchsorted", reject_sort_pipeline)
    routed, offsets, sorted_ids, inverse = _small_route(x, ids, 512)
    torch.testing.assert_close(routed, reference_routed, rtol=0, atol=0)
    torch.testing.assert_close(offsets, reference_offsets, rtol=0, atol=0)
    torch.testing.assert_close(sorted_ids, reference_ids.long(), rtol=0, atol=0)
    torch.testing.assert_close(inverse.long(), order.argsort(), rtol=0, atol=0)
    down = torch.randn_like(routed)
    weights = torch.randn(m, 10, device="cuda")
    reference = (
        (down[order.argsort()].reshape(m, 10, 2560).float() * weights[:, :, None])
        .sum(1)
        .half()
    )
    actual = _small_unroute(down, inverse, weights)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        packed = torch.ops.vllm.sm70_small_expert_route(x, ids, 512)
        out = torch.ops.vllm.sm70_small_expert_unroute(down, packed[3], weights)
    graph.replay()
    torch.testing.assert_close(out, reference, rtol=0, atol=0)
    saved = out.clone()
    graph.replay()
    torch.testing.assert_close(out, saved, rtol=0, atol=0)


@pytest.mark.skipif(
    not current_platform.is_device_capability(70), reason="CUDA SM70 required"
)
@pytest.mark.parametrize("top_k", [1, 2, 3, 4, 8, 10, 16])
@pytest.mark.parametrize("weight_dtype", [torch.float16, torch.float32])
def test_unroute_matches_torch_short_reduction_order(top_k, weight_dtype):
    torch.manual_seed(9100 + top_k)
    m, h = 5, 2560
    down = torch.randn(m * top_k, h, device="cuda", dtype=torch.float16)
    inverse = torch.randperm(m * top_k, device="cuda").int()
    weights = torch.randn(m, top_k, device="cuda", dtype=weight_dtype)
    expected = (
        (
            down[inverse.long()].reshape(m, top_k, h).float()
            * weights[:, :, None].float()
        )
        .sum(1)
        .half()
    )
    torch.testing.assert_close(
        _small_unroute(down, inverse, weights), expected, rtol=0, atol=0
    )
