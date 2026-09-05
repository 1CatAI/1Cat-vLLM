# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental grouped W13/W2: no default runtime route is changed.

Set a task-owned TORCH_EXTENSIONS_DIR and build with CUDA 12.8 on SM70.
"""

import os
from pathlib import Path

import pytest
import torch
from torch.utils.cpp_extension import load


@pytest.fixture(scope="module")
def weights():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 GPU required")
    if not os.environ.get("TORCH_EXTENSIONS_DIR"):
        pytest.skip("Set task-owned TORCH_EXTENSIONS_DIR")
    source = (
        Path(__file__).resolve().parents[3]
        / "csrc/sm70_turbomind/ops/nvfp4_grouped_decode_sm70.cu"
    )
    if not hasattr(torch.ops._C, "nvfp4_grouped_w13_sm70_out"):
        load(
            name="sm70_moe_packed_w13_screen",
            sources=[str(source)],
            extra_cuda_cflags=["-O3", "-std=c++17", "-lineinfo"],
            is_python_module=False,
        )
    torch.manual_seed(123)
    return (
        torch.randint(0, 2**31 - 1, (512, 2560, 40), device="cuda", dtype=torch.int32),
        torch.rand(512, 160, 320, device="cuda", dtype=torch.float16) * 0.01,
        torch.randint(0, 2**31 - 1, (512, 160, 320), device="cuda", dtype=torch.int32),
        torch.rand(512, 10, 2560, device="cuda", dtype=torch.float16) * 0.01,
    )


@pytest.mark.parametrize("tokens", range(1, 17))
def test_regroup_graph_covers_every_route(weights, tokens):
    """Changing counts must not consume stale groups or miss the last pack."""
    from vllm import _sm70_ops as ops

    w, s, w2, s2 = weights
    n = tokens * 10
    x = torch.randn(tokens, 2560, device="cuda", dtype=torch.float16) * 0.1
    ids = torch.zeros(n, device="cuda", dtype=torch.int32)
    out = torch.empty(n, 160, device="cuda", dtype=torch.float16)
    rows = torch.empty(n, 8, device="cuda", dtype=torch.int32)
    experts = torch.empty(n, device="cuda", dtype=torch.int32)
    sizes = torch.empty_like(experts)
    total = torch.empty(1, device="cuda", dtype=torch.int32)
    topk = torch.softmax(torch.randn(tokens, 10, device="cuda"), dim=-1)
    routed = torch.empty(n, 2560, device="cuda", dtype=torch.float16)
    reduced = torch.empty(tokens, 2560, device="cuda", dtype=torch.float16)
    reference = torch.empty_like(reduced)

    def run():
        torch.ops._C.nvfp4_grouped_w13_sm70_out(
            out, x, w, s, ids, rows, experts, sizes, total, 8, False
        )
        torch.ops._C.nvfp4_grouped_w2_sm70_out(
            reduced, routed, out, w2, s2, topk, rows, experts, sizes, total
        )

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for values in (
        torch.arange(n, dtype=torch.int32),
        torch.arange(n, dtype=torch.int32) % 10,
        torch.zeros(n, dtype=torch.int32),  # up to 20 packs of one expert
        torch.full((n,), -1, dtype=torch.int32),
        torch.arange(n, dtype=torch.int32) % 17,
    ):
        x.normal_(0, 0.1)
        ids.copy_(values)
        for t in (rows, experts, sizes, total):
            t.fill_(-9999)
        out.fill_(float("nan"))
        routed.fill_(float("nan"))
        reduced.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.isfinite(out).all()
        ops.nvfp4_moe_qpn_w2_reduce_sm70_out(reference, out, w2, s2, ids, topk)
        torch.testing.assert_close(reduced, reference, rtol=0, atol=0)
        count = total.item()
        assert 0 < count <= n
        rr, ee, ss = rows.cpu(), experts.cpu(), sizes.cpu()
        seen = []
        for group in range(count):
            size = ss[group].item()
            assert 1 <= size <= 8
            routes = rr[group, :size].tolist()
            assert all(0 <= route < n for route in routes)
            if ee[group] == 512:
                assert (values[routes] == -1).all()
            else:
                assert (values[routes] == ee[group]).all()
            seen.extend(routes)
        assert sorted(seen) == list(range(n))
        actual = out.clone()
        run()
        torch.testing.assert_close(out, actual, rtol=0, atol=0)
        if (values == -1).all():
            assert torch.count_nonzero(out) == 0


@pytest.mark.parametrize("tokens,split", [(4, 5), (8, 4), (16, 1)])
@pytest.mark.parametrize("interleaved", [False, True])
def test_same_split_matches_production(weights, tokens, split, interleaved):
    from vllm import _sm70_ops as ops

    w, s, _, _ = weights
    n = tokens * 10
    x = torch.randn(tokens, 2560, device="cuda", dtype=torch.float16) * 0.1
    ids = (torch.arange(n, device="cuda") % 13).int()
    out = torch.empty(n, 160, device="cuda", dtype=torch.float16)
    expected = torch.empty_like(out)
    rows = torch.empty(n, 8, device="cuda", dtype=torch.int32)
    experts = torch.empty(n, device="cuda", dtype=torch.int32)
    sizes = torch.empty_like(experts)
    total = torch.empty(1, device="cuda", dtype=torch.int32)
    ops.nvfp4_moe_qpn_w13_swiglu_batch_sm70_out(expected, x, w, s, ids, interleaved)
    torch.ops._C.nvfp4_grouped_w13_sm70_out(
        out, x, w, s, ids, rows, experts, sizes, total, split, interleaved
    )
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
