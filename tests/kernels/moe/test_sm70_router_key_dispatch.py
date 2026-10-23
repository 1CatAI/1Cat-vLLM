# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU route guards for the lossless FP16 E512/K10 sort-key specialization."""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.router import fused_topk_router as mod


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("rows", [1, 2, 5, 10, 16, 17])
def test_packed_half_key_preserves_dtype_and_batch_guards(monkeypatch, dtype, rows):
    calls = []

    class Kernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                calls.append((grid, kwargs))

            return launch

    monkeypatch.setattr(mod, "_sm70_qwen38_router_topk_kernel", Kernel())
    x = torch.empty(rows, 512, dtype=dtype)
    weights = torch.empty(rows, 10, dtype=torch.float32)
    ids = torch.empty(rows, 10, dtype=torch.int32)
    mod._sm70_qwen38_router_topk(weights, ids, torch.empty_like(ids), x)
    assert len(calls) == 1
    grid, kwargs = calls[0]
    assert grid == (rows,)
    assert kwargs["PACKED_HALF_KEY"] == (rows <= 16 and dtype == torch.float16)
    assert kwargs["num_warps"] == 8  # Keep the FP32 normalization reduction.


@pytest.mark.parametrize("rows", [5, 10])
@pytest.mark.parametrize("select_top16", [False, True])
def test_mtp_public_route_preserves_graph_weights_ids_and_source_rows(
    monkeypatch, rows, select_top16
):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 required")
    monkeypatch.setenv("VLLM_SM70_QWEN38_ROUTER_TOPK", "1")
    monkeypatch.setenv("VLLM_SM70_MTP_ROUTER_TOP16", str(int(select_top16)))
    x = torch.zeros(rows, 512, device="cuda", dtype=torch.float16)
    hidden = torch.zeros(rows, 2560, device="cuda", dtype=torch.float16)
    ref = (
        torch.empty(rows, 10, device="cuda", dtype=torch.float32),
        torch.empty(rows, 10, device="cuda", dtype=torch.int32),
        torch.empty(rows, 10, device="cuda", dtype=torch.int32),
    )

    def control():
        mod._sm70_qwen38_router_topk_kernel[(rows,)](
            x,
            *ref,
            E=512,
            K=10,
            M=rows,
            BLOCK_E=512,
            PACKED_HALF_KEY=False,
            num_warps=8,
        )

    def candidate():
        return mod.fused_topk(hidden, x, 10, True)

    for _ in range(3):
        control()
        candidate()
    torch.accelerator.synchronize()
    graphs = [torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()]
    with torch.cuda.graph(graphs[0]):
        control()
    with torch.cuda.graph(graphs[1]):
        actual = candidate()
    for replay in range(12):
        x.normal_(0, (0.001, 0.1, 1.0, 30.0)[replay % 4])
        if replay == 0:
            x.zero_()
            x[:, ::2] = -0.0
        elif replay == 1:
            x.copy_((torch.arange(512, device="cuda") % 7).half())
        elif replay == 2:
            x[:, 0] = float("nan")
        elif replay == 3:
            x.fill_(-float("inf"))
        for result in actual:
            result.fill_(-777)
        for graph in graphs:
            graph.replay()
        for got, expected in zip(actual, ref):
            assert torch.equal(got.view(torch.int32), expected.view(torch.int32))
