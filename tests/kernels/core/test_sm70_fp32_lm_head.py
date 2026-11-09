# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.sm70_fp32_lm_head import indexed_fp32_logits


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("rows", [1, 8])
@torch.inference_mode()
def test_indexed_logits_preserve_fp32_and_graph_replay(rows):
    torch.manual_seed(817)
    x = torch.randn(rows, 5120, device="cuda", dtype=torch.float16) * 0.1
    weight = torch.randn(1024, 5120, device="cuda", dtype=torch.float16) * 0.1
    ids = torch.randint(1024, (rows, 64), device="cuda")
    out = torch.empty(rows, 64, device="cuda", dtype=torch.float32)
    indexed_fp32_logits(x, weight, ids, out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        indexed_fp32_logits(x, weight, ids, out)
    for _ in range(3):
        x.normal_(std=0.1)
        ids.random_(1024)
        graph.replay()
        expected = (x.float() @ weight.float().t()).gather(1, ids)
        torch.testing.assert_close(out, expected, atol=3e-6, rtol=3e-6)
        assert out.dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_indexed_logits_do_not_collapse_half_rounding_boundary():
    x = torch.zeros(1, 5120, device="cuda", dtype=torch.float16)
    x[0, 0] = 1
    x[0, 1] = 2**-12
    weight = torch.zeros(64, 5120, device="cuda", dtype=torch.float16)
    weight[:, 0] = 1
    weight[:, 1] = torch.arange(64, device="cuda") / 64
    ids = torch.arange(64, device="cuda")[None]
    out = torch.empty(1, 64, device="cuda", dtype=torch.float32)
    indexed_fp32_logits(x, weight, ids, out)
    expected = x.float() @ weight.float().t()
    assert torch.equal(out, expected)
    assert torch.unique(out).numel() == 64
    assert torch.unique(out.half()).numel() == 1
