# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase-2b GPU regression for the FULL-CUDA-graph capture crash (W3): an E4M3
verify batch with >16 rows routes to grouped(16)+XQA-page4 split, and the XQA
page4 workspace is (re)built on the graph's capture stream. Before the fix it
built active_num_partitions with torch.tensor([...], device=cuda) -- a host->
device copy that is illegal mid-capture ("operation not permitted when stream is
capturing"). This test clears the stream-keyed workspace cache right before
capture so the workspace MUST be rebuilt DURING capture (no warmup on the capture
stream), for 17/18/20 rows. WRITTEN, NOT RUN on CPU: SM70-only, skips otherwise.
"""

from __future__ import annotations

import pytest
import torch

from vllm.models.qwen4_exp.nvidia.ops import qsa as qsa_ops

qsa_sparse_paged_attention = qsa_ops.qsa_sparse_paged_attention

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for E4M3 capture tests"
)

HEAD_DIM = 256


def _sm70_only():
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("QSA E4M3 page4 capture is SM70-only")


def _e4m3_inputs(rows):
    heads, page_size, topk = 6, 2052, 2051
    query = torch.randn(
        (rows, heads, HEAD_DIM), dtype=torch.float16, device="cuda"
    ).mul_(0.2)
    key = torch.randn(
        (1, page_size, 1, HEAD_DIM), dtype=torch.float16, device="cuda"
    ).mul_(0.35)
    value = torch.randn_like(key).mul_(0.3)
    k_scale = float(key.abs().max().item()) / 448.0
    v_scale = float(value.abs().max().item()) / 448.0
    key_cache = (key / k_scale).to(torch.float8_e4m3fn).view(torch.uint8)
    value_cache = (value / v_scale).to(torch.float8_e4m3fn).view(torch.uint8)
    logical_indices = torch.arange(topk, dtype=torch.int32, device="cuda").repeat(
        rows, 1
    )
    block_table = torch.zeros((1, 1), dtype=torch.int32, device="cuda")
    token_to_req = torch.zeros(rows, dtype=torch.int32, device="cuda")
    query_positions = torch.arange(topk - rows, topk, dtype=torch.int64, device="cuda")
    sequence_lengths = torch.tensor([topk], dtype=torch.int32, device="cuda")
    args = (query, key_cache, value_cache, logical_indices, block_table, token_to_req)
    kwargs = dict(
        query_positions=query_positions,
        sequence_lengths=sequence_lengths,
        kv_cache_dtype="fp8_e4m3",
        k_scale=k_scale,
        v_scale=v_scale,
    )
    return args, kwargs


@pytest.mark.parametrize("rows", [17, 18, 20])
@torch.inference_mode()
def test_e4m3_page4_workspace_capture_safe(rows, monkeypatch):
    """>16-row E4M3 verify batch captured WITHOUT warming the capture stream."""
    _sm70_only()
    torch.manual_seed(400 + rows)
    # Force the grouped(16)+XQA-page4 split that builds the XQA workspace.
    monkeypatch.setattr(qsa_ops, "_SM70_QSA_XQA_PAGE4", True)
    monkeypatch.setattr(qsa_ops, "_SM70_QSA_XQA_PAGE4_MIN_ROWS", 16)
    monkeypatch.setattr(qsa_ops, "_SM70_QSA_GROUPED_PAGE4", True)
    args, kwargs = _e4m3_inputs(rows)

    eager = qsa_sparse_paged_attention(*args, **kwargs).clone()

    # Drop every cached workspace so the graph capture stream has none: the
    # workspace is then rebuilt DURING capture, which is exactly what crashed.
    qsa_ops._SM70_QSA_XQA_PAGE4_WORKSPACES.clear()
    qsa_ops._SM70_QSA_GROUPED_PAGE4_WORKSPACES.clear()

    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = qsa_sparse_paged_attention(*args, **kwargs)
    graph.replay()
    torch.cuda.synchronize()

    assert torch.isfinite(captured).all()
    torch.testing.assert_close(captured.float(), eager.float(), atol=3e-2, rtol=3e-2)
