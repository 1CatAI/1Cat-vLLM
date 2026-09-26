# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase-2 GPU numerical-consistency tests for E4M3 QSA verify shapes used by
MTP speculative decoding. WRITTEN, NOT RUN on CPU: every test is SM70-only and
skips when CUDA is unavailable. Run inside the W3 GPU window.

Coverage: FP16 vs E4M3 sparse-paged-attention agreement across multi-row verify
batches (5/10/15/18/20 rows) that straddle the <=16-row Triton path and the
>16-row grouped-page4 + XQA-page4 split, with per-request multi-row causal
query_positions, plus a CUDA graph capture/replay equivalence check.
"""

from __future__ import annotations

import pytest
import torch

from vllm.models.qwen4_exp.nvidia.ops import qsa as qsa_ops

qsa_sparse_paged_attention = qsa_ops.qsa_sparse_paged_attention

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for E4M3 MTP verify tests"
)

HEAD_DIM = 256


def _sm70_only():
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("QSA E4M3 verify shapes are SM70-only")


def _make_cache(key, value):
    k_scale = float(key.abs().max().item()) / 448.0
    v_scale = float(value.abs().max().item()) / 448.0
    key_cache = (key / k_scale).to(torch.float8_e4m3fn).view(torch.uint8)
    value_cache = (value / v_scale).to(torch.float8_e4m3fn).view(torch.uint8)
    return key_cache, value_cache, k_scale, v_scale


@pytest.mark.parametrize("rows", [5, 10, 15, 18, 20])
@torch.inference_mode()
def test_e4m3_multi_row_verify_matches_reference(rows, monkeypatch):
    """One request with `rows` speculative verify positions; E4M3 vs a decoded
    FP32 reference. rows>16 exercises the grouped-page4 + XQA-page4 split."""
    _sm70_only()
    torch.manual_seed(100 + rows)
    heads, page_size, topk = 6, 2052, 2051
    query = torch.randn(
        (rows, heads, HEAD_DIM), dtype=torch.float16, device="cuda"
    ).mul_(0.2)
    key = torch.randn(
        (1, page_size, 1, HEAD_DIM), dtype=torch.float16, device="cuda"
    ).mul_(0.35)
    value = torch.randn_like(key).mul_(0.3)
    key_cache, value_cache, k_scale, v_scale = _make_cache(key, value)
    logical_indices = torch.arange(topk, dtype=torch.int32, device="cuda").repeat(
        rows, 1
    )
    block_table = torch.zeros((1, 1), dtype=torch.int32, device="cuda")
    token_to_req = torch.zeros(rows, dtype=torch.int32, device="cuda")
    # Causal verify positions: consecutive draft tokens at the tail.
    query_positions = torch.arange(topk - rows, topk, dtype=torch.int64, device="cuda")
    sequence_lengths = torch.tensor([topk], dtype=torch.int32, device="cuda")
    kwargs = dict(
        query_positions=query_positions,
        sequence_lengths=sequence_lengths,
        kv_cache_dtype="fp8_e4m3",
        k_scale=k_scale,
        v_scale=v_scale,
    )

    # Reference: force the Triton generic path (no grouped/XQA page4 split).
    monkeypatch.setattr(qsa_ops, "_SM70_QSA_XQA_PAGE4", False)
    monkeypatch.setattr(qsa_ops, "_SM70_QSA_GROUPED_PAGE4", False)
    reference = qsa_sparse_paged_attention(
        query,
        key_cache,
        value_cache,
        logical_indices,
        block_table,
        token_to_req,
        **kwargs,
    )
    # Actual: enable the >16-row grouped-page4 + XQA-page4 split.
    monkeypatch.setattr(qsa_ops, "_SM70_QSA_XQA_PAGE4", True)
    monkeypatch.setattr(qsa_ops, "_SM70_QSA_XQA_PAGE4_MIN_ROWS", 16)
    monkeypatch.setattr(qsa_ops, "_SM70_QSA_GROUPED_PAGE4", True)
    actual = qsa_sparse_paged_attention(
        query,
        key_cache,
        value_cache,
        logical_indices,
        block_table,
        token_to_req,
        **kwargs,
    )
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.float(), reference.float(), atol=3e-2, rtol=3e-2)


@torch.inference_mode()
def test_e4m3_multi_request_verify_split(monkeypatch):
    """46-row catch-up chunk for req 0 plus one decode row each for reqs 1..3,
    matching the grouped/XQA mixed-batch shape (49 rows > 16)."""
    _sm70_only()
    torch.manual_seed(202)
    rows, requests, heads = 49, 4, 6
    page_size, topk = 2052, 2051
    query = torch.randn(
        (rows, heads, HEAD_DIM), dtype=torch.float16, device="cuda"
    ).mul_(0.2)
    key = torch.randn(
        (requests, page_size, 1, HEAD_DIM), dtype=torch.float16, device="cuda"
    ).mul_(0.35)
    value = torch.randn_like(key).mul_(0.3)
    key_cache, value_cache, k_scale, v_scale = _make_cache(key, value)
    logical_indices = torch.arange(topk, dtype=torch.int32, device="cuda").repeat(
        rows, 1
    )
    block_table = torch.arange(requests, dtype=torch.int32, device="cuda").view(
        requests, 1
    )
    token_to_req = torch.tensor([0] * 46 + [1, 2, 3], dtype=torch.int32, device="cuda")
    query_positions = torch.full((rows,), topk - 1, dtype=torch.int64, device="cuda")
    sequence_lengths = torch.full((requests,), topk, dtype=torch.int32, device="cuda")
    kwargs = dict(
        query_positions=query_positions,
        sequence_lengths=sequence_lengths,
        kv_cache_dtype="fp8_e4m3",
        k_scale=k_scale,
        v_scale=v_scale,
    )
    monkeypatch.setattr(qsa_ops, "_SM70_QSA_XQA_PAGE4", False)
    reference = qsa_sparse_paged_attention(
        query,
        key_cache,
        value_cache,
        logical_indices,
        block_table,
        token_to_req,
        **kwargs,
    )
    monkeypatch.setattr(qsa_ops, "_SM70_QSA_XQA_PAGE4", True)
    monkeypatch.setattr(qsa_ops, "_SM70_QSA_XQA_PAGE4_MIN_ROWS", 4096)
    monkeypatch.setattr(qsa_ops, "_SM70_QSA_GROUPED_PAGE4", True)
    actual = qsa_sparse_paged_attention(
        query,
        key_cache,
        value_cache,
        logical_indices,
        block_table,
        token_to_req,
        **kwargs,
    )
    torch.testing.assert_close(actual.float(), reference.float(), atol=3e-2, rtol=3e-2)


@torch.inference_mode()
def test_e4m3_verify_cuda_graph_replay(monkeypatch):
    """A captured E4M3 verify call replays to the same result (CUDA graph)."""
    _sm70_only()
    torch.manual_seed(303)
    rows, heads, page_size, topk = 18, 6, 2052, 2051
    query = torch.randn(
        (rows, heads, HEAD_DIM), dtype=torch.float16, device="cuda"
    ).mul_(0.2)
    key = torch.randn(
        (1, page_size, 1, HEAD_DIM), dtype=torch.float16, device="cuda"
    ).mul_(0.35)
    value = torch.randn_like(key).mul_(0.3)
    key_cache, value_cache, k_scale, v_scale = _make_cache(key, value)
    logical_indices = torch.arange(topk, dtype=torch.int32, device="cuda").repeat(
        rows, 1
    )
    block_table = torch.zeros((1, 1), dtype=torch.int32, device="cuda")
    token_to_req = torch.zeros(rows, dtype=torch.int32, device="cuda")
    query_positions = torch.arange(topk - rows, topk, dtype=torch.int64, device="cuda")
    sequence_lengths = torch.tensor([topk], dtype=torch.int32, device="cuda")
    kwargs = dict(
        query_positions=query_positions,
        sequence_lengths=sequence_lengths,
        kv_cache_dtype="fp8_e4m3",
        k_scale=k_scale,
        v_scale=v_scale,
    )
    monkeypatch.setattr(qsa_ops, "_SM70_QSA_XQA_PAGE4", True)
    monkeypatch.setattr(qsa_ops, "_SM70_QSA_XQA_PAGE4_MIN_ROWS", 16)
    monkeypatch.setattr(qsa_ops, "_SM70_QSA_GROUPED_PAGE4", True)

    eager = qsa_sparse_paged_attention(
        query,
        key_cache,
        value_cache,
        logical_indices,
        block_table,
        token_to_req,
        **kwargs,
    ).clone()

    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    captured = None
    with torch.cuda.graph(graph):
        captured = qsa_sparse_paged_attention(
            query,
            key_cache,
            value_cache,
            logical_indices,
            block_table,
            token_to_req,
            **kwargs,
        )
    graph.replay()
    torch.accelerator.synchronize()
    torch.testing.assert_close(captured.float(), eager.float(), atol=3e-2, rtol=3e-2)
