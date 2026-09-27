# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared QSA key loads retain score bits and deterministic selected tokens."""

import pytest
import torch

from vllm.models.qwen4_exp.nvidia.ops import qsa


@pytest.mark.parametrize(
    "rows,expected", [(1, False), (2, False), (5, True), (10, True), (16, False)]
)
def test_mtp_indexer_shape_gate(monkeypatch, rows, expected):
    monkeypatch.setattr(qsa, "_SM70_QSA_MTP_BATCH", True)
    monkeypatch.setattr(
        qsa.current_platform, "is_device_capability", lambda cap: cap == 70
    )
    query = torch.empty(rows, 4, 128, dtype=torch.float16)
    cache = torch.empty(3, 16, 1, 128, dtype=torch.float16)
    table = torch.empty(1, 2, dtype=torch.int32)
    assert qsa._use_sm70_qsa_mtp_batch(query, cache, table, 4) is expected
    assert not qsa._use_sm70_qsa_mtp_batch(query, cache, table.expand(2, -1), 4)
    assert not qsa._use_sm70_qsa_mtp_batch(query, cache, table, 2)
    assert not qsa._use_sm70_qsa_mtp_batch(query.float(), cache, table, 4)
    monkeypatch.setattr(qsa, "_SM70_QSA_MTP_BATCH", False)
    assert not qsa._use_sm70_qsa_mtp_batch(query, cache, table, 4)


@pytest.mark.parametrize("rows,page_size", [(5, 4), (10, 16), (5, 784), (10, 784)])
def test_mtp_indexer_changed_graph_scores_and_topk_are_exact(
    monkeypatch, rows, page_size
):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 required")
    torch.manual_seed(20260927)
    context = 8192
    logical_pages = (context // 4 + 64 + page_size - 1) // page_size
    pages = logical_pages + 5
    # Noncontiguous query/key strides must remain valid, as with paged views.
    query = torch.randn(rows, 4, 256, device="cuda", dtype=torch.float16)[..., ::2]
    cache = torch.randn(pages, page_size, 1, 256, device="cuda", dtype=torch.float16)[
        ..., ::2
    ]
    table = (
        torch.randperm(pages, device="cuda")[:logical_pages].to(torch.int32).view(1, -1)
    )
    table[0, 1] = -1
    table[0, 2] = pages + 2
    request = torch.zeros(rows, device="cuda", dtype=torch.int32)
    positions = (
        torch.arange(context, context + rows, device="cuda", dtype=torch.int32)
        .flip(0)
        .contiguous()
    )
    length = torch.tensor([context + rows], device="cuda", dtype=torch.int32)
    graphs, outputs = [], []

    def call():
        scores, visible = qsa.qsa_mqa_paged(
            query, cache, table, request, positions, length, 4
        )
        selected = qsa.qsa_select_paged_tokens(
            query, cache, table, request, positions, length, 2048, 4
        )
        return scores, visible, selected

    for enabled in (False, True):
        monkeypatch.setattr(qsa, "_SM70_QSA_MTP_BATCH", enabled)
        for _ in range(3):
            call()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = call()
        graphs.append(graph)
        outputs.append(output)

    for step, scale in enumerate((0.0, 0.001, 0.1, 1.0, 3.0)):
        query.normal_(0, scale)
        cache.normal_(0, scale)
        request[-1] = -1 if step % 2 else 0
        positions.add_(1)
        length.add_(1)
        for graph in graphs:
            graph.replay()
        (expected, lengths, tokens), (actual, actual_lengths, actual_tokens) = outputs
        assert torch.equal(lengths, actual_lengths)
        assert torch.equal(tokens, actual_tokens)
        for row in range(rows):
            end = int(lengths[row])
            assert torch.equal(
                expected[row, :end].view(torch.int32),
                actual[row, :end].view(torch.int32),
            )
