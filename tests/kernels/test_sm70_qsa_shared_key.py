# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared-key scoring preserves causal selection and graph replay state."""

import pytest
import torch

from vllm.models.qwen4_exp.nvidia.ops.qsa import (
    qsa_mqa_paged,
    qsa_select_paged_tokens,
)
from vllm.models.qwen4_exp.nvidia.ops.qsa_shared_key import (
    load_operator,
    shared_key_reason,
)


@pytest.mark.parametrize("rows", [2, 5, 8])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("position_dtype", [torch.int32, torch.int64])
def test_causal_scores_selection_and_changed_input_graph(rows, strided, position_dtype):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 required")
    assert load_operator(), "Build and install the packaged QSA indexer extension"
    torch.manual_seed(921 + rows)
    columns = 2496
    width = 256 if strided else 128
    query = torch.randn(rows, 4, width, device="cuda", dtype=torch.float16)
    cache = torch.randn(columns // 16, 16, 1, width, device="cuda", dtype=torch.float16)
    if strided:
        query = query[..., ::2]
        cache = cache[..., ::2]
    table = torch.randperm(columns // 16, device="cuda", dtype=torch.int32)[None, :]
    requests = torch.zeros(rows, device="cuda", dtype=torch.int32)
    positions = 8192 + torch.arange(rows, device="cuda", dtype=position_dtype)
    lengths = torch.tensor([8192 + rows], device="cuda", dtype=torch.int32)
    args = (query, cache, table, requests, positions, lengths, 4)
    assert shared_key_reason(*args[:6]) is None
    qsa_mqa_paged(*args, shared_key_scoring=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual, visible = qsa_mqa_paged(*args, shared_key_scoring=True)
    for case in range(4):
        query.normal_()
        if case == 1:
            query.zero_()  # Ties must choose lower logical block IDs.
        if case == 2:
            positions[0] = 3  # Only one complete causal group.
            requests[-1] = -1  # Padded graph row.
        if case == 3:
            table[0, 9] = -1
        actual.fill_(float("nan"))
        graph.replay()
        reference, expected_visible = qsa_mqa_paged(*args)
        assert torch.equal(visible, expected_visible)
        for row in range(rows):
            n = min(int(visible[row]), columns)
            got, expected = actual[row, :n], reference[row, :n]
            torch.testing.assert_close(got, expected, rtol=2e-6, atol=2e-6)
            # Native selection ranks scores and breaks ties by logical index;
            # compare the final set as well as the numerical score error.
            got_ids = torch.argsort(got, stable=True, descending=True)[:512]
            ref_ids = torch.argsort(expected, stable=True, descending=True)[:512]
            assert torch.equal(got_ids.sort().values, ref_ids.sort().values)


def test_c4_falls_back_without_changing_scores():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 required")
    q = torch.randn(20, 4, 128, device="cuda", dtype=torch.float16)
    cache = torch.randn(128, 16, 1, 128, device="cuda", dtype=torch.float16)
    table = torch.arange(128, device="cuda", dtype=torch.int32).view(4, 32)
    req = torch.arange(20, device="cuda", dtype=torch.int32) // 5
    pos = torch.full((20,), 1024, device="cuda", dtype=torch.int64)
    lengths = torch.full((4,), 1025, device="cuda", dtype=torch.int32)
    args = (q, cache, table, req, pos, lengths, 4)
    assert shared_key_reason(*args[:6]) == "requires_M2_8_H4_D128"
    expected, ev = qsa_mqa_paged(*args)
    actual, av = qsa_mqa_paged(*args, shared_key_scoring=True)
    assert torch.equal(av, ev)
    mask = torch.arange(actual.shape[1], device="cuda")[None, :] < ev[:, None]
    assert torch.equal(actual[mask], expected[mask])


def test_truncated_score_width_retains_causal_lengths():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 required")
    q = torch.randn(5, 4, 128, device="cuda", dtype=torch.float16)
    cache = torch.randn(64, 16, 1, 128, device="cuda", dtype=torch.float16)
    table = torch.arange(64, device="cuda", dtype=torch.int32)[None, :]
    req = torch.tensor([0, 0, 0, -1, 0], device="cuda", dtype=torch.int32)
    pos = torch.tensor([1024, 1025, -1, 1027, -5], device="cuda", dtype=torch.int64)
    lengths = torch.tensor([1029], device="cuda", dtype=torch.int32)
    args = (q, cache, table, req, pos, lengths, 4)
    expected, ev = qsa_mqa_paged(*args, num_columns=64)
    actual, av = qsa_mqa_paged(*args, num_columns=64, shared_key_scoring=True)
    assert torch.equal(av, ev)
    mask = torch.arange(64, device="cuda")[None, :] < ev[:, None]
    torch.testing.assert_close(actual[mask], expected[mask], rtol=2e-6, atol=2e-6)


def test_short_context_selector_graph_is_exact():
    """When all keys fit, scorer rounding cannot change the selected history."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 required")
    assert load_operator()
    torch.manual_seed(2903)
    query = torch.randn(5, 4, 128, device="cuda", dtype=torch.float16)
    # Match the model's 816-token scheduler page, compressed by four.
    cache = torch.randn(12, 204, 1, 128, device="cuda", dtype=torch.float16)
    table = torch.randperm(12, device="cuda", dtype=torch.int32)[None, :]
    requests = torch.zeros(5, device="cuda", dtype=torch.int32)
    positions = torch.arange(5, device="cuda", dtype=torch.int64) + 63
    lengths = torch.tensor([68], device="cuda", dtype=torch.int32)
    args = (query, cache, table, requests, positions, lengths, 2048, 4)
    graphs, outputs = [], []
    for native in (False, True):
        qsa_select_paged_tokens(*args, shared_key_scoring=native)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs.append(qsa_select_paged_tokens(*args, shared_key_scoring=native))
        graphs.append(graph)
    for length in (68, 79, 512, 816, 1025, 2048, 0):
        query.normal_()
        cache.normal_()
        lengths.fill_(length)
        positions.copy_(torch.arange(5, device="cuda") + length - 5)
        requests.zero_()
        if length == 512:
            requests[-1] = -1
            positions[-1] = -1
        for graph in graphs:
            graph.replay()
        assert torch.equal(*outputs), f"Different selected history at length {length}"
