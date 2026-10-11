# SPDX-License-Identifier: Apache-2.0
"""Independent FP64 oracle and exact selection checks; no engine imports."""

import math

import pytest
import torch

from flashinfer.sm70 import QsaScorePlan

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
        reason="requires physical SM70; offline compilation is not correctness evidence",
    ),
]


def inputs(m, nreq, page_size, strided=False, position_dtype=torch.int64):
    torch.manual_seed(902)
    width = 256 if strided else 128
    q = torch.randn(m, 4, width, device="cuda", dtype=torch.float16)
    cache = torch.randn(nreq * 12, page_size, 1, width, device="cuda", dtype=q.dtype)
    if strided:
        q, cache = q[..., ::2], cache[..., ::2]
    table = torch.randperm(nreq * 12, device="cuda", dtype=torch.int32).view(nreq, 12)
    if strided:
        table = table.T.contiguous().T
    # Interleaving is deliberate: group membership cannot be inferred from row / 5.
    requests = torch.arange(m, device="cuda", dtype=torch.int32) % nreq
    length = 12 * page_size * 4 - 11
    positions = torch.arange(length - m, length, device="cuda", dtype=position_dtype)
    lengths = torch.full((nreq,), length, device="cuda", dtype=torch.int32)
    return q, cache, table, requests, positions, lengths


def reference(q, cache, table, requests, positions, lengths, columns):
    scores = torch.full((q.shape[0], columns), -torch.inf, device=q.device)
    visible = torch.empty(q.shape[0], dtype=torch.int32, device=q.device)
    for row in range(q.shape[0]):
        req = int(requests[row])
        length = int(lengths[req]) if 0 <= req < table.shape[0] else 0
        # CUDA/Triton signed integer division truncates toward zero.
        n = math.trunc(min(int(positions[row]) + 1, length) / 4)
        visible[row] = n
        for page_start in range(0, max(0, min(n, columns)), cache.shape[1]):
            page = int(table[req, page_start // cache.shape[1]])
            end = min(n, columns, page_start + cache.shape[1])
            if not 0 <= page < cache.shape[0]:
                continue
            keys = cache[page, : end - page_start, 0].double()
            dot = keys @ q[row].double().T
            # Inherited native scorer contract uses fmaxf(dot, 0).
            positive = torch.fmax(dot, torch.zeros_like(dot))
            scores[row, page_start:end] = (positive.sum(-1) / math.sqrt(128)).float()
    return scores, visible


def check(plan, args, columns=None):
    columns = plan.max_columns if columns is None else columns
    scores, visible = plan.run(*args, columns=columns)
    expected, expected_visible = reference(*args, columns)
    torch.testing.assert_close(visible, expected_visible, rtol=0, atol=0)
    for row in range(scores.shape[0]):
        n = max(0, min(int(visible[row]), columns))
        actual, ref = scores[row, :n], expected[row, :n]
        # Inherit the existing SM70 shared-key scorer tolerance, not the looser BF16 one.
        torch.testing.assert_close(actual, ref, rtol=2e-6, atol=2e-6)
        ids = torch.argsort(actual, descending=True, stable=True)[:512].sort().values
        ref_ids = torch.argsort(ref, descending=True, stable=True)[:512].sort().values
        assert torch.equal(ids, ref_ids)
    return scores, visible


@pytest.mark.parametrize("m,nreq", [(1, 1), (5, 1), (20, 4), (32, 1), (32, 4)])
@pytest.mark.parametrize("strided", [False, True])
def test_grouped_score_reference_and_ties(m, nreq, strided):
    args = inputs(m, nreq, 204, strided)
    plan = QsaScorePlan(32, nreq, 2448, "cuda")
    check(plan, args)
    args[0].zero_()
    args[2][0, 1] = -1
    check(plan, args, columns=777)


@pytest.mark.parametrize("position_dtype", [torch.int32, torch.int64])
def test_replay_regroups_requests_padding_and_independent_captures(position_dtype):
    args = inputs(20, 4, 4, position_dtype=position_dtype)
    cache_before = args[1].clone()
    plans = [QsaScorePlan(32, 4, 48, "cuda") for _ in range(2)]
    assert plans[0].scores.data_ptr() != plans[1].scores.data_ptr()
    graphs = []
    for plan in plans:
        plan.run(*args)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            plan.run(*args)
        graphs.append(graph)
    for iteration in range(4):
        args[0].normal_()
        args[3].copy_(torch.arange(20, device="cuda", dtype=torch.int32) // 5)
        args[3][iteration] = -1
        args[4][iteration] = -5
        args[5][iteration] = 0
        for graph in graphs:
            graph.replay()
        expected, visible = reference(*args, 48)
        mask = torch.arange(48, device="cuda")[None, :] < visible[:, None]
        for plan in plans:
            torch.testing.assert_close(plan.visible[:20], visible, rtol=0, atol=0)
            torch.testing.assert_close(
                plan.scores[:20][mask], expected[mask], rtol=2e-6, atol=2e-6
            )
    torch.testing.assert_close(args[1], cache_before, rtol=0, atol=0)
    changed = list(args)
    changed[1] = torch.randn_like(args[1])
    check(plans[0], changed)  # eager rebinding cannot retain old cache contents


def test_empty_rows_zero_columns_negative_positions_and_capture_guard():
    args = list(inputs(5, 1, 204))
    plan = QsaScorePlan(8, 1, 2448, "cuda")
    args[4].copy_(torch.tensor([-2, -3, -4, -5, -6], device="cuda"))
    check(plan, args, columns=0)
    for i in (0, 3, 4):
        args[i] = args[i][:0]
    scores, visible = plan.run(*args)
    assert scores.shape == (0, 2448) and visible.shape == (0,)
    graph = torch.cuda.CUDAGraph()
    with (
        pytest.raises(RuntimeError, match="before CUDA graph capture"),
        torch.cuda.graph(graph),
    ):
        QsaScorePlan(8, 1, 2448, "cuda")


@pytest.mark.parametrize("nreq", [1, 4])
def test_nonfinite_extreme_values_and_invalid_physical_pages(nreq):
    args = inputs(5 if nreq == 1 else 20, nreq, 4)
    args[0].fill_(65504)
    args[1].fill_(65504)
    args[1][0, 0].fill_(float("nan"))
    args[1][0, 1].fill_(float("inf"))
    args[1][0, 2].fill_(-float("inf"))
    args[2][0, 0] = 0
    args[2][0, 1] = args[1].shape[0]
    args[3][0] = -1
    plan = QsaScorePlan(32, nreq, 48, "cuda")
    check(plan, args)
