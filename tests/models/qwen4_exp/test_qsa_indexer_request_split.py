# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from tests.models.qwen4_exp.qsa_policy_utils import set_qsa_option
from vllm.models.qwen4_exp.nvidia.ops import qsa as qsa_ops
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON

pytestmark = pytest.mark.skip_global_cleanup


@pytest.mark.parametrize(
    ("query_start_loc", "expected"),
    [
        # Decode rows first, then one long prefill.
        ([0, 5, 10, 1034], [(0, 10, None), (10, 1034, 2)]),
        # Short requests on both sides of a long one stay in separate runs.
        ([0, 5, 1029, 1034], [(0, 5, None), (5, 1029, 1), (1029, 1034, None)]),
        # Two long prefills are selected one at a time.
        ([0, 600, 1200], [(0, 600, 0), (600, 1200, 1)]),
        # Empty requests are skipped.
        ([0, 0, 5, 5, 700], [(0, 5, None), (5, 700, 3)]),
        # Only short requests: one batched run.
        ([0, 5, 10, 15], [(0, 15, None)]),
    ],
)
def test_request_segments(query_start_loc, expected):
    assert qsa_ops._qsa_indexer_request_segments(query_start_loc, 512) == expected


def _enable_cublas(monkeypatch):
    set_qsa_option(monkeypatch, "qsa_indexer_cublas", True)
    set_qsa_option(monkeypatch, "qsa_cublas_min_rows", 512)
    monkeypatch.setattr(
        qsa_ops.current_platform,
        "is_device_capability",
        lambda capability: capability == 70,
    )


def _inputs(query_start_loc):
    rows = query_start_loc[-1]
    q = torch.zeros(rows, 4, 128, dtype=torch.float16)
    cache = torch.zeros(8, 64, 1, 128, dtype=torch.float16)
    num_requests = len(query_start_loc) - 1
    page_table = torch.arange(num_requests * 2, dtype=torch.int32).view(-1, 2)
    token_to_req = torch.cat(
        [
            torch.full((end - start,), request, dtype=torch.int32)
            for request, (start, end) in enumerate(
                zip(query_start_loc, query_start_loc[1:])
            )
        ]
    )
    positions = torch.arange(rows, dtype=torch.int32)
    seq_lens = torch.arange(1, num_requests + 1, dtype=torch.int32) * 1000
    return q, cache, page_table, token_to_req, positions, seq_lens


def test_long_request_is_selected_alone(monkeypatch):
    _enable_cublas(monkeypatch)
    query_start_loc = [0, 5, 1029, 1034]
    q, cache, page_table, token_to_req, positions, seq_lens = _inputs(query_start_loc)
    out = torch.empty(q.shape[0], 2051, dtype=torch.int32)
    calls = []

    def record(q_, cache_, table, to_req, pos, lens, topk, ratio, out_, **kw):
        calls.append((q_.shape[0], table.clone(), to_req.clone(), lens.clone(), kw))
        return out_

    monkeypatch.setattr(qsa_ops, "qsa_select_paged_tokens", record)

    assert qsa_ops._qsa_select_by_request(
        q,
        cache,
        page_table,
        token_to_req,
        positions,
        seq_lens,
        2048,
        4,
        out,
        torch.tensor(query_start_loc, dtype=torch.int32),
    )
    assert [c[0] for c in calls] == [5, 1024, 5]
    # Short runs keep the batch's page table and request indices.
    assert torch.equal(calls[0][1], page_table)
    assert torch.equal(calls[2][2], token_to_req[1029:])
    # The long request sees only its own table row, as when it runs alone.
    assert torch.equal(calls[1][1], page_table[1:2])
    assert torch.count_nonzero(calls[1][2]) == 0
    assert torch.equal(calls[1][3], seq_lens[1:2])
    assert all(not c[4] for c in calls)


@pytest.mark.parametrize(
    "query_start_loc",
    [
        [0, 5, 10, 15],  # no long request
        [0, 1024],  # single request: the existing path applies
    ],
)
def test_batches_without_a_long_request_in_a_multi_request_batch_are_unchanged(
    monkeypatch, query_start_loc
):
    _enable_cublas(monkeypatch)
    q, cache, page_table, token_to_req, positions, seq_lens = _inputs(query_start_loc)
    out = torch.empty(q.shape[0], 2051, dtype=torch.int32)
    assert not qsa_ops._qsa_select_by_request(
        q,
        cache,
        page_table,
        token_to_req,
        positions,
        seq_lens,
        2048,
        4,
        out,
        torch.tensor(query_start_loc, dtype=torch.int32),
    )


def test_padded_rows_keep_the_batched_path(monkeypatch):
    _enable_cublas(monkeypatch)
    query_start_loc = [0, 5, 1029]
    q, cache, page_table, token_to_req, positions, seq_lens = _inputs(query_start_loc)
    padded = torch.cat([q, q[:3]])
    out = torch.empty(padded.shape[0], 2051, dtype=torch.int32)
    assert not qsa_ops._qsa_select_by_request(
        padded,
        cache,
        page_table,
        torch.cat([token_to_req, token_to_req[:3]]),
        torch.cat([positions, positions[:3]]),
        seq_lens,
        2048,
        4,
        out,
        torch.tensor(query_start_loc, dtype=torch.int32),
    )


def test_other_platforms_keep_the_batched_path(monkeypatch):
    _enable_cublas(monkeypatch)
    monkeypatch.setattr(
        qsa_ops.current_platform, "is_device_capability", lambda capability: False
    )
    query_start_loc = [0, 5, 1029]
    q, cache, page_table, token_to_req, positions, seq_lens = _inputs(query_start_loc)
    out = torch.empty(q.shape[0], 2051, dtype=torch.int32)
    assert not qsa_ops._qsa_select_by_request(
        q,
        cache,
        page_table,
        token_to_req,
        positions,
        seq_lens,
        2048,
        4,
        out,
        torch.tensor(query_start_loc, dtype=torch.int32),
    )


requires_sm70 = pytest.mark.skipif(
    not current_platform.is_cuda()
    or not HAS_TRITON
    or not current_platform.is_device_capability(70),
    reason="the SM70 cuBLAS indexer path requires a V100",
)


@requires_sm70
def test_mixed_batch_selection_matches_each_request_alone(monkeypatch, workspace_init):
    """A long prefill selects the same blocks with or without decode rows."""

    cublas_rows = []
    original_cublas = qsa_ops._qsa_mqa_cublas

    def count_cublas(query, *args, **kwargs):
        cublas_rows.append(query.shape[0])
        return original_cublas(query, *args, **kwargs)

    monkeypatch.setattr(qsa_ops, "_qsa_mqa_cublas", count_cublas)
    torch.manual_seed(7)
    page_size, pages_per_request = 64, 40
    seq_lens_list = [6000, 8192, 9000]
    rows_per_request = [5, 1024, 5]
    query_start_loc = [0, 5, 1029, 1034]
    num_requests = len(seq_lens_list)
    cache = torch.randn(
        num_requests * pages_per_request,
        page_size,
        1,
        128,
        device="cuda",
        dtype=torch.float16,
    )
    page_table = (
        torch.randperm(num_requests * pages_per_request, device="cuda")
        .to(torch.int32)
        .view(num_requests, pages_per_request)
    )
    q = torch.randn(query_start_loc[-1], 4, 128, device="cuda", dtype=torch.float16)
    token_to_req = torch.cat(
        [
            torch.full((n,), r, device="cuda", dtype=torch.int32)
            for r, n in enumerate(rows_per_request)
        ]
    )
    positions = torch.cat(
        [
            torch.arange(s - n, s, device="cuda", dtype=torch.int32)
            for s, n in zip(seq_lens_list, rows_per_request)
        ]
    )
    seq_lens = torch.tensor(seq_lens_list, device="cuda", dtype=torch.int32)

    def select(qs, table, to_req, pos, lens, qsl=None):
        return qsa_ops.qsa_select_paged_tokens(
            qs, cache, table, to_req, pos, lens, 2048, 4, query_start_loc_cpu=qsl
        )

    split = select(
        q,
        page_table,
        token_to_req,
        positions,
        seq_lens,
        torch.tensor(query_start_loc, dtype=torch.int32),
    )
    split_cublas_rows = sum(cublas_rows)
    batched = select(q, page_table, token_to_req, positions, seq_lens)
    assert sum(cublas_rows) == split_cublas_rows, "the batched path must not use cuBLAS"
    alone = select(
        q[5:1029],
        page_table[1:2],
        torch.zeros(1024, device="cuda", dtype=torch.int32),
        positions[5:1029],
        seq_lens[1:2],
    )

    # The long request is scored by cuBLAS in the mixed batch and alone.
    assert split_cublas_rows == 1024
    assert sum(cublas_rows) == 2048
    assert torch.equal(split[5:1029], alone)
    assert torch.equal(split[:5], batched[:5])
    assert torch.equal(split[1029:], batched[1029:])
