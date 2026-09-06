# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Isolated upstream FlashInfer SM70 adapter tests, no vLLM binary required."""

import pytest
import torch

from benchmarks.kernels.benchmark_sm70_flashinfer_qsa import (
    capture,
    check_exclusive,
    make_case,
    oracle,
)
from benchmarks.kernels.flashinfer_sm70_qsa import FlashInferQSA, build


@pytest.mark.parametrize("width,splits", [(0, 1), (2, 0), (2, 65), (-1, 4)])
def test_invalid_plan(width, splits):
    with pytest.raises(ValueError):
        FlashInferQSA(torch.empty((1, 6, 256), dtype=torch.float16), width, splits)


def test_oracle_keeps_repeats_and_masks_empty_rows():
    q = torch.zeros(2, 1, 256)
    k = torch.zeros(1, 4, 1, 256)
    v = torch.zeros_like(k)
    v[0, 1] = 3
    indices = torch.tensor([[0, 1, 1, -1], [0, 1, 2, 3]], dtype=torch.int32)
    result = oracle(q, k, v, indices, torch.tensor([[0]]), torch.tensor([0, -1]))
    torch.testing.assert_close(result[0], torch.full_like(result[0], 2))
    assert result[1].count_nonzero() == 0


@pytest.fixture(scope="module")
def cuda_build():
    if not torch.cuda.is_available():
        pytest.skip("SM70 GPU required")
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("Prototype deliberately restricted to SM70")
    check_exclusive()
    torch.manual_seed(42)
    build()


@pytest.mark.parametrize(
    "rows,selected,page,kv_heads,group,splits",
    [
        (1, 1, 16, 1, 1, 1),
        (1, 5, 16, 1, 2, 8),
        (2, 31, 64, 2, 4, 4),
        (4, 33, 16, 1, 6, 32),
        (8, 257, 784, 1, 6, 16),
        (16, 2051, 784, 1, 6, 32),
        (3, 2051, 64, 2, 6, 64),
        (2, 37, 16, 2, 8, 4),
    ],
)
def test_sparse_eager_and_graph(
    cuda_build, rows, selected, page, kv_heads, group, splits
):
    case = list(make_case(rows, selected, page, kv_heads, group))
    q, k, v, indices, table, requests = case
    # Strided queries/cache and table, while maintaining vector alignment.
    q_storage = torch.empty(rows, q.shape[1] * 2, 256, device="cuda", dtype=q.dtype)
    case[0] = q_storage[:, ::2].copy_(q)
    case[1] = k.transpose(1, 2).contiguous().transpose(1, 2)
    case[2] = v.transpose(1, 2).contiguous().transpose(1, 2)
    q, k, v = case[:3]
    candidate = FlashInferQSA(q, selected, splits)
    call = lambda: candidate(*case)
    graph = capture(call, 1)
    for cycle in range(5):
        q.normal_().mul_((0.25, 1.0, 3.0, 1.0, 1.0)[cycle])
        requests.copy_(torch.randperm(rows, device="cuda"))
        table.copy_(torch.randperm(k.shape[0], device="cuda").reshape_as(table))
        indices.random_(0, 8192)
        if selected > 1:
            indices[:, 1] = indices[:, 0]  # preserve duplicate weighting
        if cycle == 1:
            indices[:, ::3] = -1
        elif cycle == 2:
            indices[:, ::3] = table.shape[1] * page + 5
            table[:, 0] = k.shape[0]  # invalid physical page
        elif cycle == 3:
            requests[0] = rows + 1
        elif cycle == 4:
            indices.fill_(-1)
            k.fill_(float("nan"))
            v.fill_(float("nan"))  # invalid slots must load zero, not page 0
        expected = oracle(*case)
        eager = call().clone()
        for workspace in (candidate.partial, candidate.lse, candidate.output):
            workspace.fill_(float("nan"))
        candidate.offsets.fill_(-7)
        candidate.metadata.fill_(-9)
        graph.replay()
        torch.accelerator.synchronize()
        torch.testing.assert_close(candidate.output, eager, atol=0, rtol=0)
        torch.testing.assert_close(
            candidate.output.float(), expected, atol=2e-3, rtol=1e-2
        )


def test_reject_unaligned_inputs(cuda_build):
    case = list(make_case(1, selected=5))
    q = case[0]
    storage = torch.empty(q.numel() + 1, dtype=q.dtype, device=q.device)
    case[0] = storage[1:].view_as(q).copy_(q)
    candidate = FlashInferQSA(case[0], 5, 1)
    with pytest.raises(RuntimeError, match="aligned"):
        candidate(*case)
