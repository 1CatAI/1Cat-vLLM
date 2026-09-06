# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native scheduler/scorer tests. These are not model-quality admission."""

import pytest
import torch

from benchmarks.kernels.benchmark_sm70_flashinfer_mqa import (
    check_schedule,
    make_inputs,
)
from benchmarks.kernels.flashinfer_sm70_mqa import FlashInferMQA, build

pytestmark = [
    pytest.mark.skip_global_cleanup,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


@pytest.fixture(scope="module", autouse=True)
def native():
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("Native Volta test")
    build()


def oracle(inputs, op):
    q, k, table, requests, positions, lengths = inputs
    expected_visible = []
    for row in range(q.shape[0]):
        req, pos = int(requests[row]), int(positions[row])
        n = 0
        if 0 <= req < len(lengths) and pos >= 0:
            n = max(
                0,
                min(
                    (pos + 1) // 4,
                    int(lengths[req]) // 4,
                    op.logits.shape[1],
                    table.shape[1] * k.shape[1],
                ),
            )
        expected_visible.append(n)
        if n:
            columns = torch.arange(n, device=q.device)
            pages = table[req, columns // k.shape[1]].long()
            live = (pages >= 0) & (pages < k.shape[0])
            keys = k[pages.clamp(0, k.shape[0] - 1), columns % k.shape[1], 0]
            scores = (keys.double() @ q[row].double().T).clamp_min(0).sum(-1)
            scores /= q.shape[2] ** 0.5
            scores[~live] = -torch.inf
            torch.testing.assert_close(
                op.logits[row, :n].double(), scores, rtol=2e-5, atol=2e-5
            )
        assert torch.isnan(op.logits[row, n:]).all(), "Unowned tail was overwritten"
    assert op.visible.cpu().tolist() == expected_visible
    check_schedule(op)


@pytest.mark.parametrize("rows", [1, 2, 4, 8, 16, 32, 33, 64])
@pytest.mark.parametrize("dim,heads", [(64, 1), (128, 4), (256, 16)])
def test_graph_dynamic_lengths_empty_rows_and_invalid_pages(rows, dim, heads):
    inputs = make_inputs(rows, 1024, dim=dim, heads=heads, table_width=3)
    q, k, table, requests, positions, lengths = inputs
    op = FlashInferMQA(q, 581, 80)
    op(*inputs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(*inputs)
    for iteration in range(4):
        lengths.copy_(
            torch.tensor(
                [0 if i % 3 == iteration else 1024 + i * 7 for i in range(rows)],
                dtype=torch.int32,
                device=q.device,
            )
        )
        positions.copy_(lengths - 1)
        if iteration == 1:
            requests[0] = -1
        elif iteration == 2:
            requests[0] = rows
            positions[-1] = -19
        else:
            requests.copy_(torch.arange(rows, device=q.device))
        table[:, 0] = -1 if iteration == 1 else k.shape[0] if iteration == 2 else 0
        op.logits.fill_(torch.nan)
        op.visible.fill_(-993)
        op.schedule.fill_(-991)
        q.mul_(-0.875)
        graph.replay()
        oracle(inputs, op)


def test_distinct_graph_instances_and_strided_inputs():
    inputs = make_inputs(8, 2048, table_width=3)
    q, k, table, requests, positions, lengths = inputs
    # Valid noncompact leading strides, including separate key/value planes.
    q_pad = torch.zeros(8, 4, 256, dtype=q.dtype, device=q.device)
    q_pad[..., :128].copy_(q)
    k_pad = torch.zeros(k.shape[0], 2, *k.shape[1:], device=k.device, dtype=k.dtype)
    k_pad[:, 0].copy_(k)
    inputs = q_pad[..., :128], k_pad[:, 0], table, requests, positions, lengths
    ops = [FlashInferMQA(inputs[0], 588, 80) for _ in range(2)]
    streams = [torch.cuda.Stream() for _ in ops]
    graphs = [torch.cuda.CUDAGraph() for _ in ops]
    torch.accelerator.synchronize()
    for stream, graph, op in zip(streams, graphs, ops):
        with torch.cuda.stream(stream):
            op(*inputs)
            stream.synchronize()
            with torch.cuda.graph(graph, stream=stream):
                op(*inputs)
    for op in ops:
        op.logits.fill_(torch.nan)
    torch.accelerator.synchronize()
    for _ in range(7):
        for stream, graph in zip(streams, graphs):
            with torch.cuda.stream(stream):
                graph.replay()
    torch.accelerator.synchronize()
    for op in ops:
        oracle(inputs, op)
    torch.testing.assert_close(
        ops[0].logits, ops[1].logits, equal_nan=True, atol=0, rtol=0
    )


def test_metadata_guard_rejects_wrong_dtype_before_launch():
    inputs = make_inputs(4, 1024, table_width=3)
    op = FlashInferMQA(inputs[0], 588, 80)
    args = list(inputs)
    args[3] = args[3].long()
    with pytest.raises(RuntimeError, match="int32"):
        op(*args)


def test_int32_and_int64_positions_agree_and_large_position_does_not_wrap():
    inputs = list(make_inputs(4, 1024, table_width=3))
    op = FlashInferMQA(inputs[0], 588, 80)
    op(*inputs)
    expected = op.logits.clone()
    inputs[4] = inputs[4].int()
    op(*inputs)
    torch.testing.assert_close(op.logits[:, :256], expected[:, :256], atol=0, rtol=0)
    inputs[4] = torch.full_like(
        inputs[4], torch.iinfo(torch.int64).max, dtype=torch.int64
    )
    op.logits.fill_(torch.nan)
    op(*inputs)
    oracle(inputs, op)


@pytest.mark.parametrize("rows,workers", [(1, 80), (4, 80), (16, 320), (64, 160)])
def test_every_live_tile_is_written_by_exactly_one_cta(rows, workers):
    inputs = make_inputs(rows, 2048, table_width=3)
    q, k, table, requests, positions, lengths = inputs
    op = FlashInferMQA(q, 588, workers)
    visits = torch.empty((rows, 10), device=q.device, dtype=torch.int32)
    for repeat in range(3):
        lengths.copy_(
            torch.tensor(
                [0 if row % 3 == repeat else 255 + 197 * row for row in range(rows)],
                device=q.device,
                dtype=torch.int32,
            )
        )
        positions.copy_(lengths - 1)
        visits.zero_()
        torch.ops._C_flashinfer_mqa_sm70.run(
            *inputs, op.logits, op.visible, op.schedule, 4, 128**0.5, workers, visits
        )
        expected = (
            torch.arange(10, device=q.device)[None] < ((op.visible + 63) // 64)[:, None]
        )
        assert torch.equal(visits, expected.int())
