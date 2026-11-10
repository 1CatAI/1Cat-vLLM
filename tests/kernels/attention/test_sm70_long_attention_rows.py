# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runtime-width K prefetch must respect paged tails, strides and live graph lengths."""

import pytest
import torch


@pytest.mark.parametrize(
    "context,page,padding,query_offset",
    [
        (129, 2048, 0, 0),
        (2049, 2048, 8, 1),
        (65536, 1648, 0, 0),
        (262144, 2048, 0, 1),
        (262144, 3296, 8, 0),
    ],
)
@pytest.mark.parametrize("query_rows", range(1, 9))
def test_runtime_rows_paged_prefetch_graph_and_oracle(
    context, page, padding, query_offset, query_rows
):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires SM70")
    from vllm.v1.attention.ops.sm70_grouped_long import resolve_long_attention

    operator, manifest = resolve_long_attention()
    if manifest is None or manifest.get("query_rows_revision", 1) < 2:
        pytest.skip("requires runtime query-row native revision 2")
    assert operator is not None
    torch.manual_seed(20261110)
    pages = (context + page - 1) // page
    storage = torch.empty(
        (pages, 2, page, 1, 256 + padding), device="cuda", dtype=torch.uint8
    )
    data = torch.randn(pages, 2, page, 1, 256, device="cuda", dtype=torch.float16)
    data[:, 1].add_(3)
    storage[..., :256].copy_(data.to(torch.float8_e4m3fn).view(torch.uint8))
    keys, values = storage[..., :256].unbind(1)
    table = torch.randperm(pages, device="cuda").int()[None].contiguous()
    query_storage = torch.randn(
        query_rows * 6 * 256 + query_offset, device="cuda", dtype=torch.float16
    )
    query = query_storage[query_offset:].view(query_rows, 6, 256)
    query.mul_(0.5)
    initial = torch.arange(context - query_rows + 1, context + 1, device="cuda").int()
    lengths = initial.clone()
    output = torch.full(
        (query_rows + 2, 6, 256), -777, device="cuda", dtype=torch.float16
    )
    partial = torch.empty((80, 8, 6, 256), device="cuda", dtype=torch.float32)
    statistics = torch.empty((80, 8, 6, 2), device="cuda", dtype=torch.float32)

    def run():
        operator(
            query,
            keys,
            values,
            output[1:-1],
            table,
            lengths,
            partial,
            statistics,
            0.0625,
            0.5,
            1.25,
        )

    run()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for state in ("live", "mixed", "zero", "restore"):
        lengths.copy_(initial)
        if state == "mixed":
            lengths[::2] = 0
        elif state == "zero":
            lengths.zero_()
        query.mul_(0.875)
        output.fill_(-777)
        graph.replay()
        assert bool(torch.isfinite(output).all())
        assert bool((output[[0, -1]] == -777).all())
        for row in range(query_rows):
            length = int(lengths[row])
            actual = output[row + 1].double()
            if length == 0:
                assert bool((actual == 0).all())
                continue
            key = keys[table[0].long()].reshape(-1, 256)[:length]
            value = values[table[0].long()].reshape(-1, 256)[:length]
            key = key.view(torch.float8_e4m3fn).double() * 0.5
            value = value.view(torch.float8_e4m3fn).double() * 1.25
            reference = ((query[row].double() @ key.T) * 0.0625).softmax(-1) @ value
            rounding_floor = (reference.half().double() - reference).norm()
            assert (actual - reference).norm() <= rounding_floor * 1.04 + 2e-6


@pytest.mark.parametrize("batch", [2, 3, 8, 32])
@pytest.mark.parametrize("width", range(1, 9))
def test_request_rows_graph_matches_independent_requests(batch, width):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires SM70")
    from vllm.v1.attention.ops.sm70_grouped_long import resolve_long_attention

    operator, manifest = resolve_long_attention()
    if manifest is None or manifest.get("query_rows_revision", 1) < 2:
        pytest.skip("requires runtime query-row native revision 2")
    torch.manual_seed(20261110)
    page = 1648
    context = 4097
    pages = (context + page - 1) // page
    cache = (
        torch.randn(batch * pages, 2, page, 1, 256, device="cuda", dtype=torch.float16)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    k, v = cache.unbind(1)
    table = torch.randperm(batch * pages, device="cuda").int().view(batch, pages)
    q = torch.randn(batch * width, 6, 256, device="cuda", dtype=torch.float16)
    initial = (
        torch.arange(context - width + 1, context + 1, device="cuda")
        .int()
        .repeat(batch)
    )
    lengths = initial.clone()
    partial = torch.empty(batch, 80, 8, 6, 256, device="cuda")
    statistics = torch.empty(batch, 80, 8, 6, 2, device="cuda")
    actual = torch.empty_like(q)
    expected = torch.empty_like(q)
    graph = torch.cuda.CUDAGraph()
    operator(q, k, v, actual, table, lengths, partial, statistics, 0.0625, 0.5, 1.25)
    with torch.cuda.graph(graph):
        operator(
            q, k, v, actual, table, lengths, partial, statistics, 0.0625, 0.5, 1.25
        )
    for state in ("live", "padded_rows", "padded_request", "empty", "restore"):
        lengths.copy_(initial)
        if state == "padded_rows":
            lengths[::width] = 0
        elif state == "padded_request":
            lengths[-width:] = 0
        elif state == "empty":
            lengths.zero_()
        graph.replay()
        for request in range(batch):
            rows = slice(request * width, (request + 1) * width)
            operator(
                q[rows],
                k,
                v,
                expected[rows],
                table[request : request + 1],
                lengths[rows],
                partial[request],
                statistics[request],
                0.0625,
                0.5,
                1.25,
            )
        assert torch.equal(actual, expected)
