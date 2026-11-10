# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batched long attention must preserve independent-request arithmetic."""

from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize(
    "batch,width,page,context,padding",
    [
        (1, 1, 1648, 3297, 8),
        (1, 5, 2048, 262144, 0),
        (3, 3, 3296, 65537, 0),
        (8, 5, 2048, 32768, 0),
        (2, 7, 1536, 8193, 0),
        (1, 8, 1648, 10003, 8),
    ],
)
def test_parallel_kv_heads_match_serial_groups_and_replay(
    batch, width, page, context, padding
):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires SM70")
    from vllm.v1.attention.ops import sm70_grouped_long as provider

    operation = provider.builtin_long_attention()
    revision = getattr(
        torch.ops._vllm_fa2_C, "sm70_grouped_long_kv_heads_revision", None
    )
    if operation is None or revision is None or revision() < 1:
        pytest.skip("requires the multi-KV grouped operator")
    torch.manual_seed(295)
    pages = (context + page - 1) // page
    storage = (
        torch.randn(
            batch * pages, 2, page, 2, 256 + padding, device="cuda", dtype=torch.float16
        )
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    k, v = storage[..., :256].unbind(1)
    # The old per-head entry requires a 16-byte base pointer; copy only its
    # reference operands so the new grid still exercises 8-byte head offsets.
    serial_k, serial_v = k.contiguous(), v.contiguous()
    q = torch.randn(batch * width, 12, 256, device="cuda", dtype=torch.float16)
    table = torch.randperm(batch * pages, device="cuda").int().view(batch, pages)
    initial = (
        torch.tensor(
            [context // (r + 1) for r in range(batch)], device="cuda", dtype=torch.int32
        )[:, None]
        + torch.arange(1 - width, 1, device="cuda", dtype=torch.int32)
    ).flatten()
    lengths = initial.clone()
    actual, expected = torch.empty_like(q), torch.empty_like(q)
    partial = torch.empty(batch * 2, 80, 8, 6, 256, device="cuda", dtype=torch.float32)
    lse = torch.empty(batch * 2, 80, 8, 6, 2, device="cuda", dtype=torch.float32)
    serial_partial = torch.empty(
        batch, 80, 8, 6, 256, device="cuda", dtype=torch.float32
    )
    serial_lse = torch.empty(batch, 80, 8, 6, 2, device="cuda", dtype=torch.float32)
    if batch == 1:
        serial_partial, serial_lse = serial_partial[0], serial_lse[0]

    def run():
        operation(q, k, v, actual, table, lengths, partial, lse, 0.0625, 0.5, 1.25)

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for phase in ("live", "padding", "empty", "restore"):
        lengths.copy_(initial)
        if phase == "padding":
            lengths[::width] = 0
            if batch > 1:
                lengths[-width:] = 0
        elif phase == "empty":
            lengths.zero_()
        q.normal_(0, 0.7)
        provider.run_six_head_groups(
            operation,
            q,
            serial_k,
            serial_v,
            expected,
            table,
            lengths,
            serial_partial,
            serial_lse,
            0.0625,
            0.5,
            1.25,
        )
        actual.fill_(float("nan"))
        graph.replay()
        assert torch.isfinite(actual).all()
        assert torch.equal(actual, expected)
    for head in range(2):
        length = int(lengths[width - 1])
        key = (
            k[table[0].long(), :, head]
            .reshape(-1, 256)[:length]
            .view(torch.float8_e4m3fn)
            .double()
            * 0.5
        )
        value = (
            v[table[0].long(), :, head]
            .reshape(-1, 256)[:length]
            .view(torch.float8_e4m3fn)
            .double()
            * 1.25
        )
        query = q[width - 1, head * 6 : (head + 1) * 6].double()
        reference = ((query @ key.T) * 0.0625).softmax(-1) @ value
        result = actual[width - 1, head * 6 : (head + 1) * 6].double()
        torch.testing.assert_close(result, reference, rtol=0.03, atol=3e-4)
        assert (result - reference).norm() / reference.norm() < 0.007


@pytest.mark.parametrize(
    "batch,context,page,heads,padding",
    [
        (2, 3297, 1648, 1, 8),
        (4, 8192, 1024, 2, 0),
        (6, 32768, 3296, 1, 0),
        (8, 32768, 3296, 1, 0),
        (16, 32768, 1536, 1, 0),
        (4, 131072, 3296, 1, 0),
        (8, 262144, 3296, 1, 0),
    ],
)
def test_long_batch_live_metadata_and_oracle(
    monkeypatch, batch, context, page, heads, padding
):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires SM70")
    from vllm.v1.attention.ops import sm70_e4m3_long as long
    from vllm.vllm_flash_attn import flash_attn_interface  # noqa: F401

    if long.long_attention_max_batch_size() < batch:
        pytest.skip("rebuild shipped long-attention batch operator")
    monkeypatch.setattr(long, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(
        long,
        "get_forward_context",
        lambda: SimpleNamespace(
            batch_descriptor=SimpleNamespace(attention_context_bucket=262144)
        ),
    )

    def declined(*args, **kwargs):
        pytest.fail("the built-in long route declined an admitted batch")

    op = long.wrap_long_attention(declined)
    torch.manual_seed(20260927)
    pages = (context + page - 1) // page
    storage = torch.empty(
        (batch * pages, 2, page, heads, 256 + padding),
        device="cuda",
        dtype=torch.uint8,
    )
    # Keep the service's interleaved KV strides; eight-byte-only alignment
    # exercises the unpaired loader as well.
    storage[..., :256].copy_(
        torch.randn(
            batch * pages,
            2,
            page,
            heads,
            256,
            device="cuda",
            dtype=torch.float16,
        )
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    k, v = storage[..., :256].unbind(1)
    table = torch.randperm(batch * pages, device="cuda").int().reshape(batch, pages)
    q = torch.randn(batch * 8, heads * 6, 256, device="cuda", dtype=torch.float16)
    initial = (
        torch.tensor(
            [max(8, context // (request + 1)) for request in range(batch)],
            device="cuda",
            dtype=torch.int32,
        )[:, None]
        + torch.arange(-7, 1, device="cuda", dtype=torch.int32)
    ).flatten()
    lengths = initial.clone()
    actual, expected = torch.empty_like(q), torch.empty_like(q)

    def call(query, blocks, rows, output):
        op(
            query,
            k,
            v,
            blocks,
            rows,
            out=output,
            softmax_scale=0.0625,
            k_scale=0.5,
            v_scale=1.25,
        )

    call(q, table, lengths, actual)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call(q, table, lengths, actual)
    for state in ("live", "padded_request", "zero_rows", "all_zero", "restore"):
        lengths.copy_(initial)
        if state == "padded_request":
            lengths[-8:] = 0
        elif state == "zero_rows":
            lengths[::8] = 0
        elif state == "all_zero":
            lengths.zero_()
        q.normal_(0, 0.7)
        for request in range(batch):
            rows = slice(request * 8, (request + 1) * 8)
            call(q[rows], table[request : request + 1], lengths[rows], expected[rows])
        actual.fill_(float("nan"))
        graph.replay()
        assert torch.isfinite(actual).all()
        assert torch.equal(actual, expected)

    # An independent FP64 oracle checks the longest and a shorter request.
    for request in (0, batch - 1):
        row = request * 8 + 7
        length = int(initial[row])
        for head in range(heads):
            key = k[table[request].long(), :, head].reshape(-1, 256)[:length]
            value = v[table[request].long(), :, head].reshape(-1, 256)[:length]
            key = key.view(torch.float8_e4m3fn).double() * 0.5
            value = value.view(torch.float8_e4m3fn).double() * 1.25
            query = q[row, head * 6 : (head + 1) * 6].double()
            reference = ((query @ key.T) * 0.0625).softmax(-1) @ value
            result = actual[row, head * 6 : (head + 1) * 6].double()
            torch.testing.assert_close(result, reference, rtol=0.03, atol=3e-4)
            assert (result - reference).norm() / reference.norm() < 0.007


@pytest.mark.parametrize("batches", [(2, 8, 4, 16, 3), (16, 8, 4, 2)])
def test_workspace_reuse_preserves_older_graphs(monkeypatch, batches):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires SM70")
    from vllm.v1.attention.ops import sm70_e4m3_long as long
    from vllm.vllm_flash_attn import flash_attn_interface  # noqa: F401

    if long.long_attention_max_batch_size() < max(batches):
        pytest.skip("rebuild shipped long-attention batch operator")
    monkeypatch.setattr(long, "_WORKSPACES", {})
    monkeypatch.setattr(long, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(
        long,
        "get_forward_context",
        lambda: SimpleNamespace(
            batch_descriptor=SimpleNamespace(attention_context_bucket=262144)
        ),
    )

    def declined(*args, **kwargs):
        pytest.fail("the built-in long route declined an admitted batch")

    op = long.wrap_long_attention(declined)
    storage = (
        torch.randn(max(batches), 2, 512, 1, 256, device="cuda", dtype=torch.float16)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    k, v = storage.unbind(1)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    cases = []
    with torch.cuda.stream(stream):
        for batch in batches:
            q = torch.randn(batch * 8, 6, 256, device="cuda", dtype=torch.float16)
            table = torch.arange(batch, device="cuda", dtype=torch.int32)[:, None]
            lengths = torch.arange(505, 513, device="cuda", dtype=torch.int32)
            lengths = lengths.repeat(batch)
            out = torch.empty_like(q)
            op(q, k, v, table, lengths, out=out, softmax_scale=0.0625)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                op(q, k, v, table, lengths, out=out, softmax_scale=0.0625)
            cases.append((graph, q, table, lengths, out))

        bank = next(iter(long._WORKSPACES.values()))
        # Descending captures only need the largest panel. Growth keeps earlier
        # buffers alive, but intermediate smaller batches do not allocate.
        assert [partial.shape[0] for partial, _ in bank] == (
            [2, 8, 16] if batches[0] == 2 else [16]
        )
        for graph, q, table, lengths, out in reversed(cases):
            q.normal_(0, 0.7)
            lengths[-8:] = 0
            expected = torch.empty_like(q)
            for request in range(table.shape[0]):
                rows = slice(request * 8, (request + 1) * 8)
                op(
                    q[rows],
                    k,
                    v,
                    table[request : request + 1],
                    lengths[rows],
                    out=expected[rows],
                    softmax_scale=0.0625,
                )
            out.fill_(float("nan"))
            graph.replay()
            assert torch.equal(out, expected)
    torch.cuda.current_stream().wait_stream(stream)
