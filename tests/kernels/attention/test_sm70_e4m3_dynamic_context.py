# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Served-window E4M3 dynamic context against same-KV FP64 attention."""

from types import SimpleNamespace

import pytest
import torch


def _routes(monkeypatch, context):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires SM70")
    from vllm.v1.attention.ops import sm70_grouped as grouped_owner
    from vllm.v1.attention.ops import sm70_grouped_long as long
    from vllm.v1.attention.ops import sm70_grouped_scalar as scalar

    monkeypatch.setattr(
        grouped_owner,
        "get_current_vllm_config_or_none",
        lambda: SimpleNamespace(model_config=SimpleNamespace(max_model_len=context)),
    )
    monkeypatch.delenv("VLLM_SM70_E4M3_LONG_ATTENTION", raising=False)
    monkeypatch.delenv("VLLM_SM70_E4M3_LONG_ATTENTION_MANIFEST", raising=False)
    assert long.builtin_dynamic_context(), "rebuild the grouped DSO"
    assert long.builtin_dynamic_context(scalar=True), "rebuild the scalar DSO"
    assert long.builtin_context_limit() == context, "rebuild the grouped DSO"
    assert long.builtin_context_limit(scalar=True) == context, "rebuild scalar DSO"
    for module in (long, scalar):
        monkeypatch.setattr(module, "is_forward_context_available", lambda: True)
        monkeypatch.setattr(
            module,
            "get_forward_context",
            lambda: SimpleNamespace(
                batch_descriptor=SimpleNamespace(attention_context_bucket=context)
            ),
        )

    def declined(*args, **kwargs):
        pytest.fail("qualified grouped input declined")

    grouped = long.wrap_long_attention(declined)
    scalar.load_scalar_tail_attention.cache_clear()
    tail = scalar.load_scalar_tail_attention("", torch.device("cuda:0"))
    assert tail is not None

    def call(q, k, v, table, lengths, out):
        kwargs = dict(out=out, softmax_scale=0.0625, k_scale=0.5, v_scale=1.25)
        if q.shape[0] == 1:
            assert tail(
                q,
                k,
                v,
                table,
                lengths,
                **kwargs,
                kv_cache_dtype="fp8_e4m3",
                window_size=(-1, -1),
                max_seq_len_hint=context,
                partition_size_hint=None,
                anchor_lens=None,
                anchored_window=0,
            )
        else:
            grouped(q, k, v, table, lengths, **kwargs)

    return call


def _inputs(context, rows, page=2048, batch=1, heads=1):
    torch.manual_seed(20261106 + rows + batch)
    pages = (context + page - 1) // page
    storage = (
        torch.randn(
            batch * pages, 2, page, heads, 256, device="cuda", dtype=torch.float16
        )
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    k, v = storage.unbind(1)
    q = torch.randn(batch * rows, heads * 6, 256, device="cuda", dtype=torch.float16)
    table = torch.randperm(batch * pages, device="cuda").int().reshape(batch, pages)
    lengths = torch.arange(
        context - rows + 1, context + 1, device="cuda", dtype=torch.int32
    ).repeat(batch)
    return q, k, v, table, lengths, torch.empty_like(q)


def _oracle(inputs):
    q, k, v, table, lengths, actual = inputs
    expected = torch.empty_like(q, dtype=torch.float64)
    batch = table.shape[0]
    rows = q.shape[0] // batch
    for request in range(batch):
        section = slice(request * rows, (request + 1) * rows)
        limit = int(lengths[section].max())
        if limit == 0:
            expected[section].zero_()
            continue
        for head in range(k.shape[2]):
            key = k[table[request].long(), :, head].reshape(-1, 256)[:limit]
            value = v[table[request].long(), :, head].reshape(-1, 256)[:limit]
            key = key.view(torch.float8_e4m3fn).double() * 0.5
            value = value.view(torch.float8_e4m3fn).double() * 1.25
            query = q[section, head * 6 : (head + 1) * 6].double()
            scores = (query @ key.T) * 0.0625
            scores.masked_fill_(
                torch.arange(limit, device="cuda")[None, None]
                >= lengths[section, None, None],
                -torch.inf,
            )
            expected[section, head * 6 : (head + 1) * 6] = (
                scores.softmax(-1).nan_to_num(0) @ value
            )
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.double(), expected, rtol=0.03, atol=3e-4)
    relative = float(
        (actual.double() - expected).norm() / expected.norm().clamp_min(1e-30)
    )
    assert relative < 0.007
    return relative


@pytest.mark.parametrize("context", [32768, 262144, 524288, 1048576])
@pytest.mark.parametrize("rows", range(1, 9))
def test_dynamic_fp64_oracle_and_live_graph(monkeypatch, context, rows):
    call = _routes(monkeypatch, context)
    inputs = _inputs(context, rows)
    call(*inputs)
    error = _oracle(inputs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call(*inputs)
    eager = torch.empty_like(inputs[-1])
    for length in (min(262145, context), 0, context):
        inputs[0].normal_(0, 0.7)
        inputs[4].fill_(length)
        # Page IDs are live graph inputs, independent of the original order.
        inputs[3].copy_(inputs[3].flip(1))
        call(*inputs[:-1], eager)
        inputs[-1].fill_(float("nan"))
        graph.replay()
        assert torch.equal(inputs[-1], eager)
        if length == 0:
            assert torch.count_nonzero(eager) == 0
    print(f"context={context} rows={rows} relative_l2={error:.8g}")


@pytest.mark.parametrize("page,heads", [(1024, 1), (3296, 2)])
@pytest.mark.parametrize("rows", [1, 7, 8])
def test_dynamic_page_rounding_and_six_head_groups(monkeypatch, page, heads, rows):
    call = _routes(monkeypatch, 1048576)
    inputs = _inputs(1048576, rows, page=page, heads=heads)
    call(*inputs)
    _oracle(inputs)


@pytest.mark.parametrize("batch", [2, 4, 8, 16])
def test_dynamic_independent_request_graphs(monkeypatch, batch):
    call = _routes(monkeypatch, 1048576)
    inputs = _inputs(1048576, 8, batch=batch)
    # Only the first request is 1M; keep independent shorter/empty requests.
    for request in range(1, batch):
        inputs[4][request * 8 : (request + 1) * 8].fill_(32768 // request)
    call(*inputs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call(*inputs)
    inputs[4][-8:] = 0
    graph.replay()
    _oracle(inputs)


def test_scalar_reads_values_beyond_old_partition_ceiling(monkeypatch):
    call = _routes(monkeypatch, 1048576)
    q, k, v, table, lengths, out = _inputs(1048576, 1)
    q.zero_()
    v.zero_()
    # Uniform softmax with a known suffix mean exposes a truncated 256K launch.
    v[table[0, 128:].long()] = 0x38  # E4M3 1.0
    call(q, k, v, table, lengths, out)
    torch.testing.assert_close(out, torch.full_like(out, 0.75 * 1.25), rtol=0, atol=0)


@pytest.mark.parametrize("context", [1023, 1024, 1025, 262145, 1048593, 2097152])
@pytest.mark.parametrize("rows", [1, 8])
def test_dynamic_partition_and_reduction_boundaries(monkeypatch, context, rows):
    call = _routes(monkeypatch, context)
    inputs = _inputs(context, rows)
    call(*inputs)
    _oracle(inputs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call(*inputs)
    inputs[4].zero_()
    graph.replay()
    assert torch.count_nonzero(inputs[-1]) == 0
    inputs[4].fill_(context)
    graph.replay()
    _oracle(inputs)


def test_scalar_reads_suffix_beyond_one_million(monkeypatch):
    context, page = 2097152, 2048
    call = _routes(monkeypatch, context)
    q, k, v, table, lengths, out = _inputs(context, 1, page=page)
    q.zero_()
    v.zero_()
    v[table[0, 1048576 // page :].long()] = 0x38
    call(q, k, v, table, lengths, out)
    torch.testing.assert_close(out, torch.full_like(out, 0.5 * 1.25), rtol=0, atol=0)


def test_native_scalar_rejects_workspace_that_cannot_cover_table(monkeypatch):
    _routes(monkeypatch, 2097152)
    q, k, v, table, lengths, out = _inputs(2097152, 1)
    p = 1024
    partial = torch.empty((1, 6, p, 256), device="cuda", dtype=torch.float32)
    maximum = torch.empty((1, 6, p), device="cuda", dtype=torch.float32)
    sums = torch.empty_like(maximum)
    active = torch.full((1,), p, device="cuda", dtype=torch.int32)
    with pytest.raises(RuntimeError, match="workspace does not cover"):
        torch.ops._vllm_fa2_C.sm70_scalar_attention_fwd(
            q,
            k,
            v,
            out,
            table,
            lengths,
            partial,
            maximum,
            sums,
            active,
            0.0625,
            0.5,
            1.25,
        )


def test_scalar_grid_and_shared_memory_do_not_limit_partition_capacity(monkeypatch):
    _routes(monkeypatch, 1048576)
    q, k, v, table, lengths, out = _inputs(1, 1)
    q.zero_()
    v.fill_(0x38)
    # Overprovision storage without allocating a 64M-token KV cache. Only the
    # first partition has work, but the launch must exceed grid.z's old bound
    # and must not request shared memory proportional to the workspace size.
    p = 65537
    partial = torch.empty((1, 6, p, 256), device="cuda", dtype=torch.float32)
    maximum = torch.empty((1, 6, p), device="cuda", dtype=torch.float32)
    sums = torch.empty_like(maximum)
    active = torch.full((1,), p, device="cuda", dtype=torch.int32)
    torch.ops._vllm_fa2_C.sm70_scalar_attention_fwd(
        q,
        k,
        v,
        out,
        table,
        lengths,
        partial,
        maximum,
        sums,
        active,
        0.0625,
        0.5,
        1.25,
    )
    torch.testing.assert_close(out, torch.full_like(out, 1.25), rtol=0, atol=0)
