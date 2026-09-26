# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Multi-KV-head arithmetic and live-metadata CUDA Graph regressions."""

from types import SimpleNamespace

import pytest
import torch

fa = pytest.importorskip("flash_attn_v100.flash_attn_interface")


def _inputs(heads, batch, length, page, cache_dtype=torch.float8_e4m3fn):
    torch.manual_seed(4619)
    blocks = (length + page - 1) // page
    storage = (
        torch.randn(
            batch * blocks, 2, page, heads, 256, device="cuda", dtype=torch.float16
        )
        * 0.25
    )
    storage = storage.to(cache_dtype)
    if cache_dtype != torch.float16:
        storage = storage.view(torch.uint8)
    # Reverse physical pages and keep the standard interleaved K/V strides.
    table = torch.arange(
        batch * blocks - 1, -1, -1, device="cuda", dtype=torch.int32
    ).reshape(batch, blocks)
    lengths = torch.full((batch,), length, device="cuda", dtype=torch.int32)
    query = torch.randn(batch, heads * 6, 256, device="cuda", dtype=torch.float16)
    return query, storage[:, 0], storage[:, 1], table, lengths


def _reference(
    query,
    key,
    value,
    table,
    lengths,
    k_scale=0.75,
    v_scale=1.25,
    cache_dtype=torch.float8_e4m3fn,
):
    result = torch.zeros_like(query, dtype=torch.float64)
    for row, length in enumerate(lengths.cpu().tolist()):
        if not length:
            continue
        seq = min(row, table.shape[0] - 1)
        for head in range(key.shape[2]):
            k = key[table[seq].long(), :, head].reshape(-1, 256)[:length]
            v = value[table[seq].long(), :, head].reshape(-1, 256)[:length]
            k = k.view(cache_dtype).double() * k_scale
            v = v.view(cache_dtype).double() * v_scale
            q = query[row, head * 6 : (head + 1) * 6].double()
            scores = (q @ k.T) * 0.0625
            result[row, head * 6 : (head + 1) * 6] = scores.softmax(-1) @ v
    return result


def _replay_check(call, query, lengths, output, reference, initial_length):
    call()  # Allocate the fixed workspace before capture.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for length in (initial_length, max(1, initial_length - 17)):
        query.normal_(0, 0.7)
        lengths.fill_(length)
        if lengths.numel() > 1:
            lengths[0] = 0
        output.fill_(float("nan"))
        graph.replay()
        expected = reference()
        assert torch.isfinite(output).all()
        torch.testing.assert_close(output.double(), expected, rtol=0.03, atol=3e-4)
        relative_l2 = (output.double() - expected).norm() / expected.norm().clamp_min(
            1e-12
        )
        assert relative_l2.item() < 0.007


@pytest.mark.parametrize("heads", [1, 2, 4])
@pytest.mark.parametrize("batch", [1, 2, 4, 8, 16, 32])
def test_e4m3_xqa_multihead_graph(heads, batch):
    q, k, v, table, lengths = _inputs(heads, batch, 2049, 800)
    out = torch.empty_like(q)

    def call():
        fa.flash_attn_decode_paged_xqa(
            q,
            k,
            v,
            table,
            lengths,
            out=out,
            kv_cache_dtype="fp8_e4m3",
            k_scale=0.75,
            v_scale=1.25,
            max_seq_len_hint=2049,
            workspace_seq_capacity_hint=table.shape[1] * 800,
            partition_size_hint=64 if batch == 1 else None,
            batch_context_routing=True,
        )

    _replay_check(
        call, q, lengths, out, lambda: _reference(q, k, v, table, lengths), 2049
    )


@pytest.mark.parametrize("heads", [1, 2, 4])
@pytest.mark.parametrize("page,length", [(848, 32768), (1568, 131072), (800, 262144)])
def test_e4m3_xqa_long_multihead_graph(heads, page, length):
    q, k, v, table, lengths = _inputs(heads, 1, length, page)
    out = torch.empty_like(q)

    def call():
        fa.flash_attn_decode_paged_xqa(
            q,
            k,
            v,
            table,
            lengths,
            out=out,
            kv_cache_dtype="fp8_e4m3",
            k_scale=0.75,
            v_scale=1.25,
            max_seq_len_hint=length,
            workspace_seq_capacity_hint=table.shape[1] * page,
            partition_size_hint=64,
            batch_context_routing=True,
        )

    _replay_check(
        call, q, lengths, out, lambda: _reference(q, k, v, table, lengths), length
    )


@pytest.mark.parametrize("heads", [2, 4])
@pytest.mark.parametrize("cache_dtype", [torch.float16, torch.float8_e5m2])
def test_other_dtypes_long_multihead_graph(heads, cache_dtype):
    length, page = 32768, 800
    q, k, v, table, lengths = _inputs(heads, 1, length, page, cache_dtype)
    out = torch.empty_like(q)
    fp16 = cache_dtype == torch.float16
    k_scale, v_scale = (1.0, 1.0) if fp16 else (0.75, 1.25)

    def call():
        fa.flash_attn_decode_paged_xqa(
            q,
            k,
            v,
            table,
            lengths,
            out=out,
            kv_cache_dtype="auto" if fp16 else "fp8_e5m2",
            k_scale=k_scale,
            v_scale=v_scale,
            max_seq_len_hint=length,
            workspace_seq_capacity_hint=table.shape[1] * page,
            batch_context_routing=True,
        )

    _replay_check(
        call,
        q,
        lengths,
        out,
        lambda: _reference(q, k, v, table, lengths, k_scale, v_scale, cache_dtype),
        length,
    )


@pytest.mark.parametrize("heads", [2, 4])
@pytest.mark.parametrize("rows", [1, 3, 8])
@pytest.mark.parametrize("page", [1024, 2048, 3296, 4096, 8192])
def test_builtin_long_multihead_graph(monkeypatch, heads, rows, page):
    from vllm.v1.attention.ops import sm70_e4m3_long as long
    from vllm.v1.attention.ops import sm70_e4m3_scalar as scalar
    from vllm.vllm_flash_attn import flash_attn_interface  # noqa: F401

    for module in (long, scalar):
        monkeypatch.setattr(module, "is_forward_context_available", lambda: True)
        monkeypatch.setattr(
            module,
            "get_forward_context",
            lambda: SimpleNamespace(
                batch_descriptor=SimpleNamespace(attention_context_bucket=262144)
            ),
        )
    length = 262144
    _, k, v, table, _ = _inputs(heads, 1, length, page)
    q = torch.randn(rows, heads * 6, 256, device="cuda", dtype=torch.float16)
    lengths = torch.full((rows,), length, device="cuda", dtype=torch.int32)
    out = torch.empty_like(q)

    def declined(*args, **kwargs):
        pytest.fail("built-in long path declined the multi-head layout")

    if rows == 1:
        run = scalar.load_scalar_tail_attention("", q.device)
        assert run is not None

        def call():
            assert run(
                q,
                k,
                v,
                table,
                lengths,
                out=out,
                softmax_scale=0.0625,
                k_scale=0.75,
                v_scale=1.25,
                kv_cache_dtype="fp8_e4m3",
                window_size=(-1, -1),
                max_seq_len_hint=length,
                partition_size_hint=None,
                anchor_lens=None,
                anchored_window=0,
            )
    else:
        run = long.wrap_long_attention(declined)

        def call():
            return run(
                q,
                k,
                v,
                table,
                lengths,
                out=out,
                softmax_scale=0.0625,
                k_scale=0.75,
                v_scale=1.25,
            )

    _replay_check(
        call, q, lengths, out, lambda: _reference(q, k, v, table, lengths), length
    )


@pytest.mark.parametrize("heads", [1, 2, 4])
@pytest.mark.parametrize("page,length", [(256, 2049), (848, 32768), (800, 262144)])
def test_e4m3_grouped_multihead_graph(heads, page, length):
    _, k, v, table, _ = _inputs(heads, 1, length, page)
    q = torch.randn(8, heads * 6, 256, device="cuda", dtype=torch.float16)
    lengths = torch.arange(length - 7, length + 1, device="cuda", dtype=torch.int32)
    out = torch.empty_like(q)

    def call():
        fa.flash_attn_grouped_e4m3_fp32_paged(
            q,
            k,
            v,
            table,
            lengths,
            out=out,
            softmax_scale=0.0625,
            k_scale=0.75,
            v_scale=1.25,
        )

    _replay_check(
        call, q, lengths, out, lambda: _reference(q, k, v, table, lengths), length
    )
