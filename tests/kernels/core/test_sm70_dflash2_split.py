# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise live graph lengths, page indirection, and window-edge masking."""

import math

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="SM70 attention specialization",
)


def _dense(q, k, v, table, length, scale):
    if length == 0:
        return torch.zeros_like(q, dtype=torch.float64)
    first = max(0, length - 8 - 2047)
    positions = torch.arange(first, length, device=q.device)
    pages = table[0, positions // k.shape[1]].long()
    keys = k[pages, positions % k.shape[1]].repeat_interleave(4, 1).double()
    values = v[pages, positions % v.shape[1]].repeat_interleave(4, 1).double()
    query_positions = torch.arange(length - 8, length, device=q.device)
    mask = (positions[None] >= query_positions[:, None] - 2047) & (
        positions[None] <= query_positions[:, None] + 2047
    )
    scores = torch.einsum("bqhd,khd->bhqk", q.double(), keys) * scale
    scores.masked_fill_(~mask[None, None], -float("inf"))
    return torch.einsum("bhqk,khd->bqhd", scores.softmax(-1), values)


@pytest.mark.parametrize("through_api", [False, True])
@pytest.mark.parametrize("page", [832, 1024, 1648, 2048])
@pytest.mark.parametrize("scale", [1 / math.sqrt(128), 0.1])
def test_live_window_graph(page, scale, through_api):
    from flash_attn_v100.sm70_dflash2_split import forward

    if through_api:
        from flash_attn_v100 import flash_attn_prefill_paged

        def call(q, k, v, table, lengths, scale, output):
            return flash_attn_prefill_paged(
                q,
                k,
                v,
                table,
                lengths,
                softmax_scale=scale,
                out=output,
                causal=False,
                window_size=(2047, 2047),
            )
    else:
        call = forward

    torch.manual_seed(123)
    pages = math.ceil(10240 / page)
    capacity = pages * page
    q = torch.zeros(1, 8, 8, 128, device="cuda", dtype=torch.float16)
    # Match the model's interleaved K/V allocation, including noncontiguous
    # page strides, instead of testing only independent contiguous caches.
    kv = torch.zeros(pages, 2, page, 2, 128, device="cuda", dtype=torch.float16)
    k, v = kv[:, 0], kv[:, 1]
    table = torch.randperm(pages, device="cuda").int()[None]
    lengths = torch.zeros(1, device="cuda", dtype=torch.int32)
    output = torch.empty_like(q)
    call(q, k, v, table, lengths, scale, output)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call(q, k, v, table, lengths, scale, output)
    for length in (0, 8, 127, 128, 129, 1024, 2047, 2048, 2055, 4096, 8192):
        logical = torch.full(
            (capacity, 2, 128), float("nan"), device="cuda", dtype=torch.float16
        )
        logical[:length] = 60000
        logical[max(0, length - 8 - 2047) : length] = 1
        if length:
            for row in range(8):
                logical[max(0, length - 8 + row - 2047)] = (row + 1) * 128
        v[table[0].long()] = logical.reshape_as(v)
        lengths.fill_(length)
        graph.replay()
        truth = _dense(q, k, v, table, length, scale)
        assert torch.isfinite(output).all()
        torch.testing.assert_close(output.double(), truth, atol=0.003, rtol=0.001)
    # Reuse the same captured graph on nonuniform attention scores and values.
    q.normal_()
    k.normal_()
    v.normal_()
    for length in (1032, 8200):
        lengths.fill_(length)
        graph.replay()
        truth = _dense(q, k, v, table, length, scale)
        torch.testing.assert_close(output.double(), truth, atol=0.001, rtol=0.002)


@pytest.mark.parametrize("page", [832, 1648])
def test_hybrid_split_route_without_native_symbol(monkeypatch, page):
    from flash_attn_v100 import flash_attn_interface as interface
    from flash_attn_v100 import sm70_dflash2_split as split

    calls = []
    original = split.forward

    def record(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(split, "forward", record)
    monkeypatch.delattr(
        interface.flash_attn_v100_cuda, "dflash2_paged_bmhd_fwd", raising=False
    )
    query = torch.randn(1, 8, 8, 128, dtype=torch.float16, device="cuda")
    kv = torch.randn(2, 2, page, 2, 128, dtype=torch.float16, device="cuda")
    table = torch.tensor([[1, 0]], dtype=torch.int32, device="cuda")
    lengths = torch.tensor([1032], dtype=torch.int32, device="cuda")
    output = torch.empty_like(query)
    arguments = dict(out=output, causal=False, window_size=(2047, 2047))
    interface.flash_attn_prefill_paged(
        query,
        kv[:, 0],
        kv[:, 1],
        table,
        lengths,
        dflash2_window_split=False,
        **arguments,
    )
    baseline = output.clone()
    assert calls == []
    graph = torch.cuda.CUDAGraph()
    interface.flash_attn_prefill_paged(
        query,
        kv[:, 0],
        kv[:, 1],
        table,
        lengths,
        dflash2_window_split=True,
        **arguments,
    )
    with torch.cuda.graph(graph):
        interface.flash_attn_prefill_paged(
            query,
            kv[:, 0],
            kv[:, 1],
            table,
            lengths,
            dflash2_window_split=True,
            **arguments,
        )
    graph.replay()
    torch.testing.assert_close(output, baseline, atol=0.001, rtol=0.002)
    assert len(calls) == 2
