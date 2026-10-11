# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Installed codec bindings through conversion, long attention and the engine.

Frozen numerical limits: conversion is byte-exact against codec reconstruction;
compensated grouped attention uses the existing 1.02 * FP16 rounding floor +
2e-6 relative L2 contract. Engine prefill/XQA use their existing 0.004 absolute,
0.015 relative output bound. These are operator gates, not model quality claims.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.kv_cache_view import KVCacheView
from vllm.v1.attention.kv_codecs import (
    FP8_E4M3,
    FP8_E5M2,
    FP16,
    INT8,
    INT8_PER_TOKEN_HEAD,
)

package = pytest.importorskip("flash_attn_v100")
CODECS = (FP16, FP8_E4M3, FP8_E5M2, INT8, INT8_PER_TOKEN_HEAD)
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="SM70 native qualification",
)


def cache(codec, pages=5, page=32, heads=1, dim=256):
    carrier = torch.zeros(
        (pages, 2, page, heads, dim + (4 if codec.per_token_head else 0)),
        device="cuda",
        dtype=codec.storage_dtype,
    )
    view = KVCacheView.bind(carrier, codec, dim)
    for data, scales in ((view.key, view.key_scale), (view.value, view.value_scale)):
        x = torch.randn(data.shape, device="cuda", dtype=torch.float16)
        if scales is not None:
            x *= torch.logspace(-2, 1, page, device="cuda")[None, :, None, None]
            raw, scale = codec.encode(x)
            data.copy_(raw)
            scales.copy_(scale)
        elif codec.quantized:
            data.copy_(x.to(codec.element_dtype).view(codec.storage_dtype))
        else:
            data.copy_(x)
    return carrier, view


def reference(
    q, view, table, lengths, *, grouped=False, query_positions=None, window=(-1, -1)
):
    rows = []
    for row, length in enumerate(lengths.cpu().tolist()):
        if not length:
            rows.append(torch.zeros_like(q[row], device="cpu", dtype=torch.float64))
            continue
        req = row // (q.shape[0] // table.shape[0]) if grouped else row
        t = torch.arange(length, device="cuda")
        indices = (table[req, t // view.key.shape[1]].long(), t % view.key.shape[1])
        decoded = []
        for data, scales in (
            (view.key, view.key_scale),
            (view.value, view.value_scale),
        ):
            raw = data[indices].cpu()
            if view.codec.element_dtype is not None:
                raw = raw.view(view.codec.element_dtype)
            val = raw.double()
            if scales is not None:
                val *= scales[indices].cpu().double()[..., None]
            decoded.append(val.repeat_interleave(q.shape[1] // val.shape[1], 1))
        k, v = decoded
        logits = (
            torch.einsum("hd,thd->ht", q[row].cpu().double(), k) * q.shape[-1] ** -0.5
        )
        if query_positions is not None:
            position = int(query_positions[row])
            valid = torch.ones(length, dtype=torch.bool)
            if window[0] >= 0:
                valid &= torch.arange(length) >= position - window[0]
            if window[1] >= 0:
                valid &= torch.arange(length) <= position + window[1]
            logits.masked_fill_(~valid, float("-inf"))
        rows.append(torch.einsum("ht,thd->hd", logits.softmax(-1), v))
    return torch.stack(rows)


@pytest.mark.parametrize("codec", CODECS, ids=lambda c: c.name)
@pytest.mark.parametrize("dim,heads,batch", [(64, 1, 1), (128, 2, 2), (256, 2, 1)])
def test_public_bridge_codec_pages_padding_and_replay(codec, dim, heads, batch):
    torch.manual_seed(811)
    _, view = cache(codec, pages=7, heads=heads, dim=dim)
    table = torch.tensor([[4, 1], [6, 2]][:batch], device="cuda", dtype=torch.int32)
    lengths = torch.full((batch,), 47, device="cuda", dtype=torch.int32)
    ko = torch.full(
        (batch * 4, 16, heads, dim), -19, device="cuda", dtype=torch.float16
    )
    vo = torch.empty_like(ko)
    ks, vs = (0.5, 2.0) if not codec.per_token_head else (1.0, 1.0)
    if not codec.quantized:
        ks = vs = 1.0

    def run():
        package.paged_kv_to_fp16(
            view.key,
            view.value,
            table,
            lengths,
            ko,
            vo,
            ks,
            vs,
            kv_cache_dtype=codec.name,
            **view.scale_kwargs,
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for length in (47, 33, 64):
        lengths.fill_(length)
        table.copy_(table.flip(1))
        graph.replay()
        for actual, data, scales, tensor_scale in (
            (ko, view.key, view.key_scale, ks),
            (vo, view.value, view.value_scale, vs),
        ):
            t = torch.arange(length, device="cuda")
            indices = (table[:, t // 32].long(), (t % 32).expand(batch, -1))
            scale = tensor_scale if scales is None else scales[indices]
            expected = codec.dequantize(data[indices], scale, torch.float16)
            dense = actual.view(batch, 64, heads, dim)
            assert torch.equal(dense[:, :length], expected)
            assert torch.count_nonzero(dense[:, length : (length + 15) // 16 * 16]) == 0


@pytest.mark.parametrize("codec", CODECS, ids=lambda c: c.name)
@pytest.mark.parametrize(
    "rows,batch,heads", [(2, 1, 1), (5, 1, 2), (8, 1, 1), (8, 2, 1)]
)
def test_long_native_codec_binding_and_replay(codec, rows, batch, heads):
    from vllm.v1.attention.ops.sm70_grouped_long import (
        builtin_codec_attention,
        run_six_head_groups,
    )

    torch.manual_seed(812)
    _, view = cache(codec, pages=12, heads=heads)
    table = torch.tensor(
        [[9, 4, 1], [8, 3, 2]][:batch], device="cuda", dtype=torch.int32
    )
    lengths = torch.arange(97 - rows, 97, device="cuda", dtype=torch.int32).repeat(
        batch
    )
    q = torch.randn((batch * rows, heads * 6, 256), device="cuda", dtype=torch.float16)
    out = torch.empty_like(q)
    partial = torch.empty((batch, 80, 8, 6, 256), device="cuda", dtype=torch.float32)
    lse = torch.empty((batch, 80, 8, 6, 2), device="cuda", dtype=torch.float32)
    op = builtin_codec_attention("sm70_grouped_long", codec)
    assert op is not None, "Normal FA2 artifact is missing the shared long binding"

    def run():
        run_six_head_groups(
            op,
            q,
            view.key,
            view.value,
            out,
            table,
            lengths,
            partial[0] if batch == 1 else partial,
            lse[0] if batch == 1 else lse,
            0.0625,
            1.0,
            1.0,
            codec=codec.native_id,
            key_scales=view.key_scale,
            value_scales=view.value_scale,
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for zero in (False, True):
        if zero:
            lengths[::rows] = 0
        table.copy_(table.flip(1))
        graph.replay()
        expected = reference(q, view, table, lengths, grouped=True)
        actual = out.cpu().double()
        error = (actual - expected).norm() / expected.norm()
        floor = (expected.half().double() - expected).norm() / expected.norm()
        assert torch.isfinite(actual).all()
        assert error <= 1.02 * floor + 2e-6, (error.item(), floor.item())
        assert torch.count_nonzero(actual[lengths.cpu() == 0]) == 0


@pytest.mark.parametrize("codec", (INT8, INT8_PER_TOKEN_HEAD), ids=lambda c: c.name)
@pytest.mark.parametrize("rows", [1, 5, 8, 32, 64])
def test_full_engine_reads_int8_view_for_decode_verify_and_prefix(codec, rows):
    from vllm.v1.attention.backends.flash_v100.impl import FlashAttnV100Impl

    torch.manual_seed(813)
    carrier, view = cache(codec, pages=5, page=32)
    impl = FlashAttnV100Impl(
        num_heads=6,
        head_size=256,
        scale=0.0625,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype=codec.name,
    )
    assert impl._supports_flash_v100_path()
    q = torch.randn((rows, 6, 256), device="cuda", dtype=torch.float16)
    raw = torch.zeros((rows, 1, 256), device="cuda", dtype=torch.float16)
    table = torch.tensor([[4, 2, 1]], device="cuda", dtype=torch.int32)
    lengths = torch.tensor([96], device="cuda", dtype=torch.int32)
    qsl = torch.tensor([0, rows], dtype=torch.int32)
    metadata = SimpleNamespace(
        num_actual_tokens=rows,
        max_query_len=rows,
        max_seq_len=96,
        num_reqs=1,
        max_model_len=262144,
        causal=True,
        query_start_loc=qsl.cuda(),
        query_start_loc_cpu=qsl,
        seq_lens=lengths,
        seq_lens_cpu=lengths.cpu(),
        block_table=table,
        is_dflash_selector_target=rows in (5, 8),
        smallq_decode_block_table=table.expand(rows, -1).contiguous(),
        smallq_decode_seq_lens=torch.arange(
            97 - rows, 97, device="cuda", dtype=torch.int32
        ),
        smallq_query_start_loc=qsl.cuda(),
        smallq_decode_max_seq_len_hint=96,
        smallq_decode_workspace_seq_capacity_hint=96,
        flash_v100_decode_max_seq_len_hint=96,
        flash_v100_decode_workspace_seq_capacity_hint=96,
    )
    layer = SimpleNamespace(
        _k_scale_float=1.0, _v_scale_float=1.0, layer_name="operator_contract"
    )
    out = torch.empty_like(q)
    impl.forward(layer, q, raw, raw, carrier, metadata, out)
    expected = reference(q, view, table, metadata.smallq_decode_seq_lens, grouped=True)
    torch.testing.assert_close(out.cpu().double(), expected, atol=0.004, rtol=0.015)
    # The passed carrier must be interpreted by the shared view, not raw unbind.
    assert impl._kv_cache_view.key.data_ptr() == view.key.data_ptr()
    assert impl._kv_cache_view.key_scale.data_ptr() == view.key_scale.data_ptr()


@pytest.mark.parametrize("codec", CODECS, ids=lambda c: c.name)
@pytest.mark.parametrize("variant", ("paged", "bhmd", "splitkv", "bfla", "wmma"))
def test_public_prefill_algorithms_bind_each_codec(codec, variant):
    torch.manual_seed(816)
    _, view = cache(codec, pages=10)
    table = torch.tensor([[9, 7, 3, 5, 1, 8, 2, 4]], device="cuda", dtype=torch.int32)
    lengths = torch.tensor([233], device="cuda", dtype=torch.int32)
    rows = 1 if variant == "wmma" else 8
    q = torch.randn((rows, 6, 256), device="cuda", dtype=torch.float16)
    out = torch.empty_like(q)
    mask = torch.ones((1, 1, 1, 8), device="cuda", dtype=torch.int32)
    kwargs = dict(kv_cache_dtype=codec.name, **view.scale_kwargs)

    def run():
        if variant == "wmma":
            return package.flash_attn_decode_paged_wmma(
                q, view.key, view.value, table, lengths, out=out, **kwargs
            )
        qb, ob = q.unsqueeze(0), out.unsqueeze(0)
        if variant == "bhmd":
            qb = qb.permute(0, 2, 1, 3).contiguous()
            ob = ob.permute(0, 2, 1, 3).contiguous()
        name = "flash_attn_prefill_paged" + (
            "" if variant == "paged" else "_" + variant
        )
        extra = (
            {"split_kv_tokens": 64, "max_seq_len_hint": 256}
            if variant == "splitkv"
            else {}
        )
        args = (mask, 32) if variant == "bfla" else ()
        result = getattr(package, name)(
            qb, view.key, view.value, table, lengths, *args, out=ob, **kwargs, **extra
        )
        if variant == "bhmd":
            out.copy_(result.permute(0, 2, 1, 3).squeeze(0))

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for length in (233, 201):
        lengths.fill_(length)
        table.copy_(table.flip(1))
        graph.replay()
        row_lengths = torch.arange(
            length - rows + 1, length + 1, device="cuda", dtype=torch.int32
        )
        expected = reference(q, view, table, row_lengths, grouped=True)
        torch.testing.assert_close(out.cpu().double(), expected, atol=0.004, rtol=0.015)


@pytest.mark.parametrize("codec", CODECS, ids=lambda c: c.name)
def test_compact_tail_uses_shared_codec_and_noncontiguous_pages(codec):
    from vllm.v1.attention.ops.sm70_grouped_long import builtin_codec_attention

    torch.manual_seed(815)
    _, view = cache(codec, pages=4, page=1024)
    table = torch.tensor([[3, 1]], device="cuda", dtype=torch.int32)
    lengths = torch.tensor([1025], device="cuda", dtype=torch.int32)
    q = torch.randn((1, 6, 256), device="cuda", dtype=torch.float16)
    out = torch.empty_like(q)
    buffers = (
        torch.empty((1, 6, 256, 256), device="cuda", dtype=torch.float32),
        torch.empty((1, 6, 256), device="cuda", dtype=torch.float32),
        torch.empty((1, 6, 256), device="cuda", dtype=torch.float32),
        torch.full((1,), 256, device="cuda", dtype=torch.int32),
    )
    op = builtin_codec_attention("sm70_scalar_attention", codec)
    assert op is not None

    def run():
        op(
            q,
            view.key,
            view.value,
            out,
            table,
            lengths,
            *buffers,
            0.0625,
            1.0,
            1.0,
            codec.native_id,
            view.key_scale,
            view.value_scale,
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for length in (1023, 1024, 1025, 2048, 0):
        lengths.fill_(length)
        table.copy_(table.flip(1))
        graph.replay()
        expected = reference(q, view, table, lengths)
        torch.testing.assert_close(out.cpu().double(), expected, atol=0.004, rtol=0.015)


@pytest.mark.parametrize("codec", CODECS, ids=lambda c: c.name)
@pytest.mark.parametrize(
    "dim,heads,rows,causal,window",
    [
        (64, 2, 17, True, (-1, -1)),
        (128, 2, 8, False, (31, 31)),
        (256, 1, 33, True, (-1, -1)),
    ],
)
def test_prefill_codec_geometry_window_and_split_merge(
    codec, dim, heads, rows, causal, window
):
    torch.manual_seed(817)
    _, view = cache(codec, pages=7, heads=heads, dim=dim)
    table = torch.tensor([[6, 1, 5, 2]], device="cuda", dtype=torch.int32)
    lengths = torch.tensor([113], device="cuda", dtype=torch.int32)
    q = torch.randn((rows, 4 * heads, dim), device="cuda", dtype=torch.float16)
    row_lengths = (
        torch.arange(114 - rows, 114, device="cuda", dtype=torch.int32)
        if causal
        else lengths.expand(rows)
    )
    expected = reference(
        q,
        view,
        table,
        row_lengths,
        grouped=True,
        query_positions=None if causal else range(113 - rows, 113),
        window=window,
    )
    arguments = dict(
        kv_cache_dtype=codec.name,
        **view.scale_kwargs,
        causal=causal,
        window_size=window,
    )
    output = package.flash_attn_prefill_paged(
        q.unsqueeze(0),
        view.key,
        view.value,
        table,
        lengths,
        **arguments,
    )
    torch.testing.assert_close(
        output.squeeze(0).cpu().double(), expected, atol=0.004, rtol=0.015
    )
    if dim == 256:
        # The ordinary kernel's BLOCK_M is 32; the merge kernel's is 16.
        # More than 32 query rows must still be completely written.
        output.fill_(float("nan"))
        package.flash_attn_prefill_paged_splitkv(
            q.unsqueeze(0),
            view.key,
            view.value,
            table,
            lengths,
            out=output,
            split_kv_tokens=64,
            max_seq_len_hint=128,
            **arguments,
        )
        torch.testing.assert_close(
            output.squeeze(0).cpu().double(), expected, atol=0.004, rtol=0.015
        )
