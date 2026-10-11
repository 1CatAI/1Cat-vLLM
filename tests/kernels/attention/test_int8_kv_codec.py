# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU contracts for the shared writer, Triton attention and native decode.

Numerical limits are frozen with the candidate before GPU qualification. These
operator cases do not establish model quality or end-to-end speed parity.
"""

import pytest
import torch

from vllm.v1.attention.kv_cache_view import KVCacheView
from vllm.v1.attention.kv_codecs import INT8, INT8_PER_TOKEN_HEAD
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash_per_token_head_quant,
)
from vllm.v1.attention.ops.triton_unified_attention import unified_attention
from vllm.v1.kv_cache_interface import get_kv_quant_mode

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


def _cache(codec, blocks, block, heads, dim):
    carrier = torch.zeros(
        (blocks, 2, block, heads, dim + 4), dtype=torch.int8, device="cuda"
    )
    return carrier, KVCacheView.bind(carrier, codec, dim)


def _write(codec, key, value, view, slots):
    triton_reshape_and_cache_flash_per_token_head_quant(
        key,
        value,
        view.key,
        view.value,
        view.key_scale,
        view.value_scale,
        slots,
        round_int8_even=codec is INT8,
    )


@pytest.mark.parametrize("codec", [INT8, INT8_PER_TOKEN_HEAD], ids=lambda c: c.name)
@pytest.mark.parametrize("dim", [32, 64, 80, 128, 256])
def test_writer_rounding_negative_slots_and_graph_replay(codec, dim):
    carrier, view = _cache(codec, 3, 16, 2, dim)
    row = torch.tensor([127.0, 0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 0.0])
    host = row.repeat((dim + 7) // 8)[:dim].expand(3, 2, dim).clone()
    key = host.cuda().half()
    value = (host * 2).cuda().half()
    slots = torch.tensor([33, -1, 5], dtype=torch.int64, device="cuda")
    _write(codec, key, value, view, slots)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _write(codec, key, value, view, slots)
    for multiplier, indices in [(1, [33, -1, 5]), (3, [17, 35, -1])]:
        key.copy_((host * multiplier).half())
        value.copy_((host * (multiplier + 1)).half())
        slots.copy_(torch.tensor(indices, device="cuda"))
        before = carrier.clone()
        graph.replay()
        expected = before.clone()
        expected_view = KVCacheView.bind(expected, codec, dim)
        for token, slot in enumerate(indices):
            if slot < 0:
                continue
            page, offset = divmod(slot, 16)
            for data, scales, source in (
                (expected_view.key, expected_view.key_scale, key),
                (expected_view.value, expected_view.value_scale, value),
            ):
                encoded, scale = codec.encode(source[token].cpu())
                data[page, offset].copy_(encoded)
                scales[page, offset].copy_(scale)
        assert torch.equal(carrier, expected)


@pytest.mark.parametrize("codec", [INT8, INT8_PER_TOKEN_HEAD], ids=lambda c: c.name)
@pytest.mark.parametrize("length", [15, 16, 17, 255, 256, 257, 1023, 1024, 1025])
@pytest.mark.parametrize("algorithm", ["triton", "scalar", "xqa"])
def test_paged_attention_noncontiguous_pages_and_partition_edges(
    codec, length, algorithm
):
    dim, heads, gqa, block = 256 if algorithm == "xqa" else 128, 2, 4, 16
    generator = torch.Generator().manual_seed(20261011)
    key = torch.randn(length, heads, dim, generator=generator).half().cuda()
    value = (torch.randn(length, heads, dim, generator=generator) * 0.75).half().cuda()
    query = torch.randn(1, heads * gqa, dim, generator=generator).half().cuda()
    blocks = (length + block - 1) // block
    _, view = _cache(codec, blocks + 3, block, heads, dim)
    order = torch.randperm(blocks, generator=generator) + 2
    table = order.to(device="cuda", dtype=torch.int32)[None]
    token = torch.arange(length, device="cuda")
    slots = table[0, token // block].long() * block + token % block
    _write(codec, key, value, view, slots)
    out = torch.empty_like(query)
    lengths = torch.tensor([length], dtype=torch.int32, device="cuda")
    if algorithm != "triton":
        from flash_attn_v100 import flash_attn_decode_paged, flash_attn_decode_paged_xqa

        operation = (
            flash_attn_decode_paged_xqa
            if algorithm == "xqa"
            else flash_attn_decode_paged
        )
        assert codec.name in operation.kv_cache_codecs, (
            "The installed native artifact must advertise the tested codec"
        )
        operation(
            query,
            view.key,
            view.value,
            table,
            lengths,
            out=out,
            kv_cache_dtype=codec.name,
            softmax_scale=dim**-0.5,
            k_scale_cache=view.key_scale,
            v_scale_cache=view.value_scale,
            max_seq_len_hint=length,
            workspace_seq_capacity_hint=length,
            partition_size_hint=256,
        )
    else:
        unified_attention(
            q=query,
            k=view.key,
            v=view.value,
            out=out,
            cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
            max_seqlen_q=1,
            seqused_k=lengths,
            max_seqlen_k=length,
            softmax_scale=dim**-0.5,
            causal=True,
            window_size=(-1, -1),
            block_table=table,
            softcap=0.0,
            q_descale=None,
            k_descale=None,
            v_descale=None,
            kv_quant_mode=get_kv_quant_mode(codec.name),
            k_scale_cache=view.key_scale,
            v_scale_cache=view.value_scale,
        )
    kq, ks = codec.encode(key.cpu())
    vq, vs = codec.encode(value.cpu())
    k = codec.dequantize(kq, ks, torch.float32).double().repeat_interleave(gqa, 1)
    v = codec.dequantize(vq, vs, torch.float32).double().repeat_interleave(gqa, 1)
    q = query.cpu().double()[0]
    logits = torch.einsum("hd,thd->ht", q, k) * dim**-0.5
    expected = torch.einsum("ht,thd->hd", logits.softmax(-1), v)
    # Shared attention arithmetic, after quantization: absolute+relative bound.
    torch.testing.assert_close(out.cpu()[0].double(), expected, atol=0.004, rtol=0.015)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("codec", [INT8, INT8_PER_TOKEN_HEAD], ids=lambda c: c.name)
@pytest.mark.parametrize("gqa,partition", [(4, 256), (6, 256), (6, 512), (8, 1024)])
def test_xqa_graph_changes_lengths_pages_queries_and_scales(codec, gqa, partition):
    from flash_attn_v100 import flash_attn_decode_paged_xqa

    dim, heads, block, capacity, batch = 256, 2, 16, 1025, 2
    generator = torch.Generator().manual_seed(20261011)
    blocks = (capacity + block - 1) // block
    _, view = _cache(codec, batch * blocks + 2, block, heads, dim)
    query = torch.randn(batch, heads * gqa, dim, generator=generator).half().cuda()
    key = torch.randn(batch * capacity, heads, dim, generator=generator)
    value = torch.randn(batch * capacity, heads, dim, generator=generator)
    key *= torch.linspace(0.25, 4, batch * capacity)[:, None, None]
    value *= torch.linspace(8, 0.125, batch * capacity)[:, None, None]
    key, value = key.half().cuda(), value.half().cuda()
    order = (torch.randperm(batch * blocks, generator=generator) + 2).reshape(batch, -1)
    table = order.to(device="cuda", dtype=torch.int32)
    token = torch.arange(capacity, device="cuda")
    slots = (table[:, token // block].long() * block + token % block).flatten()
    _write(codec, key, value, view, slots)
    lengths = torch.tensor([capacity, 17], device="cuda", dtype=torch.int32)
    out = torch.empty_like(query)

    def run():
        flash_attn_decode_paged_xqa(
            query,
            view.key,
            view.value,
            table,
            lengths,
            out=out,
            kv_cache_dtype=codec.name,
            k_scale_cache=view.key_scale,
            v_scale_cache=view.value_scale,
            max_seq_len_hint=capacity,
            workspace_seq_capacity_hint=capacity,
            partition_size_hint=partition,
        )

    run()
    torch.accelerator.synchronize()
    if gqa == 6 and partition == 256:
        # Observe GPU launches, not just the Python binding or route name.
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as trace:
            run()
            torch.accelerator.synchronize()
        kernels = [
            event.name
            for event in trace.events()
            if event.device_type == torch.autograd.DeviceType.CUDA
        ]
        assert any(
            "flash_attention_decode_xqa_tc_partition_kernel_256_wide" in k
            for k in kernels
        )
        assert not any("flash_attention_decode_partition_kernel" in k for k in kernels)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for turn, live_lengths in enumerate(([15, 513], [1025, 257], [512, 16])):
        lengths.copy_(torch.tensor(live_lengths, device="cuda", dtype=torch.int32))
        table.copy_(order.flip(0) if turn % 2 == 0 else order)
        query.mul_(-0.75)
        # Changes to scale contents must be read at replay, not bound at capture.
        view.key_scale.mul_(0.875)
        view.value_scale.mul_(1.125)
        graph.replay()
        torch.accelerator.synchronize()
        for row, length in enumerate(live_lengths):
            positions = torch.arange(length, device="cuda")
            pages = table[row, positions // block].long()
            offsets = positions % block
            decoded = []
            for data, scales in (
                (view.key, view.key_scale),
                (view.value, view.value_scale),
            ):
                decoded.append(
                    (
                        data[pages, offsets].double()
                        * scales[pages, offsets, :, None].double()
                    )
                    .cpu()
                    .repeat_interleave(gqa, 1)
                )
            k, v = decoded
            logits = (
                torch.einsum("hd,thd->ht", query[row].cpu().double(), k) * dim**-0.5
            )
            expected = torch.einsum("ht,thd->hd", logits.softmax(-1), v)
            torch.testing.assert_close(
                out[row].cpu().double(), expected, atol=0.004, rtol=0.015
            )
        assert torch.isfinite(out).all()


@pytest.mark.parametrize("codec", [INT8, INT8_PER_TOKEN_HEAD], ids=lambda c: c.name)
@pytest.mark.parametrize("invalid_plane", ["key_scale", "value_scale"])
def test_xqa_preserves_invalid_rows_without_contaminating_other_heads(
    codec, invalid_plane
):
    from flash_attn_v100 import flash_attn_decode_paged_xqa

    _, view = _cache(codec, 1, 16, 2, 256)
    view.key.fill_(1)
    view.value.fill_(1)
    view.key_scale.fill_(0.125)
    view.value_scale.fill_(0.25)
    getattr(view, invalid_plane)[0, 3, 0] = float("nan")
    query = torch.ones((1, 8, 256), device="cuda", dtype=torch.float16)
    out = flash_attn_decode_paged_xqa(
        query,
        view.key,
        view.value,
        torch.zeros((1, 1), device="cuda", dtype=torch.int32),
        torch.tensor([16], device="cuda", dtype=torch.int32),
        kv_cache_dtype=codec.name,
        k_scale_cache=view.key_scale,
        v_scale_cache=view.value_scale,
        max_seq_len_hint=16,
        workspace_seq_capacity_hint=16,
    )
    assert out[0, :4].isnan().all()
    assert torch.equal(out[0, 4:], torch.full_like(out[0, 4:], 0.25))
