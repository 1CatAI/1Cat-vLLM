# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU contracts for the shared writer, Triton attention and native scalar.

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
@pytest.mark.parametrize("native", [False, True], ids=["triton", "native_scalar"])
def test_paged_attention_noncontiguous_pages_and_partition_edges(codec, length, native):
    dim, heads, gqa, block = 128, 2, 4, 16
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
    if native:
        from flash_attn_v100 import flash_attn_decode_paged

        assert codec.name in flash_attn_decode_paged.kv_cache_codecs, (
            "The installed native artifact must advertise the tested codec"
        )
        flash_attn_decode_paged(
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
