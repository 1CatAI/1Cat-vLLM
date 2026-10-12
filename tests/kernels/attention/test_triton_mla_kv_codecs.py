# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Concatenated MLA codecs share native writers, gathers and Triton decode."""

from types import SimpleNamespace

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.v1.attention.backends.mla.triton_mla import TritonMLAImpl
from vllm.v1.attention.kv_cache_view import KVCacheView
from vllm.v1.attention.kv_codecs import kv_codecs_for_representation

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="Concatenated MLA qualification on SM70",
)
CODECS = kv_codecs_for_representation("mla_concat")


def allocate(codec, dim, page, padding=16, dense_dtype=None, blocks=8):
    slot = dim + (4 if codec.native_id == 3 else 0)
    dtype = (
        torch.float8_e4m3fn
        if codec.native_id == 1
        else (dense_dtype or codec.storage_dtype)
        if codec.native_id == 0
        else codec.storage_dtype
    )
    size = torch.empty(0, dtype=dtype).element_size()
    stride = page * slot + padding // size
    owner = torch.empty(blocks * stride, dtype=dtype, device="cuda")
    owner.view(torch.uint8).fill_(43)
    carrier = owner.as_strided((blocks, page, slot), (stride, slot, 1))
    scale = torch.tensor(0.03125, device="cuda", dtype=torch.float32)
    view = KVCacheView.bind_mla_concat(carrier, codec, dim, scale)
    return owner, carrier, view, scale


def reconstructed(view, scale):
    if view.codec.native_id == 3:
        return (view.key.squeeze(2).double() * view.key_scale.double()).half().double()
    if view.codec.native_id == 1:
        return (
            (view.key.view(torch.float8_e4m3fn).float() * scale)
            .half()
            .squeeze(2)
            .double()
        )
    return view.key.squeeze(2).double()


@pytest.mark.parametrize("codec", CODECS, ids=lambda c: c.name)
@pytest.mark.parametrize("page", [16, 64])
@pytest.mark.parametrize("dim", [320, 576])
def test_native_concat_writer_prefix_gather_and_changed_graph(codec, page, dim):
    torch.manual_seed(20261012)
    owner, cache, view, scale = allocate(codec, dim, page)
    input_dtype = codec.storage_dtype if codec.native_id == 0 else torch.float16
    values = torch.randn((8 * page, dim), device="cuda", dtype=input_dtype) * 0.2
    slots = torch.arange(8 * page, device="cuda", dtype=torch.int64)
    ops.concat_and_cache_mla(
        values[:, :-64], values[:, -64:], cache, slots, codec.name, scale
    )
    before = owner.clone()
    changed = values[:4].clone()
    changed_slots = torch.tensor([page + 3, -2, 8 * page, -1], device="cuda")

    def write():
        ops.concat_and_cache_mla(
            changed[:, :-64], changed[:, -64:], cache, changed_slots, codec.name, scale
        )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        write()
    changed.zero_()
    graph.replay()
    torch.accelerator.synchronize()
    expected = before.clone()
    expected_carrier = expected.as_strided(cache.shape, cache.stride())
    expected_view = KVCacheView.bind_mla_concat(expected_carrier, codec, dim, scale)
    expected_view.key[1, 3].zero_()
    if codec.native_id == 3:
        expected_view.key_scale[1, 3].fill_(1e-6)
        q, scales = codec.encode(values)
        actual_q = view.key.squeeze(2).reshape(8 * page, dim)
        actual_s = view.key_scale.reshape(8 * page)
        selected = torch.arange(8 * page, device="cuda") != page + 3
        assert torch.equal(actual_q[selected], q[selected])
        assert torch.equal(actual_s[selected], scales[selected])
    assert torch.equal(owner.view(torch.uint8), expected.view(torch.uint8))

    table = torch.tensor([[5, 1, 6], [2, 7, 3]], device="cuda", dtype=torch.int32)
    counts = torch.tensor([0, 19, 26], device="cuda", dtype=torch.int32)
    token_to_seq = torch.tensor([0] * 19 + [1] * 7, device="cuda", dtype=torch.int32)
    starts = torch.tensor([3, 2], device="cuda", dtype=torch.int32)
    output = torch.empty((29, dim), device="cuda", dtype=input_dtype)
    output.fill_(42)

    def gather():
        ops.gather_and_maybe_dequant_cache(
            cache, output, table, counts, token_to_seq, 26, codec.name, scale, starts
        )

    gather()
    gather_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gather_graph):
        gather()
    table.copy_(table.flip(-1))
    gather_graph.replay()
    keys = reconstructed(view, scale)
    rows = []
    for seq, length in ((0, 19), (1, 7)):
        for token in range(length):
            logical = token + int(starts[seq])
            rows.append(keys[int(table[seq, logical // page]), logical % page])
    torch.testing.assert_close(output[:26].double(), torch.stack(rows), atol=0, rtol=0)
    assert torch.all(output[26:] == 42)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("codec", CODECS, ids=lambda c: c.name)
@pytest.mark.parametrize("rank,rope", [(256, 32), (512, 64)])
@pytest.mark.parametrize("heads", [1, 16])
@pytest.mark.parametrize("page", [16, 64])
def test_actual_triton_mla_forward_codec_masks_and_changed_graph(
    codec, rank, rope, heads, page, query_dtype=torch.float16
):
    torch.manual_seed(20261012)
    dim = rank + rope
    _, cache, view, scale = allocate(codec, dim, page)
    input_dtype = codec.storage_dtype if codec.native_id == 0 else torch.float16
    values = torch.randn((8 * page, dim), device="cuda", dtype=input_dtype) * 0.2
    slots = torch.arange(8 * page, device="cuda", dtype=torch.int64)
    ops.concat_and_cache_mla(
        values[:, :rank], values[:, rank:], cache, slots, codec.name, scale
    )
    query = torch.randn((2, heads, dim), device="cuda", dtype=query_dtype) * 0.2
    table = torch.tensor([[5, 1, 6, 2], [3, 7, 2, 1]], device="cuda", dtype=torch.int32)
    lengths = torch.tensor([3, 49], device="cuda", dtype=torch.int32)
    metadata = SimpleNamespace(
        max_seq_len=1536, decode=SimpleNamespace(block_table=table, seq_lens=lengths)
    )
    impl = object.__new__(TritonMLAImpl)
    impl.head_size, impl.kv_lora_rank = dim, rank
    impl.kv_cache_dtype, impl.scale, impl._sm_count = codec.name, dim**-0.5, 80
    layer = SimpleNamespace(_k_scale=scale)
    impl.forward_mqa(query, cache, metadata, layer)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output, lse = impl.forward_mqa(query, cache, metadata, layer)
    for repeat in range(2):
        table.copy_(table.flip(-1))
        query.mul_(0.95)
        lengths[0] = 0 if repeat == 0 else 17
        graph.replay()
        if codec.native_id == 3:
            # INT8 scales participate after QK and in weighted PV, without a
            # premature dense half conversion in the independent reference.
            keys = view.key.squeeze(2).double() * view.key_scale.double()
        else:
            keys = reconstructed(view, scale)
        expected = torch.zeros_like(output, dtype=torch.float64)
        for seq in range(2):
            visible = int(lengths[seq])
            if visible:
                rows = torch.stack(
                    [
                        keys[int(table[seq, token // page]), token % page]
                        for token in range(visible)
                    ]
                )
                logits = query[seq].double() @ rows.T * impl.scale
                expected[seq] = logits.softmax(-1) @ rows[:, :rank]
                torch.testing.assert_close(
                    lse[seq].float(),
                    logits.logsumexp(-1).float(),
                    atol=0.004,
                    rtol=0.03,
                )
            else:
                assert torch.isneginf(lse[seq]).all()
        assert torch.isfinite(output).all()
        torch.testing.assert_close(output.double(), expected, atol=0.004, rtol=0.03)


@pytest.mark.parametrize("codec", CODECS, ids=lambda c: c.name)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("is_neox", [False, True])
def test_fused_rope_writer_matches_shared_encoder(codec, dtype, is_neox):
    torch.manual_seed(20261012)
    owner, cache, view, scale = allocate(codec, 576, 16, dense_dtype=dtype)
    query = torch.randn((5, 4, 64), device="cuda", dtype=dtype) * 0.2
    rope = torch.randn((5, 64), device="cuda", dtype=dtype) * 0.2
    latent = torch.randn((5, 512), device="cuda", dtype=dtype) * 0.2
    angles = torch.randn((20, 32), device="cuda", dtype=torch.float32)
    table = torch.cat((angles.cos(), angles.sin()), -1).to(dtype)
    positions = torch.tensor([3, 8, 1, 7, 5], device="cuda", dtype=torch.int64)
    slots = torch.tensor([17, 51, -1, 128, 9], device="cuda", dtype=torch.int64)
    original_q, original_k, original_owner = query.clone(), rope.clone(), owner.clone()

    def fused():
        ops.concat_and_cache_mla_rope_fused(
            positions,
            query,
            rope,
            latent,
            table,
            is_neox,
            slots,
            cache,
            codec.name,
            scale,
        )

    fused()
    valid = (slots >= 0) & (slots < 128)
    cos, sin = table[positions, :32].float(), table[positions, 32:].float()

    def rotate(source):
        expanded_cos, expanded_sin = cos, sin
        if source.ndim == 3:
            expanded_cos, expanded_sin = cos[:, None], sin[:, None]
        x = source[..., :32] if is_neox else source[..., ::2]
        y = source[..., 32:] if is_neox else source[..., 1::2]
        # Native Half/BF16 operators preserve their existing intermediate rounding.
        a = (x * expanded_cos.to(dtype)).to(dtype)
        b = (y * expanded_sin.to(dtype)).to(dtype)
        c = (y * expanded_cos.to(dtype)).to(dtype)
        d = (x * expanded_sin.to(dtype)).to(dtype)
        out = source.clone()
        if is_neox:
            out[..., :32], out[..., 32:] = a - b, c + d
        else:
            out[..., ::2], out[..., 1::2] = a - b, c + d
        return out

    torch.testing.assert_close(
        query[valid], rotate(original_q)[valid], atol=0.004, rtol=0.03
    )
    torch.testing.assert_close(
        rope[valid], rotate(original_k)[valid], atol=0.004, rtol=0.03
    )
    assert torch.equal(query[~valid], original_q[~valid])
    assert torch.equal(rope[~valid], original_k[~valid])
    expected = original_owner.clone()
    expected_cache = expected.as_strided(cache.shape, cache.stride())
    # This comparison uses the actual rotated K, isolating the cache-byte contract
    # from the bounded RoPE arithmetic differences.
    ops.concat_and_cache_mla(latent, rope, expected_cache, slots, codec.name, scale)
    assert torch.equal(owner.view(torch.uint8), expected.view(torch.uint8))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fused()
    query.zero_()
    rope.zero_()
    graph.replay()
    expected.copy_(original_owner)
    ops.concat_and_cache_mla(latent, rope, expected_cache, slots, codec.name, scale)
    assert torch.equal(owner.view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.parametrize("codec", CODECS, ids=lambda c: c.name)
def test_bf16_query_actual_triton_mla(codec):
    test_actual_triton_mla_forward_codec_masks_and_changed_graph(
        codec, 512, 64, 16, 16, query_dtype=torch.bfloat16
    )


@pytest.mark.parametrize("codec", CODECS, ids=lambda c: c.name)
def test_single_partition_mla_retains_small_weighted_tail_mass(codec):
    from vllm.v1.attention.ops.triton_decode_attention import decode_attention_fwd

    rank, dim, page, tokens = 256, 288, 64, 8192
    _, cache, view, scale = allocate(codec, dim, page, blocks=tokens // page)
    dtype = codec.storage_dtype if codec.native_id == 0 else torch.float16
    values = torch.full((tokens, dim), 0.2, device="cuda", dtype=dtype)
    values[:, rank:].zero_()
    # Non-contiguous page order, with the dominant logit encountered first.
    table = torch.arange(tokens // page - 1, -1, -1, device="cuda", dtype=torch.int32)[
        None
    ]
    values[(tokens // page - 1) * page, :rank].neg_()
    slots = torch.arange(tokens, device="cuda", dtype=torch.int64)
    ops.concat_and_cache_mla(
        values[:, :rank], values[:, rank:], cache, slots, codec.name, scale
    )
    query = torch.zeros((1, 16, dim), device="cuda", dtype=torch.float16)
    query[..., 0] = -6 * dim**0.5 / float(values[0, 0])
    output = torch.empty((1, 16, rank), device="cuda", dtype=query.dtype)
    lse = torch.empty((1, 16), device="cuda", dtype=query.dtype)
    partial = torch.empty((1, 16, 1, rank + 1), device="cuda", dtype=torch.float32)
    lengths = torch.tensor([tokens], device="cuda", dtype=torch.int32)
    decode_attention_fwd(
        query,
        view.key,
        view.key[..., :rank],
        output,
        lse,
        table,
        lengths,
        partial,
        1,
        dim**-0.5,
        page,
        k_scale=scale,
        v_scale=scale,
        is_mla=True,
        cache_view=view,
    )
    rows = (
        view.key.squeeze(2).double() * view.key_scale.double()
        if codec.native_id == 3
        else reconstructed(view, scale)
    ).reshape(tokens, dim)
    dominant, tail = rows[(tokens // page - 1) * page], rows[0]
    dominant_logit = query[0, :, 0].double() * dominant[0] * dim**-0.5
    tail_logit = query[0, :, 0].double() * tail[0] * dim**-0.5
    tail_mass = (tokens - 1) * torch.exp(tail_logit - dominant_logit)
    expected = (dominant[:rank] + tail_mass[:, None] * tail[:rank]) / (
        1 + tail_mass[:, None]
    )
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output[0].double(), expected, atol=0.004, rtol=0.03)


@pytest.mark.parametrize(
    "value,next_value", [(4062, 4064), (4064, 4062), (5000, 65504), (65504, 5000)]
)
def test_int8_large_finite_row_does_not_overflow_weighted_partial(value, next_value):
    from vllm.v1.attention.kv_codecs import resolve_kv_codec
    from vllm.v1.attention.ops.triton_decode_attention import decode_attention_fwd

    codec = resolve_kv_codec("int8", representation="mla_concat")
    _, cache, view, scale = allocate(codec, 288, 16)
    values = torch.full((16, 288), value, device="cuda", dtype=torch.float16)
    slots = torch.arange(16, device="cuda", dtype=torch.int64)
    query = torch.zeros((1, 16, 288), device="cuda", dtype=torch.float16)
    output = torch.empty((1, 16, 256), device="cuda", dtype=query.dtype)
    partial = torch.empty((1, 16, 1, 257), device="cuda", dtype=torch.float32)
    lse = torch.empty((1, 16), device="cuda", dtype=query.dtype)
    table = torch.zeros((1, 1), device="cuda", dtype=torch.int32)
    lengths = torch.tensor([16], device="cuda", dtype=torch.int32)

    def execute():
        ops.concat_and_cache_mla(
            values[:, :256], values[:, 256:], cache, slots, "int8", scale
        )
        decode_attention_fwd(
            query,
            view.key,
            view.key[..., :256],
            output,
            lse,
            table,
            lengths,
            partial,
            1,
            288**-0.5,
            16,
            k_scale=scale,
            v_scale=scale,
            is_mla=True,
            cache_view=view,
        )

    execute()
    assert torch.isfinite(output).all()
    torch.testing.assert_close(
        output, torch.full_like(output, value), atol=0.004, rtol=0.03
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        execute()
    values.fill_(next_value)
    graph.replay()
    assert torch.isfinite(output).all()
    torch.testing.assert_close(
        output, torch.full_like(output, next_value), atol=0.004, rtol=0.03
    )
