# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mixed MLA keeps grouped latent and exact BF16 RoPE in shared algorithms."""

import pytest
import torch

from vllm.models.deepseek_v4.common.ops.cache_utils import (
    dequantize_and_gather_k_cache_triton,
    mixed_mla_cache_view,
    quantize_and_insert_k_cache,
)
from vllm.models.deepseek_v4.common.ops.sparse_decode_bmm import (
    sparse_attn_decode_bmm,
    sparse_decode_bmm_workspace_specs,
)
from vllm.models.deepseek_v4.sm70.sparse_kernels import (
    sm70_sparse_attention_paged,
    sm70_sparse_attention_paged_splitk,
    sm70_sparse_attention_paged_splitk_qk_dsplit,
)
from vllm.v1.attention.kv_cache_view import MLAPageLayout
from vllm.v1.attention.kv_codecs import kv_codecs_for_representation

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="Mixed MLA qualification on SM70",
)
CODECS = kv_codecs_for_representation("mla_mixed")


def cache(codec, page, padding=0):
    layout = MLAPageLayout(448, 64, scale_bytes=codec.scale_dtype.itemsize)
    stride = layout.page_size_bytes(page) + padding
    storage = torch.full((4 * stride,), 43, dtype=codec.storage_dtype, device="cuda")
    carrier = storage.as_strided(
        (4, page, layout.slot_bytes), (stride, layout.slot_bytes, 1)
    )
    return carrier, mixed_mla_cache_view(carrier, codec.name), storage


def encode_reference(codec, values):
    latent = values[..., :448].float().reshape(-1, 7, 64)
    if codec.native_id == 3:
        raw, scales = codec.encode(values[..., :448])
    else:
        exponents = torch.ceil(torch.log2(latent.abs().amax(-1).clamp_min(1e-4) / 448))
        scales = (exponents + 127).clamp(0, 255).byte()
        raw = (latent / torch.exp2(exponents)[..., None]).clamp(-448, 448)
        raw = raw.to(codec.element_dtype).view(codec.storage_dtype).reshape(-1, 448)
    return raw, scales, values[..., 448:].to(torch.bfloat16)


def decoded_reference(view, *, half_scale_first):
    raw = view.key[:, :, 0].view(view.codec.element_dtype).float().reshape(-1, 7, 64)
    scales = view.key_scale.reshape(-1, 7)
    if view.codec.native_id == 1:
        scales = torch.exp2(scales.float() - 127)
    if half_scale_first and view.codec.native_id == 1:
        latent = (raw.half() * scales[..., None].half()).reshape(-1, 448)
    else:
        latent = (raw * scales[..., None]).reshape(-1, 448).half()
    rope = view.unquantized_field("rope")[:, :, 0].reshape(-1, 64).half()
    return torch.cat((latent, rope), dim=-1).double()


@pytest.mark.parametrize("codec", CODECS, ids=lambda c: c.name)
@pytest.mark.parametrize("page,padding", [(64, 0), (128, 576)])
@pytest.mark.parametrize("input_dtype", [torch.float16, torch.bfloat16])
def test_mixed_writer_gather_exact_fields_padding_and_graph(
    codec, page, padding, input_dtype
):
    torch.manual_seed(20261012)
    carrier, view, storage = cache(codec, page, padding)
    values = torch.randn((9, 512), device="cuda").to(input_dtype)
    values[:, :448] *= torch.arange(1, 8, device="cuda").repeat_interleave(64)[None, :]
    slots = torch.tensor(
        [0, page - 1, page, 2 * page + 3, -1, -2, 4 * page, 1, 2], device="cuda"
    )

    def launch():
        quantize_and_insert_k_cache(
            values, carrier, slots, block_size=page, kv_cache_dtype=codec.name
        )

    launch()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    for repeat in range(2):
        values.mul_(0.95)
        values[repeat].zero_()
        slots.copy_(slots.flip(0))
        old = storage.clone()
        graph.replay()
        raw, scales, rope = encode_reference(codec, values)
        touched = set()
        for row, slot in enumerate(slots.cpu().tolist()):
            if not 0 <= slot < 4 * page:
                continue
            block, offset = divmod(slot, page)
            touched.add(slot)
            assert torch.equal(view.key[block, offset, 0], raw[row])
            assert torch.equal(view.key_scale[block, offset], scales[row])
            assert torch.equal(
                view.unquantized_field("rope")[block, offset, 0], rope[row]
            )
        untouched = torch.ones((4, page), dtype=torch.bool, device="cuda")
        untouched.reshape(-1)[list(touched)] = False
        old_view = mixed_mla_cache_view(
            old.as_strided(carrier.shape, carrier.stride()), codec.name
        )
        assert torch.equal(view.key[untouched], old_view.key[untouched])
        assert torch.equal(view.key_scale[untouched], old_view.key_scale[untouched])
        assert torch.equal(
            view.unquantized_field("rope")[untouched],
            old_view.unquantized_field("rope")[untouched],
        )
        for block in range(4):
            start = block * carrier.stride(0) + view.page_size_bytes
            assert torch.equal(
                storage[start : (block + 1) * carrier.stride(0)],
                old[start : (block + 1) * carrier.stride(0)],
            )
        # Non-contiguous logical page table restores latent/scales/RoPE together.
        table = torch.tensor([[2, 0, 3, 1]], dtype=torch.int32, device="cuda")
        lengths = torch.tensor([4 * page], dtype=torch.int32, device="cuda")
        out = torch.empty((1, 4 * page, 512), dtype=torch.float16, device="cuda")
        # Fill untouched logical slots first; the writer graph above still checks
        # invalid-slot isolation by comparing all owner bytes to its previous state.
        full_values = (
            torch.randn((4 * page, 512), device="cuda", dtype=input_dtype) * 0.2
        )
        quantize_and_insert_k_cache(
            full_values,
            carrier,
            torch.arange(4 * page, device="cuda"),
            block_size=page,
            kv_cache_dtype=codec.name,
        )
        dequantize_and_gather_k_cache_triton(
            out, carrier, lengths, None, table, page, 0, kv_cache_dtype=codec.name
        )
        expected = (
            decoded_reference(view, half_scale_first=False)
            .reshape(4, page, 512)[table[0].long()]
            .reshape_as(out)
        )
        torch.testing.assert_close(out.double(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("codec", CODECS, ids=lambda c: c.name)
@pytest.mark.parametrize(
    "algorithm,rows,block_d",
    [
        ("paged", 1, 64),
        ("paged", 7, 64),
        ("splitk", 8, 64),
        ("dsplit", 8, 32),
        ("dsplit", 8, 64),
        ("dsplit", 8, 128),
        ("bmm", 1, 64),
        ("bmm", 8, 64),
    ],
)
@pytest.mark.parametrize("has_extra", [False, True])
def test_mixed_algorithms_share_groups_masks_and_changed_input_graphs(
    codec, algorithm, rows, block_d, has_extra
):
    torch.manual_seed(20261012)
    page = 64
    main, main_view, _ = cache(codec, page, 576)
    extra, extra_view, _ = cache(codec, page, 0)
    for carrier in (main, extra):
        values = torch.randn((4 * page, 512), device="cuda", dtype=torch.float16) * 0.15
        values[:, :448] *= torch.arange(1, 8, device="cuda").repeat_interleave(64)[
            None, :
        ]
        quantize_and_insert_k_cache(
            values,
            carrier,
            torch.arange(4 * page, device="cuda"),
            block_size=page,
            kv_cache_dtype=codec.name,
        )
    query = torch.randn((rows, 12, 512), device="cuda", dtype=torch.float16) * 0.2
    sink = torch.randn((12,), device="cuda")
    main_indices = torch.randperm(4 * page, device="cuda")[:19].int().repeat(rows, 1)
    main_indices[:, -2:] = torch.tensor([-1, 4 * page + 1], device="cuda")
    extra_indices = torch.randperm(4 * page, device="cuda")[:11].int().repeat(rows, 1)
    extra_indices[:, -1] = -2
    main_indices[main_indices == 0] = -1
    extra_indices[extra_indices == 0] = -1
    main_view.unquantized_field("rope")[0, 0].fill_(float("nan"))
    extra_view.unquantized_field("rope")[0, 0].fill_(float("nan"))
    main_lens = torch.full((rows,), 19, dtype=torch.int32, device="cuda")
    extra_lens = torch.full((rows,), 11, dtype=torch.int32, device="cuda")
    output = torch.empty_like(query)
    partials = 2 + int(has_extra)
    states = [torch.empty((rows, 12, partials), device="cuda") for _ in range(2)]
    accum = torch.empty((rows, 12, partials, 512), device="cuda")
    qk = torch.empty((rows, 12, partials, 512 // block_d, 16), device="cuda")
    probabilities = torch.empty(
        (rows, 12, partials, 16), dtype=torch.float16, device="cuda"
    )
    workspace = [
        torch.empty(shape, dtype=dtype, device="cuda")
        for shape, dtype in sparse_decode_bmm_workspace_specs(
            rows, 12, 512, 19, 11 if has_extra else 0, torch.float16
        )
    ]

    def launch():
        args = (
            query,
            main,
            main_indices,
            main_lens,
            512**-0.5,
            sink,
            output,
            extra if has_extra else None,
            extra_indices if has_extra else None,
            extra_lens if has_extra else None,
        )
        if algorithm == "paged":
            sm70_sparse_attention_paged(*args, kv_cache_dtype=codec.name)
        elif algorithm == "splitk":
            sm70_sparse_attention_paged_splitk(
                *args, *states, accum, kv_cache_dtype=codec.name
            )
        elif algorithm == "dsplit":
            sm70_sparse_attention_paged_splitk_qk_dsplit(
                *args,
                qk,
                *states,
                probabilities,
                accum,
                block_d=block_d,
                kv_cache_dtype=codec.name,
            )
        else:
            sparse_attn_decode_bmm(
                query,
                main,
                main_indices,
                main_lens,
                extra if has_extra else None,
                extra_indices if has_extra else None,
                extra_lens if has_extra else None,
                512**-0.5,
                sink,
                output,
                *workspace,
                kv_cache_dtype=codec.name,
            )

    launch()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    for repeat in range(2):
        main_indices.copy_(main_indices.flip(-1))
        main_lens[repeat % rows] = 0
        extra_lens[repeat % rows] = 0
        query.mul_(0.95)
        graph.replay()
        keys = decoded_reference(main_view, half_scale_first=algorithm != "bmm")
        extra_keys = decoded_reference(extra_view, half_scale_first=algorithm != "bmm")
        expected = torch.zeros_like(query, dtype=torch.float64)
        for row in range(rows):
            ids = main_indices[row, : main_lens[row]].long()
            selected = keys[ids[(ids >= 0) & (ids < 4 * page)]]
            if has_extra:
                ids = extra_indices[row, : extra_lens[row]].long()
                selected = torch.cat(
                    (selected, extra_keys[ids[(ids >= 0) & (ids < 4 * page)]])
                )
            logits = query[row].double() @ selected.T * (512**-0.5)
            logits = torch.cat((logits, sink.double()[:, None]), dim=-1)
            expected[row] = logits.softmax(-1)[:, :-1] @ selected
        assert torch.isfinite(output).all()
        # Existing BMM qualification is tighter than generic paged MLA.
        atol, rtol = (0.002, 0.002) if algorithm == "bmm" else (0.004, 0.03)
        torch.testing.assert_close(output.double(), expected, atol=atol, rtol=rtol)
