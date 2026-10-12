# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One NoPE MLA algorithm family for declared grouped cache codecs.

The existing MLA bound stays atol=0.004, rtol=0.03. Cache publication is exact;
task quality and engine performance are separate gates.
"""

import pytest
import torch

from vllm.models.glm5next.sm70.fp8_kv import (
    sm70_glm5_kv_insert,
    sm70_glm5_sparse_attention_paged,
    sm70_glm5_sparse_attention_paged_batched_gemm,
    sm70_glm5_sparse_attention_paged_gemm,
)
from vllm.v1.attention.kv_cache_view import KVCacheView, MLAPageLayout
from vllm.v1.attention.kv_codecs import kv_codecs_for_representation

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="Grouped MLA qualification on SM70",
)
CODECS = kv_codecs_for_representation("mla_latent")


def cache(codec, page, padding):
    layout = MLAPageLayout(512, scale_bytes=codec.scale_dtype.itemsize)
    stride = layout.page_size_bytes(page) + padding
    storage = torch.full((4 * stride,), 57, dtype=codec.storage_dtype, device="cuda")
    carrier = storage.as_strided(
        (4, page, layout.slot_bytes), (stride, layout.slot_bytes, 1)
    )
    return carrier, KVCacheView.bind_mla(carrier, codec, 512)


@pytest.mark.parametrize("codec", CODECS, ids=lambda c: c.name)
@pytest.mark.parametrize("page", [16, 64])
@pytest.mark.parametrize("padding", [0, 128])
def test_mla_group_writer_exact_scales_slots_and_graph_replay(codec, page, padding):
    torch.manual_seed(20261012)
    carrier, view = cache(codec, page, padding)
    values = torch.randn((11, 512), device="cuda", dtype=torch.float16)
    values *= torch.arange(1, 9, device="cuda").repeat_interleave(64)[None, :]
    slots = torch.tensor(
        [0, page - 1, page, 2 * page + 1, 3 * page, -1, 4 * page, 1, 2, 3, 4],
        device="cuda",
        dtype=torch.int64,
    )

    def launch():
        sm70_glm5_kv_insert(values, carrier, slots, kv_cache_dtype=codec.name)

    launch()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    for repeat in range(2):
        values.mul_(0.9)
        values[repeat].zero_()
        slots.copy_(slots.flip(0))
        before = carrier.clone()
        page_bytes = carrier.as_strided((4, carrier.stride(0)), (carrier.stride(0), 1))
        padding_before = page_bytes[:, page * carrier.shape[-1] :].clone()
        graph.replay()
        assert torch.equal(page_bytes[:, page * carrier.shape[-1] :], padding_before)
        for row, slot in enumerate(slots.cpu().tolist()):
            if slot < 0 or slot >= 4 * page:
                continue
            block, offset = divmod(slot, page)
            if codec.native_id == 3:
                raw, scale = codec.encode(values[row])
            else:
                grouped = values[row].float().reshape(8, 64)
                exponent = torch.ceil(
                    torch.log2(grouped.abs().amax(-1).clamp_min(1e-4) / 448)
                )
                scale = (exponent + 127).clamp(0, 255).byte()
                raw = (
                    (grouped / torch.exp2(exponent)[:, None])
                    .clamp(-448, 448)
                    .to(codec.element_dtype)
                    .view(torch.uint8)
                    .reshape(512)
                )
            assert torch.equal(view.key[block, offset, 0], raw)
            assert torch.equal(view.key_scale[block, offset], scale)
        # No invalid slot may modify page padding or unassigned payload/scale rows.
        touched = {slot for slot in slots.cpu().tolist() if 0 <= slot < 4 * page}
        old = KVCacheView.bind_mla(before, codec, 512)
        for slot in range(4 * page):
            if slot not in touched:
                block, offset = divmod(slot, page)
                assert torch.equal(view.key[block, offset], old.key[block, offset])
                assert torch.equal(
                    view.key_scale[block, offset], old.key_scale[block, offset]
                )


@pytest.mark.parametrize("codec", CODECS, ids=lambda c: c.name)
@pytest.mark.parametrize(
    "algorithm,rows", [("paged", 1), ("paged", 9), ("gemm", 1), ("bmm", 8)]
)
@pytest.mark.parametrize("page,padding", [(16, 128), (64, 0)])
def test_mla_algorithms_share_codecs_masks_and_changed_input_graphs(
    codec, algorithm, rows, page, padding
):
    torch.manual_seed(20261012)
    carrier, view = cache(codec, page, padding)
    values = torch.randn((4 * page, 512), device="cuda", dtype=torch.float16) * 0.25
    sm70_glm5_kv_insert(
        values,
        carrier,
        torch.arange(4 * page, device="cuda"),
        kv_cache_dtype=codec.name,
    )
    query = torch.randn((rows, 12, 512), device="cuda", dtype=torch.float16) * 0.2
    width = 35
    indices = torch.randperm(4 * page, device="cuda")[:width].int().repeat(rows, 1)
    indices[:, -1] = -1
    lengths = torch.full((rows,), width, device="cuda", dtype=torch.int32)
    output = torch.empty_like(query)
    gathered = torch.empty((rows, width, 512), device="cuda", dtype=torch.float16)
    scores = torch.empty((rows, 12, width), device="cuda", dtype=torch.float16)
    probs = torch.empty_like(scores)

    def launch():
        args = (query, carrier, indices, lengths, 512**-0.5, output)
        if algorithm == "paged":
            sm70_glm5_sparse_attention_paged(*args, kv_cache_dtype=codec.name)
        elif algorithm == "gemm":
            sm70_glm5_sparse_attention_paged_gemm(
                *args, gathered[0], scores[0], probs[0], kv_cache_dtype=codec.name
            )
        else:
            sm70_glm5_sparse_attention_paged_batched_gemm(
                *args, gathered, scores, probs, kv_cache_dtype=codec.name
            )

    launch()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    for repeat in range(2):
        indices.copy_(indices.flip(-1))
        lengths[repeat % rows] = 0
        query.mul_(0.95)
        graph.replay()
        decoded = codec.dequantize(view.key[:, :, 0], view.key_scale, torch.float64)
        decoded = decoded.reshape(-1, 512)
        reference = torch.zeros_like(query, dtype=torch.float64)
        for row in range(rows):
            selected = indices[row, : lengths[row]].long()
            selected = selected[(selected >= 0) & (selected < 4 * page)]
            if selected.numel():
                logits = query[row].double() @ decoded[selected].T * (512**-0.5)
                reference[row] = logits.softmax(-1) @ decoded[selected]
        assert torch.isfinite(output).all()
        torch.testing.assert_close(output.double(), reference, atol=0.004, rtol=0.03)
