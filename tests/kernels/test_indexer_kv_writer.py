# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native group writers, physical pages and mutable graph inputs."""

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.v1.attention.kv_cache_view import KVCacheView
from vllm.v1.attention.kv_codecs import kv_codecs_for_representation

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 7,
    reason="requires SM70/SM75 native cache operators",
)


def _cache(codec, head_size, padding=64):
    pages, block = 3, 16
    row_bytes = head_size + (head_size // 128) * 4
    stride = block * row_bytes + padding
    storage = torch.full((pages * stride,), 77, device="cuda", dtype=torch.uint8)
    carrier = storage.as_strided((pages, block, row_bytes), (stride, row_bytes, 1))
    return storage, carrier, KVCacheView.bind_indexer(carrier, codec, head_size)


def _reference(source, codec):
    groups = source.cpu().float().unflatten(-1, (-1, 128))
    if codec.element_dtype == torch.int8:
        scale = (groups.abs().amax(-1) / 127).clamp_min(1e-6)
        payload = (groups / scale[..., None]).round().clamp(-128, 127).to(torch.int8)
    else:
        scale = groups.abs().amax(-1).clamp_min(1e-4) / 448
        payload = (groups / scale[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return payload.flatten(-2).view(torch.uint8), scale


@pytest.mark.parametrize(
    "codec", kv_codecs_for_representation("indexer"), ids=lambda c: c.name
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("head_size", [128, 256])
def test_native_writer_uses_group_scales_and_physical_page_strides(
    codec, dtype, head_size
):
    torch.manual_seed(1178)
    source = torch.randn(12, head_size, device="cuda", dtype=dtype)[::2]
    source[:, 128:].mul_(8)
    source[0].zero_()
    slots = torch.tensor([0, 15, 17, 32, 47, -1], device="cuda", dtype=torch.int64)
    storage, cache, view = _cache(codec, head_size)
    before = storage.clone()
    ops.indexer_k_quant_and_cache(
        source, cache, slots, 128, "float32", cache_dtype=codec.name
    )
    expected, scales = _reference(source, codec)
    assert view.key_scale is not None
    for row, slot in enumerate(slots.cpu().tolist()):
        if slot < 0:
            continue
        page, position = divmod(slot, 16)
        assert torch.equal(
            view.key[page, position, 0].view(torch.uint8).cpu(), expected[row]
        )
        torch.testing.assert_close(
            view.key_scale[page, position].cpu(),
            scales[row],
            rtol=0 if codec.name == "int8" else 1e-7,
            atol=0,
        )
    for page in range(3):
        end = (page + 1) * view.page_stride_bytes
        assert torch.equal(storage[end - 64 : end], before[end - 64 : end])
    assert torch.equal(
        view.key[0, 1].view(torch.uint8),
        torch.full_like(view.key[0, 1], 77, dtype=torch.uint8),
    )


def test_int8_writer_rounds_ties_and_propagates_invalid_groups():
    codec = next(c for c in kv_codecs_for_representation("indexer") if c.name == "int8")
    _, cache, view = _cache(codec, 256)
    source = torch.zeros(2, 256, device="cuda", dtype=torch.float32)
    source[0, :7] = torch.tensor([127, 0.5, 1.5, 2.5, -0.5, -1.5, -2.5], device="cuda")
    source[0, 128:135] = source[0, :7] * 2
    source[1, 0] = float("inf")
    source[1, 128] = 127
    ops.indexer_k_quant_and_cache(
        source,
        cache,
        torch.tensor([0, 1], device="cuda"),
        128,
        "float32",
        cache_dtype="int8",
    )
    assert view.key[0, 0, 0, :7].tolist() == [127, 0, 2, 2, 0, -2, -2]
    assert view.key_scale is not None
    assert view.key_scale[0, 0].tolist() == [1.0, 2.0]
    assert torch.isnan(view.key_scale[0, 1, 0])
    assert view.key_scale[0, 1, 1].item() == 1.0


@pytest.mark.parametrize(
    "codec", kv_codecs_for_representation("indexer"), ids=lambda c: c.name
)
def test_native_writer_replays_with_changed_source_and_slots(codec):
    _, cache, view = _cache(codec, 256)
    source = torch.zeros(2, 256, device="cuda", dtype=torch.float32)
    source[0].fill_(127)
    source[1].fill_(254)
    slots = torch.tensor([0, 17], device="cuda", dtype=torch.int64)

    def write():
        ops.indexer_k_quant_and_cache(
            source, cache, slots, 128, "float32", cache_dtype=codec.name
        )

    write()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        write()
    source.mul_(-0.5)
    slots.copy_(torch.tensor([32, 47], device="cuda"))
    graph.replay()
    expected, scales = _reference(source, codec)
    assert view.key_scale is not None
    for row, position in enumerate((0, 15)):
        assert torch.equal(
            view.key[2, position, 0].view(torch.uint8).cpu(), expected[row]
        )
        torch.testing.assert_close(
            view.key_scale[2, position].cpu(),
            scales[row],
            rtol=0 if codec.name == "int8" else 1e-7,
            atol=0,
        )


def test_writer_executes_the_native_codec_binding():
    codec = next(c for c in kv_codecs_for_representation("indexer") if c.name == "int8")
    _, cache, _ = _cache(codec, 128)
    source = torch.ones(2, 128, device="cuda")
    slots = torch.tensor([0, 1], device="cuda")
    assert ops._indexer_kv_writer() is not None
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as trace:
        ops.indexer_k_quant_and_cache(
            source, cache, slots, 128, "float32", cache_dtype="int8"
        )
        torch.accelerator.synchronize()
    assert any(
        "indexer_k_quant_and_cache_kernel" in event.name for event in trace.events()
    )
