# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Grouped cache identity, independent scales and borrowed physical regions."""

import pytest
import torch

from vllm.v1.attention.kv_cache_view import KVCacheView
from vllm.v1.attention.kv_codecs import (
    FP8_E4M3,
    INT8,
    KV_REPRESENTATION_CODECS,
    resolve_kv_codec,
)

pytestmark = pytest.mark.cpu_test


def test_indexer_identity_and_rounding_keep_the_group_axis():
    codec = resolve_kv_codec("int8", representation="indexer")
    assert codec is not None
    assert codec.cache_identity != INT8.cache_identity
    fp8 = resolve_kv_codec("fp8", representation="indexer")
    assert fp8 is not None
    assert codec.cache_identity != fp8.cache_identity
    first = torch.zeros(128)
    first[:7] = torch.tensor([127, 0.5, 1.5, 2.5, -0.5, -1.5, -2.5])
    source = torch.cat([first, 2 * first])[None, :]
    encoded, scales = codec.encode(source)
    assert encoded[0, :7].tolist() == [127, 0, 2, 2, 0, -2, -2]
    assert torch.equal(encoded[:, :128], encoded[:, 128:])
    assert scales.tolist() == [[1.0, 2.0]]
    restored = codec.dequantize(encoded, scales, torch.float32)
    assert restored[0, :7].tolist() == [127, 0, 2, 2, 0, -2, -2]
    assert torch.equal(restored[:, 128:], 2 * restored[:, :128])


@pytest.mark.parametrize("codec", KV_REPRESENTATION_CODECS, ids=lambda c: c.name)
@pytest.mark.parametrize("head_size", [128, 256])
def test_indexer_view_preserves_padding_scale_bytes_and_restore(codec, head_size):
    block, pages, groups = 64, 3, head_size // 128
    occupied = block * (head_size + groups * 4)
    stride = occupied + 64
    raw = torch.full((pages * stride,), 77, dtype=torch.uint8)

    def bind(storage):
        carrier = storage.as_strided(
            (pages, block, head_size + groups * 4),
            (stride, head_size + groups * 4, 1),
        )
        return KVCacheView.bind_indexer(carrier, codec, head_size)

    view = bind(raw)
    assert view.key is view.value and view.key_scale is view.value_scale
    assert view.key_scale is not None
    assert view.key_scale.shape == (pages, block, groups)
    assert view.page_stride_bytes == stride
    assert view.page_size_bytes == occupied
    assert view.key.untyped_storage().data_ptr() == raw.data_ptr()
    for page in range(pages):
        view.key[page].fill_(page + 1)
        view.key_scale[page].fill_(page + 0.25)
        base = page * stride
        assert torch.all(raw[base : base + block * head_size] == page + 1)
        assert torch.all(
            raw[base + block * head_size : base + occupied].view(torch.float32)
            == page + 0.25
        )
        assert torch.all(raw[base + occupied : base + stride] == 77)
    copy = raw.clone()
    restored = bind(copy)
    assert torch.equal(copy, raw)
    assert torch.equal(restored.key, view.key)
    assert torch.equal(restored.key_scale, view.key_scale)


def test_indexer_and_ordinary_layouts_cannot_be_bound_interchangeably():
    codec = resolve_kv_codec("int8", representation="indexer")
    assert codec is not None
    with pytest.raises(ValueError, match="ordinary"):
        KVCacheView.bind(torch.zeros(1, 2, 16, 1, 132, dtype=torch.int8), codec, 128)
    with pytest.raises(ValueError, match="Indexer"):
        KVCacheView.bind_indexer(torch.zeros(1, 16, 132, dtype=torch.uint8), INT8, 128)
    with pytest.raises(ValueError, match="Indexer"):
        KVCacheView.bind_indexer(
            torch.zeros(1, 16, 132, dtype=torch.uint8), FP8_E4M3, 128
        )


def test_old_binary_rejects_int8_before_accessing_cache(monkeypatch):
    from vllm import _custom_ops as ops

    monkeypatch.setattr(ops, "_indexer_kv_writer", lambda: None)
    ops.require_indexer_cache_codec("fp8")
    cache = torch.full((1, 16, 132), 77, dtype=torch.uint8)
    with pytest.raises(RuntimeError, match="native cache writer ABI 1"):
        ops.indexer_k_quant_and_cache(
            torch.ones(1, 128),
            cache,
            torch.tensor([0]),
            128,
            "float32",
            cache_dtype="int8",
        )
    assert torch.all(cache == 77)
