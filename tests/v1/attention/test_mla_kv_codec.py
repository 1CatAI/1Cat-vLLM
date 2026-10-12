# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Grouped MLA views keep latent, group scales and RoPE in one owner."""

from dataclasses import replace

import pytest
import torch

from vllm.v1.attention.kv_cache_view import KVCacheView, MLAPageLayout
from vllm.v1.attention.kv_codecs import (
    INT8,
    kv_codecs_for_representation,
    resolve_kv_codec,
)
from vllm.v1.kv_cache_interface import MLAAttentionSpec

pytestmark = pytest.mark.cpu_test
CODECS = (
    *kv_codecs_for_representation("mla_latent"),
    *kv_codecs_for_representation("mla_mixed"),
)


@pytest.mark.parametrize("codec", CODECS, ids=lambda c: f"{c.representation}_{c.name}")
@pytest.mark.parametrize("page", [16, 64])
def test_grouped_mla_regions_restore_padding_and_unquantized_fields(codec, page):
    rope = 64 if codec.unquantized_fields else 0
    latent = 448 if rope else 512
    layout = MLAPageLayout(latent, rope, scale_bytes=codec.scale_dtype.itemsize)
    stride = layout.page_size_bytes(page) + 128
    storage = torch.full((3 * stride,), 77, dtype=codec.storage_dtype)

    def bind(owner):
        carrier = owner.as_strided(
            (3, page, layout.slot_bytes), (stride, layout.slot_bytes, 1)
        )
        return KVCacheView.bind_mla(carrier, codec, latent, rope)

    view = bind(storage)
    assert view.page_size_bytes == page * layout.slot_bytes
    assert view.page_stride_bytes == stride
    assert view.key.shape == (3, page, 1, latent)
    assert view.key_scale.shape == (3, page, latent // 64)
    assert view.key_scale.dtype == codec.scale_dtype
    assert view.key_scale.stride(-2) == 8
    assert view.key.stride(1) == layout.data_row_bytes
    assert view.codec.cache_identity != INT8.cache_identity
    for index in range(3):
        view.key[index].fill_(index + 1)
        view.key_scale[index].fill_(index + 2)
        if rope:
            view.unquantized_field("rope")[index].fill_(index + 0.5)
        assert torch.all(
            storage[
                index * stride + layout.page_size_bytes(page) : (index + 1) * stride
            ]
            == 77
        )
    restored = bind(storage.clone())
    assert torch.equal(restored.key, view.key)
    assert torch.equal(restored.key_scale, view.key_scale)
    if rope:
        assert torch.equal(
            restored.unquantized_field("rope"), view.unquantized_field("rope")
        )


def test_mla_spec_uses_declared_layout_and_its_own_codec():
    layout = MLAPageLayout(512, scale_bytes=4)
    spec = MLAAttentionSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.int8,
        cache_dtype_str="int8",
        kv_cache_layout=layout,
    )
    assert spec.page_size_bytes == 64 * 544
    assert spec.layout_cache_dtype("fp8") == "int8"
    assert MLAAttentionSpec.merge([spec, spec]) == spec
    assert replace(spec, compress_ratio=4).page_size_bytes == 16 * 544
    with pytest.raises(AssertionError, match="same quantization"):
        MLAAttentionSpec.merge([spec, replace(spec, kv_cache_layout=None)])


def test_mla_grouped_identity_cannot_alias_ordinary_or_other_fields():
    latent = resolve_kv_codec("int8", representation="mla_latent")
    mixed = resolve_kv_codec("int8", representation="mla_mixed")
    assert len({INT8.cache_identity, latent.cache_identity, mixed.cache_identity}) == 3
    assert resolve_kv_codec("fp8_ds_mla", representation="mla_mixed") is not None
    assert resolve_kv_codec("int8_per_token_head", representation="mla_latent") is None
    with pytest.raises(ValueError, match="representation"):
        KVCacheView.bind_mla(torch.empty(3, 64, 544, dtype=torch.int8), INT8, 512)
    with pytest.raises(ValueError, match="representation"):
        KVCacheView.bind_mla(torch.empty(3, 64, 544, dtype=torch.int8), mixed, 512)
    with pytest.raises(ValueError, match="aligned data"):
        KVCacheView.bind_mla(torch.empty(3, 64, 520, dtype=torch.int8), latent, 512)


@pytest.mark.parametrize("cache_dtype,size", [("fp8", 520), ("int8", 544)])
def test_mla_layer_emits_provider_layout_without_model_name(cache_dtype, size):
    from types import SimpleNamespace

    from vllm.model_executor.layers.attention.mla_attention import MLAAttention
    from vllm.models.glm5next.sm70.sparse import Glm5NextSM70SparseBackend

    layer = object.__new__(MLAAttention)
    torch.nn.Module.__init__(layer)
    layer.kv_cache_dtype, layer.head_size = cache_dtype, 512
    layer.attn_backend = Glm5NextSM70SparseBackend
    spec = layer.get_kv_cache_spec(
        SimpleNamespace(
            cache_config=SimpleNamespace(block_size=64, cache_dtype="auto"),
            model_config=SimpleNamespace(dtype=torch.float16),
        )
    )
    assert spec.kv_cache_layout is not None
    assert spec.page_size_bytes == 64 * size
    assert spec.cache_dtype_str == cache_dtype
    assert spec.layout_cache_dtype("auto") == cache_dtype
