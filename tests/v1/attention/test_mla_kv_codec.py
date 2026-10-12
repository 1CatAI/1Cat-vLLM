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
from vllm.v1.kv_cache_interface import MLAAttentionSpec, SlidingWindowMLASpec

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


@pytest.mark.parametrize("cache_dtype,size", [("fp8_ds_mla", 584), ("int8", 608)])
def test_mixed_mla_flattened_pages_require_explicit_capacity(cache_dtype, size):
    codec = resolve_kv_codec(cache_dtype, representation="mla_mixed")
    page = 64
    storage = torch.full((3, page * size + 576), 77, dtype=codec.storage_dtype)
    view = KVCacheView.bind_mla(storage, codec, 448, 64, page_tokens=page)
    assert view.page_size_bytes == page * size
    assert view.page_stride_bytes == storage.stride(0)
    view.key.fill_(3)
    view.key_scale.fill_(2)
    view.unquantized_field("rope").fill_(1)
    assert torch.all(storage[:, page * size :] == 77)
    with pytest.raises(ValueError, match="explicit token capacity"):
        KVCacheView.bind_mla(storage, codec, 448, 64)
    with pytest.raises(ValueError, match="reserve payload/scales"):
        KVCacheView.bind_mla(
            storage[:, : page * size - 1], codec, 448, 64, page_tokens=page
        )


@pytest.mark.parametrize("cache_dtype,size", [("fp8_ds_mla", 584), ("int8", 608)])
def test_mixed_mla_engine_specs_share_declared_layout(monkeypatch, cache_dtype, size):
    from types import SimpleNamespace

    from vllm.models.deepseek_v4.attention import DeepseekV4MLAAttention
    from vllm.models.deepseek_v4.sm70.sparse import DeepseekV4SM70SparseBackend
    from vllm.v1.attention.backends.mla import sparse_swa

    codec = resolve_kv_codec(cache_dtype, representation="mla_mixed")
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(static_forward_context={}),
        cache_config=SimpleNamespace(cache_dtype=cache_dtype, block_size=512),
    )
    monkeypatch.setattr(sparse_swa, "get_current_vllm_config", lambda: config)
    swa = sparse_swa.DeepseekV4SWACache(
        512, 128, codec.storage_dtype, "swa", config.cache_config
    )
    spec = swa.get_kv_cache_spec(config)
    assert isinstance(spec, SlidingWindowMLASpec)
    assert spec.real_page_size_bytes == 64 * size
    assert spec.page_size_bytes == ((64 * size + 575) // 576) * 576
    assert spec.layout_cache_dtype("auto") == cache_dtype
    assert SlidingWindowMLASpec.merge([spec, spec]) == spec
    assert sparse_swa.DeepseekSparseSWABackend.get_kv_cache_shape(
        3, 64, 1, 512, cache_dtype
    ) == (3, 64, size)
    with pytest.raises(AssertionError, match="same quantization"):
        SlidingWindowMLASpec.merge([spec, replace(spec, kv_cache_layout=None)])
    with pytest.raises(ValueError, match="storage codec"):
        replace(spec, cache_dtype_str=None).layout_cache_dtype("int8")

    layer = object.__new__(DeepseekV4MLAAttention)
    torch.nn.Module.__init__(layer)
    layer.head_dim, layer.kv_cache_dtype, layer.kv_codec = 512, cache_dtype, codec
    layer.kv_cache_layout = DeepseekV4SM70SparseBackend.get_kv_cache_layout(
        512, cache_dtype
    )
    for ratio in (4, 128):
        layer.compress_ratio = ratio
        compressed = layer.get_kv_cache_spec(config)
        assert compressed.real_page_size_bytes == (512 // ratio) * size
        assert compressed.layout_cache_dtype("fp8") == cache_dtype
        assert compressed.page_size_bytes >= compressed.real_page_size_bytes
        assert DeepseekV4SM70SparseBackend.get_kv_cache_shape(
            3, 512 // ratio, 1, 512, cache_dtype
        ) == (3, 512 // ratio, size)
    layer.compress_ratio = 1
    assert layer.get_kv_cache_spec(config) is None
