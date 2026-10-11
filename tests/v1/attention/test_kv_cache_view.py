# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Byte addressing, restore and subdivision contracts, independent of kernels."""

import pytest
import torch

from vllm.v1.attention.kv_cache_view import (
    KVCacheView,
    KVPageLayout,
    bind_planar_kv_cache,
)
from vllm.v1.attention.kv_codecs import (
    FP8_E4M3,
    FP8_E5M2,
    FP16,
    INT8,
    INT8_PER_TOKEN_HEAD,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec, get_kv_quant_mode

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("block", [16, 32, 48, 848])
@pytest.mark.parametrize("heads,dim", [(1, 32), (3, 64), (2, 128), (1, 256)])
def test_planar_regions_are_aligned_disjoint_and_in_the_budget(block, heads, dim):
    layout = KVPageLayout(block, heads, dim, dim)
    spec = FullAttentionSpec(
        block_size=block,
        num_kv_heads=heads,
        head_size=dim,
        dtype=torch.int8,
        kv_quant_mode=get_kv_quant_mode("int8"),
    )
    assert spec.page_size_bytes == layout.page_size_bytes
    raw = torch.full((2 * spec.page_size_bytes,), 99, dtype=torch.int8)
    carrier = bind_planar_kv_cache(
        raw,
        num_blocks=2,
        scheduler_block_size=block,
        kernel_block_size=block,
        num_heads=heads,
        head_size=dim,
        page_size_bytes=spec.page_size_bytes,
    )
    before = raw.clone()
    view = KVCacheView.bind(carrier, INT8, dim)
    assert torch.equal(raw, before), "Binding must not initialize/overwrite scales"
    for tensor in (view.key, view.value, view.key_scale, view.value_scale):
        assert tensor.data_ptr() % 16 == 0
        assert tensor.untyped_storage().data_ptr() == raw.data_ptr()
    view.key.fill_(1)
    view.value.fill_(2)
    view.key_scale.fill_(0.25)
    view.value_scale.fill_(0.5)
    page = raw[: spec.page_size_bytes]
    assert torch.all(page[: layout.value_offset] == 1)
    assert torch.all(page[layout.value_offset : layout.key_scale_offset] == 2)
    assert torch.all(
        page[layout.key_scale_offset : layout.value_scale_offset].view(torch.float32)
        == 0.25
    )
    assert torch.all(page[layout.value_scale_offset :].view(torch.float32) == 0.5)


@pytest.mark.parametrize("kernel_block", [16, 32, 64])
@pytest.mark.parametrize("padding", [0, 64, 128])
@pytest.mark.parametrize("members", [1, 3])
def test_split_padded_pages_copy_and_restore_all_payload_and_scales(
    kernel_block, padding, members
):
    block, heads, dim = 64, 2, 128
    layout = KVPageLayout(block, heads, dim, dim)
    page_bytes = layout.scheduler_page_stride(layout.page_size_bytes + padding)
    raw = torch.full((3 * page_bytes * members,), 17, dtype=torch.int8)

    def bind(storage, member):
        carrier = bind_planar_kv_cache(
            storage,
            num_blocks=3,
            scheduler_block_size=block,
            kernel_block_size=kernel_block,
            num_heads=heads,
            head_size=dim,
            page_size_bytes=page_bytes,
            member_index=member,
            member_count=members,
        )
        return KVCacheView.bind(carrier, INT8, dim)

    for member in range(members):
        view = bind(raw, member)
        for page in range(view.key.shape[0]):
            view.key[page].fill_(page + member)
            view.value[page].fill_(-page - member)
            view.key_scale[page].fill_(0.25 + page + member)
            view.value_scale[page].fill_(0.5 + page + member)

    # Offload/restore and non-contiguous prefix pages use scheduler byte pages.
    page_order = torch.tensor([2, 0, 1])
    restored = raw.view(3, page_bytes * members)[page_order].clone().flatten()
    parts = block // kernel_block
    for member in range(members):
        before, after = bind(raw, member), bind(restored, member)
        for name in ("key", "value", "key_scale", "value_scale"):
            original = getattr(before, name)
            expected = original.reshape(3, parts, *original.shape[1:])[page_order]
            assert torch.equal(getattr(after, name), expected.flatten(0, 1))


@pytest.mark.parametrize("hnd", [False, True])
def test_legacy_inline_scale_uses_actual_strides_and_storage_offset(hnd):
    n, block, heads, dim = 3, 16, 2, 64
    shape = (n, 2, heads, block, dim + 4) if hnd else (n, 2, block, heads, dim + 4)
    page = 2 * block * heads * (dim + 4)
    raw = torch.full((16 + n * (page + 32),), 71, dtype=torch.int8)
    strides = list(torch.empty(shape, device="meta").stride())
    strides[0] = page + 32
    carrier = raw.as_strided(shape, strides, 16)
    if hnd:
        carrier = carrier.permute(0, 1, 3, 2, 4)
    view = KVCacheView.bind(carrier, INT8_PER_TOKEN_HEAD, dim)
    view.key_scale.fill_(0.25)
    view.value_scale.fill_(0.5)
    assert torch.all(carrier[:, 0, :, :, dim:].contiguous().view(torch.float32) == 0.25)
    assert torch.all(carrier[:, 1, :, :, dim:].contiguous().view(torch.float32) == 0.5)
    assert torch.all(view.key == 71) and torch.all(view.value == 71)
    snapshot = raw.clone()
    KVCacheView.bind(carrier, INT8_PER_TOKEN_HEAD, dim)
    assert torch.equal(raw, snapshot)


@pytest.mark.parametrize("codec", [FP16, FP8_E4M3, FP8_E5M2])
def test_dense_formats_keep_the_original_views(codec):
    cache = torch.zeros((3, 2, 16, 2, 128), dtype=codec.storage_dtype)
    view = KVCacheView.bind(cache, codec, 128)
    assert view.key.data_ptr() == cache[:, 0].data_ptr()
    assert view.value.data_ptr() == cache[:, 1].data_ptr()
    assert view.key.stride() == cache[:, 0].stride()
    assert view.key_scale is None and view.value_scale is None


def test_scale_binding_rebinds_storage_without_mutating_restored_scales():
    from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl

    impl = TritonAttentionImpl.__new__(TritonAttentionImpl)
    impl.kv_cache_dtype, impl.head_size = "int8", 64
    first = torch.zeros((2, 2, 16, 1, 68), dtype=torch.int8)
    second = torch.zeros_like(first)
    KVCacheView.bind(first, INT8, 64).key_scale.fill_(0.25)
    KVCacheView.bind(second, INT8, 64).key_scale.fill_(0.5)
    impl._ensure_scale_caches(first)
    assert torch.all(impl._k_scale_cache == 0.25)
    impl._ensure_scale_caches(second)
    assert torch.all(impl._k_scale_cache == 0.5)
    first.set_(second)
    impl._ensure_scale_caches(first)
    assert (
        impl._k_scale_cache.data_ptr()
        == KVCacheView.bind(second, INT8, 64).key_scale.data_ptr()
    )


def test_new_cache_mode_and_version_participate_in_config_identity(monkeypatch):
    from dataclasses import replace

    from vllm.config.cache import CacheConfig
    from vllm.v1.attention import kv_codecs

    config = CacheConfig(cache_dtype="int8")
    assert get_kv_quant_mode(config.cache_dtype).is_per_token_head
    baseline = config.compute_hash()
    assert baseline != CacheConfig(cache_dtype="int8_per_token_head").compute_hash()
    monkeypatch.setattr(kv_codecs, "INT8", replace(INT8, encoding_version=2))
    assert config.compute_hash() != baseline


def test_planar_padding_is_never_silently_changed():
    layout = KVPageLayout(64, 1, 128, 128)
    assert (
        layout.scheduler_page_stride(layout.page_size_bytes + 16)
        == layout.page_size_bytes + 16
    )
    with pytest.raises(ValueError, match="sixteen-byte aligned"):
        layout.scheduler_page_stride(layout.page_size_bytes + 4)
    raw = torch.zeros(layout.page_size_bytes + 16, dtype=torch.int8)
    with pytest.raises(ValueError, match="split into aligned"):
        bind_planar_kv_cache(
            raw,
            num_blocks=1,
            scheduler_block_size=64,
            kernel_block_size=16,
            num_heads=1,
            head_size=128,
            page_size_bytes=raw.numel(),
        )
