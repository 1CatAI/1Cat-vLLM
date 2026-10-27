# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVQuantMode,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    get_kv_quant_mode,
)

pytestmark = pytest.mark.skip_global_cleanup


@pytest.mark.parametrize("mode", list(KVQuantMode))
def test_layout_cache_dtype_round_trips_every_quant_mode(mode):
    """Every KVQuantMode must have a spelling that maps back to itself.

    ``layout_cache_dtype`` is called with the engine-wide ``kv_cache_dtype``
    during KV allocation. A mode without an entry raises KeyError there, which
    reports the enum value rather than the mode the user configured.
    """
    spec = AttentionSpec(
        block_size=16,
        num_kv_heads=4,
        head_size=128,
        dtype=torch.uint8,
        kv_quant_mode=mode,
    )
    spelling = spec.layout_cache_dtype("auto")
    assert get_kv_quant_mode(spelling) is mode


def test_layout_cache_dtype_keeps_matching_configuration():
    """A spec that already matches the engine dtype keeps its exact spelling."""
    spec = AttentionSpec(
        block_size=16,
        num_kv_heads=4,
        head_size=128,
        dtype=torch.uint8,
        kv_quant_mode=KVQuantMode.NVFP4,
    )
    assert spec.layout_cache_dtype("nvfp4") == "nvfp4"


@pytest.mark.parametrize(
    "spec",
    [
        # GLM-5.3 sparse MLA: 512 E4M3 bytes + eight UE8M0 scales per token.
        MLAAttentionSpec(
            block_size=64,
            num_kv_heads=1,
            head_size=512,
            dtype=torch.uint8,
            cache_dtype_str="fp8_e4m3",
            model_version="glm5_next",
        ),
        # DeepSeek-V4 compressed attention and its sliding-window cache.
        MLAAttentionSpec(
            block_size=256,
            num_kv_heads=1,
            head_size=512,
            dtype=torch.uint8,
            compress_ratio=4,
            cache_dtype_str="fp8_ds_mla",
            alignment=576,
            model_version="deepseek_v4",
        ),
        SlidingWindowMLASpec(
            block_size=64,
            num_kv_heads=1,
            head_size=512,
            dtype=torch.uint8,
            sliding_window=128,
            cache_dtype_str="fp8_ds_mla",
            alignment=576,
            model_version="deepseek_v4",
        ),
    ],
)
def test_mla_specs_carry_the_codec_of_their_cache_dtype(spec):
    """An MLA spec that names its cache dtype keeps that layout at allocation.

    The KV allocator reshapes each carrier with the backend shape for
    ``layout_cache_dtype``. A packed fp8 MLA spec left at ``KVQuantMode.NONE``
    was reshaped as "auto": GLM-5.3 failed to boot, and the DeepSeek-V4
    sliding-window cache silently got 512- instead of 584-byte token rows.
    """
    assert spec.kv_quant_mode is get_kv_quant_mode(spec.cache_dtype_str)
    assert spec.layout_cache_dtype(spec.cache_dtype_str) == spec.cache_dtype_str
    resized = spec.copy_with_new_block_size(2 * spec.block_size)
    assert resized.kv_quant_mode is spec.kv_quant_mode


def test_mla_spec_rejects_a_codec_that_contradicts_its_cache_dtype():
    with pytest.raises(ValueError, match="contradicts"):
        MLAAttentionSpec(
            block_size=64,
            num_kv_heads=1,
            head_size=512,
            dtype=torch.uint8,
            cache_dtype_str="fp8_e4m3",
            kv_quant_mode=KVQuantMode.NVFP4,
        )


def test_deepseek_v4_swa_carrier_keeps_its_584_byte_rows_after_merge():
    """The sliding-window group spec is rebuilt by merge, without a mode.

    Its carrier must still get the packed fp8_ds_mla row: a 512-wide "auto"
    view also fits the 576-aligned page and would go unnoticed.
    """
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWABackend

    layers = [
        SlidingWindowMLASpec(
            block_size=64,
            num_kv_heads=1,
            head_size=512,
            dtype=torch.uint8,
            sliding_window=128,
            cache_dtype_str="fp8_ds_mla",
            alignment=576,
            model_version="deepseek_v4",
        )
        for _ in range(2)
    ]
    group = SlidingWindowMLASpec.merge(layers)
    assert group.kv_quant_mode is get_kv_quant_mode("fp8_ds_mla")
    shape = DeepseekSparseSWABackend.get_kv_cache_shape(
        num_blocks=3,
        block_size=group.block_size,
        num_kv_heads=1,
        head_size=512,
        cache_dtype_str=group.layout_cache_dtype("fp8_ds_mla"),
    )
    assert shape == (3, 64, 584)


@pytest.mark.parametrize("block_size", [32, 64, 128, 256])
def test_copy_with_new_block_size_keeps_padding_valid(block_size):
    """Growing the block grows the payload; the padded page must keep up.

    ``page_size_padded`` is a byte count. ``copy_with_new_block_size`` used to
    copy it verbatim, so a large enough block made the copy's payload exceed
    its own padding and ``page_size_bytes`` raised a bare AssertionError.
    """
    base = AttentionSpec(
        block_size=16,
        num_kv_heads=4,
        head_size=128,
        dtype=torch.uint8,
        kv_quant_mode=KVQuantMode.NVFP4,
    )
    padded = AttentionSpec(
        block_size=16,
        num_kv_heads=4,
        head_size=128,
        dtype=torch.uint8,
        kv_quant_mode=KVQuantMode.NVFP4,
        page_size_padded=base.real_page_size_bytes * 4,
    )
    grown = padded.copy_with_new_block_size(block_size)
    assert grown.page_size_bytes >= grown.real_page_size_bytes
    assert grown.block_size == block_size


def test_copy_with_new_block_size_preserves_ample_padding():
    """A copy that still fits keeps the original padded page exactly."""
    base = AttentionSpec(
        block_size=16,
        num_kv_heads=4,
        head_size=128,
        dtype=torch.uint8,
        kv_quant_mode=KVQuantMode.NVFP4,
    )
    padded = AttentionSpec(
        block_size=16,
        num_kv_heads=4,
        head_size=128,
        dtype=torch.uint8,
        kv_quant_mode=KVQuantMode.NVFP4,
        page_size_padded=base.real_page_size_bytes * 8,
    )
    grown = padded.copy_with_new_block_size(32)
    assert grown.page_size_bytes == padded.page_size_padded
