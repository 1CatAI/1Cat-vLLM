# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Byte budgets and scale-view aliasing of existing dense KV formats."""

import pytest
import torch

from vllm.v1.kv_cache_codec import KVCacheCodec, KVQuantMode, get_kv_quant_mode
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
    TQFullAttentionSpec,
)

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("spec_type", [FullAttentionSpec, SlidingWindowSpec])
@pytest.mark.parametrize(
    "dtype,mode,expected_payload,expected_page",
    [
        (torch.float16, KVQuantMode.NONE, 24576, 24576),
        (torch.uint8, KVQuantMode.FP8_PER_TENSOR, 12288, 12288),
        (torch.int8, KVQuantMode.INT8_PER_TOKEN_HEAD, 12288, 12544),
        (torch.uint8, KVQuantMode.FP8_PER_TOKEN_HEAD, 12288, 12544),
        (torch.uint8, KVQuantMode.NVFP4, 6912, 6912),
    ],
)
def test_page_budget_includes_inline_scales(
    spec_type, dtype, mode, expected_payload, expected_page
):
    kwargs = {"sliding_window": 1024} if spec_type is SlidingWindowSpec else {}
    spec = spec_type(
        block_size=16,
        num_kv_heads=2,
        head_size=256,
        head_size_v=128,
        dtype=dtype,
        kv_quant_mode=mode,
        **kwargs,
    )
    assert spec.real_page_size_bytes == expected_payload
    assert spec.page_size_bytes == expected_page
    assert spec.copy_with_new_block_size(32).page_size_bytes == 2 * expected_page
    assert spec.codec.bytes_per_token(2, 256, 128) * 16 == expected_page


def test_padding_cannot_hide_missing_scale_budget():
    kwargs = dict(
        block_size=16,
        num_kv_heads=2,
        head_size=256,
        dtype=torch.int8,
        kv_quant_mode=KVQuantMode.INT8_PER_TOKEN_HEAD,
    )
    assert FullAttentionSpec(**kwargs, page_size_padded=32768).page_size_bytes == 32768
    with pytest.raises(AssertionError):
        _ = FullAttentionSpec(**kwargs, page_size_padded=16384).page_size_bytes


@pytest.mark.parametrize(
    "cache_dtype,dtype",
    [("int8_per_token_head", torch.int8), ("fp8_per_token_head", torch.uint8)],
)
@pytest.mark.parametrize("block_size,num_heads,head_size", [(16, 2, 256), (32, 3, 64)])
def test_scale_views_alias_only_the_inline_scale_bytes(
    cache_dtype, dtype, block_size, num_heads, head_size
):
    codec = KVCacheCodec(dtype, get_kv_quant_mode(cache_dtype))
    num_blocks = 3
    budget = num_blocks * block_size * codec.bytes_per_token(num_heads, head_size)
    raw = torch.full((budget,), 0x35, dtype=torch.uint8)
    cache = raw.view(dtype).view(
        num_blocks, 2, block_size, num_heads, head_size + codec.scale_padding_elements
    )
    layout = codec.per_token_head_scale_layout(block_size, num_heads, head_size)
    k_scale = torch.as_strided(
        raw.view(torch.float32),
        size=(num_blocks, block_size, num_heads),
        stride=layout.strides,
        storage_offset=layout.k_offset,
    )
    v_scale = torch.as_strided(
        raw.view(torch.float32),
        size=k_scale.shape,
        stride=layout.strides,
        storage_offset=layout.v_offset,
    )
    k_scale.copy_(torch.arange(k_scale.numel()).view(k_scale.shape) + 1)
    v_scale.copy_(-k_scale)
    assert torch.all(cache[..., :head_size] == 0x35)
    scales = cache[..., head_size:].contiguous().view(torch.float32).squeeze(-1)
    torch.testing.assert_close(scales[:, 0], k_scale, rtol=0, atol=0)
    torch.testing.assert_close(scales[:, 1], v_scale, rtol=0, atol=0)
    assert v_scale[-1, -1, -1].storage_offset() * 4 + 4 == budget


def test_specialized_layouts_keep_their_own_payload_accounting():
    kwargs = dict(block_size=16, num_kv_heads=1, head_size=256, dtype=torch.uint8)
    assert TQFullAttentionSpec(**kwargs, tq_slot_size=192).page_size_bytes == 3072
    assert (
        MLAAttentionSpec(**kwargs, cache_dtype_str="fp8_ds_mla").page_size_bytes
        == 10496
    )


def test_quant_mode_reexports_preserve_existing_imports():
    from vllm.v1 import kv_cache_interface

    assert kv_cache_interface.KVQuantMode is KVQuantMode
    assert kv_cache_interface.get_kv_quant_mode is get_kv_quant_mode
