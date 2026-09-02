import pytest
import torch

from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    FullAttentionSpec,
    KVQuantMode,
    get_kv_quant_mode,
    make_int8_block32_kv_cache_views,
)


def test_int8_block32_mode_and_page_size() -> None:
    spec = AttentionSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=64,
        dtype=torch.int8,
        kv_quant_mode=KVQuantMode.INT8_BLOCK32,
    )

    payload_bytes = 2 * 16 * 2 * 64
    scale_bytes = (
        2 * 2 * (64 // 32) * torch.empty([], dtype=torch.float16).element_size()
    )
    assert get_kv_quant_mode("int8_block32") == KVQuantMode.INT8_BLOCK32
    owner_bytes = torch.empty([], dtype=torch.int32).element_size()
    assert spec.page_size_bytes == payload_bytes + scale_bytes + owner_bytes


def test_int8_block32_cache_views_share_one_allocation() -> None:
    num_blocks, block_size, num_heads, head_size = 3, 16, 2, 64
    scale_bytes = 2 * num_blocks * num_heads * (head_size // 32) * 2
    owner_bytes = num_blocks * torch.empty([], dtype=torch.int32).element_size()
    raw = torch.zeros(
        2 * num_blocks * block_size * num_heads * head_size + scale_bytes + owner_bytes,
        dtype=torch.int8,
    )

    key, value, key_scales, value_scales, page_owners = (
        make_int8_block32_kv_cache_views(
            raw,
            num_blocks=num_blocks,
            block_size=block_size,
            num_kv_heads=num_heads,
            head_size=head_size,
        )
    )
    key[1, 2, 0, 3] = -17
    value_scales[2, 1, 1] = 0.5
    page_owners[1] = 7

    assert key.shape == (num_blocks, block_size, num_heads, head_size)
    assert value.shape == key.shape
    assert key_scales.shape == (num_blocks, num_heads, head_size // 32)
    assert value_scales.shape == key_scales.shape

    side_payload_bytes = block_size * num_heads * head_size
    side_scale_bytes = num_heads * (head_size // 32) * 2
    page_bytes = 2 * side_payload_bytes + 2 * side_scale_bytes + 4
    assert key.stride(0) == page_bytes
    assert value.stride(0) == page_bytes
    assert key_scales.stride(0) == page_bytes // 2
    assert value_scales.stride(0) == page_bytes // 2
    assert page_owners.stride(0) == page_bytes // 4
    key_byte = page_bytes + 2 * num_heads * head_size + 3
    assert raw[key_byte] == -17
    assert value_scales[2, 1, 1] == torch.tensor(0.5, dtype=torch.float16)
    owner_byte = page_bytes + 2 * side_payload_bytes + 2 * side_scale_bytes
    owner_from_raw = raw[owner_byte : owner_byte + 4].view(torch.int32)
    assert owner_from_raw.item() == 7


def test_int8_block32_rejects_padded_or_asymmetric_pages() -> None:
    padded = AttentionSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=64,
        dtype=torch.int8,
        kv_quant_mode=KVQuantMode.INT8_BLOCK32,
        page_size_padded=8192,
    )
    with pytest.raises(ValueError, match="does not support padded pages"):
        _ = padded.page_size_bytes

    with pytest.raises(ValueError, match="equal key and value head sizes"):
        FullAttentionSpec(
            block_size=16,
            num_kv_heads=2,
            head_size=64,
            head_size_v=32,
            dtype=torch.int8,
            kv_quant_mode=KVQuantMode.INT8_BLOCK32,
        )


def test_int8_block32_rejects_partial_channel_blocks() -> None:
    raw = torch.zeros(4096, dtype=torch.int8)
    with pytest.raises(ValueError, match="head size divisible by 32"):
        make_int8_block32_kv_cache_views(
            raw,
            num_blocks=1,
            block_size=16,
            num_kv_heads=1,
            head_size=80,
        )
