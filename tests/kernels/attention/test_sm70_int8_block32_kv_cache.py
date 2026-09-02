import math

import pytest
import torch

from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVQuantMode,
    make_int8_block32_kv_cache_views,
)


def _require_sm70_extension():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("This test requires a CUDA SM70 device")
    return pytest.importorskip("flash_attn_v100_cuda")


def test_int8_block32_publish_requantize_and_decode() -> None:
    extension = _require_sm70_extension()
    torch.manual_seed(7)

    num_blocks = 2
    block_size = 16
    num_kv_heads = 1
    num_query_heads = 2
    head_size = 64
    num_tokens = 20
    spec = AttentionSpec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=torch.int8,
        kv_quant_mode=KVQuantMode.INT8_BLOCK32,
    )
    raw = torch.zeros(
        num_blocks * spec.page_size_bytes,
        dtype=torch.int8,
        device="cuda",
    )
    (
        key_cache,
        value_cache,
        key_scales,
        value_scales,
        page_owners,
    ) = make_int8_block32_kv_cache_views(
        raw,
        num_blocks=num_blocks,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
    )

    key = torch.randn(
        num_tokens, num_kv_heads, head_size, dtype=torch.float16, device="cuda"
    )
    value = torch.randn_like(key)
    # Force several scale increases in one publication batch. The kernel must
    # compute the final scale first instead of repeatedly rounding old codes.
    key[:15, 0, :32] = 0.05
    key[0, 0, :32] = 0.1
    key[1, 0, :32] = 0.2
    key[2, 0, :32] = 0.3
    value[:15, 0, 32:] = 0.05
    value[0, 0, 32:] = 0.1
    value[1, 0, 32:] = 0.2
    value[2, 0, 32:] = 0.3
    # A later call expands both scales again and exercises historical-page
    # re-quantization across publication calls.
    key[15, 0, :32] = 12.0
    value[15, 0, 32:] = -10.0

    # Logical page zero maps to physical page one. Logical page one maps to zero.
    slot_mapping = torch.cat(
        (
            torch.arange(16, 32, device="cuda"),
            torch.arange(0, 4, device="cuda"),
        )
    ).to(torch.int64)
    # Interleave writes to the two physical pages and defer the scale-growing
    # token. Publication must elect one owner per page rather than assuming
    # page runs are contiguous in slot_mapping.
    first_indices = torch.tensor(
        [0, 16, 1, 17, 2, 18, 3, 19, *range(4, 15)],
        dtype=torch.long,
        device="cuda",
    )
    extension.int8_block32_reshape_and_cache(
        key[first_indices],
        value[first_indices],
        key_cache,
        value_cache,
        key_scales,
        value_scales,
        page_owners,
        slot_mapping[first_indices],
    )
    key_scale_before = key_scales[1, 0].repeat_interleave(32).float()
    value_scale_before = value_scales[1, 0].repeat_interleave(32).float()
    expected_key_codes = (
        torch.round(key[:15, 0].float() / key_scale_before)
        .clamp(-127, 127)
        .to(torch.int8)
    )
    expected_value_codes = (
        torch.round(value[:15, 0].float() / value_scale_before)
        .clamp(-127, 127)
        .to(torch.int8)
    )
    torch.testing.assert_close(key_cache[1, :15, 0], expected_key_codes)
    torch.testing.assert_close(value_cache[1, :15, 0], expected_value_codes)

    key_before_growth = (
        key_cache[1, :15].float() * key_scales[1].repeat_interleave(32, dim=-1).float()
    )
    value_before_growth = (
        value_cache[1, :15].float()
        * value_scales[1].repeat_interleave(32, dim=-1).float()
    )

    extension.int8_block32_reshape_and_cache(
        key[15:16],
        value[15:16],
        key_cache,
        value_cache,
        key_scales,
        value_scales,
        page_owners,
        slot_mapping[15:16],
    )

    key_after_growth = (
        key_cache[1, :15].float() * key_scales[1].repeat_interleave(32, dim=-1).float()
    )
    value_after_growth = (
        value_cache[1, :15].float()
        * value_scales[1].repeat_interleave(32, dim=-1).float()
    )
    torch.testing.assert_close(
        key_after_growth,
        key_before_growth,
        atol=float(key_scales[1].max()),
        rtol=0.0,
    )
    torch.testing.assert_close(
        value_after_growth,
        value_before_growth,
        atol=float(value_scales[1].max()),
        rtol=0.0,
    )

    assert torch.all(key_scales > 0)
    assert torch.all(value_scales > 0)
    assert key_scales[1, 0, 0] >= torch.tensor(
        12.0 / 127.0, dtype=torch.float16, device="cuda"
    )

    query = torch.randn(
        1, num_query_heads, head_size, dtype=torch.float16, device="cuda"
    )
    block_table = torch.tensor([[1, 0]], dtype=torch.int32, device="cuda")
    seq_lens = torch.tensor([num_tokens], dtype=torch.int32, device="cuda")
    output = torch.empty_like(query)
    softmax_scale = 1.0 / math.sqrt(head_size)
    extension.int8_block32_decode_paged(
        query,
        key_cache,
        value_cache,
        key_scales,
        value_scales,
        block_table,
        seq_lens,
        output,
        softmax_scale,
    )

    key_scale_values = key_scales.repeat_interleave(32, dim=-1).unsqueeze(1)
    value_scale_values = value_scales.repeat_interleave(32, dim=-1).unsqueeze(1)
    dequant_key = (key_cache.float() * key_scale_values.float()).half().float()
    dequant_value = (value_cache.float() * value_scale_values.float()).half().float()
    token_indices = torch.arange(num_tokens, device="cuda")
    physical_pages = block_table[0, token_indices // 16].to(torch.long)
    page_offsets = token_indices % 16
    key_sequence = dequant_key[physical_pages, page_offsets, 0]
    value_sequence = dequant_value[physical_pages, page_offsets, 0]
    scores = torch.einsum("bhd,td->bht", query.float(), key_sequence)
    probabilities = torch.softmax(scores * softmax_scale, dim=-1)
    reference = torch.einsum("bht,td->bhd", probabilities, value_sequence)

    torch.testing.assert_close(output.float(), reference, atol=2e-3, rtol=2e-3)

    chunk_query = torch.randn(
        3, num_query_heads, head_size, dtype=torch.float16, device="cuda"
    )
    chunk_output = torch.empty_like(chunk_query)
    query_start_loc = torch.tensor([0, 3], dtype=torch.int32, device="cuda")
    extension.int8_block32_prefill_paged(
        chunk_query,
        key_cache,
        value_cache,
        key_scales,
        value_scales,
        block_table,
        seq_lens,
        query_start_loc,
        chunk_output,
        softmax_scale,
    )

    chunk_reference = torch.empty_like(chunk_output, dtype=torch.float32)
    for query_idx in range(3):
        visible_tokens = num_tokens - 3 + query_idx + 1
        chunk_scores = torch.einsum(
            "hd,td->ht",
            chunk_query[query_idx].float(),
            key_sequence[:visible_tokens],
        )
        chunk_probabilities = torch.softmax(chunk_scores * softmax_scale, dim=-1)
        chunk_reference[query_idx] = torch.einsum(
            "ht,td->hd",
            chunk_probabilities,
            value_sequence[:visible_tokens],
        )

    torch.testing.assert_close(
        chunk_output.float(), chunk_reference, atol=2e-3, rtol=2e-3
    )
