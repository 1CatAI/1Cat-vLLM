# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import pytest
import torch


def _make_sequential_cache(
    *,
    seq_len: int,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_blocks = (seq_len + block_size - 1) // block_size
    key_cache = torch.randn(
        num_blocks,
        block_size,
        num_kv_heads,
        head_dim,
        dtype=dtype,
        device=device,
    )
    value_cache = torch.randn_like(key_cache)
    block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).view(1, -1)
    return key_cache, value_cache, block_table


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("query_len", [5, 16])
@pytest.mark.parametrize("seq_len", [4097])
@torch.inference_mode()
def test_flash_v100_mtp_smallq_decode_matches_causal_prefill(
    query_len: int,
    seq_len: int,
) -> None:
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("FlashAttention-V100 regression is SM70/V100 only")

    flash_attn_v100 = pytest.importorskip("flash_attn_v100")
    torch.manual_seed(1234)
    device = "cuda"
    dtype = torch.float16
    block_size = 16
    num_heads = 8
    num_kv_heads = 1
    head_dim = 256
    scale = head_dim**-0.5

    key_cache, value_cache, block_table = _make_sequential_cache(
        seq_len=seq_len,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=dtype,
        device=device,
    )
    query = torch.randn(
        query_len,
        num_heads,
        head_dim,
        dtype=dtype,
        device=device,
    )
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=device)

    prefill_out = flash_attn_v100.flash_attn_prefill_paged(
        query.unsqueeze(0),
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        softmax_scale=scale,
        kv_cache_dtype="auto",
        k_scale=1.0,
        v_scale=1.0,
        causal=True,
    )[0]

    # This is the same transformation used by Flash-V100 small-query MTP
    # verification: each causal query row becomes an independent decode row
    # whose visible KV length increases by one.
    decode_block_table = block_table.repeat(query_len, 1).contiguous()
    decode_seq_lens = torch.arange(
        seq_len - query_len + 1,
        seq_len + 1,
        dtype=torch.int32,
        device=device,
    )
    smallq_out = flash_attn_v100.flash_attn_decode_paged(
        query,
        key_cache,
        value_cache,
        decode_block_table,
        decode_seq_lens,
        softmax_scale=scale,
        kv_cache_dtype="auto",
        k_scale=1.0,
        v_scale=1.0,
        window_size=(-1, -1),
        max_seq_len_hint=seq_len,
        workspace_seq_capacity_hint=seq_len,
    )

    torch.cuda.synchronize()
    max_diff = (smallq_out - prefill_out).abs().max().item()
    assert max_diff <= 1.5e-2
    torch.testing.assert_close(smallq_out, prefill_out, atol=1.5e-2, rtol=1e-2)
