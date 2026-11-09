# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compacting draft pages preserves the sliding-window arithmetic."""

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("kv_heads", [1, 2, 4])
@pytest.mark.parametrize("seq_len", [4096, 4111])
@torch.inference_mode()
def test_compact_sliding_pages_preserve_output_bits(kv_heads, seq_len):
    import flash_attn_v100

    torch.manual_seed(690)
    query = torch.randn(1, 8, 8, 128, device="cuda", dtype=torch.float16)
    capacity = ((seq_len + 2047) // 2048) * 2048
    key = torch.randn(capacity, kv_heads, 128, device="cuda", dtype=torch.float16)
    value = torch.randn_like(key)
    lengths = torch.tensor([seq_len], dtype=torch.int32, device="cuda")
    outputs = []
    for block_size in (2048, 1024):
        blocks = (seq_len + block_size - 1) // block_size
        cache_shape = (blocks, block_size, kv_heads, 128)
        table = torch.arange(blocks, device="cuda", dtype=torch.int32)[None]
        outputs.append(
            flash_attn_v100.flash_attn_prefill_paged(
                query,
                key[: blocks * block_size].view(cache_shape),
                value[: blocks * block_size].view(cache_shape),
                table,
                lengths,
                causal=False,
                window_size=(2048, 2048),
            )
        )
    assert torch.isfinite(outputs[0]).all()
    assert torch.equal(outputs[0].view(torch.int16), outputs[1].view(torch.int16))
