# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.models.deepseek_v4.sm70.indexer import (
    sm70_indexer_decode_logits,
    sm70_indexer_prefill_logits,
)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="requires NVIDIA V100/SM70",
)
def test_sm70_indexer_prefill_dequantizes_fp8_storage_as_uint8() -> None:
    torch.manual_seed(20260803)
    num_queries = 3
    num_heads = 4
    num_keys = 7
    head_dim = 128

    q = torch.randn(
        (num_queries, num_heads, head_dim), device="cuda", dtype=torch.float16
    )
    weights = torch.randn((num_queries, num_heads), device="cuda", dtype=torch.float32)
    # E4M3FN bit pattern 0x38 is exactly 1.0. Constructing the tensor through
    # uint8 also avoids requiring native FP8 conversion support on SM70.
    k_bits = torch.full((num_keys, head_dim), 0x38, device="cuda", dtype=torch.uint8)
    k_quant = k_bits.view(torch.float8_e4m3fn)
    scales = torch.linspace(0.5, 2.0, num_keys, device="cuda", dtype=torch.float32)

    actual = sm70_indexer_prefill_logits(q, k_quant, scales, weights)

    weighted_q = torch.sum(q.float() * weights[:, :, None], dim=1).half()
    k_reference = scales[:, None].expand(num_keys, head_dim).half()
    expected = torch.mm(weighted_q, k_reference.T, out_dtype=torch.float32)
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="requires NVIDIA V100/SM70",
)
def test_sm70_indexer_decode_valid_logits_do_not_depend_on_workspace_width() -> None:
    torch.manual_seed(20260804)
    num_queries = 8
    num_heads = 4
    num_keys = 7
    head_dim = 128
    cache_block_size = 64

    q = torch.randn(
        (num_queries, num_heads, head_dim), device="cuda", dtype=torch.float16
    )
    weights = torch.randn((num_queries, num_heads), device="cuda", dtype=torch.float16)
    cache = torch.zeros(
        (1, cache_block_size, head_dim + 4), device="cuda", dtype=torch.uint8
    )
    cache[..., :head_dim].fill_(0x38)
    cache[..., head_dim:].view(torch.float32).fill_(1.0)
    seq_lens = torch.full((1, num_queries), num_keys, device="cuda", dtype=torch.int32)
    block_table = torch.zeros((1, 1), device="cuda", dtype=torch.int32)

    narrow = sm70_indexer_decode_logits(
        q, cache, weights, seq_lens, block_table, max_seq_len=16
    )
    wide = sm70_indexer_decode_logits(
        q, cache, weights, seq_lens, block_table, max_seq_len=64
    )

    assert narrow.shape == (num_queries, 16)
    assert wide.shape == (num_queries, 64)
    torch.testing.assert_close(narrow[:, :num_keys], wide[:, :num_keys], rtol=0, atol=0)
