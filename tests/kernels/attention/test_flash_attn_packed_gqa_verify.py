# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The packed-GQA verify split matches FA2's causal paged attention."""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.backends.flash_attn import packed_gqa_verify_attention

try:
    from vllm.vllm_flash_attn import flash_attn_varlen_func, is_fa_version_supported
except ImportError:
    pytest.skip("vllm_flash_attn is not available", allow_module_level=True)

if not current_platform.is_cuda() or not is_fa_version_supported(2):
    pytest.skip("needs FlashAttention 2", allow_module_level=True)

BLOCK_SIZE = 16


@pytest.mark.parametrize("context_lens", [[1000], [17, 4097, 31]])
@pytest.mark.parametrize("query_len", [2, 4])
@pytest.mark.parametrize("num_heads", [(12, 2), (16, 4), (8, 1)])
@pytest.mark.parametrize("head_size", [128, 256])
@torch.inference_mode()
def test_packed_gqa_verify_matches_causal_attention(
    context_lens: list[int],
    query_len: int,
    num_heads: tuple[int, int],
    head_size: int,
):
    torch.set_default_device("cuda")
    set_random_seed(0)
    num_query_heads, num_kv_heads = num_heads
    num_reqs = len(context_lens)
    seq_lens = [context + query_len for context in context_lens]
    blocks_per_req = max((seq + BLOCK_SIZE - 1) // BLOCK_SIZE for seq in seq_lens)
    num_blocks = num_reqs * blocks_per_req
    key_cache = torch.randn(num_blocks, BLOCK_SIZE, num_kv_heads, head_size).half()
    value_cache = torch.randn_like(key_cache)
    block_table = torch.randperm(num_blocks, dtype=torch.int32).view(num_reqs, -1)
    query = torch.randn(num_reqs * query_len, num_query_heads, head_size).half()

    # The new tokens' keys and values are the last query_len cache positions.
    new_positions = [torch.arange(c, s) for c, s in zip(context_lens, seq_lens)]
    slots = torch.cat(
        [
            block_table[req, positions // BLOCK_SIZE].long() * BLOCK_SIZE
            + positions % BLOCK_SIZE
            for req, positions in enumerate(new_positions)
        ]
    )
    key = key_cache.view(-1, num_kv_heads, head_size)[slots]
    value = value_cache.view(-1, num_kv_heads, head_size)[slots]
    query_start_loc = torch.arange(num_reqs + 1, dtype=torch.int32) * query_len
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32)
    scale = head_size**-0.5

    expected = flash_attn_varlen_func(
        q=query,
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=query_start_loc,
        max_seqlen_q=query_len,
        seqused_k=seq_lens_t,
        max_seqlen_k=max(seq_lens),
        softmax_scale=scale,
        causal=True,
        block_table=block_table,
        fa_version=2,
    )
    output = torch.empty_like(query)
    packed_gqa_verify_attention(
        output,
        query,
        key,
        value,
        key_cache,
        value_cache,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens_t,
        max_seq_len=max(seq_lens),
        query_len=query_len,
        block_table=block_table,
        softmax_scale=scale,
        logits_soft_cap=0.0,
        max_num_splits=0,
        fa_version=2,
    )
    torch.testing.assert_close(output, expected, atol=2e-3, rtol=2e-3)
