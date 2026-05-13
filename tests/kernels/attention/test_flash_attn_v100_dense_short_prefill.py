# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import pytest
import torch


def _require_flash_attn_v100():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for flash_attn_v100")
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (7, 0):
        pytest.skip("flash_attn_v100 kernels are only validated on SM70")
    return pytest.importorskip("flash_attn_v100")


def _ref_gqa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    causal: bool,
) -> torch.Tensor:
    q_heads = q.shape[2]
    kv_heads = k.shape[2]
    if q_heads != kv_heads:
        repeat = q_heads // kv_heads
        k = k.repeat_interleave(repeat, dim=2)
        v = v.repeat_interleave(repeat, dim=2)

    q_ref = q.float().transpose(1, 2)
    k_ref = k.float().transpose(1, 2)
    v_ref = v.float().transpose(1, 2)
    q_len = q.shape[1]
    seq_len = k.shape[1]
    scores = torch.matmul(q_ref, k_ref.transpose(-1, -2)) * softmax_scale
    if causal:
        q_pos = torch.arange(q_len, device=q.device) + max(seq_len - q_len, 0)
        k_pos = torch.arange(seq_len, device=q.device)
        mask = k_pos.view(1, 1, 1, seq_len) > q_pos.view(1, 1, q_len, 1)
        scores = scores.masked_fill(mask, float("-inf"))
    out = torch.matmul(torch.softmax(scores, dim=-1), v_ref)
    return out.transpose(1, 2).to(torch.float16)


def _make_paged_cache(
    k: torch.Tensor,
    v: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    seq_len = k.shape[1]
    num_blocks = (seq_len + block_size - 1) // block_size
    key_cache = torch.zeros(
        (num_blocks, block_size, k.shape[2], k.shape[3]),
        device=k.device,
        dtype=k.dtype,
    )
    value_cache = torch.zeros_like(key_cache)
    for block_idx in range(num_blocks):
        start = block_idx * block_size
        end = min(start + block_size, seq_len)
        key_cache[block_idx, :end - start] = k[0, start:end]
        value_cache[block_idx, :end - start] = v[0, start:end]
    block_table = torch.arange(num_blocks, device=k.device,
                               dtype=torch.int32).view(1, num_blocks)
    seq_lens = torch.tensor([seq_len], device=k.device, dtype=torch.int32)
    return key_cache, value_cache, block_table, seq_lens


@pytest.mark.parametrize(
    ("seq_len", "head_dim", "causal"),
    [
        (1, 256, True),
        (5, 64, False),
        (9, 256, True),
        (15, 256, True),
        (57, 128, True),
        (69, 256, False),
        (181, 128, True),
    ],
)
def test_flash_attn_v100_dense_prefill_tail_lengths_match_reference(
    seq_len: int,
    head_dim: int,
    causal: bool,
):
    flash_attn_v100 = _require_flash_attn_v100()

    torch.manual_seed(seq_len * 1000 + head_dim + int(causal))
    device = torch.device("cuda")
    q_heads = 6
    kv_heads = 1
    q = torch.randn((1, seq_len, q_heads, head_dim),
                    device=device,
                    dtype=torch.float16)
    k = torch.randn((1, seq_len, kv_heads, head_dim),
                    device=device,
                    dtype=torch.float16)
    v = torch.randn_like(k)
    softmax_scale = 1.0 / math.sqrt(head_dim)

    actual = flash_attn_v100.flash_attn_func(
        q,
        k,
        v,
        causal=causal,
        softmax_scale=softmax_scale,
    )
    expected = _ref_gqa_attention(q, k, v, softmax_scale, causal)

    assert not torch.isnan(actual).any()
    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-2)


@pytest.mark.parametrize(
    ("seq_len", "query_len", "head_dim", "causal"),
    [
        (5, 5, 64, False),
        (9, 9, 256, True),
        (69, 16, 256, True),
        (181, 32, 128, True),
    ],
)
def test_flash_attn_v100_paged_prefill_tail_lengths_match_reference(
    seq_len: int,
    query_len: int,
    head_dim: int,
    causal: bool,
):
    flash_attn_v100 = _require_flash_attn_v100()

    torch.manual_seed(seq_len * 1000 + query_len * 10 + head_dim)
    device = torch.device("cuda")
    q_heads = 6
    kv_heads = 1
    block_size = 16
    q = torch.randn((1, query_len, q_heads, head_dim),
                    device=device,
                    dtype=torch.float16)
    k = torch.randn((1, seq_len, kv_heads, head_dim),
                    device=device,
                    dtype=torch.float16)
    v = torch.randn_like(k)
    softmax_scale = 1.0 / math.sqrt(head_dim)
    key_cache, value_cache, block_table, seq_lens = _make_paged_cache(
        k, v, block_size)

    actual = flash_attn_v100.flash_attn_prefill_paged(
        q,
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        softmax_scale=softmax_scale,
        causal=causal,
    )
    expected = _ref_gqa_attention(q, k, v, softmax_scale, causal)

    assert not torch.isnan(actual).any()
    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-2)
