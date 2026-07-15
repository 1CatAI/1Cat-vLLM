# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch


def _require_sm70_flash_attn_v100():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("fixed BM32 entry is SM70-only")
    return pytest.importorskip("flash_attn_v100")


def _make_inputs(*, query_len: int = 32, page_size: int = 784):
    torch.manual_seed(20260715)
    batch_size = 1
    num_q_heads = 6
    num_kv_heads = 2
    head_dim = 256
    seq_len = 913
    key_cache = torch.randn(
        (2, page_size, num_kv_heads, head_dim),
        dtype=torch.float16,
        device="cuda",
    )
    value_cache = torch.randn_like(key_cache)
    query = torch.randn(
        (batch_size, num_q_heads, query_len, head_dim),
        dtype=torch.float16,
        device="cuda",
    )
    block_table = torch.tensor([[1, 0]], dtype=torch.int32, device="cuda")
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device="cuda")
    return query, key_cache, value_cache, block_table, seq_lens


@torch.inference_mode()
def test_fixed_bm32_entry_is_exact_and_ignores_dispatch_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    flash_attn_v100 = _require_sm70_flash_attn_v100()
    query, key_cache, value_cache, block_table, seq_lens = _make_inputs()

    monkeypatch.setenv("VLLM_FLASH_V100_PREFILL_D256_LOW_SMEM", "1")
    monkeypatch.setenv("VLLM_FLASH_V100_PREFILL_D256_BM32_PHASE", "1")
    monkeypatch.setenv("VLLM_FLASH_V100_PREFILL_D256_BM32_ALL_P", "1")
    monkeypatch.setenv("VLLM_FLASH_V100_PREFILL_D256_BM32_PAIR_SCRATCH", "1")
    expected = flash_attn_v100.flash_attn_prefill_paged_bhmd(
        query,
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        causal=True,
    )

    monkeypatch.setenv("VLLM_FLASH_V100_PREFILL_D256_LOW_SMEM", "0")
    monkeypatch.setenv("VLLM_FLASH_V100_PREFILL_D256_BM32_PHASE", "0")
    monkeypatch.setenv("VLLM_FLASH_V100_PREFILL_D256_BM32_ALL_P", "0")
    monkeypatch.setenv("VLLM_FLASH_V100_PREFILL_D256_BM32_PAIR_SCRATCH", "0")
    out = torch.empty_like(query)
    softmax_lse = torch.empty(
        query.shape[:-1],
        dtype=torch.float32,
        device=query.device,
    )
    actual, actual_lse = (
        flash_attn_v100.flash_attn_prefill_paged_d256_bm32_allp_pair_scratch(
            query,
            key_cache,
            value_cache,
            block_table,
            seq_lens,
            out=out,
            softmax_lse=softmax_lse,
        )
    )

    torch.cuda.synchronize()
    assert actual.data_ptr() == out.data_ptr()
    assert actual_lse.data_ptr() == softmax_lse.data_ptr()
    assert torch.equal(actual, expected)


@torch.inference_mode()
def test_fixed_bm32_entry_rejects_unqualified_shapes() -> None:
    flash_attn_v100 = _require_sm70_flash_attn_v100()
    query, key_cache, value_cache, block_table, seq_lens = _make_inputs(
        query_len=16
    )

    with pytest.raises(RuntimeError, match="M >= 32"):
        flash_attn_v100.flash_attn_prefill_paged_d256_bm32_allp_pair_scratch(
            query,
            key_cache,
            value_cache,
            block_table,
            seq_lens,
        )

    query, key_cache, value_cache, block_table, seq_lens = _make_inputs(
        page_size=16
    )
    with pytest.raises(RuntimeError, match="page size 784"):
        flash_attn_v100.flash_attn_prefill_paged_d256_bm32_allp_pair_scratch(
            query,
            key_cache,
            value_cache,
            block_table,
            seq_lens,
        )
