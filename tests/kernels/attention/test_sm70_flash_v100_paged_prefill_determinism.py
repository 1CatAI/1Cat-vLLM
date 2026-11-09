# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import pytest
import torch


def _make_paged_inputs(
    *,
    seq_len: int,
    block_size: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    num_query_heads = 12
    num_kv_heads = 2
    head_dim = 128
    num_blocks = (seq_len + block_size - 1) // block_size

    query = torch.randn(
        1,
        seq_len,
        num_query_heads,
        head_dim,
        dtype=torch.float16,
        device="cuda",
    )
    key_cache = torch.full(
        (num_blocks, block_size, num_kv_heads, head_dim),
        100.0,
        dtype=torch.float16,
        device="cuda",
    )
    value_cache = key_cache.clone()
    key = torch.randn(
        seq_len,
        num_kv_heads,
        head_dim,
        dtype=torch.float16,
        device="cuda",
    )
    value = torch.randn_like(key)

    for token_idx in range(seq_len):
        block_idx, block_offset = divmod(token_idx, block_size)
        key_cache[block_idx, block_offset] = key[token_idx]
        value_cache[block_idx, block_offset] = value[token_idx]

    block_table = torch.arange(num_blocks, dtype=torch.int32, device="cuda").unsqueeze(
        0
    )
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device="cuda")
    return query, key_cache, value_cache, block_table, seq_lens


def _assert_bit_exact_replays(outputs: list[torch.Tensor], label: str) -> None:
    reference = outputs[0]
    max_diff = max((reference - output).abs().max().item() for output in outputs[1:])
    assert all(torch.equal(reference, output) for output in outputs[1:]), (
        f"{label} replay changed; max_abs_diff={max_diff}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("seq_len", [300, 512])
@torch.inference_mode()
def test_sm70_flash_v100_paged_prefill_d128_replay_is_bit_exact(
    seq_len: int,
) -> None:
    """Guard the physical-V100 regression reported in #98.

    The direct extension replay separates a kernel error from the public API's
    BMHD-to-BHMD conversion and output copy. Synchronizing each launch makes
    this a strict deterministic-replay test rather than a stream-order test.
    """
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("FlashAttention-V100 regression is SM70/V100 only")

    flash_attn_v100 = pytest.importorskip("flash_attn_v100")
    from flash_attn_v100 import flash_attn_interface

    torch.manual_seed(0)
    query, key_cache, value_cache, block_table, seq_lens = _make_paged_inputs(
        seq_len=seq_len,
        block_size=256,
    )
    query_bhmd = query.permute(0, 2, 1, 3).contiguous()
    direct_outputs: list[torch.Tensor] = []
    public_outputs: list[torch.Tensor] = []

    for _ in range(6):
        direct_output = flash_attn_interface.flash_attn_v100_cuda.prefill_paged_fwd(
            query_bhmd,
            key_cache,
            value_cache,
            None,
            block_table,
            seq_lens,
            query.shape[-1] ** -0.5,
            "auto",
            1.0,
            1.0,
            True,
            -1,
            -1,
            None,
            0,
        )
        torch.accelerator.synchronize()
        direct_outputs.append(direct_output)

        public_output = flash_attn_v100.flash_attn_prefill_paged(
            query,
            key_cache,
            value_cache,
            block_table,
            seq_lens,
            causal=True,
        )
        torch.accelerator.synchronize()
        public_outputs.append(public_output)

    _assert_bit_exact_replays(direct_outputs, "prefill_paged_fwd")
    _assert_bit_exact_replays(public_outputs, "flash_attn_prefill_paged")
    assert torch.equal(direct_outputs[0].permute(0, 2, 1, 3), public_outputs[0])
