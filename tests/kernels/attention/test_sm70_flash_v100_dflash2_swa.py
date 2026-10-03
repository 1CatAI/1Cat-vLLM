# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Correctness tests for the DFlash2 draft sliding-window attention shape."""

from __future__ import annotations

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("seq_len", [4096, 4111])
@torch.inference_mode()
def test_sm70_dflash2_noncausal_swa_matches_reference(seq_len: int) -> None:
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("FlashAttention-V100 regression is SM70/V100 only")

    flash_attn_v100 = pytest.importorskip("flash_attn_v100")
    torch.manual_seed(0)

    q_len = 8
    num_query_heads = 8
    num_kv_heads = 2
    head_dim = 128
    block_size = 16
    window = 2048
    num_blocks = (seq_len + block_size - 1) // block_size
    query = torch.randn(
        1,
        q_len,
        num_query_heads,
        head_dim,
        device="cuda",
        dtype=torch.float16,
    )
    key_cache = torch.randn(
        num_blocks,
        block_size,
        num_kv_heads,
        head_dim,
        device="cuda",
        dtype=torch.float16,
    )
    value_cache = torch.randn_like(key_cache)
    block_table = torch.arange(num_blocks, device="cuda", dtype=torch.int32).unsqueeze(
        0
    )
    seq_lens = torch.tensor([seq_len], device="cuda", dtype=torch.int32)

    actual = flash_attn_v100.flash_attn_prefill_paged(
        query,
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        causal=False,
        window_size=(window, window),
    )

    key = key_cache.reshape(-1, num_kv_heads, head_dim)[:seq_len]
    value = value_cache.reshape(-1, num_kv_heads, head_dim)[:seq_len]
    key = key.repeat_interleave(num_query_heads // num_kv_heads, dim=1)
    value = value.repeat_interleave(num_query_heads // num_kv_heads, dim=1)
    query_positions = torch.arange(seq_len - q_len, seq_len, device="cuda")
    key_positions = torch.arange(seq_len, device="cuda")
    mask = (key_positions[None, :] >= query_positions[:, None] - window) & (
        key_positions[None, :] <= query_positions[:, None] + window
    )
    scores = torch.einsum("qhd,khd->hqk", query[0].float(), key.float())
    scores *= head_dim**-0.5
    scores.masked_fill_(~mask.unsqueeze(0), float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    reference = torch.einsum("hqk,khd->qhd", probs, value.float())

    torch.testing.assert_close(actual[0].float(), reference, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("batch_size", [1, 2, 4])
@torch.inference_mode()
def test_sm70_dflash2_window_split_replay_changes_lengths(batch_size: int) -> None:
    """A captured sliding grid must follow live lengths and physical pages."""
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("FlashAttention-V100 regression is SM70/V100 only")
    flash_attn_v100 = pytest.importorskip("flash_attn_v100")
    torch.manual_seed(20261004)
    page_size = 2048
    page_count = 128
    query = torch.randn(batch_size, 8, 8, 128, device="cuda", dtype=torch.float16)
    key = torch.randn(page_count, page_size, 2, 128, device="cuda", dtype=query.dtype)
    value = torch.randn_like(key)
    pages = torch.stack(
        [torch.randperm(page_count, device="cuda") for _ in range(batch_size)]
    ).int()
    lengths = torch.full((batch_size,), 1024, device="cuda", dtype=torch.int32)
    output = torch.empty_like(query)

    def run() -> None:
        flash_attn_v100.flash_attn_prefill_paged(
            query,
            key,
            value,
            pages,
            lengths,
            out=output,
            causal=False,
            window_size=(2047, 2047),
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    # Empty partitions, 176-token split boundaries, partial pages, a full
    # window, and the maximum context all reuse the same captured graph.
    for sequence_length in [8, 175, 176, 177, 1024, 2047, 2048, 2055, 8192, 262144]:
        live_lengths = [max(8, sequence_length - 17 * b) for b in range(batch_size)]
        lengths.copy_(torch.tensor(live_lengths, device="cuda", dtype=torch.int32))
        graph.replay()
        for request, length in enumerate(live_lengths):
            positions = torch.arange(max(0, length - 8 - 2047), length, device="cuda")
            physical = pages[request, positions // page_size].long()
            k = key[physical, positions % page_size].repeat_interleave(4, dim=1)
            v = value[physical, positions % page_size].repeat_interleave(4, dim=1)
            qpos = torch.arange(length - 8, length, device="cuda")
            mask = (positions[None] >= qpos[:, None] - 2047) & (
                positions[None] <= qpos[:, None] + 2047
            )
            scores = torch.einsum("qhd,khd->hqk", query[request].float(), k.float())
            scores *= 128**-0.5
            scores.masked_fill_(~mask[None], -float("inf"))
            reference = torch.einsum("hqk,khd->qhd", scores.softmax(-1), v.float())
            torch.testing.assert_close(
                output[request].float(), reference, atol=6e-3, rtol=2e-2
            )
