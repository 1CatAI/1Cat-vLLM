# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native page4 QSA for ranks with six query heads per KV head or a multiple.

A TP2 rank of Qwen3.8-Flash-Next carries 12 query heads on one KV head, a
single-rank stage 24 query heads on two. The native grouped/XQA page4 routes run
each KV head's block of query heads as consecutive 6-head pseudo-rows against
that head's slice of the K/V-interleaved cache. The result must match the Triton
split-K route and an FP32 reference.
"""

from __future__ import annotations

from unittest import mock

import pytest
import torch

from tests.kernels.flash_v100_page4 import require_flash_v100_page4
from vllm.models.qwen4_exp.nvidia.ops import qsa as qsa_ops

PAGE_SIZE = 16
HEAD_DIM = 256
GROUPS = 512


def _selection(positions: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """QSA selection rows: 512 complete 4-token groups, then the open group."""
    rows = []
    for position in positions.tolist():
        complete = (position + 1) // 4
        middle = torch.randperm(complete - 2, generator=generator)[: GROUPS - 2] + 1
        groups = torch.cat([torch.tensor([0, complete - 1]), middle]).sort().values
        tokens = (groups[:, None] * 4 + torch.arange(4)).reshape(-1)
        tail = torch.arange(complete * 4, position + 1)
        row = torch.full((GROUPS * 4 + 3,), -1, dtype=torch.int32)
        row[: tokens.numel()] = tokens.to(torch.int32)
        row[tokens.numel() : tokens.numel() + tail.numel()] = tail.to(torch.int32)
        rows.append(row)
    return torch.stack(rows)


def _reference(query, key_cache, value_cache, indices, block_table) -> torch.Tensor:
    valid = indices >= 0
    tokens = indices.clamp(min=0).long()
    slots = block_table[0, tokens // PAGE_SIZE].long() * PAGE_SIZE + tokens % PAGE_SIZE
    kv_heads = key_cache.shape[2]
    group = query.shape[1] // kv_heads
    out = []
    for kv_head in range(kv_heads):
        keys = key_cache[:, :, kv_head].reshape(-1, HEAD_DIM)[slots].float()
        values = value_cache[:, :, kv_head].reshape(-1, HEAD_DIM)[slots].float()
        heads = query[:, kv_head * group : (kv_head + 1) * group].float()
        scores = torch.einsum("rhd,rtd->rht", heads, keys) * HEAD_DIM**-0.5
        scores = scores.masked_fill(~valid[:, None, :], float("-inf"))
        out.append(torch.einsum("rht,rtd->rhd", scores.softmax(-1), values))
    return torch.cat(out, dim=1)


# Query/KV heads per rank: TP4 (6/1), TP2 (12/1) and TP1 (24/2) of
# Qwen3.8-Flash-Next, plus 24 query heads on one KV head.
@pytest.mark.parametrize(("heads", "kv_heads"), [(6, 1), (12, 1), (24, 2), (24, 1)])
@pytest.mark.parametrize("rows", [64, 69])
@torch.no_grad()  # inference tensors would leak into the cached page4 workspaces
def test_sm70_qsa_page4_head_groups_match_triton(
    heads: int, kv_heads: int, rows: int
) -> None:
    require_flash_v100_page4()
    torch.manual_seed(11)
    generator = torch.Generator().manual_seed(11)
    context = 6000
    blocks = context // PAGE_SIZE + 4
    query = torch.randn((rows, heads, HEAD_DIM), dtype=torch.float16, device="cuda")
    # The serving layout: K and V interleaved inside every cache block.
    kv_cache = torch.randn(
        (blocks, 2, PAGE_SIZE, kv_heads, HEAD_DIM), dtype=torch.float16, device="cuda"
    )
    key_cache, value_cache = kv_cache.unbind(1)
    block_table = torch.randperm(blocks, generator=generator).to(torch.int32)
    block_table = block_table[None, :].cuda()
    positions = torch.arange(context - rows, context, dtype=torch.int64)
    indices = _selection(positions, generator).cuda()
    positions = positions.cuda()
    requests = torch.zeros(rows, dtype=torch.int32, device="cuda")
    lengths = torch.tensor([context], dtype=torch.int32, device="cuda")

    triton_out = torch.empty_like(query)
    qsa_ops.qsa_sparse_paged_attention(
        query, key_cache, value_cache, indices, block_table, requests, out=triton_out
    )
    native_out = torch.empty_like(query)
    grouped = mock.patch.object(
        qsa_ops,
        "_qsa_sparse_paged_attention_sm70_grouped_page4",
        wraps=qsa_ops._qsa_sparse_paged_attention_sm70_grouped_page4,
    )
    with grouped as grouped_route:
        qsa_ops.qsa_sparse_paged_attention(
            query,
            key_cache,
            value_cache,
            indices,
            block_table,
            requests,
            out=native_out,
            query_positions=positions,
            sequence_lengths=lengths,
        )
    assert grouped_route.call_count == kv_heads
    for call in grouped_route.call_args_list:
        assert call.args[0].shape[1:] == (6, HEAD_DIM)
        assert call.args[1].shape[2] == 1

    reference = _reference(query, key_cache, value_cache, indices, block_table)
    native = native_out.float()
    torch.testing.assert_close(native, triton_out.float(), atol=2e-3, rtol=0)
    torch.testing.assert_close(native, reference, atol=2e-3, rtol=0)
