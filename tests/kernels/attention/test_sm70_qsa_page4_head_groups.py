# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native page4 QSA for ranks with a multiple of six query heads.

A TP2 rank of Qwen3.8-Flash-Next carries 12 query heads, a single-rank stage 24.
The native grouped/XQA page4 routes run them as consecutive 6-head pseudo-rows.
The result must match the Triton split-K route and an FP32 reference.
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
    keys = key_cache.reshape(-1, HEAD_DIM)[slots].float()
    values = value_cache.reshape(-1, HEAD_DIM)[slots].float()
    scores = torch.einsum("rhd,rtd->rht", query.float(), keys) * HEAD_DIM**-0.5
    scores = scores.masked_fill(~valid[:, None, :], float("-inf"))
    return torch.einsum("rht,rtd->rhd", scores.softmax(-1), values)


@pytest.mark.parametrize("heads", [12, 24])
@pytest.mark.parametrize("rows", [64, 69])
@torch.no_grad()  # inference tensors would leak into the cached page4 workspaces
def test_sm70_qsa_page4_head_groups_match_triton(heads: int, rows: int) -> None:
    require_flash_v100_page4()
    torch.manual_seed(11)
    generator = torch.Generator().manual_seed(11)
    context = 6000
    blocks = context // PAGE_SIZE + 4
    query = torch.randn((rows, heads, HEAD_DIM), dtype=torch.float16, device="cuda")
    key_cache = torch.randn(
        (blocks, PAGE_SIZE, 1, HEAD_DIM), dtype=torch.float16, device="cuda"
    )
    value_cache = torch.randn_like(key_cache)
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
    assert grouped_route.call_count == 1
    pseudo_query = grouped_route.call_args.args[0]
    assert pseudo_query.shape[1:] == (6, HEAD_DIM)

    reference = _reference(query, key_cache, value_cache, indices, block_table)
    native = native_out.float()
    torch.testing.assert_close(native, triton_out.float(), atol=2e-3, rtol=0)
    torch.testing.assert_close(native, reference, atol=2e-3, rtol=0)
