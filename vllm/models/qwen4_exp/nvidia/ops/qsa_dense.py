# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dense causal QSA decode attention for contexts within the indexer budget."""

import torch

DENSE_HEAD_SIZE = 256
DENSE_SPLITS = 32
DENSE_WARPS = 16
DENSE_ROWS = 32


def qsa_dense_supported(
    max_query_len: int,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    cache_dtype: torch.dtype,
    num_requests: int,
) -> bool:
    """Shape admission; requests may hold at most ``max_query_len`` tokens."""
    if not hasattr(torch.ops._C, "qsa_dense_decode_sm70_out"):
        return False
    if (
        cache_dtype != torch.float16
        or head_size != DENSE_HEAD_SIZE
        or num_heads % num_kv_heads
        or num_requests <= 0
    ):
        return False
    return (num_heads // num_kv_heads) * max_query_len <= DENSE_ROWS


def qsa_dense_decode(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    positions: torch.Tensor,
    output: torch.Tensor,
    output_gate: torch.Tensor | None,
    num_requests: int,
) -> None:
    kv_heads = key_cache.shape[2]
    workspace_o = torch.empty(
        num_requests * kv_heads * DENSE_SPLITS * DENSE_ROWS * DENSE_HEAD_SIZE,
        dtype=torch.float32,
        device=query.device,
    )
    workspace_ml = torch.empty(
        num_requests * kv_heads * DENSE_SPLITS * DENSE_ROWS * 2,
        dtype=torch.float32,
        device=query.device,
    )
    torch.ops._C.qsa_dense_decode_sm70_out(
        output,
        query,
        key_cache,
        value_cache,
        block_table.to(torch.int32),
        token_to_req.to(torch.int32),
        positions.to(torch.int32),
        output_gate,
        workspace_o,
        workspace_ml,
        num_requests,
        DENSE_SPLITS,
        DENSE_WARPS,
    )
