# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paged cache layout and contiguous views, separate from KV encoding math.

The existing FP16-only view admission is retained during extraction. Encoded
cache tiles must pass through their reader before becoming dense FP16 views.
"""

from __future__ import annotations

import torch

from vllm.v1.attention.backends.flash_v100.metadata import _as_flash_v100_metadata
from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadata


def _split_paged_kv_cache(
    kv_cache: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(kv_cache, (list, tuple)):
        if len(kv_cache) != 2:
            raise ValueError(
                f"Unexpected KV cache tuple/list length {len(kv_cache)}; expected 2"
            )
        return kv_cache[0], kv_cache[1]

    if kv_cache.ndim < 2:
        raise ValueError(
            f"Unexpected KV cache shape {tuple(kv_cache.shape)}; "
            "expected dimension 2 at axis 0 or 1"
        )

    # Standard vLLM paged KV layout is [num_blocks, 2, block_size, heads, dim].
    # Prefer axis 1 so num_blocks == 2 does not get mistaken for K/V.
    if kv_cache.shape[1] == 2:
        return kv_cache.unbind(1)
    if kv_cache.shape[0] == 2:
        return kv_cache.unbind(0)

    raise ValueError(
        f"Unexpected KV cache shape {tuple(kv_cache.shape)}; "
        "expected dimension 2 at axis 0 or 1"
    )


def _same_storage(left: torch.Tensor, right: torch.Tensor) -> bool:
    return left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()


def _contiguous_paged_start_block(
    key_cache: torch.Tensor,
    block_table_row: torch.Tensor,
    seq_len: int,
    block_size: int,
    attn_metadata: TritonAttentionMetadata,
    seq_idx: int,
) -> tuple[int, int] | None:
    if seq_len <= 0 or block_size <= 0:
        return None
    num_blocks = (seq_len + block_size - 1) // block_size
    if num_blocks <= 0 or num_blocks > int(block_table_row.shape[0]):
        return None

    cache_key = (
        int(seq_idx),
        int(seq_len),
        int(block_size),
        int(block_table_row.data_ptr()),
        int(key_cache.data_ptr()),
    )
    contig_cache = getattr(attn_metadata, "flash_v100_contig_dense_cache", None)
    if contig_cache is None:
        contig_cache = {}
        _as_flash_v100_metadata(
            attn_metadata
        ).flash_v100_contig_dense_cache = contig_cache

    start_block = contig_cache.get(cache_key)
    if start_block is None:
        blocks_cpu = block_table_row[:num_blocks].detach().cpu()
        if int(blocks_cpu[0].item()) < 0:
            contig_cache[cache_key] = -1
            return None
        if num_blocks > 1:
            expected = blocks_cpu[0] + torch.arange(
                num_blocks,
                dtype=blocks_cpu.dtype,
                device=blocks_cpu.device,
            )
            if not bool(torch.equal(blocks_cpu, expected)):
                contig_cache[cache_key] = -1
                return None
        start_block = int(blocks_cpu[0].item())
        contig_cache[cache_key] = start_block

    if start_block < 0:
        return None
    if start_block + num_blocks > int(key_cache.shape[0]):
        return None

    return start_block, num_blocks


def _contiguous_paged_kv_view(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table_row: torch.Tensor,
    seq_len: int,
    block_size: int,
    attn_metadata: TritonAttentionMetadata,
    seq_idx: int,
    allow_copy: bool,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return a dense [1, N, Hkv, D] K/V view for physically contiguous pages."""
    if key_cache.dtype != torch.float16 or value_cache.dtype != torch.float16:
        return None
    if key_cache.shape != value_cache.shape:
        return None
    if not allow_copy and (
        not key_cache.is_contiguous() or not value_cache.is_contiguous()
    ):
        return None

    start_info = _contiguous_paged_start_block(
        key_cache,
        block_table_row,
        seq_len,
        block_size,
        attn_metadata,
        seq_idx,
    )
    if start_info is None:
        return None
    start_block, num_blocks = start_info

    num_kv_heads = key_cache.shape[2]
    head_dim = key_cache.shape[3]
    end_block = start_block + num_blocks
    key_block_slice = key_cache[start_block:end_block]
    value_block_slice = value_cache[start_block:end_block]
    key_flat = key_block_slice.reshape(-1, num_kv_heads, head_dim)
    value_flat = value_block_slice.reshape(-1, num_kv_heads, head_dim)
    return (
        key_flat[:seq_len].unsqueeze(0),
        value_flat[:seq_len].unsqueeze(0),
    )


def _contiguous_paged_kv_bhmd(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table_row: torch.Tensor,
    seq_len: int,
    block_size: int,
    attn_metadata: TritonAttentionMetadata,
    seq_idx: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return dense [1, Hkv, N, D] K/V tensors for contiguous paged cache."""
    if key_cache.dtype != torch.float16 or value_cache.dtype != torch.float16:
        return None
    if key_cache.shape != value_cache.shape:
        return None

    start_info = _contiguous_paged_start_block(
        key_cache,
        block_table_row,
        seq_len,
        block_size,
        attn_metadata,
        seq_idx,
    )
    if start_info is None:
        return None
    start_block, num_blocks = start_info

    num_kv_heads = key_cache.shape[2]
    head_dim = key_cache.shape[3]
    end_block = start_block + num_blocks
    key_blocks = key_cache[start_block:end_block]
    value_blocks = value_cache[start_block:end_block]
    key_bhmd = (
        key_blocks.permute(2, 0, 1, 3)
        .reshape(1, num_kv_heads, -1, head_dim)[:, :, :seq_len, :]
        .contiguous()
    )
    value_bhmd = (
        value_blocks.permute(2, 0, 1, 3)
        .reshape(1, num_kv_heads, -1, head_dim)[:, :, :seq_len, :]
        .contiguous()
    )
    return key_bhmd, value_bhmd
