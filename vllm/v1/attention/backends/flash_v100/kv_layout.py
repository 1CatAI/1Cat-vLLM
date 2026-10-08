# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paged/contiguous KV-cache views, gathers and query-length helpers."""

from __future__ import annotations

import torch

from vllm.logger import init_logger
from vllm.v1.attention.backends.flash_v100 import masks as _masks
from vllm.v1.attention.backends.flash_v100 import metadata as _metadata
from vllm.v1.attention.backends.flash_v100 import ops as _ops
from vllm.v1.attention.backends.flash_v100 import routing as _routing
from vllm.v1.attention.backends.flash_v100 import workspace as _workspace
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionMetadata,
)
from vllm.v1.attention.kv_codecs import (
    resolve_kv_codec,
)

logger = init_logger("vllm.v1.attention.backends.flash_attn_v100")
_warned_prefill_gather_oom = False
_prefill_gather_dense_workspaces: dict[
    tuple[int, int, torch.dtype, int, int, int],
    tuple[torch.Tensor, torch.Tensor],
] = {}


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


def _has_prefix_context(attn_metadata: TritonAttentionMetadata) -> bool:
    """Return True if any sequence has KV context before current query tokens."""
    query_start_loc_cpu = getattr(attn_metadata, "query_start_loc_cpu", None)
    seq_lens_cpu = getattr(attn_metadata, "seq_lens_cpu", None)
    if query_start_loc_cpu is not None and seq_lens_cpu is not None:
        query_lens = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        return bool(torch.any(query_lens != seq_lens_cpu).item())

    query_lens = attn_metadata.query_start_loc[1:] - attn_metadata.query_start_loc[:-1]
    return not torch.equal(query_lens, attn_metadata.seq_lens)


def _metadata_expects_more_query_tokens_than_available(
    attn_metadata: TritonAttentionMetadata,
    available_query_tokens: int,
) -> bool:
    """Return True when per-layer Q/K/V tensors are shorter than query metadata.

    Hybrid Qwen3.5/3.6 routes can feed a full-attention layer only the live
    query-token subset while the batch-level metadata still describes the
    wider request span. That shape is not a dense raw-QKV prefill; it must use
    the prefix/live-token compatible path.
    """
    query_start_loc_cpu = getattr(attn_metadata, "query_start_loc_cpu", None)
    query_start_loc = (
        query_start_loc_cpu
        if query_start_loc_cpu is not None
        else attn_metadata.query_start_loc
    )
    if len(query_start_loc) <= 1:
        return False
    expected_query_tokens = int(query_start_loc[-1].item())
    return available_query_tokens < expected_query_tokens


def _normalize_query_start_loc_for_available_tokens(
    query_start_loc: torch.Tensor,
    available_query_tokens: int,
) -> torch.Tensor:
    """Project metadata query spans onto the tokens actually present in Q/K/V.

    This is only needed when a hybrid/model-specific path feeds a full-attention
    layer a live-token subset instead of the full batch span described by the
    shared metadata.
    """
    num_seqs = len(query_start_loc) - 1
    if num_seqs <= 0:
        return query_start_loc

    expected_query_tokens = int(query_start_loc[-1].item())
    if available_query_tokens >= expected_query_tokens:
        return query_start_loc

    if available_query_tokens <= 0:
        return query_start_loc.new_zeros(query_start_loc.shape)

    if num_seqs == 1:
        return query_start_loc.new_tensor([0, available_query_tokens])

    if available_query_tokens == num_seqs:
        return torch.arange(
            num_seqs + 1,
            dtype=query_start_loc.dtype,
            device=query_start_loc.device,
        )

    if available_query_tokens % num_seqs == 0:
        q_per_seq = available_query_tokens // num_seqs
        orig_query_lens = query_start_loc[1:] - query_start_loc[:-1]
        if int(orig_query_lens.min().item()) >= q_per_seq:
            return torch.arange(
                0,
                available_query_tokens + 1,
                q_per_seq,
                dtype=query_start_loc.dtype,
                device=query_start_loc.device,
            )

    raise RuntimeError(
        "FLASH_ATTN_V100 received fewer layer query tokens than query metadata "
        "describes, and the per-sequence live-token layout could not be "
        "reconstructed safely."
    )


def _extract_contiguous_kv_from_paged_cache(
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    total_tokens: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract contiguous K/V from paged KV cache.

    Uses the CUDA extension when available and falls back to a Python path.
    """

    paged_kv_utils = _ops._get_paged_kv_utils()

    key_cache, value_cache = _split_paged_kv_cache(kv_cache)

    if paged_kv_utils is not None and key_cache.dtype != torch.uint8:
        if hasattr(paged_kv_utils, "paged_kv_to_contiguous"):
            k_cont, v_cont = paged_kv_utils.paged_kv_to_contiguous(
                key_cache, value_cache, block_table, seq_lens
            )
        else:
            k_cont = paged_kv_utils.paged_to_contiguous(
                key_cache, block_table, seq_lens
            )
            v_cont = paged_kv_utils.paged_to_contiguous(
                value_cache, block_table, seq_lens
            )
        if total_tokens is None:
            total_tokens = int(seq_lens.sum().item())
        return k_cont[:total_tokens], v_cont[:total_tokens]

    # Slow Python fallback.
    batch_size = block_table.shape[0]
    if total_tokens is None:
        total_tokens = int(seq_lens.sum().item())

    k_cont = torch.empty(
        (total_tokens, num_kv_heads, head_dim),
        dtype=key_cache.dtype,
        device=key_cache.device,
    )
    v_cont = torch.empty(
        (total_tokens, num_kv_heads, head_dim),
        dtype=value_cache.dtype,
        device=value_cache.device,
    )

    token_offset = 0
    for batch_idx in range(batch_size):
        seq_len = int(seq_lens[batch_idx].item())
        if seq_len == 0:
            continue

        num_blocks = (seq_len + block_size - 1) // block_size
        for block_idx in range(num_blocks):
            physical_block_idx = int(block_table[batch_idx, block_idx].item())
            start_token = block_idx * block_size
            end_token = min(start_token + block_size, seq_len)
            n = end_token - start_token

            k_cont[token_offset : token_offset + n] = key_cache[physical_block_idx, :n]
            v_cont[token_offset : token_offset + n] = value_cache[
                physical_block_idx, :n
            ]
            token_offset += n

    return k_cont, v_cont


def _dequantize_fp8_contiguous_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: float,
    v_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not _routing._uses_fp8_kv_cache(kv_cache_dtype):
        return key, value
    codec = resolve_kv_codec(kv_cache_dtype)
    if codec is None or not codec.quantized:
        raise ValueError(f"Unsupported FLASH_ATTN_V100 fp8 dtype: {kv_cache_dtype}")
    return (
        codec.dequantize(key, k_scale, torch.float16),
        codec.dequantize(value, v_scale, torch.float16),
    )


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
        _metadata._as_flash_v100_metadata(
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


def _get_prefill_gather_dense_workspace(
    key_cache: torch.Tensor,
    required_blocks: int,
    max_blocks: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    global _warned_prefill_gather_oom

    if required_blocks <= 0:
        return None
    device_index = key_cache.device.index
    if device_index is None:
        device_index = (
            torch.accelerator.current_device_index() if key_cache.is_cuda else -1
        )
    stream_id = (
        int(torch.cuda.current_stream(key_cache.device).cuda_stream)
        if key_cache.is_cuda
        else 0
    )
    cache_key = (
        device_index,
        stream_id,
        key_cache.dtype,
        int(key_cache.shape[1]),
        int(key_cache.shape[2]),
        int(key_cache.shape[3]),
    )
    workspace = _prefill_gather_dense_workspaces.get(cache_key)
    if workspace is not None and workspace[0].shape[0] >= required_blocks:
        return workspace[0][:required_blocks], workspace[1][:required_blocks]
    if _routing._is_cuda_graph_capturing(key_cache):
        return None

    previous_capacity = workspace[0].shape[0] if workspace is not None else 0
    capacity = max(required_blocks, previous_capacity * 2)
    if max_blocks is not None:
        # Doubling must not reserve more than the block table can ever address
        # (max_model_len worth of pages); beyond that the overshoot is pure
        # loss against the KV cache.
        capacity = min(capacity, max(required_blocks, int(max_blocks)))
    shape = (capacity, *key_cache.shape[1:])

    def _allocate() -> tuple[torch.Tensor, ...]:
        key_out = torch.empty(shape, dtype=key_cache.dtype, device=key_cache.device)
        return key_out, torch.empty_like(key_out)

    workspace = None
    _prefill_gather_dense_workspaces.pop(cache_key, None)
    allocated = _workspace._allocate_growing_workspace(
        _allocate, on_cuda=key_cache.is_cuda
    )
    if allocated is None:
        if not _warned_prefill_gather_oom:
            logger.warning(
                "Insufficient memory for the long-prefill dense KV workspace; "
                "falling back to direct paged attention."
            )
            _warned_prefill_gather_oom = True
        return None
    key_out, value_out = allocated
    _prefill_gather_dense_workspaces[cache_key] = key_out, value_out
    return key_out[:required_blocks], value_out[:required_blocks]


def _gather_paged_kv_to_exact_dense(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table_row: torch.Tensor,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Gather one logical paged sequence into reusable dense K/V storage."""
    if (
        seq_len <= 0
        or key_cache.dtype != torch.float16
        or value_cache.dtype != torch.float16
        or key_cache.shape != value_cache.shape
        or key_cache.ndim != 4
        or block_table_row.ndim != 1
        or block_table_row.device != key_cache.device
        or block_table_row.dtype not in (torch.int32, torch.int64)
    ):
        return None

    block_size = int(key_cache.shape[1])
    required_blocks = _masks._cdiv_int(seq_len, block_size)
    if required_blocks > int(block_table_row.shape[0]):
        return None
    workspace = _get_prefill_gather_dense_workspace(
        key_cache,
        required_blocks,
        max_blocks=int(block_table_row.shape[0]),
    )
    if workspace is None:
        return None

    key_pages, value_pages = workspace
    page_indices = block_table_row[:required_blocks]
    torch.index_select(key_cache, 0, page_indices, out=key_pages)
    torch.index_select(value_cache, 0, page_indices, out=value_pages)
    num_kv_heads = int(key_cache.shape[2])
    head_dim = int(key_cache.shape[3])
    key_dense = key_pages.flatten(0, 1)[:seq_len].reshape(
        1, seq_len, num_kv_heads, head_dim
    )
    value_dense = value_pages.flatten(0, 1)[:seq_len].reshape(
        1, seq_len, num_kv_heads, head_dim
    )
    return key_dense, value_dense
