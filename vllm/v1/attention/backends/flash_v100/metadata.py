# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Format-independent, graph-stable metadata for Flash-V100 attention."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from typing import cast

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadata

_MIXED_ROWS_PLAN_ATTR = "_flash_v100_mixed_decode_rows_plan"
_MIXED_ROWS_GROUP = 8


@triton.jit
def _sm70_prepare_smallq_decode_metadata_kernel(
    out_block_table_ptr,
    out_seq_lens_ptr,
    out_query_start_loc_ptr,
    block_table_ptr,
    seq_lens_ptr,
    query_start_loc_ptr,
    block_table_stride,
    out_block_table_stride,
    num_reqs,
    num_query_tokens,
    real_num_query_tokens,
    block_cols,
    REQ_BLOCK: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    token_idx = tl.program_id(0)

    # Find the request that owns this flattened query token. Equal trailing
    # boundaries are CUDA-graph padding; clamping maps them to the final padded
    # request, matching repeat_query_lens[-1] += padding_tokens.
    req_offsets = tl.arange(0, REQ_BLOCK)
    req_mask = req_offsets < num_reqs
    query_ends = tl.load(
        query_start_loc_ptr + req_offsets + 1,
        mask=req_mask,
        other=0x7FFFFFFF,
    )
    req_idx = tl.sum((token_idx >= query_ends).to(tl.int32), axis=0)
    req_idx = tl.minimum(req_idx, num_reqs - 1)

    query_start = tl.load(query_start_loc_ptr + req_idx)
    query_end = tl.load(query_start_loc_ptr + req_idx + 1)
    query_len = query_end - query_start
    seq_len = tl.load(seq_lens_ptr + req_idx)
    effective_seq_len = tl.maximum(seq_len, query_len)
    decode_seq_len = effective_seq_len - query_len + token_idx - query_start + 1
    is_padding = token_idx >= real_num_query_tokens
    tl.store(
        out_seq_lens_ptr + token_idx,
        tl.where(is_padding, 0, decode_seq_len),
    )

    cols = tl.arange(0, BLOCK_COLS)
    col_mask = cols < block_cols
    block_ids = tl.load(
        block_table_ptr + req_idx * block_table_stride + cols,
        mask=col_mask,
        other=0,
    )
    block_ids = tl.maximum(block_ids, 0)
    block_ids = tl.where(is_padding, 0, block_ids)
    tl.store(
        out_block_table_ptr + token_idx * out_block_table_stride + cols,
        block_ids,
        mask=col_mask,
    )

    # The same launch also refreshes the graph-stable query boundaries.
    if token_idx == 0:
        boundary_offsets = tl.arange(0, REQ_BLOCK)
        boundary_mask = boundary_offsets < num_reqs + 1
        boundaries = tl.load(
            query_start_loc_ptr + boundary_offsets,
            mask=boundary_mask,
            other=0,
        )
        tl.store(
            out_query_start_loc_ptr + boundary_offsets,
            boundaries,
            mask=boundary_mask,
        )


def _sm70_prepare_smallq_decode_metadata(
    out_block_table: torch.Tensor,
    out_seq_lens: torch.Tensor,
    out_query_start_loc: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    *,
    num_reqs: int,
    num_query_tokens: int,
    real_num_query_tokens: int,
) -> None:
    """Materialize persistent Flash-V100 verifier metadata in one launch."""
    if num_reqs <= 0 or num_query_tokens <= 0:
        raise ValueError("small-query metadata requires positive request/token counts")
    block_cols = int(block_table.shape[1])
    if block_cols <= 0:
        raise ValueError("small-query block table must have at least one column")
    if out_block_table.shape[0] < num_query_tokens:
        raise ValueError("small-query output block table is too small")
    if out_seq_lens.numel() < num_query_tokens:
        raise ValueError("small-query output sequence buffer is too small")
    if out_query_start_loc.numel() < num_reqs + 1:
        raise ValueError("small-query output boundary buffer is too small")

    _sm70_prepare_smallq_decode_metadata_kernel[(num_query_tokens,)](
        out_block_table,
        out_seq_lens,
        out_query_start_loc,
        block_table,
        seq_lens,
        query_start_loc,
        block_table.stride(0),
        out_block_table.stride(0),
        num_reqs,
        num_query_tokens,
        real_num_query_tokens,
        block_cols,
        REQ_BLOCK=triton.next_power_of_2(num_reqs + 1),
        BLOCK_COLS=triton.next_power_of_2(block_cols),
        num_warps=1,
    )


@triton.jit
def _load_sm70_smallq_i32_ptr(ptrs, group_id):
    ptr = tl.load(ptrs + group_id)
    return tl.cast(ptr, tl.pointer_type(tl.int32))


@triton.jit
def _sm70_prepare_grouped_smallq_decode_metadata_kernel(
    out_block_table_ptrs,
    out_seq_lens_ptrs,
    out_query_start_loc_ptrs,
    block_table_ptrs,
    block_table_strides,
    out_block_table_strides,
    block_col_counts,
    seq_lens_ptr,
    query_start_loc_ptr,
    num_reqs,
    real_num_query_tokens,
    REQ_BLOCK: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    """Materialize every target full-attention group's verifier metadata."""
    group_id = tl.program_id(0)
    token_idx = tl.program_id(1)
    out_block_table = _load_sm70_smallq_i32_ptr(out_block_table_ptrs, group_id)
    out_seq_lens = _load_sm70_smallq_i32_ptr(out_seq_lens_ptrs, group_id)
    out_query_start_loc = _load_sm70_smallq_i32_ptr(out_query_start_loc_ptrs, group_id)
    block_table = _load_sm70_smallq_i32_ptr(block_table_ptrs, group_id)
    block_table_stride = tl.load(block_table_strides + group_id)
    out_block_table_stride = tl.load(out_block_table_strides + group_id)
    block_cols = tl.load(block_col_counts + group_id)

    req_offsets = tl.arange(0, REQ_BLOCK)
    req_mask = req_offsets < num_reqs
    query_ends = tl.load(
        query_start_loc_ptr + req_offsets + 1,
        mask=req_mask,
        other=0x7FFFFFFF,
    )
    req_idx = tl.sum((token_idx >= query_ends).to(tl.int32), axis=0)
    req_idx = tl.minimum(req_idx, num_reqs - 1)

    query_start = tl.load(query_start_loc_ptr + req_idx)
    query_end = tl.load(query_start_loc_ptr + req_idx + 1)
    query_len = query_end - query_start
    seq_len = tl.load(seq_lens_ptr + req_idx)
    effective_seq_len = tl.maximum(seq_len, query_len)
    decode_seq_len = effective_seq_len - query_len + token_idx - query_start + 1
    is_padding = token_idx >= real_num_query_tokens
    tl.store(
        out_seq_lens + token_idx,
        tl.where(is_padding, 0, decode_seq_len),
    )

    cols = tl.arange(0, BLOCK_COLS)
    col_mask = cols < block_cols
    block_ids = tl.load(
        block_table + req_idx * block_table_stride + cols,
        mask=col_mask,
        other=0,
    )
    block_ids = tl.maximum(block_ids, 0)
    block_ids = tl.where(is_padding, 0, block_ids)
    tl.store(
        out_block_table + token_idx * out_block_table_stride + cols,
        block_ids,
        mask=col_mask,
    )

    if token_idx == 0:
        boundary_offsets = tl.arange(0, REQ_BLOCK)
        boundary_mask = boundary_offsets < num_reqs + 1
        boundaries = tl.load(
            query_start_loc_ptr + boundary_offsets,
            mask=boundary_mask,
            other=0,
        )
        tl.store(
            out_query_start_loc + boundary_offsets,
            boundaries,
            mask=boundary_mask,
        )


@dataclass
class DFlash2SmallQGroupDescriptor:
    """Persistent pointer tables for grouped Flash-V100 verifier metadata."""

    key: tuple[object, ...]
    block_table_ptrs: torch.Tensor
    out_block_table_ptrs: torch.Tensor
    out_seq_lens_ptrs: torch.Tensor
    out_query_start_loc_ptrs: torch.Tensor
    block_table_strides: torch.Tensor
    out_block_table_strides: torch.Tensor
    block_col_counts: torch.Tensor


@dataclass(frozen=True)
class DFlash2SmallQPreparedMetadata:
    """Graph-stable buffers already refreshed by the grouped launch."""

    builder_id: int
    num_reqs: int
    num_query_tokens: int
    max_seq_len_hint: int
    workspace_seq_capacity_hint: int
    partition_size_hint: int | None = None


def _sm70_prepare_grouped_smallq_decode_metadata(
    out_block_tables: list[torch.Tensor],
    out_seq_lens: list[torch.Tensor],
    out_query_start_locs: list[torch.Tensor],
    block_tables: list[torch.Tensor],
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    *,
    num_reqs: int,
    num_query_tokens: int,
    real_num_query_tokens: int,
    descriptor: DFlash2SmallQGroupDescriptor | None = None,
) -> DFlash2SmallQGroupDescriptor:
    """Refresh N full-attention cache groups in one Triton launch."""
    num_groups = len(block_tables)
    if num_groups <= 0:
        raise ValueError("grouped small-query metadata requires at least one group")
    if not (
        len(out_block_tables)
        == len(out_seq_lens)
        == len(out_query_start_locs)
        == num_groups
    ):
        raise ValueError("grouped small-query metadata lists must have equal lengths")
    if num_reqs <= 0 or num_query_tokens <= 0:
        raise ValueError("grouped small-query metadata requires positive sizes")
    if not 0 <= real_num_query_tokens <= num_query_tokens:
        raise ValueError("real query token count exceeds grouped launch size")

    block_col_counts = [int(table.shape[1]) for table in block_tables]
    if any(cols <= 0 for cols in block_col_counts):
        raise ValueError("grouped small-query block tables cannot be empty")
    max_block_cols = max(block_col_counts)
    device = block_tables[0].device
    if (
        seq_lens.device != device
        or query_start_loc.device != device
        or seq_lens.dtype != torch.int32
        or query_start_loc.dtype != torch.int32
        or not seq_lens.is_contiguous()
        or not query_start_loc.is_contiguous()
        or seq_lens.numel() < num_reqs
        or query_start_loc.numel() < num_reqs + 1
    ):
        raise ValueError("grouped small-query input metadata contract mismatch")
    key: tuple[object, ...] = (
        device.type,
        device.index,
        tuple(block_col_counts),
        tuple(table.data_ptr() for table in block_tables),
        tuple(table.data_ptr() for table in out_block_tables),
        tuple(tensor.data_ptr() for tensor in out_seq_lens),
        tuple(tensor.data_ptr() for tensor in out_query_start_locs),
        tuple(table.stride(0) for table in block_tables),
        tuple(table.stride(0) for table in out_block_tables),
    )
    if descriptor is None or descriptor.key != key:
        for group, (
            block_table,
            out_block_table,
            out_seq_len,
            out_query_start,
        ) in enumerate(
            zip(
                block_tables,
                out_block_tables,
                out_seq_lens,
                out_query_start_locs,
                strict=True,
            )
        ):
            block_cols = block_col_counts[group]
            if (
                block_table.device != device
                or out_block_table.device != device
                or out_seq_len.device != device
                or out_query_start.device != device
                or block_table.dtype != torch.int32
                or out_block_table.dtype != torch.int32
                or out_seq_len.dtype != torch.int32
                or out_query_start.dtype != torch.int32
                or block_table.ndim != 2
                or out_block_table.ndim != 2
                or block_table.shape[0] < num_reqs
                or out_block_table.shape[0] < num_query_tokens
                or out_block_table.shape[1] < block_cols
                or block_table.stride(1) != 1
                or out_block_table.stride(1) != 1
                or not out_seq_len.is_contiguous()
                or not out_query_start.is_contiguous()
                or out_seq_len.numel() < num_query_tokens
                or out_query_start.numel() < num_reqs + 1
            ):
                raise ValueError(
                    f"grouped small-query metadata contract mismatch for group {group}"
                )
        descriptor = DFlash2SmallQGroupDescriptor(
            key=key,
            block_table_ptrs=torch.tensor(
                [table.data_ptr() for table in block_tables],
                dtype=torch.uint64,
                device=device,
            ),
            out_block_table_ptrs=torch.tensor(
                [table.data_ptr() for table in out_block_tables],
                dtype=torch.uint64,
                device=device,
            ),
            out_seq_lens_ptrs=torch.tensor(
                [tensor.data_ptr() for tensor in out_seq_lens],
                dtype=torch.uint64,
                device=device,
            ),
            out_query_start_loc_ptrs=torch.tensor(
                [tensor.data_ptr() for tensor in out_query_start_locs],
                dtype=torch.uint64,
                device=device,
            ),
            block_table_strides=torch.tensor(
                [table.stride(0) for table in block_tables],
                dtype=torch.int64,
                device=device,
            ),
            out_block_table_strides=torch.tensor(
                [table.stride(0) for table in out_block_tables],
                dtype=torch.int64,
                device=device,
            ),
            block_col_counts=torch.tensor(
                block_col_counts,
                dtype=torch.int32,
                device=device,
            ),
        )

    _sm70_prepare_grouped_smallq_decode_metadata_kernel[(num_groups, num_query_tokens)](
        descriptor.out_block_table_ptrs,
        descriptor.out_seq_lens_ptrs,
        descriptor.out_query_start_loc_ptrs,
        descriptor.block_table_ptrs,
        descriptor.block_table_strides,
        descriptor.out_block_table_strides,
        descriptor.block_col_counts,
        seq_lens,
        query_start_loc,
        num_reqs,
        real_num_query_tokens,
        REQ_BLOCK=triton.next_power_of_2(num_reqs + 1),
        BLOCK_COLS=triton.next_power_of_2(max_block_cols),
        num_warps=1,
    )
    return descriptor


class FlashAttnV100Metadata(TritonAttentionMetadata):
    """Static view of Flash-V100 fields attached to Triton metadata."""

    query_start_loc_cpu: torch.Tensor
    seq_lens_cpu: torch.Tensor
    causal: bool
    max_model_len: int
    flash_v100_cudagraph_capture: bool
    flash_v100_batch_context_routing: bool
    flash_v100_contig_dense_cache: dict[tuple[int, int, int, int, int], int]
    prefix_anchor_lens: torch.Tensor | None = None
    decode_sliding_window: int | None = None
    flash_v100_decode_max_seq_len_hint: int | None
    flash_v100_decode_workspace_seq_capacity_hint: int | None
    flash_v100_static_decode_seq_hint: int | None
    flash_v100_decode_active_num_partitions: torch.Tensor | None
    ddtree_parent_ids: torch.Tensor | None
    ddtree_parent_ids_cpu: torch.Tensor | None
    ddtree_num_tree_tokens_cpu: torch.Tensor | None
    ddtree_seq_lens_restored_for_triton: bool
    ddtree_query_start_loc_restored_for_triton: bool
    smallq_decode_block_table: torch.Tensor | None
    smallq_decode_seq_lens: torch.Tensor | None
    smallq_query_start_loc: torch.Tensor | None
    smallq_decode_max_seq_len_hint: int | None
    smallq_decode_workspace_seq_capacity_hint: int | None
    smallq_decode_partition_size_hint: int | None
    is_dflash_selector_target: bool


def _as_flash_v100_metadata(
    attn_metadata: TritonAttentionMetadata,
) -> FlashAttnV100Metadata:
    # The inherited Triton builder creates the object; this backend attaches
    # the fields above before any Flash-V100 path consumes them.
    return cast(FlashAttnV100Metadata, attn_metadata)


class _MixedDecodeRowsPlan:
    """Layout of the small-query rows of one mixed prefill/decode step.

    The host side depends only on ``query_start_loc`` and the sequence-length
    shadow, so it is built once per step and shared by every attention layer of
    the group instead of re-deriving it (lists, host-to-device copies, gathers)
    in each layer. Visible KV lengths are never taken from the host shadow: it
    can be an upper bound under async speculative decoding, so every row length
    is derived on the device from the authoritative ``seq_lens``.

    Tokens are ordered by request and then by position inside the request.
    A request with ``q`` query tokens is split into ``ceil(q / 8)`` groups of
    eight rows for the request-major grouped operator; the causal boundary of
    each row is carried by its own length, so slicing a longer span is exact.
    """

    __slots__ = (
        "rows",
        "max_query_len",
        "max_seq_len_hint",
        "num_groups",
        "src_idx",
        "token_req",
        "token_delta",
        "dst_idx",
        "group_req",
        "_token_lengths",
        "_group_lengths",
        "_group_table",
    )

    def __init__(
        self,
        rows: tuple[int, ...],
        qsl: list[int],
        seq_lens_host: list[int],
        device: torch.device,
    ) -> None:
        src: list[int] = []
        req: list[int] = []
        delta: list[int] = []
        dst: list[int] = []
        group_req: list[int] = []
        max_query_len = 0
        max_seq_len = 0
        for i in rows:
            q_len = qsl[i + 1] - qsl[i]
            max_query_len = max(max_query_len, q_len)
            max_seq_len = max(max_seq_len, int(seq_lens_host[i]))
            base = len(group_req) * _MIXED_ROWS_GROUP
            group_req.extend([i] * -(-q_len // _MIXED_ROWS_GROUP))
            for j in range(q_len):
                src.append(qsl[i] + j)
                req.append(i)
                delta.append(1 + j - q_len)
                dst.append(base + j)
        self.rows = rows
        self.max_query_len = max_query_len
        self.max_seq_len_hint = max_seq_len
        self.num_groups = len(group_req)
        packed = torch.tensor(
            src + req + delta + dst + group_req,
            dtype=torch.int64,
            device="cpu",
            pin_memory=device.type == "cuda",
        ).to(device, non_blocking=True)
        n = len(src)
        self.src_idx = packed[:n]
        self.token_req = packed[n : 2 * n]
        self.token_delta = packed[2 * n : 3 * n].to(torch.int32)
        self.dst_idx = packed[3 * n : 4 * n]
        self.group_req = packed[4 * n :]
        self._token_lengths: torch.Tensor | None = None
        self._group_lengths: torch.Tensor | None = None
        self._group_table: torch.Tensor | None = None

    def token_lengths(self, seq_lens: torch.Tensor) -> torch.Tensor:
        """Visible KV length of every selected query token (int32, [T])."""
        if self._token_lengths is None:
            self._token_lengths = (
                seq_lens.index_select(0, self.token_req).to(torch.int32)
                + self.token_delta
            )
        return self._token_lengths

    def group_lengths(self, seq_lens: torch.Tensor) -> torch.Tensor:
        """Row lengths of the padded eight-row groups (zero on padding rows)."""
        if self._group_lengths is None:
            lengths = torch.zeros(
                self.num_groups * _MIXED_ROWS_GROUP,
                dtype=torch.int32,
                device=seq_lens.device,
            )
            lengths.index_copy_(0, self.dst_idx, self.token_lengths(seq_lens))
            self._group_lengths = lengths
        return self._group_lengths

    def group_table(self, block_table: torch.Tensor) -> torch.Tensor:
        """One block-table row per eight-row group ([G, columns])."""
        if self._group_table is None:
            self._group_table = block_table.index_select(0, self.group_req)
        return self._group_table


def _mixed_decode_rows_plan(
    attn_metadata: TritonAttentionMetadata,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    max_query_len: int,
    device: torch.device,
) -> _MixedDecodeRowsPlan | None:
    """Select the resident decode/verification rows of a mixed batch.

    Returns ``None`` when the batch has no such row, or only such rows (that
    shape is the uniform small-query batch and has its own route). The result is
    cached on the step's metadata object, which every layer of the group shares.
    """
    cached = getattr(attn_metadata, _MIXED_ROWS_PLAN_ATTR, False)
    if cached is not False:
        return cast(_MixedDecodeRowsPlan | None, cached)
    num_seqs = len(query_start_loc) - 1
    qsl = query_start_loc[: num_seqs + 1].tolist()
    seq_lens_host = seq_lens[:num_seqs].tolist()
    rows = tuple(
        i
        for i in range(num_seqs)
        if 1 <= qsl[i + 1] - qsl[i] <= max_query_len
        and int(seq_lens_host[i]) > qsl[i + 1] - qsl[i]
    )
    plan = (
        _MixedDecodeRowsPlan(rows, qsl, seq_lens_host, device)
        if rows and len(rows) != num_seqs
        else None
    )
    with suppress(AttributeError):
        setattr(attn_metadata, _MIXED_ROWS_PLAN_ATTR, plan)
    return plan
