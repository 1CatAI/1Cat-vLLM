# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device-side metadata for small-query (DFlash2/MTP) decode groups."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm.config.sm70_dflash2 import (
    capture_sm70_dflash2_config,
    sm70_dflash2_enabled,
)
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.flash_v100 import metadata as _metadata

logger = init_logger("vllm.v1.attention.backends.flash_attn_v100")


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


def prepare_dflash2_smallq_group_metadata(
    *,
    builders_by_group: list[tuple[int, _metadata.FlashAttnV100MetadataBuilder]],
    block_tables: tuple[torch.Tensor, ...],
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    query_start_loc_cpu: torch.Tensor,
    num_reqs: int,
    num_query_tokens: int,
    max_seq_len_hint: int,
    workspace_seq_capacity_cap: int | None,
    descriptor: DFlash2SmallQGroupDescriptor | None,
) -> (
    tuple[
        dict[int, DFlash2SmallQPreparedMetadata],
        DFlash2SmallQGroupDescriptor,
    ]
    | None
):
    """Refresh all pure-DFlash2 target full-attention metadata in one launch."""

    def fallback(reason: str) -> None:
        logger.info_once(
            "DFlash2 grouped small-query metadata fell back to per-group launches: %s",
            reason,
        )

    if (
        not sm70_dflash2_enabled(
            "fused_smallq_metadata",
            capture_sm70_dflash2_config(
                getattr(builders_by_group[0][1], "vllm_config", None)
            )
            if builders_by_group
            else None,
        )
        or not sm70_dflash2_enabled(
            "grouped_smallq_metadata",
            capture_sm70_dflash2_config(
                getattr(builders_by_group[0][1], "vllm_config", None)
            )
            if builders_by_group
            else None,
        )
        or not builders_by_group
        or num_reqs <= 0
        or num_query_tokens <= 1
        or max_seq_len_hint <= 0
    ):
        fallback("route-or-shape guard")
        return None
    if (
        seq_lens.device.type != "cuda"
        or seq_lens.dtype != torch.int32
        or query_start_loc.device != seq_lens.device
        or query_start_loc.dtype != torch.int32
        or query_start_loc_cpu.device.type != "cpu"
        or query_start_loc_cpu.dtype != torch.int32
        or seq_lens.numel() < num_reqs
        or query_start_loc.numel() < num_reqs + 1
        or query_start_loc_cpu.numel() < num_reqs + 1
    ):
        fallback("common tensor contract")
        return None

    real_num_query_tokens = int(query_start_loc_cpu[num_reqs].item())
    if not 0 < real_num_query_tokens <= num_query_tokens:
        fallback("real query-token count")
        return None

    input_tables: list[torch.Tensor] = []
    output_tables: list[torch.Tensor] = []
    output_seq_lens: list[torch.Tensor] = []
    output_query_start_locs: list[torch.Tensor] = []
    builder_ids: list[int] = []
    workspace_hints: list[int] = []
    seen_builders: set[int] = set()
    for group_id, builder in builders_by_group:
        builder_id = id(builder)
        if builder_id in seen_builders:
            continue
        seen_builders.add(builder_id)
        if (
            not builder._use_sm70_dflash2_fused_smallq_metadata
            or group_id < 0
            or group_id >= len(block_tables)
            or builder._smallq_decode_block_table is None
            or builder._smallq_decode_seq_lens is None
            or builder._smallq_query_start_loc is None
            or builder._smallq_buffer_shape is None
        ):
            fallback("builder route or persistent buffers")
            return None

        input_table = block_tables[group_id]
        output_table = builder._smallq_decode_block_table
        output_seq = builder._smallq_decode_seq_lens
        output_query = builder._smallq_query_start_loc
        token_capacity, req_capacity, builder_block_cols = builder._smallq_buffer_shape
        input_block_cols = int(input_table.shape[1])
        if (
            input_table.device != seq_lens.device
            or input_table.dtype != torch.int32
            or input_table.ndim != 2
            or input_table.shape[0] < num_reqs
            or not input_table.is_contiguous()
            or input_block_cols <= 0
            or builder_block_cols != input_block_cols
            or token_capacity < num_query_tokens
            or req_capacity < num_reqs
            or output_table.dtype != torch.int32
            or output_seq.dtype != torch.int32
            or output_query.dtype != torch.int32
            or not output_table.is_contiguous()
            or not output_seq.is_contiguous()
            or not output_query.is_contiguous()
        ):
            fallback(
                "cache-group tensor contract "
                f"(group={group_id}, input_shape={tuple(input_table.shape)}, "
                f"input_dtype={input_table.dtype}, input_device={input_table.device}, "
                f"input_stride={input_table.stride()}, "
                f"input_contiguous={input_table.is_contiguous()}, "
                f"buffer_shape={builder._smallq_buffer_shape}, "
                f"output_dtype={output_table.dtype}, "
                f"output_stride={output_table.stride()}, "
                f"output_contiguous={output_table.is_contiguous()}, "
                f"seq_dtype={output_seq.dtype}, "
                f"seq_contiguous={output_seq.is_contiguous()}, "
                f"query_dtype={output_query.dtype}, "
                f"query_contiguous={output_query.is_contiguous()}, "
                f"num_reqs={num_reqs}, num_query_tokens={num_query_tokens}, "
                f"input_block_cols={input_block_cols}, "
                f"builder_block_cols={builder_block_cols})"
            )
            return None

        raw_seq_capacity = input_block_cols * int(builder.block_size)
        if workspace_seq_capacity_cap is not None:
            raw_seq_capacity = min(
                raw_seq_capacity,
                max(max_seq_len_hint, int(workspace_seq_capacity_cap)),
            )
        input_tables.append(input_table)
        output_tables.append(output_table)
        output_seq_lens.append(output_seq)
        output_query_start_locs.append(output_query)
        builder_ids.append(builder_id)
        workspace_hints.append(raw_seq_capacity)

    if not input_tables:
        fallback("no distinct full-attention builders")
        return None
    descriptor = _sm70_prepare_grouped_smallq_decode_metadata(
        output_tables,
        output_seq_lens,
        output_query_start_locs,
        input_tables,
        seq_lens[:num_reqs],
        query_start_loc[: num_reqs + 1],
        num_reqs=num_reqs,
        num_query_tokens=num_query_tokens,
        real_num_query_tokens=real_num_query_tokens,
        descriptor=descriptor,
    )
    prepared = {
        builder_id: DFlash2SmallQPreparedMetadata(
            builder_id=builder_id,
            num_reqs=num_reqs,
            num_query_tokens=num_query_tokens,
            max_seq_len_hint=max_seq_len_hint,
            workspace_seq_capacity_hint=workspace_hint,
        )
        for builder_id, workspace_hint in zip(builder_ids, workspace_hints)
    }
    return prepared, descriptor
