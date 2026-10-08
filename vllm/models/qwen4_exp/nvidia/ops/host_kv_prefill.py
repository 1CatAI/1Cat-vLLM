# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage host history once per prefill owner in the existing miss workspace."""

import torch

from vllm.logger import init_logger
from vllm.models.deepseek_v4.common.ops.fp8_software import (
    fp8_e4m3fn_bits_to_fp32_bitcast,
)
from vllm.models.qwen4_exp.nvidia.ops.qsa import qsa_sparse_paged_attention
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)
PREFILL_MIN_ROWS = 512
PREFILL_QUERY_ROWS = 256


def grouped_prefill_workspace_bytes(rows: int) -> int:
    """Planner capacity per CUDA stream, excluding borrowed history scratch."""
    from vllm.models.qwen4_exp.nvidia.ops import qsa

    if rows < PREFILL_MIN_ROWS:
        return 0
    groups = rows // qsa._SM70_QSA_GROUPED_PAGE4_QUERIES
    capacity = 1 << (groups - 1).bit_length()
    return capacity * (
        qsa._SM70_QSA_GROUPED_PAGE4_OUTPUT_PAGES * 8
        + 4
        + qsa._SM70_QSA_GROUPED_PAGE4_QUERIES * 6 * 4
    )


def _grouped_prefill_reason(
    query, key, value, indices, table, requests, positions, lengths
):
    from vllm.models.qwen4_exp.nvidia.ops import qsa
    from vllm.platforms import current_platform

    if not current_platform.is_device_capability(70):
        return "requires_sm70", None
    if not qsa._qsa_xqa_page4_shape_supported(
        query, key, value, indices, table, requests, positions, lengths
    ):
        return "unsupported_grouped_page4_layout", None
    try:
        from flash_attn_v100.flash_attn_interface import flash_attn_v100_cuda
    except ImportError:
        return "flash_v100_provider_unavailable", None
    if not qsa._qsa_grouped_page4_supported(flash_attn_v100_cuda, "auto"):
        return "grouped_page4_abi_unavailable", None
    return None, flash_attn_v100_cuda


@triton.jit
def _prefill_selected_counts(
    Positions,
    Requests,
    Lengths,
    Counts,
    Rows: tl.constexpr,
    Width: tl.constexpr,
    NumRequests: tl.constexpr,
):
    rows = tl.program_id(0) * 256 + tl.arange(0, 256)
    request = tl.load(Requests + rows, rows < Rows, other=-1)
    position = tl.load(Positions + rows, rows < Rows, other=-1)
    length = tl.load(
        Lengths + tl.maximum(request, 0),
        (rows < Rows) & (request >= 0) & (request < NumRequests),
        other=0,
    )
    visible = tl.maximum(tl.minimum(position + 1, length), 0)
    count = tl.minimum(visible // 4, Width // 4) * 4
    count += tl.minimum(visible % 4, Width % 4)
    tl.store(Counts + rows, count, rows < Rows)


def prefill_staging_reason(state, table, rows: int) -> str | None:
    if state.rows != 32:
        return "requires_32_row_host_reader_workspace"
    if rows < PREFILL_MIN_ROWS:
        return "query_rows_below_prefill_band"
    if table.ndim != 2 or not all(table.shape):
        return "empty_or_invalid_request_table"
    required = table.numel() * 2 * state.page_size * state.dim
    if required > state.staging.numel():
        return "request_history_exceeds_shared_miss_workspace"
    return None


@triton.jit
def _stage_request_history(
    History,
    Scales,
    Table,
    Lengths,
    Output,
    Remapped,
    TableStride: tl.constexpr,
    Columns: tl.constexpr,
    Blocks: tl.constexpr,
    Page: tl.constexpr,
    Dim: tl.constexpr,
    FP8: tl.constexpr,
):
    logical_block, tile = tl.program_id(0), tl.program_id(1)
    request, logical_page = logical_block // Columns, logical_block % Columns
    physical = tl.load(Table + request * TableStride + logical_page)
    if tile == 0:
        tl.store(
            Remapped + logical_block,
            tl.where((physical >= 0) & (physical < Blocks), logical_block, -1),
        )
    length = tl.load(Lengths + request)
    tokens = tile * 16 + tl.arange(0, 16)
    kinds = tl.arange(0, 2)
    dims = tl.arange(0, Dim)
    valid = (
        (physical >= 0)
        & (physical < Blocks)
        & (tokens < Page)
        & (logical_page * Page + tokens < length)
    )
    physical = tl.maximum(physical, 0).to(tl.int64)
    codes = tl.load(
        History
        + ((physical * 2 + kinds[None, :, None]) * Page + tokens[:, None, None]) * Dim
        + dims[None, None, :],
        valid[:, None, None],
        other=0,
    )
    if FP8:
        scales = tl.load(
            Scales + (physical * Page + tokens[:, None]) * 2 + kinds[None, :],
            valid[:, None],
            other=0,
        )
        values = (fp8_e4m3fn_bits_to_fp32_bitcast(codes) * scales[:, :, None]).to(
            tl.float16
        )
    else:
        values = codes.to(tl.float16)
    tl.store(
        Output
        + ((logical_block * 2 + kinds[None, :, None]) * Page + tokens[:, None, None])
        * Dim
        + dims[None, None, :],
        values,
        (tokens < Page)[:, None, None],
    )


def stage_prefill_history(state, table, lengths):
    """Map logical request pages into scratch without allocating another bank."""
    pages = table.numel()
    required = pages * 2 * state.page_size * state.dim
    if required > state.staging.numel():
        raise ValueError("Prefill history exceeds the fixed miss workspace")
    if table.shape[0] != lengths.numel():
        raise ValueError("Prefill request table and sequence lengths disagree")
    cache = state.staging.reshape(-1)[:required].view(
        pages, 2, state.page_size, 1, state.dim
    )
    remapped = torch.empty(table.shape, device=table.device, dtype=torch.int32)
    _stage_request_history[(pages, triton.cdiv(state.page_size, 16))](
        state.history,
        state.scales,
        table,
        lengths,
        cache,
        remapped,
        table.stride(0),
        table.shape[1],
        state.blocks,
        state.page_size,
        state.dim,
        state.fp8,
        num_warps=4,
    )
    return cache, remapped


def host_qsa_prefill(
    query,
    state,
    indices,
    table,
    requests,
    positions,
    lengths,
    out,
    gate=None,
    *,
    grouped_page4=False,
):
    """Stage once; optionally consume canonical group-and-tail selections.

    Only the QSA indexer's compact four-token groups followed by the causal
    tail satisfy the native planner contract. Arbitrary causal selections
    must continue through the generic sparse attention implementation.
    """
    reason = prefill_staging_reason(state, table, query.shape[0])
    if reason is not None:
        raise ValueError(f"Host prefill not admitted: {reason}")
    cache, remapped = stage_prefill_history(state, table, lengths)
    counts = torch.empty(query.shape[0], device=query.device, dtype=torch.int32)
    _prefill_selected_counts[(triton.cdiv(query.shape[0], 256),)](
        positions,
        requests,
        lengths,
        counts,
        query.shape[0],
        state.width,
        lengths.numel(),
        num_warps=4,
    )
    key, value = cache.unbind(1)
    if grouped_page4:
        from vllm.models.qwen4_exp.nvidia.ops import qsa

        reason, provider = _grouped_prefill_reason(
            query, key, value, indices, remapped, requests, positions, lengths
        )
        if reason is None:
            assert provider is not None
            complete = query.shape[0] // 8 * 8
            qsa._qsa_sparse_paged_attention_sm70_grouped_page4(
                query[:complete],
                key,
                value,
                indices[:complete],
                remapped,
                requests[:complete],
                positions[:complete],
                lengths,
                out[:complete],
                "auto",
                1.0,
                1.0,
                provider,
            )
            if gate is not None:
                qsa._qsa_output_gate(
                    out[:complete], gate[:complete].view_as(query[:complete])
                )
            if complete == query.shape[0]:
                return out
            qsa_sparse_paged_attention(
                query[complete:],
                key,
                value,
                indices[complete:],
                remapped,
                requests[complete:],
                out[complete:],
                output_gate=gate[complete:] if gate is not None else None,
                query_positions=positions[complete:],
                sequence_lengths=lengths,
                causal_mask=True,
                selected_counts=counts[complete:],
            )
            return out
        logger.info_once("QSA host prefill grouped route skipped: %s", reason)
    # Full 32-row groups use the same eight splits for every size through 256.
    # Keep a final short group separate: its split and warp profiles differ.
    complete = query.shape[0] // state.rows * state.rows
    for start in range(0, complete, PREFILL_QUERY_ROWS):
        stop = min(start + PREFILL_QUERY_ROWS, complete)
        qsa_sparse_paged_attention(
            query[start:stop],
            key,
            value,
            indices[start:stop],
            remapped,
            requests[start:stop],
            out[start:stop],
            output_gate=gate[start:stop] if gate is not None else None,
            query_positions=positions[start:stop],
            sequence_lengths=lengths,
            causal_mask=True,
            selected_counts=counts[start:stop],
        )
    if complete < query.shape[0]:
        qsa_sparse_paged_attention(
            query[complete:],
            key,
            value,
            indices[complete:],
            remapped,
            requests[complete:],
            out[complete:],
            output_gate=gate[complete:] if gate is not None else None,
            query_positions=positions[complete:],
            sequence_lengths=lengths,
            causal_mask=True,
            selected_counts=counts[complete:],
        )
    return out
