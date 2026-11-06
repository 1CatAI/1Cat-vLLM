# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused PLE short-conv group metadata for pure MTP verify batches (SM70).

``PleShortConvAttentionMetadataBuilder.build`` prepares the speculative conv
metadata of every PLE short-conv KV-cache group separately: a host-to-device
mask copy, request-index copies, the Mamba block-table gather, index gathers
and the persistent CUDA-graph buffer copies and fills (about 14 GPU operations
per group). ``ONECAT_MTP_SHORTCONV_GROUP_META=1`` writes the graph buffers of
all these groups with one pointer-table kernel for FULL-graph pure speculative
verify batches (live speculative rows first, graph-padding rows with query
length 0 after them, no prefill and no non-speculative decode row), and the
builders return the metadata over those buffers.

Per group and padded request row ``r < num_reqs`` the kernel writes the
per-group path's values: the conv-state slot ``block_table[r, s_r]`` (column
0 of ``mamba_get_block_table_tensor``: align mode ``s_r`` is the request's
Mamba state column, ``ceil((computed + query_len) / B) - 1`` = the per-group
path's ``(seq_len - 1) // B``; mode ``none`` reads column 0) and NULL_BLOCK_ID
on padding rows; the speculative row mask; the query start offsets with the
tail set to the total; the accepted counts with the tail set to 1. The token
index (a fresh ``arange`` the graph captured) and the empty non-speculative
index are returned as cached tensors of the same values. A batch outside the
precondition keeps the per-group path; every decision is counted in
``sm70_fuse47.ROUTE_COUNTS`` under the unit name ``pg1b``.
"""

import os
from dataclasses import dataclass

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

ENV = "ONECAT_MTP_SHORTCONV_GROUP_META"
# Token-index tensors of the prepared metadata, kept for the process lifetime
# (never freed or rebuilt): one arange per (device, length) and one empty index
# per device. CUDA-graph replay reads the tensors captured by the per-group
# path, not these; they carry the same values for the eager metadata object.
_TOKEN_INDEX: dict[tuple[str, int | None, int], torch.Tensor] = {}
_EMPTY_INDEX: dict[tuple[str, int | None], torch.Tensor] = {}


def token_index(device: torch.device, length: int) -> torch.Tensor:
    key = (device.type, device.index, length)
    t = _TOKEN_INDEX.get(key)
    if t is None:
        t = torch.arange(length, dtype=torch.int32, device=device)
        _TOKEN_INDEX[key] = t
    return t


def empty_index(device: torch.device) -> torch.Tensor:
    key = (device.type, device.index)
    t = _EMPTY_INDEX.get(key)
    if t is None:
        t = torch.empty(0, dtype=torch.int32, device=device)
        _EMPTY_INDEX[key] = t
    return t


def enabled() -> bool:
    return os.getenv(ENV, "0").strip().lower() in ("1", "true", "yes", "on")


def note_route(route: str) -> None:
    from vllm.model_executor.layers import sm70_fuse47

    sm70_fuse47.note_route("pg1b", route)


def pure_spec_block_reason(
    num_decode_draft_tokens_cpu: torch.Tensor,
    query_start_loc_cpu: torch.Tensor,
    num_actual_tokens: int,
) -> str | None:
    """Return why the batch does not match the pure MTP graph layout."""
    num_rows = num_decode_draft_tokens_cpu.numel()
    if (
        num_decode_draft_tokens_cpu.ndim != 1
        or query_start_loc_cpu.numel() != num_rows + 1
    ):
        return "shape"
    spec_mask = num_decode_draft_tokens_cpu >= 0
    num_spec = int(spec_mask.sum().item())
    if num_spec == 0:
        return "no_spec"
    if not bool(spec_mask[:num_spec].all().item()) or bool(
        spec_mask[num_spec:].any().item()
    ):
        return "spec_not_prefix"
    if int(num_decode_draft_tokens_cpu[:num_spec].sum().item()) == 0:
        return "no_drafts"
    query_lens = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
    if bool((query_lens[num_spec:] != 0).any().item()):
        return "non_spec_rows"
    if bool((query_lens[:num_spec] <= 0).any().item()):
        return "empty_spec_row"
    if (
        num_spec > num_actual_tokens
        or int(query_start_loc_cpu[-1].item()) > num_actual_tokens
    ):
        return "tokens"
    return None


@dataclass
class ShortConvGroupDescriptor:
    """Persistent pointer tables for one runner's PLE short-conv groups."""

    key: tuple[object, ...]
    block_table_ptrs: torch.Tensor
    state_ptrs: torch.Tensor
    mask_ptrs: torch.Tensor
    query_start_ptrs: torch.Tensor
    accepted_ptrs: torch.Tensor
    block_table_strides: torch.Tensor


@dataclass(frozen=True)
class PreparedShortConvMetadata:
    """What a builder needs to return the metadata over its written buffers."""

    num_spec_decodes: int
    num_spec_decode_tokens: int
    num_reqs: int
    spec_token_indx: torch.Tensor
    non_spec_token_indx: torch.Tensor


@triton.jit
def _load_i32_ptr(ptr_to_ptr):
    return tl.cast(tl.load(ptr_to_ptr), tl.pointer_type(tl.int32))


@triton.jit
def _ple_shortconv_group_metadata_kernel(
    block_table_ptrs,
    state_ptrs,
    mask_ptrs,
    query_start_ptrs,
    accepted_ptrs,
    block_table_strides,
    state_start_indices,
    req_index_mapping,
    query_start_src,
    accepted_src,
    num_spec_decodes,
    batch_size,
    NULL_ID: tl.constexpr,
    BLOCK: tl.constexpr,
    USE_STATE_START: tl.constexpr,
):
    """One program per group: conv-state slots and the three graph buffers."""
    group_id = tl.program_id(0)
    block_table = _load_i32_ptr(block_table_ptrs + group_id)
    state_out = _load_i32_ptr(state_ptrs + group_id)
    # torch.bool storage is one byte holding 0 or 1.
    mask_out = tl.cast(tl.load(mask_ptrs + group_id), tl.pointer_type(tl.int8))
    query_out = _load_i32_ptr(query_start_ptrs + group_id)
    accepted_out = _load_i32_ptr(accepted_ptrs + group_id)
    stride = tl.load(block_table_strides + group_id)

    rows = tl.arange(0, BLOCK)
    row_mask = rows < batch_size
    live = row_mask & (rows < num_spec_decodes)
    columns = rows * 0
    live_state = live
    if USE_STATE_START:
        req = tl.load(req_index_mapping + rows, mask=live, other=0)
        live_state = live & (req >= 0)
        starts = tl.load(state_start_indices + req, mask=live_state, other=-1)
        live_state = live_state & (starts >= 0) & (starts < stride)
        columns = starts
    state = tl.load(
        block_table + rows * stride + columns, mask=live_state, other=NULL_ID
    )
    tl.store(state_out + rows, state, mask=row_mask)
    tl.store(mask_out + rows, (rows < num_spec_decodes).to(tl.int8), mask=row_mask)
    accepted = tl.load(accepted_src + rows, mask=live, other=1)
    tl.store(accepted_out + rows, accepted, mask=row_mask)
    query_mask = rows < batch_size + 1
    query = tl.load(
        query_start_src + tl.minimum(rows, num_spec_decodes), mask=query_mask, other=0
    )
    tl.store(query_out + rows, query, mask=query_mask)


def prepare_ple_shortconv_group_metadata(
    *,
    builders_by_group: list,
    block_tables: tuple[torch.Tensor, ...],
    num_accepted_tokens: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_spec_decodes: int,
    num_spec_decode_tokens: int,
    num_reqs: int,
    descriptor: ShortConvGroupDescriptor | None,
    state_start_indices: torch.Tensor | None = None,
    req_index_mapping: torch.Tensor | None = None,
) -> tuple[dict[int, PreparedShortConvMetadata], ShortConvGroupDescriptor] | None:
    """Write every PLE short-conv group's graph metadata in one launch, or
    return None (the caller keeps the per-group path)."""
    if not builders_by_group or num_spec_decodes <= 0 or num_reqs < num_spec_decodes:
        return None
    dev = num_accepted_tokens.device
    if dev.type != "cuda" or num_accepted_tokens.dtype != torch.int32:
        return None
    if num_accepted_tokens.ndim != 1 or num_accepted_tokens.numel() < num_reqs:
        return None
    if (
        query_start_loc.device != dev
        or query_start_loc.dtype != torch.int32
        or query_start_loc.numel() < num_spec_decodes + 1
    ):
        return None
    use_state_start = state_start_indices is not None
    if use_state_start != (req_index_mapping is not None):
        return None
    if use_state_start:
        assert state_start_indices is not None and req_index_mapping is not None
        if (
            state_start_indices.device != dev
            or state_start_indices.dtype != torch.int32
            or req_index_mapping.device != dev
            or req_index_mapping.dtype != torch.int32
            or req_index_mapping.numel() < num_spec_decodes
        ):
            return None
    mode = builders_by_group[0][1].vllm_config.cache_config.mamba_cache_mode
    if mode not in ("none", "align") or use_state_start != (mode == "align"):
        return None

    tables, states, masks, queries, accepted, ids, groups = [], [], [], [], [], [], []
    for group_id, builder in builders_by_group:
        if id(builder) in ids:
            continue
        if (
            not builder.use_full_cuda_graph
            or not builder.use_spec_decode
            or builder.vllm_config.cache_config.mamba_cache_mode != mode
            or num_reqs > builder.decode_cudagraph_max_bs
            or num_spec_decode_tokens > builder.decode_cudagraph_max_tokens
            or group_id < 0
            or group_id >= len(block_tables)
        ):
            return None
        table = block_tables[group_id]
        if (
            table.device != dev
            or table.dtype != torch.int32
            or table.ndim != 2
            or table.shape[0] < num_reqs
            or table.shape[1] < 1
            or not table.is_contiguous()
        ):
            return None
        bufs = (
            builder.spec_state_indices_tensor,
            builder.spec_sequence_masks,
            builder.spec_query_start_loc,
            builder.num_accepted_tokens,
        )
        if any(b.device != dev or not b.is_contiguous() for b in bufs):
            return None
        if (
            bufs[0].dtype != torch.int32
            or bufs[1].dtype != torch.bool
            or bufs[2].dtype != torch.int32
            or bufs[3].dtype != torch.int32
            or bufs[0].numel() < num_reqs
            or bufs[1].numel() < num_reqs
            or bufs[2].numel() < num_reqs + 1
            or bufs[3].numel() < num_reqs
        ):
            return None
        tables.append(table)
        states.append(bufs[0])
        masks.append(bufs[1])
        queries.append(bufs[2])
        accepted.append(bufs[3])
        ids.append(id(builder))
        groups.append(group_id)

    key = (
        dev.type,
        dev.index,
        tuple(groups),
        tuple(ids),
        tuple(t.data_ptr() for t in tables),
        tuple(t.stride(0) for t in tables),
        tuple(t.data_ptr() for t in states + masks + queries + accepted),
        use_state_start,
    )
    if descriptor is None or descriptor.key != key:

        def ptrs(ts: list[torch.Tensor]) -> torch.Tensor:
            return torch.tensor(
                [t.data_ptr() for t in ts], dtype=torch.uint64, device=dev
            )

        descriptor = ShortConvGroupDescriptor(
            key=key,
            block_table_ptrs=ptrs(tables),
            state_ptrs=ptrs(states),
            mask_ptrs=ptrs(masks),
            query_start_ptrs=ptrs(queries),
            accepted_ptrs=ptrs(accepted),
            block_table_strides=torch.tensor(
                [t.stride(0) for t in tables], dtype=torch.int64, device=dev
            ),
        )

    _ple_shortconv_group_metadata_kernel[(len(tables),)](
        descriptor.block_table_ptrs,
        descriptor.state_ptrs,
        descriptor.mask_ptrs,
        descriptor.query_start_ptrs,
        descriptor.accepted_ptrs,
        descriptor.block_table_strides,
        num_accepted_tokens if state_start_indices is None else state_start_indices,
        num_accepted_tokens if req_index_mapping is None else req_index_mapping,
        query_start_loc,
        num_accepted_tokens,
        num_spec_decodes,
        num_reqs,
        NULL_ID=NULL_BLOCK_ID,
        BLOCK=triton.next_power_of_2(num_reqs + 1),
        USE_STATE_START=use_state_start,
        num_warps=1,
    )
    prepared = PreparedShortConvMetadata(
        num_spec_decodes=num_spec_decodes,
        num_spec_decode_tokens=num_spec_decode_tokens,
        num_reqs=num_reqs,
        spec_token_indx=token_index(dev, num_spec_decode_tokens),
        non_spec_token_indx=empty_index(dev),
    )
    return {builder_id: prepared for builder_id in ids}, descriptor
