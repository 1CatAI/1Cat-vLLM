# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Graph-stable direct reads of authoritative device QSA history."""

import importlib

import torch

from vllm.config import get_current_vllm_config_or_none
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

# QSA owners execute serially on the model stream, like the existing shared
# host staging buffer. Allocate before capture and reuse across target layers.
_WORKSPACES: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}


def initialize_device_history_attention(state, enabled: bool | None) -> None:
    cfg = get_current_vllm_config_or_none()
    if enabled is None:
        enabled = bool(cfg and cfg.kernel_config.sm70_qsa_device_history)
    reason = None
    if not enabled:
        reason = "user_override"
    elif getattr(state, "is_speculative_draft", False) or (
        cfg and cfg.is_speculative_draft
    ):
        reason = "speculative_draft_unqualified"
    elif not state.device_reference and not cfg_allows_host_native(cfg):
        reason = "history_on_host"
    elif not current_platform.is_device_capability(70):
        reason = "requires_SM70"
    elif not 0 < state.width <= 4096:
        reason = "selection_width_outside_1_4096"
    else:
        try:
            importlib.import_module("vllm._sm70_qsa_device_C")
        except ImportError:
            reason = "native_extension_unavailable"
        else:
            if not state.device_reference and not hasattr(
                torch.ops.vllm_sm70_qsa_device, "run_hot"
            ):
                reason = "native_hot_cache_unavailable"
    state.device_history_reason = reason
    # Host-authoritative history: resolve selected pages into the protected
    # hot cache, then read hot tokens on device and the rest from mapped host.
    state.device_history_hot = reason is None and not state.device_reference
    state.device_history_workspace = None
    if reason is not None:
        return
    key = (state.history.device, state.width, getattr(state, "resolve_rows", 20))
    if key not in _WORKSPACES:
        # The protected launch policy uses 16-column tiles and up to 64 splits,
        # including at short widths. Bound all admitted M values before capture.
        reference_elements = max(
            rows * 6 * _attention_profile(rows, state.width)[2] for rows in range(1, 21)
        )
        native_rows = max(20, getattr(state, "resolve_rows", 20))
        native_elements = native_rows * 6 * ((state.width + 63) // 64)
        # Native split states contain both a maximum and a denominator. Keep
        # enough room for the arithmetic-preserving diagnostic reference too.
        _WORKSPACES[key] = (
            torch.empty(
                max(reference_elements, native_elements) * 256,
                dtype=torch.float32,
                device=state.history.device,
            ),
            torch.empty(
                max(reference_elements, 2 * native_elements),
                dtype=torch.float32,
                device=state.history.device,
            ),
        )
    state.device_history_workspace = _WORKSPACES[key]


# Rows per native launch over staged prefill history (the op's maximum).
PREFILL_ROWS = 64


def prefill_history_workspace(state):
    """Native reader workspace for staged prefill history (PREFILL_ROWS rows).

    Staged history is device memory, so this also serves owners the
    host-history reader does not admit (the speculative draft).
    """
    if not current_platform.is_device_capability(70) or not 0 < state.width <= 4096:
        return None
    try:
        importlib.import_module("vllm._sm70_qsa_device_C")
    except ImportError:
        return None
    key = (state.history.device, state.width, "prefill")
    if key not in _WORKSPACES:
        elements = PREFILL_ROWS * 6 * ((state.width + 63) // 64)
        _WORKSPACES[key] = (
            torch.empty(
                elements * 256, dtype=torch.float32, device=state.history.device
            ),
            torch.empty(2 * elements, dtype=torch.float32, device=state.history.device),
        )
    return _WORKSPACES[key]


def cfg_allows_host_native(cfg) -> bool:
    return bool(cfg is None or cfg.kernel_config.sm70_qsa_host_native)


def device_history_attention(
    query, state, indices, table, requests, positions, lengths, out, gate
):
    workspace = state.device_history_workspace
    if workspace is None:
        return False
    hot = getattr(state, "device_history_hot", False)
    rows_limit = state.resolve_rows if hot else 20
    if hot and (query.shape[0] > rows_limit or indices.shape[1] != state.width):
        state.device_history_reason = "host_native_requires_staging_geometry"
        return False
    if not (
        query.dtype == torch.float16
        and 0 < query.shape[0] <= rows_limit
        and query.shape[1:] == (6, 256)
    ):
        state.device_history_reason = "requires_M1_resolve_rows_H6_D256"
        return False
    if (
        query.data_ptr() % 16
        or query.stride(2) != 1
        or query.stride(0) % 8
        or query.stride(1) % 8
    ):
        state.device_history_reason = "query_alignment"
        return False
    if not (
        indices.dtype == torch.int32
        and indices.stride(1) == 1
        and table.dtype == torch.int32
        and table.stride(1) == 1
        and requests.dtype == torch.int32
        and positions.dtype == torch.int64
        and lengths.dtype == torch.int32
        and requests.is_contiguous()
        and positions.is_contiguous()
        and lengths.is_contiguous()
        and 0 < indices.shape[1] <= state.width
    ):
        state.device_history_reason = "metadata_layout"
        return False
    gate = gate.view_as(query) if gate is not None else None
    if hot:
        resolved = state.resolve_hot(indices, table, requests, positions, lengths)
        torch.ops.vllm_sm70_qsa_device.run_hot(
            query,
            state.history,
            state.scales,
            indices,
            table,
            requests,
            positions,
            lengths,
            out,
            gate,
            *workspace,
            state.hot_values,
            resolved,
        )
        state.device_history_reason = None
        logger.info_once(
            "Using SM70 native QSA attention over host history with a device "
            "hot cache (FP32 PV)"
        )
        return True
    torch.ops.vllm_sm70_qsa_device.run(
        query,
        state.history,
        state.scales,
        indices,
        table,
        requests,
        positions,
        lengths,
        out,
        gate,
        *workspace,
    )
    state.device_history_reason = None
    logger.info_once("Using SM70 native QSA device-history attention (FP32 PV)")
    return True


def _attention_profile(rows, width):
    from vllm.triton_utils import triton

    from .qsa import (
        _qsa_sparse_launch_profile,
        _use_sm70_qsa_two_warp_partial,
    )

    heads, dim, block_m = 6, 256, 8
    block_n, target, warps = _qsa_sparse_launch_profile(rows, block_m, True)
    if _use_sm70_qsa_two_warp_partial(rows, heads, dim):
        warps = 2
    tiles = triton.cdiv(width, block_n)
    splits = min(1 << (tiles.bit_length() - 1), target)
    return block_m, block_n, splits, warps


def _direct_history_triton(
    query, state, indices, table, requests, positions, lengths, out, gate, workspace
):
    """Remove placement dependencies while retaining protected QSA arithmetic."""
    from vllm.triton_utils import triton

    from .qsa import _qsa_merge_splitk_kernel, _qsa_sparse_paged_gqa_splitk_kernel

    rows, heads, dim = query.shape
    block_m, block_n, splits, warps = _attention_profile(rows, indices.shape[1])
    tiles = triton.cdiv(indices.shape[1], block_n)
    if splits == 1:
        partial = lse = out
    else:
        partial = workspace[0][: splits * query.numel()].view(splits, *query.shape)
        lse = workspace[1][: splits * rows * heads].view(splits, rows, heads)
    # Views only change the base address; both planes retain page-major strides.
    keys, values = state.history[:, 0], state.history[:, 1]
    _qsa_sparse_paged_gqa_splitk_kernel[(rows, 1, splits)](
        query,
        keys,
        values,
        indices,
        table,
        requests,
        partial,
        lse,
        out,
        None,
        gate,
        query.stride(0),
        query.stride(1),
        keys.stride(0),
        keys.stride(1),
        keys.stride(2),
        values.stride(0),
        values.stride(1),
        values.stride(2),
        indices.stride(0),
        table.stride(0),
        out.stride(0),
        out.stride(1),
        gate.stride(0) if gate is not None else 0,
        gate.stride(1) if gate is not None else 0,
        rows,
        state.blocks,
        table.shape[0],
        1.0,
        1.0,
        TOPK=indices.shape[1],
        PAGE_SIZE=state.page_size,
        PAGE_TABLE_WIDTH=table.shape[1],
        GROUP_SIZE=heads,
        HEAD_DIM=dim,
        NUM_QUERY_HEADS=heads,
        NUM_SPLITS=splits,
        NUM_TILES=tiles,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        KV_E4M3=False,
        HISTORY_SCALES=state.scales,
        HISTORY_POSITIONS=positions,
        HISTORY_LENGTHS=lengths,
        DEVICE_HISTORY=True,
        HISTORY_E4M3=state.fp8,
        num_warps=warps,
        num_stages=2,
    )
    if splits > 1:
        _qsa_merge_splitk_kernel[(rows, heads)](
            partial,
            lse,
            out,
            None,
            gate,
            out.stride(0),
            out.stride(1),
            gate.stride(0) if gate is not None else 0,
            gate.stride(1) if gate is not None else 0,
            rows,
            1.0,
            HEAD_DIM=dim,
            NUM_QUERY_HEADS=heads,
            NUM_SPLITS=splits,
            BLOCK_SPLITS=triton.next_power_of_2(splits),
            KV_E4M3=False,
            num_warps=2,
            num_stages=1,
        )
