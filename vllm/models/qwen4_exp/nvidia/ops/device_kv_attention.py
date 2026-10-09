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
    elif not state.device_reference:
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
    state.device_history_reason = reason
    state.device_history_workspace = None
    if reason is not None:
        return
    key = (state.history.device, state.width)
    if key not in _WORKSPACES:
        # The protected launch policy uses 16-column tiles and up to 64 splits,
        # including at short widths. Bound all admitted M values before capture.
        reference_elements = max(
            rows * 6 * _attention_profile(rows, state.width)[2] for rows in range(1, 21)
        )
        native_elements = 20 * 6 * ((state.width + 63) // 64)
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


def device_history_attention(
    query, state, indices, table, requests, positions, lengths, out, gate
):
    workspace = state.device_history_workspace
    if workspace is None:
        return False
    if not (
        query.dtype == torch.float16
        and 0 < query.shape[0] <= 20
        and query.shape[1:] == (6, 256)
    ):
        state.device_history_reason = "requires_M1_20_H6_D256"
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
