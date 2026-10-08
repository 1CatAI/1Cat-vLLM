# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Graph-stable direct reads of authoritative device QSA history."""

import importlib

import torch

from vllm.config import get_current_vllm_config_or_none
from vllm.platforms import current_platform

# QSA owners execute serially on the model stream, like the existing shared
# host staging buffer. Allocate before capture; reuse across target and draft.
_WORKSPACES: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}


def initialize_device_history_attention(state, enabled: bool | None) -> None:
    cfg = get_current_vllm_config_or_none()
    if enabled is None:
        enabled = bool(cfg and cfg.kernel_config.sm70_qsa_device_history)
    reason = None
    if not enabled:
        reason = "user_override"
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
        splits = (state.width + 63) // 64
        _WORKSPACES[key] = (
            torch.empty(
                20 * 6 * splits * 256, dtype=torch.float32, device=state.history.device
            ),
            torch.empty(
                20 * 6 * splits * 2, dtype=torch.float32, device=state.history.device
            ),
        )
    state.device_history_workspace = _WORKSPACES[key]


def device_history_attention(
    query, state, indices, table, requests, positions, lengths, out, gate
):
    workspace = state.device_history_workspace
    if workspace is None:
        return False
    if not (0 < query.shape[0] <= 20 and query.shape[1:] == (6, 256)):
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
    return True
