# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Debug and profiling switches for the Flash-V100 backend."""

from __future__ import annotations

import torch

from vllm.v1.attention.backends.flash_v100 import config as _config
from vllm.v1.attention.backends.flash_v100.plan.events import (
    DiagnosticMessage,
    diagnostic_messages,
)


def sm70_profile_trace(message: str, *args: object) -> None:
    if _config.registered("VLLM_SM70_PROFILE_TRACE"):
        if args:
            message = message % args
        diagnostic_messages.emit(
            DiagnosticMessage("SM70 Flash-V100 trace: %s", (message,))
        )


_draft_graph_debug_counts: dict[str, int] = {}


def draft_graph_debug_enabled() -> bool:
    return _config.raw("VLLM_FLASH_V100_DRAFT_GRAPH_DEBUG", "0") == "1"


def _draft_graph_debug_limit() -> int:
    return int(_config.raw("VLLM_FLASH_V100_DRAFT_GRAPH_DEBUG_LIMIT", "12"))


def format_tensor_debug(tensor: torch.Tensor | None, name: str) -> str:
    if tensor is None:
        return f"{name}=None"

    values = ""
    if tensor.numel() > 0 and not (
        tensor.is_cuda and torch.cuda.is_current_stream_capturing()
    ):
        try:
            flat = tensor.detach().reshape(-1)[: min(8, tensor.numel())]
            values = f" vals={flat.cpu().tolist()}"
        except Exception as exc:  # pragma: no cover - diagnostic only.
            values = f" vals=<unavailable:{type(exc).__name__}>"

    return (
        f"{name}=shape={tuple(tensor.shape)} dtype={tensor.dtype} "
        f"ptr=0x{tensor.data_ptr():x} storage=0x"
        f"{tensor.untyped_storage().data_ptr():x} "
        f"offset={tensor.storage_offset()}{values}"
    )


def draft_graph_debug_log(key: str, message: str, *args: object) -> None:
    if not draft_graph_debug_enabled():
        return
    count = _draft_graph_debug_counts.get(key, 0)
    if count >= _draft_graph_debug_limit():
        return
    _draft_graph_debug_counts[key] = count + 1
    if args:
        message = message % args
    diagnostic_messages.emit(
        DiagnosticMessage(
            "FLASH_ATTN_V100 draft graph debug[%s#%d]: %s", (key, count, message)
        )
    )


def graph_metadata_debug_log(key: str, message: str, *args: object) -> None:
    if not draft_graph_debug_enabled():
        return
    count = _draft_graph_debug_counts.get(key, 0)
    if count >= _draft_graph_debug_limit():
        return
    _draft_graph_debug_counts[key] = count + 1
    if args:
        message = message % args
    diagnostic_messages.emit(
        DiagnosticMessage(
            "FLASH_ATTN_V100 graph metadata debug[%s#%d]: %s", (key, count, message)
        )
    )


# Public owner operations; legacy bindings are installed by package assembly.
LEGACY_ALIASES = {
    "_graph_metadata_debug_log": "graph_metadata_debug_log",
    "_draft_graph_debug_log": "draft_graph_debug_log",
    "_sm70_profile_trace": "sm70_profile_trace",
    "_format_tensor_debug": "format_tensor_debug",
    "_draft_graph_debug_enabled": "draft_graph_debug_enabled",
}
