# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP16 paged-KV verification with FP32 probability and PV arithmetic."""

from typing import Any

import torch

from vllm.config import get_current_vllm_config_or_none
from vllm.platforms import current_platform
from vllm.v1.attention.ops.sm70_grouped import (  # noqa: F401
    FP16_MAX_GROUPS as MAX_GROUPS,
)
from vllm.v1.attention.ops.sm70_grouped import MAX_CONTEXT as MAX_CONTEXT
from vllm.v1.attention.ops.sm70_grouped import (
    grouped_fp16_fp32_reason as grouped_fp16_fp32_reason,
)
from vllm.v1.attention.ops.sm70_grouped import model_context_limit
from vllm.v1.attention.ops.sm70_workspaces import workspace_cache

OPERATOR = "sm70_grouped_fp16_fwd"
_WORKSPACES: dict[tuple, list[tuple[torch.Tensor, torch.Tensor]]] = {}


def grouped_context_limit() -> int:
    """FP16 owns its capacity independently of the E4M3 long/scalar providers."""
    revision = getattr(
        torch.ops._vllm_fa2_C, "sm70_grouped_fp16_context_revision", None
    )
    if revision is None or revision() < 1:
        return MAX_CONTEXT
    capacity = workspace_cache("grouped_fp16_context", {}).get("capacity")
    return model_context_limit(capacity, fallback=MAX_CONTEXT)


def clear_grouped_fp16_workspaces():
    _WORKSPACES.clear()


def load_grouped_fp16_fp32():
    if not current_platform.is_device_capability(70):
        return None
    from vllm.vllm_flash_attn.flash_attn_interface import ensure_fa2_library_loaded

    ensure_fa2_library_loaded()
    operator = getattr(torch.ops._vllm_fa2_C, OPERATOR, None)
    if operator is None:
        return None
    # Model initialization owns the config; forwards borrow its resources.
    workspace_cache("grouped_fp16_context", {})["capacity"] = model_context_limit(
        fallback=MAX_CONTEXT
    )
    kv_rev = getattr(torch.ops._vllm_fa2_C, "sm70_grouped_fp16_kv_heads_revision", None)
    page_rev = getattr(torch.ops._vllm_fa2_C, "sm70_grouped_fp16_page_revision", None)
    rows_rev = getattr(
        torch.ops._vllm_fa2_C, "sm70_grouped_fp16_query_rows_revision", None
    )
    runner: Any = _run
    runner.multi_kv_heads = bool(kv_rev is not None and kv_rev() >= 1)
    runner.runtime_pages = bool(page_rev is not None and page_rev() >= 1)
    if rows_rev is not None and rows_rev() >= 1:
        runner.request_query_rows = (2, 3, 4, 5, 6, 7, 8)
    elif hasattr(runner, "request_query_rows"):
        delattr(runner, "request_query_rows")
    return runner


def short_split_capability():
    cfg = get_current_vllm_config_or_none()
    enabled = cfg is None or cfg.kernel_config.sm70_fp16_grouped_short_splits
    revision = getattr(
        torch.ops._vllm_fa2_C, "sm70_grouped_fp16_short_split_revision", None
    )
    supported = revision is not None and revision() >= 1
    reason = None
    if not enabled:
        reason = "disabled_by_policy"
    elif not supported:
        reason = "operator_missing:sm70_grouped_fp16_short_split_revision"
    return dict(
        enabled=enabled and supported,
        reason=reason,
        supported=supported,
        runtime_guards="FP16 q8/B1, device context 129..2048; retain K64 elsewhere",
    )


def _run(q, k, v, table, row_lengths, *, out, softmax_scale):
    cache = workspace_cache("grouped_fp16", _WORKSPACES)
    panels = table.shape[0] * k.shape[2]
    key = (q.device, torch.cuda.current_stream(q.device).cuda_stream)
    bank = cache.setdefault(key, [])
    for partial, lse in bank:
        if partial.shape[0] >= panels:
            break
    else:
        partial = torch.empty(
            (panels, 80, 8, 6, 256), dtype=torch.float32, device=q.device
        )
        lse = torch.empty((panels, 80, 8, 6, 2), dtype=torch.float32, device=q.device)
        bank.append((partial, lse))
    partial, lse = (
        (partial[0], lse[0]) if panels == 1 else (partial[:panels], lse[:panels])
    )
    capability = short_split_capability()
    arguments = (q, k, v, out, table, row_lengths, partial, lse, softmax_scale)
    if capability["supported"]:
        torch.ops._vllm_fa2_C.sm70_grouped_fp16_fwd(*arguments, capability["enabled"])
    else:
        torch.ops._vllm_fa2_C.sm70_grouped_fp16_fwd(*arguments)
    return out
