# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capability-gated shared-key scoring and fused Top-512 for QSA decode."""

import importlib
from functools import lru_cache

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.attention.ops.sm70_workspaces import retain_for_capture, workspace_cache

logger = init_logger(__name__)

_HIST_BINS = 512
_TOPK = 512
_MAX_TIE_CAP = 8192
_MAX_ROWS = 64
_STANDALONE_TOPK_WS: dict[torch.device, tuple[torch.Tensor, ...]] = {}


def _fused_topk_workspace(device: torch.device) -> tuple[torch.Tensor, ...]:
    cache = workspace_cache("sm70_qsa_indexer_topk", _STANDALONE_TOPK_WS)
    ws = cache.get(device)
    if ws is None:
        ws = (
            torch.zeros((_MAX_ROWS, _HIST_BINS), dtype=torch.int32, device=device),
            torch.empty((_MAX_ROWS, _TOPK), dtype=torch.int32, device=device),
            torch.empty((_MAX_ROWS, _MAX_TIE_CAP), dtype=torch.int32, device=device),
            torch.empty((_MAX_ROWS, _MAX_TIE_CAP), dtype=torch.int32, device=device),
            torch.zeros((_MAX_ROWS, 8), dtype=torch.int32, device=device),
        )
        cache[device] = ws
    retain_for_capture(cache, ws, ws[0])
    return ws


@lru_cache(maxsize=1)
def load_operator():
    try:
        importlib.import_module("vllm._sm70_qsa_indexer_C")
    except ImportError:
        return False

    @torch.library.register_fake("vllm_sm70_qsa_indexer::run")
    def fake(
        scores,
        visible,
        q,
        cache,
        table,
        requests,
        positions,
        lengths,
        divisor,
        compress,
    ):
        return None

    if hasattr(torch.ops.vllm_sm70_qsa_indexer, "run_topk"):

        @torch.library.register_fake("vllm_sm70_qsa_indexer::run_topk")
        def fake_topk(
            output,
            scores,
            visible,
            ws_hist,
            ws_greater,
            ws_tie_keys,
            ws_tie_idx,
            ws_meta,
            q,
            cache,
            table,
            requests,
            positions,
            lengths,
            divisor,
            compress,
        ):
            return None

    return True


def shared_key_reason(
    q, cache, table, requests, positions, lengths, *, min_rows: int = 2
):
    if not current_platform.is_device_capability(70):
        return "requires_SM70"
    if q.dtype != torch.float16 or cache.dtype != torch.float16:
        return "requires_FP16_query_and_indexer_cache"
    if not (min_rows <= q.shape[0] <= _MAX_ROWS and q.shape[1:] == (4, 128)):
        return f"requires_M{min_rows}_{_MAX_ROWS}_H4_D128"
    if not all(x.dtype == torch.int32 for x in (table, requests, lengths)):
        return "requires_int32_request_metadata"
    if positions.dtype not in (torch.int32, torch.int64):
        return "requires_integer_positions"
    if not all(x.is_contiguous() for x in (requests, positions, lengths)):
        return "noncontiguous_row_metadata"
    if not load_operator():
        return "native_extension_unavailable"
    return None


def maybe_shared_key_scores(
    q, cache, table, requests, positions, lengths, compress, columns, divisor
):
    reason = shared_key_reason(q, cache, table, requests, positions, lengths)
    if reason is not None:
        logger.info_once("SM70 QSA shared-key scorer fallback: %s.", reason)
        return None
    scores = torch.empty((q.shape[0], columns), dtype=torch.float32, device=q.device)
    visible = torch.empty(q.shape[0], dtype=torch.int32, device=q.device)
    torch.ops.vllm_sm70_qsa_indexer.run(
        scores,
        visible,
        q,
        cache,
        table,
        requests,
        positions,
        lengths,
        float(divisor),
        compress,
    )
    logger.info_once("Using SM70 QSA shared-key indexer for M=%d.", q.shape[0])
    return scores, visible


def maybe_shared_key_topk(
    q,
    cache,
    table,
    requests,
    positions,
    lengths,
    compress,
    columns,
    divisor,
    out_blocks,
) -> bool:
    if (
        out_blocks.ndim != 2
        or out_blocks.shape != (q.shape[0], _TOPK)
        or out_blocks.dtype != torch.int32
        or not out_blocks.is_contiguous()
    ):
        return False
    reason = shared_key_reason(
        q, cache, table, requests, positions, lengths, min_rows=1
    )
    if reason is not None:
        return False
    if not hasattr(torch.ops.vllm_sm70_qsa_indexer, "run_topk"):
        return False
    ws_hist, ws_greater, ws_tie_keys, ws_tie_idx, ws_meta = _fused_topk_workspace(
        q.device
    )
    scores = torch.empty((q.shape[0], columns), dtype=torch.float32, device=q.device)
    visible = torch.empty(q.shape[0], dtype=torch.int32, device=q.device)
    torch.ops.vllm_sm70_qsa_indexer.run_topk(
        out_blocks,
        scores,
        visible,
        ws_hist,
        ws_greater,
        ws_tie_keys,
        ws_tie_idx,
        ws_meta,
        q,
        cache,
        table,
        requests,
        positions,
        lengths,
        float(divisor),
        compress,
    )
    logger.info_once(
        "Using SM70 QSA fused histogram + multi-CTA exact Top-512 for M=%d.",
        q.shape[0],
    )
    return True
