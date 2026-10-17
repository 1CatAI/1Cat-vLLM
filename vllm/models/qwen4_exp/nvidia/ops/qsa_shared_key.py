# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capability-gated shared-key scoring for small QSA verification batches."""

import importlib
from functools import lru_cache

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)


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

    if hasattr(torch.ops.vllm_sm70_qsa_indexer, "run_staged"):

        @torch.library.register_fake("vllm_sm70_qsa_indexer::run_staged")
        def fake_staged(
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
            staged,
            page_slots,
        ):
            return None

    if hasattr(torch.ops.vllm_sm70_qsa_indexer, "run_staged_banked"):

        @torch.library.register_fake("vllm_sm70_qsa_indexer::run_staged_banked")
        def fake_banked(
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
            staged,
            page_slots,
            spill,
            spill_slots,
        ):
            return None

        if hasattr(torch.ops.vllm_sm70_qsa_indexer, "run_staged_banked_batch"):
            torch.library.register_fake(
                "vllm_sm70_qsa_indexer::run_staged_banked_batch"
            )(fake_banked)

    return True


def shared_key_reason(q, cache, table, requests, positions, lengths):
    if not current_platform.is_device_capability(70):
        return "requires_SM70"
    if q.dtype != torch.float16 or cache.dtype != torch.float16:
        return "requires_FP16_query_and_indexer_cache"
    if not (2 <= q.shape[0] <= 8 and q.shape[1:] == (4, 128)):
        return "requires_M2_8_H4_D128"
    if table.shape[0] != 1:
        return "multiple_requests"
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
    q,
    cache,
    table,
    requests,
    positions,
    lengths,
    compress,
    columns,
    divisor,
    staging_workspace=None,
):
    reason = shared_key_reason(q, cache, table, requests, positions, lengths)
    if reason is not None:
        logger.info_once("SM70 QSA shared-key scorer fallback: %s.", reason)
        return None
    scores = torch.empty((q.shape[0], columns), dtype=torch.float32, device=q.device)
    visible = torch.empty(q.shape[0], dtype=torch.int32, device=q.device)
    staged = (
        staging_workspace.prepare(
            cache, table, requests, positions, lengths, compress, columns
        )
        if staging_workspace is not None
        and table.shape[0] <= 8
        and hasattr(torch.ops.vllm_sm70_qsa_indexer, "run_staged")
        else ()
    )
    op = (
        torch.ops.vllm_sm70_qsa_indexer.run_staged_banked
        if len(staged) == 4
        else torch.ops.vllm_sm70_qsa_indexer.run_staged
        if staged
        else torch.ops.vllm_sm70_qsa_indexer.run
    )
    op(
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
        *staged,
    )
    logger.info_once("Using SM70 QSA shared-key indexer for M=%d.", q.shape[0])
    return scores, visible


def maybe_shared_key_batch_scores(
    q,
    cache,
    table,
    requests,
    positions,
    lengths,
    compress,
    columns,
    divisor,
    query_start_loc_cpu,
    staging_workspace=None,
):
    """Reuse shared-key scoring without changing batched top-k.

    Query starts are the owner's host layout, including capture padding.
    All device metadata remains live across graph replays. An unqualified
    group sends the entire batch to the existing scorer.
    """
    if (
        table.shape[0] < 2
        or not 1 <= q.shape[0] <= 64
        or query_start_loc_cpu.device.type != "cpu"
        or query_start_loc_cpu.ndim != 1
        or query_start_loc_cpu.dtype not in (torch.int32, torch.int64)
    ):
        return None
    starts = query_start_loc_cpu.tolist()
    if len(starts) != table.shape[0] + 1 or starts[0] != 0 or starts[-1] != q.shape[0]:
        return None
    if any(end < start for start, end in zip(starts, starts[1:])):
        return None
    groups = [
        (i, start, end)
        for i, (start, end) in enumerate(zip(starts, starts[1:]))
        if end > start
    ]
    if not groups or any(not 2 <= end - start <= 8 for _, start, end in groups):
        return None
    for i, start, end in groups:
        if (
            shared_key_reason(
                q[start:end],
                cache,
                table[i : i + 1],
                requests[start:end],
                positions[start:end],
                lengths[i : i + 1],
            )
            is not None
        ):
            return None
    scores = torch.empty((q.shape[0], columns), dtype=torch.float32, device=q.device)
    visible = torch.empty(q.shape[0], dtype=torch.int32, device=q.device)
    staged = (
        staging_workspace.prepare(
            cache, table, requests, positions, lengths, compress, columns
        )
        if staging_workspace is not None
        and table.shape[0] <= 8
        and hasattr(torch.ops.vllm_sm70_qsa_indexer, "run_staged")
        else ()
    )
    if (
        len(staged) == 4
        and len(groups) == table.shape[0]
        and all(end - start == 5 for _, start, end in groups)
        and hasattr(torch.ops.vllm_sm70_qsa_indexer, "run_staged_banked_batch")
    ):
        torch.ops.vllm_sm70_qsa_indexer.run_staged_banked_batch(
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
            *staged,
        )
        logger.info_once("Using one SM70 shared-key grid for MTP4 request groups.")
        return scores, visible
    op = (
        torch.ops.vllm_sm70_qsa_indexer.run_staged_banked
        if len(staged) == 4
        else torch.ops.vllm_sm70_qsa_indexer.run_staged
        if staged
        else torch.ops.vllm_sm70_qsa_indexer.run
    )
    for i, start, end in groups:
        local_requests = (requests[start:end] - i).contiguous()
        op(
            scores[start:end],
            visible[start:end],
            q[start:end],
            cache,
            table[i : i + 1],
            local_requests,
            positions[start:end],
            lengths[i : i + 1],
            float(divisor),
            compress,
            *staged,
        )
    logger.info_once(
        "Using SM70 shared-key indexer for %d request groups.", len(groups)
    )
    if staged:
        logger.info_once("Using bounded SM70 indexer page staging with paged overflow.")
    return scores, visible
