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
