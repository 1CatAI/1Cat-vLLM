# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared-key QSA scoring admitted by tensor and installed native capability.

Models provide compressed FP16 keys, four query heads of width 128 and causal
row metadata. The native provider tiles queries independently of model names,
speculative width and context length. Unsupported inputs retain caller fallback.
"""

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


@lru_cache(maxsize=1)
def _native_query_limit():
    # Older packaged binaries implement a single query tile. Importing newer
    # Python must not send additional rows to that ABI's M<=8 implementation.
    native = importlib.import_module("vllm._sm70_qsa_indexer_C")
    return native.MAX_QUERY_ROWS if hasattr(native, "MAX_QUERY_ROWS") else 8


def shared_key_reason(q, cache, table, requests, positions, lengths):
    if not current_platform.is_device_capability(70):
        return "requires_SM70"
    if q.dtype != torch.float16 or cache.dtype != torch.float16:
        return "requires_FP16_query_and_indexer_cache"
    if q.ndim != 3 or q.shape[1:] != (4, 128):
        return "requires_H4_D128"
    if q.shape[0] < 2:
        return "requires_multiple_query_rows"
    if cache.ndim != 4 or cache.shape[2:] != (1, 128):
        return "requires_single_head_D128_cache"
    if table.ndim != 2 or table.shape[0] != 1:
        return "multiple_requests"
    if not q.is_cuda or any(
        x.device != q.device for x in (cache, table, requests, positions, lengths)
    ):
        return "requires_same_CUDA_device"
    if not all(x.dtype == torch.int32 for x in (table, requests, lengths)):
        return "requires_int32_request_metadata"
    if positions.dtype not in (torch.int32, torch.int64):
        return "requires_integer_positions"
    if not all(x.is_contiguous() for x in (requests, positions, lengths)):
        return "noncontiguous_row_metadata"
    if requests.numel() < q.shape[0] or positions.numel() < q.shape[0]:
        return "insufficient_row_metadata"
    if lengths.numel() < 1:
        return "missing_request_length"
    if not load_operator():
        return "native_extension_unavailable"
    if q.shape[0] > _native_query_limit():
        return "native_query_row_limit"
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
