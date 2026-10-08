# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dynamic feature policies and trace destination handling."""

from __future__ import annotations

import json
import os

from vllm.logger import init_logger
from vllm.v1.attention.backends.flash_v100 import config as _config
from vllm.v1.attention.backends.flash_v100.routing import _VALID_DECODE_PARTITION_SIZES

logger = init_logger("vllm.v1.attention.backends.flash_attn_v100")


def _mtp_context_bucket_partition_size_hint() -> int | None:
    raw = _config.raw("VLLM_SM70_MTP_CONTEXT_BUCKET_PARTITION_SIZE")
    if raw is None:
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            "VLLM_SM70_MTP_CONTEXT_BUCKET_PARTITION_SIZE must be one of "
            f"{_VALID_DECODE_PARTITION_SIZES}, got {raw!r}"
        ) from exc
    if value not in _VALID_DECODE_PARTITION_SIZES:
        raise ValueError(
            "VLLM_SM70_MTP_CONTEXT_BUCKET_PARTITION_SIZE must be one of "
            f"{_VALID_DECODE_PARTITION_SIZES}, got {value}"
        )
    return value


def _mtp5_xqa_dual_cta_partition_size_hint() -> int | None:
    if _config.raw("VLLM_FLASH_V100_XQA_MTP5_DUAL_CTA", "1") != "1":
        return None
    raw = _config.raw("VLLM_FLASH_V100_XQA_MTP5_PARTITION_SIZE", "1024")
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            "VLLM_FLASH_V100_XQA_MTP5_PARTITION_SIZE must be one of "
            f"{_VALID_DECODE_PARTITION_SIZES}, got {raw!r}"
        ) from exc
    if value not in _VALID_DECODE_PARTITION_SIZES:
        raise ValueError(
            "VLLM_FLASH_V100_XQA_MTP5_PARTITION_SIZE must be one of "
            f"{_VALID_DECODE_PARTITION_SIZES}, got {value}"
        )
    return value


def _ddtree_trace_event(event: str, payload: dict[str, object]) -> None:
    trace_path = _config.raw("VLLM_DFLASH_DDTREE_TRACE_JSONL")
    if not trace_path:
        return
    record = {"event": event, "pid": os.getpid(), **payload}
    try:
        with open(trace_path, "a", encoding="utf-8") as trace_file:
            json.dump(record, trace_file, ensure_ascii=True, sort_keys=True)
            trace_file.write("\n")
    except OSError:
        logger.exception("Failed to write DDTree trace event to %s", trace_path)


def _ddtree_trace_enabled() -> bool:
    return bool(_config.raw("VLLM_DFLASH_DDTREE_TRACE_JSONL"))


def _dflash_prefix_dump_enabled() -> bool:
    return _config.raw("VLLM_FLASH_V100_DFLASH_PREFIX_DUMP", "0") == "1"


def _dflash_ddtree_triton_branch_attn_enabled() -> bool:
    return _config.raw("VLLM_DFLASH_DDTREE_TRITON_BRANCH_ATTN", "1") != "0"


def _dflash_ddtree_triton_branch_attn_strict() -> bool:
    return _config.raw("VLLM_DFLASH_DDTREE_TRITON_BRANCH_ATTN_STRICT", "0") == "1"


def _dflash_ddtree_worker_profile_enabled() -> bool:
    return _config.raw("VLLM_DFLASH_DDTREE_WORKER_PROFILE", "0") == "1"
