# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dynamic feature policies and trace destination handling."""

from __future__ import annotations

import json
import os

from vllm.logger import init_logger
from vllm.v1.attention.backends.flash_v100 import config as _config

logger = init_logger("vllm.v1.attention.backends.flash_attn_v100")


def context_bucket_partition_size_hint() -> int | None:
    from vllm.config.execution_policy import graph_policy

    return graph_policy().mtp_partition_hint()


def dual_cta_partition_size_hint() -> int | None:
    if not _config.options().value("xqa_mtp5_dual_cta"):
        return None
    return _config.options().value("xqa_mtp5_partition_size")


def trace_event(event: str, payload: dict[str, object]) -> None:
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


def trace_enabled() -> bool:
    return bool(_config.raw("VLLM_DFLASH_DDTREE_TRACE_JSONL"))


def prefix_dump_enabled() -> bool:
    return _config.trace().flash_v100.value("dflash_prefix_dump")


def branch_attn_enabled() -> bool:
    return _config.raw("VLLM_DFLASH_DDTREE_TRITON_BRANCH_ATTN", "1") != "0"


def branch_attn_strict() -> bool:
    return _config.raw("VLLM_DFLASH_DDTREE_TRITON_BRANCH_ATTN_STRICT", "0") == "1"


def worker_profile_enabled() -> bool:
    return _config.raw("VLLM_DFLASH_DDTREE_WORKER_PROFILE", "0") == "1"


COMPATIBILITY_ALIASES = {
    "_mtp_context_bucket_partition_size_hint": "context_bucket_partition_size_hint",
    "_mtp5_xqa_dual_cta_partition_size_hint": "dual_cta_partition_size_hint",
    "_ddtree_trace_event": "trace_event",
    "_ddtree_trace_enabled": "trace_enabled",
    "_dflash_prefix_dump_enabled": "prefix_dump_enabled",
    "_dflash_ddtree_triton_branch_attn_enabled": "branch_attn_enabled",
    "_dflash_ddtree_triton_branch_attn_strict": "branch_attn_strict",
    "_dflash_ddtree_worker_profile_enabled": "worker_profile_enabled",
}
