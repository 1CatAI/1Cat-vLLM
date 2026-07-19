# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flash Attention V100 backend for SM70.

Selecting this backend keeps both prefill and decode on Flash-V100 by default.
The upstream Triton prefill path is still available as an explicit diagnostic
fallback with VLLM_FLASH_V100_PREFILL_USE_TRITON=1, but mixed Triton-prefill
plus Flash-decode runs do not count as the final SM70 FlashAttention route.
"""

from __future__ import annotations

import atexit
import inspect
import json
import os
import time
from collections.abc import Callable
from functools import partial
from typing import cast

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionCGSupport, AttentionType
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionImpl,
    TritonAttentionMetadata,
    TritonAttentionMetadataBuilder,
)

logger = init_logger(__name__)


class FlashAttnV100Metadata(TritonAttentionMetadata):
    """Static view of Flash-V100 fields attached to Triton metadata."""

    query_start_loc_cpu: torch.Tensor
    seq_lens_cpu: torch.Tensor
    causal: bool
    max_model_len: int
    flash_v100_cudagraph_capture: bool
    flash_v100_contig_dense_cache: dict[tuple[int, int, int, int, int], int]
    flash_v100_decode_max_seq_len_hint: int | None
    flash_v100_decode_workspace_seq_capacity_hint: int | None
    flash_v100_static_decode_seq_hint: int | None
    flash_v100_decode_active_num_partitions: torch.Tensor | None
    ddtree_parent_ids: torch.Tensor | None
    ddtree_parent_ids_cpu: torch.Tensor | None
    ddtree_num_tree_tokens_cpu: torch.Tensor | None
    ddtree_seq_lens_restored_for_triton: bool
    ddtree_query_start_loc_restored_for_triton: bool
    smallq_decode_block_table: torch.Tensor | None
    smallq_decode_seq_lens: torch.Tensor | None
    smallq_query_start_loc: torch.Tensor | None
    smallq_decode_max_seq_len_hint: int | None
    smallq_decode_workspace_seq_capacity_hint: int | None
    smallq_decode_partition_size_hint: int | None


def _as_flash_v100_metadata(
    attn_metadata: TritonAttentionMetadata,
) -> FlashAttnV100Metadata:
    # The inherited Triton builder creates the object; this backend attaches
    # the fields above before any Flash-V100 path consumes them.
    return cast(FlashAttnV100Metadata, attn_metadata)


def _sm70_profile_trace(message: str, *args: object) -> None:
    if envs.VLLM_SM70_PROFILE_TRACE:
        if args:
            message = message % args
        logger.info("SM70 Flash-V100 trace: %s", message)


# Lazy imports: only resolve optional CUDA extensions when needed.
_flash_attn_func = None
_flash_attn_bhmd_func = None
_flash_attn_decode_paged = None
_flash_attn_decode_paged_xqa = None
_flash_attn_decode_paged_wmma = None
_flash_attn_prefill_paged = None
_flash_attn_prefill_paged_bhmd = None
_flash_attn_prefill_paged_bfla = None
_flash_attn_prefill_paged_splitkv = None
_fp8_e5m2_paged_kv_to_fp16 = None
_fp8_e5m2_paged_kv_to_fp16_checked = False
_flash_attn_turboquant_decode_paged = None
_flash_attn_turboquant_decode_checked = False
_paged_kv_utils = None
_warned_feature_fallback = False
_warned_decode_fallback = False
_warned_decode_strict_fallback = False
_logged_prefill_flash = False
_logged_prefill_prefix_flash = False
_logged_prefill_prefix_contig_dense = False
_logged_prefill_prefix_bfla = False
_logged_prefill_prefix_splitkv = False
_logged_prefill_paged_cache = False
_logged_prefill_smallq_decode = False
_logged_prefill_smallq_decode_xqa = False
_logged_prefill_triton_safe = False
_logged_decode_flash = False
_logged_decode_dense_reference = False
_logged_decode_dense_cache = False
_logged_decode_paged_prefill = False
_logged_decode_paged_prefill_bhmd = False
_logged_decode_paged_prefill_bhmd_q_clone = False
_logged_decode_wmma_wrapper = False
_logged_fp8_kv_prefill = False
_logged_fp8_kv_decode = False
_logged_fp8_prefill_bridge = False
_logged_prefill_compare = False
_logged_dflash_prefix_dump = False
_logged_prefill_ddtree_dense = False
_logged_prefill_ddtree_triton = False
_logged_prefill_ddtree_triton_fallback = False
_route_summary_registered = False
_route_counts: dict[str, int] = {}
_decode_active_trace_signatures: set[tuple[object, ...]] = set()
_draft_graph_debug_counts: dict[str, int] = {}
_DEFAULT_DECODE_PARTITION_SIZE = 256
_VALID_DECODE_PARTITION_SIZES = (256, 512, 1024)
_DEFAULT_Q4_XQA_MIN_SEQ_LEN = 32768
_DEFAULT_FP8_XQA_MIN_SEQ_LEN = 8192
_FP8_PREFILL_BRIDGE_PAGE_SIZE = 784
_fp8_prefill_bridge_workspaces: dict[
    tuple[int, int, int, int],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
] = {}


def _split_paged_kv_cache(
    kv_cache: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(kv_cache, (list, tuple)):
        if len(kv_cache) != 2:
            raise ValueError(
                f"Unexpected KV cache tuple/list length {len(kv_cache)}; expected 2"
            )
        return kv_cache[0], kv_cache[1]

    if kv_cache.ndim < 2:
        raise ValueError(
            f"Unexpected KV cache shape {tuple(kv_cache.shape)}; "
            "expected dimension 2 at axis 0 or 1"
        )

    # Standard vLLM paged KV layout is [num_blocks, 2, block_size, heads, dim].
    # Prefer axis 1 so num_blocks == 2 does not get mistaken for K/V.
    if kv_cache.shape[1] == 2:
        return kv_cache.unbind(1)
    if kv_cache.shape[0] == 2:
        return kv_cache.unbind(0)

    raise ValueError(
        f"Unexpected KV cache shape {tuple(kv_cache.shape)}; "
        "expected dimension 2 at axis 0 or 1"
    )


def _draft_graph_debug_enabled() -> bool:
    return os.getenv("VLLM_FLASH_V100_DRAFT_GRAPH_DEBUG", "0") == "1"


def _draft_graph_debug_limit() -> int:
    return int(os.getenv("VLLM_FLASH_V100_DRAFT_GRAPH_DEBUG_LIMIT", "12"))


def _dflash_prefix_dump_enabled() -> bool:
    return os.getenv("VLLM_FLASH_V100_DFLASH_PREFIX_DUMP", "0") == "1"


def _dflash_ddtree_triton_branch_attn_enabled() -> bool:
    return os.getenv("VLLM_DFLASH_DDTREE_TRITON_BRANCH_ATTN", "1") != "0"


def _dflash_ddtree_triton_branch_attn_strict() -> bool:
    return os.getenv("VLLM_DFLASH_DDTREE_TRITON_BRANCH_ATTN_STRICT", "0") == "1"


def _dflash_ddtree_worker_profile_enabled() -> bool:
    return os.getenv("VLLM_DFLASH_DDTREE_WORKER_PROFILE", "0") == "1"


def _format_tensor_debug(tensor: torch.Tensor | None, name: str) -> str:
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


def _draft_graph_debug_log(key: str, message: str, *args: object) -> None:
    if not _draft_graph_debug_enabled():
        return
    count = _draft_graph_debug_counts.get(key, 0)
    if count >= _draft_graph_debug_limit():
        return
    _draft_graph_debug_counts[key] = count + 1
    if args:
        message = message % args
    logger.info(
        "FLASH_ATTN_V100 draft graph debug[%s#%d]: %s",
        key,
        count,
        message,
    )


def _graph_metadata_debug_log(key: str, message: str, *args: object) -> None:
    if not _draft_graph_debug_enabled():
        return
    count = _draft_graph_debug_counts.get(key, 0)
    if count >= _draft_graph_debug_limit():
        return
    _draft_graph_debug_counts[key] = count + 1
    if args:
        message = message % args
    logger.info(
        "FLASH_ATTN_V100 graph metadata debug[%s#%d]: %s",
        key,
        count,
        message,
    )


def _decode_dynamic_partitions_enabled() -> bool:
    return os.getenv("VLLM_FLASH_V100_DECODE_DYNAMIC_PARTITIONS", "1") != "0"


def _decode_partition_size_for_metadata(
    max_seq_len_hint: int | None = None,
) -> int:
    raw = os.getenv("VLLM_FLASH_V100_DECODE_PARTITION_SIZE")
    if raw is None:
        return _select_default_decode_partition_size(max_seq_len_hint)
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            "VLLM_FLASH_V100_DECODE_PARTITION_SIZE must be one of "
            f"{_VALID_DECODE_PARTITION_SIZES}, got {raw!r}"
        ) from exc
    if value not in _VALID_DECODE_PARTITION_SIZES:
        raise ValueError(
            "VLLM_FLASH_V100_DECODE_PARTITION_SIZE must be one of "
            f"{_VALID_DECODE_PARTITION_SIZES}, got {value}"
        )
    return value


def _mtp_context_bucket_partition_size_hint() -> int | None:
    raw = os.getenv("VLLM_SM70_MTP_CONTEXT_BUCKET_PARTITION_SIZE")
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
    if os.getenv("VLLM_FLASH_V100_XQA_MTP5_DUAL_CTA", "1") != "1":
        return None
    raw = os.getenv("VLLM_FLASH_V100_XQA_MTP5_PARTITION_SIZE", "1024")
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


def _select_default_decode_partition_size(
    max_seq_len_hint: int | None,
) -> int:
    if max_seq_len_hint is None:
        return _DEFAULT_DECODE_PARTITION_SIZE

    seq_len = max(1, int(max_seq_len_hint))
    if seq_len >= 32768:
        return 1024
    return _DEFAULT_DECODE_PARTITION_SIZE


def _decode_xqa_q4_min_seq_len() -> int:
    raw = os.getenv("VLLM_FLASH_V100_DECODE_XQA_Q4_MIN_SEQ_LEN")
    if raw is None:
        return _DEFAULT_Q4_XQA_MIN_SEQ_LEN
    try:
        return max(1, int(raw))
    except ValueError as exc:
        raise ValueError(
            f"VLLM_FLASH_V100_DECODE_XQA_Q4_MIN_SEQ_LEN must be an integer, got {raw!r}"
        ) from exc


def _decode_fp8_xqa_min_seq_len() -> int:
    raw = os.getenv("VLLM_FLASH_V100_DECODE_FP8_XQA_MIN_SEQ_LEN")
    if raw is None:
        return _DEFAULT_FP8_XQA_MIN_SEQ_LEN
    try:
        return max(1, int(raw))
    except ValueError as exc:
        raise ValueError(
            "VLLM_FLASH_V100_DECODE_FP8_XQA_MIN_SEQ_LEN must be an integer, "
            f"got {raw!r}"
        ) from exc


def _decode_fp8_xqa_allowed(
    attn_metadata: TritonAttentionMetadata,
    query: torch.Tensor,
) -> bool:
    if bool(getattr(attn_metadata, "flash_v100_cudagraph_capture", False)):
        return True
    if _is_cuda_graph_capturing(query):
        hint_names = (
            "flash_v100_static_decode_seq_hint",
            "flash_v100_decode_workspace_seq_capacity_hint",
            "flash_v100_decode_max_seq_len_hint",
        )
    else:
        hint_names = (
            "flash_v100_decode_max_seq_len_hint",
            "flash_v100_static_decode_seq_hint",
            "flash_v100_decode_workspace_seq_capacity_hint",
        )
    for name in hint_names:
        seq_hint = getattr(attn_metadata, name, None)
        if seq_hint is not None:
            return int(seq_hint) >= _decode_fp8_xqa_min_seq_len()
    return False


def _decode_xqa_allowed_for_q_per_kv(
    q_per_kv: int,
    attn_metadata: TritonAttentionMetadata,
) -> bool:
    if q_per_kv in (6, 8):
        return True
    if q_per_kv != 4:
        return False

    seq_hint = getattr(
        attn_metadata,
        "flash_v100_decode_workspace_seq_capacity_hint",
        None,
    )
    if seq_hint is None:
        seq_hint = getattr(
            attn_metadata,
            "flash_v100_static_decode_seq_hint",
            None,
        )
    if seq_hint is None:
        seq_hint = getattr(
            attn_metadata,
            "flash_v100_decode_max_seq_len_hint",
            None,
        )
    if seq_hint is None:
        return False
    return int(seq_hint) >= _decode_xqa_q4_min_seq_len()


def _same_storage(left: torch.Tensor, right: torch.Tensor) -> bool:
    return left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()


def _is_cuda_graph_capturing(tensor: torch.Tensor) -> bool:
    return bool(tensor.is_cuda and torch.cuda.is_current_stream_capturing())


def _route_summary_enabled() -> bool:
    return (
        os.getenv("VLLM_FLASH_V100_ROUTE_SUMMARY", "0") == "1"
        or os.getenv("VLLM_FLASH_V100_DEBUG_ROUTE_SUMMARY", "0") == "1"
    )


def _log_route_summary() -> None:
    if _route_counts:
        logger.info(
            "FLASH_ATTN_V100 route summary: %s",
            json.dumps(_route_counts, sort_keys=True),
        )


def _record_route(route: str) -> None:
    global _route_summary_registered
    if not _route_summary_enabled():
        return
    _route_counts[route] = _route_counts.get(route, 0) + 1
    if not _route_summary_registered:
        atexit.register(_log_route_summary)
        _route_summary_registered = True


def _ddtree_trace_event(event: str, payload: dict[str, object]) -> None:
    trace_path = os.getenv("VLLM_DFLASH_DDTREE_TRACE_JSONL")
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
    return bool(os.getenv("VLLM_DFLASH_DDTREE_TRACE_JSONL"))


def _decode_active_trace_enabled() -> bool:
    return os.getenv("VLLM_FLASH_V100_TRACE_DECODE_ACTIVE", "0") == "1"


def _decode_active_value(active_num_partitions: object) -> int | None:
    if not isinstance(active_num_partitions, torch.Tensor):
        return None
    if active_num_partitions.numel() == 0:
        return None
    return int(active_num_partitions.detach().reshape(-1)[0].item())


def _trace_decode_active(
    *,
    route: str,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    attn_metadata: TritonAttentionMetadata,
    window_size: tuple[int, int],
) -> None:
    if not _decode_active_trace_enabled():
        return
    if torch.cuda.is_current_stream_capturing():
        return

    active_value = _decode_active_value(
        getattr(attn_metadata, "flash_v100_decode_active_num_partitions", None)
    )
    seq_len = int(seq_lens[: query.shape[0]].max().item())
    partition_size = _decode_partition_size_for_metadata(seq_len)
    expected_active = max(1, (seq_len + partition_size - 1) // partition_size)
    max_seq_hint = getattr(
        attn_metadata,
        "flash_v100_decode_max_seq_len_hint",
        None,
    )
    workspace_hint = getattr(
        attn_metadata,
        "flash_v100_decode_workspace_seq_capacity_hint",
        None,
    )
    static_hint = getattr(
        attn_metadata,
        "flash_v100_static_decode_seq_hint",
        None,
    )
    workspace_partitions = (
        max(1, (int(workspace_hint) + partition_size - 1) // partition_size)
        if workspace_hint is not None
        else None
    )
    signature = (
        route,
        int(query.shape[0]),
        int(query.shape[1]),
        int(key_cache.shape[2]),
        int(query.shape[2]),
        int(key_cache.shape[1]),
        seq_len,
        partition_size,
        active_value,
        expected_active,
        workspace_partitions,
        window_size,
    )
    if signature in _decode_active_trace_signatures:
        return
    _decode_active_trace_signatures.add(signature)
    logger.info(
        "FLASH_ATTN_V100 decode active trace: route=%s q=%d heads_q=%d "
        "heads_kv=%d head_dim=%d page_size=%d seq_len=%d partition=%d "
        "active=%s expected_active=%d workspace_partitions=%s "
        "max_seq_hint=%s workspace_hint=%s static_hint=%s window=%s",
        route,
        query.shape[0],
        query.shape[1],
        key_cache.shape[2],
        query.shape[2],
        key_cache.shape[1],
        seq_len,
        partition_size,
        active_value,
        expected_active,
        workspace_partitions,
        max_seq_hint,
        workspace_hint,
        static_hint,
        window_size,
    )


def _trace_decode_active_metadata(
    *,
    stage: str,
    max_seq_len_hint: int,
    workspace_seq_capacity_hint: int | None,
    static_decode_seq_hint: int | None,
    active: int,
    partition_size: int,
) -> None:
    if not _decode_active_trace_enabled():
        return

    expected_active = max(
        1,
        (int(max_seq_len_hint) + partition_size - 1) // partition_size,
    )
    workspace_partitions = (
        max(
            1,
            (int(workspace_seq_capacity_hint) + partition_size - 1) // partition_size,
        )
        if workspace_seq_capacity_hint is not None
        else None
    )
    signature = (
        "metadata",
        stage,
        active,
        partition_size,
        workspace_partitions,
    )
    if signature in _decode_active_trace_signatures:
        return
    _decode_active_trace_signatures.add(signature)
    logger.info(
        "FLASH_ATTN_V100 decode active metadata: stage=%s seq_len_hint=%d "
        "partition=%d active=%d expected_active=%d workspace_partitions=%s "
        "workspace_hint=%s static_hint=%s",
        stage,
        max_seq_len_hint,
        partition_size,
        active,
        expected_active,
        workspace_partitions,
        workspace_seq_capacity_hint,
        static_decode_seq_hint,
    )


def _uses_fp8_kv_cache(kv_cache_dtype: str) -> bool:
    return isinstance(kv_cache_dtype, str) and kv_cache_dtype.startswith("fp8")


def _log_fp8_kv_cache_route(stage: str, kv_cache_dtype: str, route: str) -> None:
    global _logged_fp8_kv_decode, _logged_fp8_kv_prefill

    if not _uses_fp8_kv_cache(kv_cache_dtype):
        return
    if stage not in ("prefill", "decode"):
        raise ValueError(f"Unsupported FP8 KV cache route stage: {stage}")
    _record_route(f"fp8_kv_{stage}")
    _record_route(f"fp8_kv_{stage}_{route}")
    if stage == "prefill":
        if _logged_fp8_kv_prefill:
            return
        logger.info(
            "FLASH_ATTN_V100 FP8 KV cache prefill path active "
            "(kv_cache_dtype=%s, route=%s).",
            kv_cache_dtype,
            route,
        )
        _logged_fp8_kv_prefill = True
        return
    if stage == "decode":
        if _logged_fp8_kv_decode:
            return
        logger.info(
            "FLASH_ATTN_V100 FP8 KV cache decode path active "
            "(kv_cache_dtype=%s, route=%s).",
            kv_cache_dtype,
            route,
        )
        _logged_fp8_kv_decode = True
        return


def _callable_accepts_keyword(fn: object, name: str) -> bool:
    if not callable(fn):
        return False
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return name in params or any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()
    )


def _get_flash_ops():
    """Lazy-load flash_attn_v100 ops if available."""
    global _flash_attn_func, _flash_attn_bhmd_func
    global _flash_attn_decode_paged, _flash_attn_decode_paged_xqa
    global _flash_attn_prefill_paged
    global _flash_attn_decode_paged_wmma, _flash_attn_prefill_paged_bhmd
    global _flash_attn_prefill_paged_bfla, _flash_attn_prefill_paged_splitkv
    if (
        _flash_attn_func is None
        or _flash_attn_decode_paged is None
        or _flash_attn_prefill_paged is None
    ):
        try:
            from flash_attn_v100 import (
                flash_attn_bhmd_func,
                flash_attn_decode_paged,
                flash_attn_func,
                flash_attn_prefill_paged,
            )

            _flash_attn_func = flash_attn_func
            _flash_attn_bhmd_func = flash_attn_bhmd_func
            _flash_attn_decode_paged = flash_attn_decode_paged
            _flash_attn_prefill_paged = flash_attn_prefill_paged
            try:
                from flash_attn_v100 import flash_attn_decode_paged_xqa

                _flash_attn_decode_paged_xqa = flash_attn_decode_paged_xqa
            except ImportError:
                _flash_attn_decode_paged_xqa = None
            try:
                from flash_attn_v100 import flash_attn_decode_paged_wmma

                _flash_attn_decode_paged_wmma = flash_attn_decode_paged_wmma
            except ImportError:
                _flash_attn_decode_paged_wmma = None
            try:
                from flash_attn_v100 import flash_attn_prefill_paged_bhmd

                _flash_attn_prefill_paged_bhmd = flash_attn_prefill_paged_bhmd
            except ImportError:
                _flash_attn_prefill_paged_bhmd = None
            try:
                from flash_attn_v100 import flash_attn_prefill_paged_bfla

                _flash_attn_prefill_paged_bfla = flash_attn_prefill_paged_bfla
            except ImportError:
                _flash_attn_prefill_paged_bfla = None
            try:
                from flash_attn_v100 import flash_attn_prefill_paged_splitkv

                _flash_attn_prefill_paged_splitkv = flash_attn_prefill_paged_splitkv
            except ImportError:
                _flash_attn_prefill_paged_splitkv = None
        except ImportError:
            _flash_attn_func = None
            _flash_attn_bhmd_func = None
            _flash_attn_decode_paged = None
            _flash_attn_decode_paged_xqa = None
            _flash_attn_decode_paged_wmma = None
            _flash_attn_prefill_paged = None
            _flash_attn_prefill_paged_bhmd = None
            _flash_attn_prefill_paged_bfla = None
            _flash_attn_prefill_paged_splitkv = None
    return (
        _flash_attn_func,
        _flash_attn_bhmd_func,
        _flash_attn_decode_paged,
        _flash_attn_decode_paged_xqa,
        _flash_attn_decode_paged_wmma,
        _flash_attn_prefill_paged,
        _flash_attn_prefill_paged_bhmd,
        _flash_attn_prefill_paged_bfla,
        _flash_attn_prefill_paged_splitkv,
    )


def _get_fp8_e5m2_paged_kv_bridge_op():
    global _fp8_e5m2_paged_kv_to_fp16
    global _fp8_e5m2_paged_kv_to_fp16_checked
    if not _fp8_e5m2_paged_kv_to_fp16_checked:
        _fp8_e5m2_paged_kv_to_fp16_checked = True
        try:
            from flash_attn_v100 import fp8_e5m2_paged_kv_to_fp16

            _fp8_e5m2_paged_kv_to_fp16 = fp8_e5m2_paged_kv_to_fp16
        except ImportError:
            _fp8_e5m2_paged_kv_to_fp16 = None
    return _fp8_e5m2_paged_kv_to_fp16


def _get_fp8_prefill_bridge_workspace(
    key_cache: torch.Tensor,
    required_blocks: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    device_index = (
        key_cache.device.index
        if key_cache.device.index is not None
        else torch.accelerator.current_device_index()
    )
    stream_id = int(torch.cuda.current_stream(key_cache.device).cuda_stream)
    cache_key = (
        device_index,
        stream_id,
        int(key_cache.shape[2]),
        int(key_cache.shape[3]),
    )
    workspace = _fp8_prefill_bridge_workspaces.get(cache_key)
    if workspace is not None and workspace[0].shape[0] >= required_blocks:
        return (
            workspace[0][:required_blocks],
            workspace[1][:required_blocks],
            workspace[2][:, :required_blocks],
        )

    if torch.cuda.is_current_stream_capturing():
        return None

    previous_capacity = workspace[0].shape[0] if workspace is not None else 0
    capacity = max(required_blocks, previous_capacity * 2)
    shape = (
        capacity,
        _FP8_PREFILL_BRIDGE_PAGE_SIZE,
        key_cache.shape[2],
        key_cache.shape[3],
    )
    key_out = torch.empty(shape, dtype=torch.float16, device=key_cache.device)
    value_out = torch.empty_like(key_out)
    block_table = torch.arange(
        capacity,
        dtype=torch.int32,
        device=key_cache.device,
    ).unsqueeze(0)
    _fp8_prefill_bridge_workspaces[cache_key] = (
        key_out,
        value_out,
        block_table,
    )
    return (
        key_out[:required_blocks],
        value_out[:required_blocks],
        block_table[:, :required_blocks],
    )


def flash_v100_dense_prefill_available() -> bool:
    flash_attn_func, _, _, _, _, _, _, _, _ = _get_flash_ops()
    return flash_attn_func is not None


def flash_v100_dense_prefill(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_actual_tokens: int,
    softmax_scale: float,
    causal: bool = True,
    window_size: tuple[int, int] = (-1, -1),
) -> torch.Tensor:
    """Run Flash-V100 dense raw-QKV prefill without backend metadata coupling."""
    flash_attn_func, _, _, _, _, _, _, _, _ = _get_flash_ops()
    if flash_attn_func is None:
        raise RuntimeError("flash_attn_v100 dense prefill op is unavailable")

    query = query[:num_actual_tokens]
    key = key[:num_actual_tokens]
    value = value[:num_actual_tokens]
    out_view = output[:num_actual_tokens]

    num_seqs = len(query_start_loc) - 1
    if num_seqs == 0:
        return output

    seq_lens = query_start_loc[1:] - query_start_loc[:-1]
    run_start = 0
    while run_start < num_seqs:
        run_seq_len = int(seq_lens[run_start].item())
        run_end = run_start + 1
        while run_end < num_seqs and int(seq_lens[run_end].item()) == run_seq_len:
            run_end += 1

        if run_seq_len > 0:
            tok_start = int(query_start_loc[run_start].item())
            tok_end = int(query_start_loc[run_end].item())
            batch_size = run_end - run_start

            q_batch = query[tok_start:tok_end].view(
                batch_size, run_seq_len, query.shape[1], query.shape[2]
            )
            k_batch = key[tok_start:tok_end].view(
                batch_size, run_seq_len, key.shape[1], key.shape[2]
            )
            v_batch = value[tok_start:tok_end].view(
                batch_size, run_seq_len, value.shape[1], value.shape[2]
            )

            out_batch = flash_attn_func(
                q_batch,
                k_batch,
                v_batch,
                causal=causal,
                softmax_scale=softmax_scale,
                window_size=window_size,
            )
            out_view[tok_start:tok_end].copy_(
                out_batch.view(
                    tok_end - tok_start, out_batch.shape[2], out_batch.shape[3]
                )
            )

        run_start = run_end

    return output


def _get_flash_turboquant_decode_op():
    """Lazy-load the optional TurboQuant decode op without touching base ops."""
    global _flash_attn_turboquant_decode_checked
    global _flash_attn_turboquant_decode_paged
    if not _flash_attn_turboquant_decode_checked:
        try:
            from flash_attn_v100 import (
                flash_attn_turboquant_decode_paged,
                flash_attn_turboquant_decode_paged_available,
            )

            if flash_attn_turboquant_decode_paged_available():
                _flash_attn_turboquant_decode_paged = flash_attn_turboquant_decode_paged
            else:
                _flash_attn_turboquant_decode_paged = None
        except ImportError:
            _flash_attn_turboquant_decode_paged = None
        _flash_attn_turboquant_decode_checked = True
    return _flash_attn_turboquant_decode_paged


def flash_v100_turboquant_decode_available() -> bool:
    return _get_flash_turboquant_decode_op() is not None


def flash_v100_turboquant_decode(
    q_rot: torch.Tensor,
    kv_cache: torch.Tensor,
    output: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    centroids: torch.Tensor,
    softmax_scale: float,
    mse_bits: int,
    value_quant_bits: int,
    norm_correction: bool,
    num_kv_splits: int,
) -> torch.Tensor:
    """Run Flash-V100 decode directly over TurboQuant packed paged cache."""
    op = _get_flash_turboquant_decode_op()
    if op is None:
        raise RuntimeError("flash_attn_v100 TurboQuant decode op is unavailable")
    return op(
        q_rot,
        kv_cache,
        block_table,
        seq_lens,
        centroids,
        softmax_scale=softmax_scale,
        out=output,
        mse_bits=mse_bits,
        value_quant_bits=value_quant_bits,
        norm_correction=norm_correction,
        num_kv_splits=num_kv_splits,
    )


def _get_paged_kv_utils():
    """Lazy-load paged KV extraction CUDA extension."""
    global _paged_kv_utils
    if _paged_kv_utils is None:
        try:
            from flash_attn_v100 import paged_kv_utils

            _paged_kv_utils = paged_kv_utils
        except ImportError:
            try:
                import paged_kv_utils

                _paged_kv_utils = paged_kv_utils
            except ImportError:
                _paged_kv_utils = None
    return _paged_kv_utils


def _has_prefix_context(attn_metadata: TritonAttentionMetadata) -> bool:
    """Return True if any sequence has KV context before current query tokens."""
    query_start_loc_cpu = getattr(attn_metadata, "query_start_loc_cpu", None)
    seq_lens_cpu = getattr(attn_metadata, "seq_lens_cpu", None)
    if query_start_loc_cpu is not None and seq_lens_cpu is not None:
        query_lens = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        return bool(torch.any(query_lens != seq_lens_cpu).item())

    query_lens = attn_metadata.query_start_loc[1:] - attn_metadata.query_start_loc[:-1]
    return not torch.equal(query_lens, attn_metadata.seq_lens)


def _metadata_expects_more_query_tokens_than_available(
    attn_metadata: TritonAttentionMetadata,
    available_query_tokens: int,
) -> bool:
    """Return True when per-layer Q/K/V tensors are shorter than query metadata.

    Hybrid Qwen3.5/3.6 routes can feed a full-attention layer only the live
    query-token subset while the batch-level metadata still describes the
    wider request span. That shape is not a dense raw-QKV prefill; it must use
    the prefix/live-token compatible path.
    """
    query_start_loc_cpu = getattr(attn_metadata, "query_start_loc_cpu", None)
    query_start_loc = (
        query_start_loc_cpu
        if query_start_loc_cpu is not None
        else attn_metadata.query_start_loc
    )
    if len(query_start_loc) <= 1:
        return False
    expected_query_tokens = int(query_start_loc[-1].item())
    return available_query_tokens < expected_query_tokens


def _normalize_query_start_loc_for_available_tokens(
    query_start_loc: torch.Tensor,
    available_query_tokens: int,
) -> torch.Tensor:
    """Project metadata query spans onto the tokens actually present in Q/K/V.

    This is only needed when a hybrid/model-specific path feeds a full-attention
    layer a live-token subset instead of the full batch span described by the
    shared metadata.
    """
    num_seqs = len(query_start_loc) - 1
    if num_seqs <= 0:
        return query_start_loc

    expected_query_tokens = int(query_start_loc[-1].item())
    if available_query_tokens >= expected_query_tokens:
        return query_start_loc

    if available_query_tokens <= 0:
        return query_start_loc.new_zeros(query_start_loc.shape)

    if num_seqs == 1:
        return query_start_loc.new_tensor([0, available_query_tokens])

    if available_query_tokens == num_seqs:
        return torch.arange(
            num_seqs + 1,
            dtype=query_start_loc.dtype,
            device=query_start_loc.device,
        )

    if available_query_tokens % num_seqs == 0:
        q_per_seq = available_query_tokens // num_seqs
        orig_query_lens = query_start_loc[1:] - query_start_loc[:-1]
        if int(orig_query_lens.min().item()) >= q_per_seq:
            return torch.arange(
                0,
                available_query_tokens + 1,
                q_per_seq,
                dtype=query_start_loc.dtype,
                device=query_start_loc.device,
            )

    raise RuntimeError(
        "FLASH_ATTN_V100 received fewer layer query tokens than query metadata "
        "describes, and the per-sequence live-token layout could not be "
        "reconstructed safely."
    )


def _extract_contiguous_kv_from_paged_cache(
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    total_tokens: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract contiguous K/V from paged KV cache.

    Uses the CUDA extension when available and falls back to a Python path.
    """

    paged_kv_utils = _get_paged_kv_utils()

    key_cache, value_cache = _split_paged_kv_cache(kv_cache)

    if paged_kv_utils is not None and key_cache.dtype != torch.uint8:
        if hasattr(paged_kv_utils, "paged_kv_to_contiguous"):
            k_cont, v_cont = paged_kv_utils.paged_kv_to_contiguous(
                key_cache, value_cache, block_table, seq_lens
            )
        else:
            k_cont = paged_kv_utils.paged_to_contiguous(
                key_cache, block_table, seq_lens
            )
            v_cont = paged_kv_utils.paged_to_contiguous(
                value_cache, block_table, seq_lens
            )
        if total_tokens is None:
            total_tokens = int(seq_lens.sum().item())
        return k_cont[:total_tokens], v_cont[:total_tokens]

    # Slow Python fallback.
    batch_size = block_table.shape[0]
    if total_tokens is None:
        total_tokens = int(seq_lens.sum().item())

    k_cont = torch.empty(
        (total_tokens, num_kv_heads, head_dim),
        dtype=key_cache.dtype,
        device=key_cache.device,
    )
    v_cont = torch.empty(
        (total_tokens, num_kv_heads, head_dim),
        dtype=value_cache.dtype,
        device=value_cache.device,
    )

    token_offset = 0
    for batch_idx in range(batch_size):
        seq_len = int(seq_lens[batch_idx].item())
        if seq_len == 0:
            continue

        num_blocks = (seq_len + block_size - 1) // block_size
        for block_idx in range(num_blocks):
            physical_block_idx = int(block_table[batch_idx, block_idx].item())
            start_token = block_idx * block_size
            end_token = min(start_token + block_size, seq_len)
            n = end_token - start_token

            k_cont[token_offset : token_offset + n] = key_cache[physical_block_idx, :n]
            v_cont[token_offset : token_offset + n] = value_cache[
                physical_block_idx, :n
            ]
            token_offset += n

    return k_cont, v_cont


def _fp8_dtype_from_cache_dtype(kv_cache_dtype: str) -> torch.dtype:
    if kv_cache_dtype in ("fp8", "fp8_e4m3"):
        return torch.float8_e4m3fn
    if kv_cache_dtype == "fp8_e5m2":
        return torch.float8_e5m2
    raise ValueError(f"Unsupported FLASH_ATTN_V100 fp8 dtype: {kv_cache_dtype}")


def _dequantize_fp8_contiguous_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: float,
    v_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not _uses_fp8_kv_cache(kv_cache_dtype):
        return key, value
    fp8_dtype = _fp8_dtype_from_cache_dtype(kv_cache_dtype)
    key = key.view(fp8_dtype).to(torch.float16) * k_scale
    value = value.view(fp8_dtype).to(torch.float16) * v_scale
    return key, value


def _contiguous_paged_start_block(
    key_cache: torch.Tensor,
    block_table_row: torch.Tensor,
    seq_len: int,
    block_size: int,
    attn_metadata: TritonAttentionMetadata,
    seq_idx: int,
) -> tuple[int, int] | None:
    if seq_len <= 0 or block_size <= 0:
        return None
    num_blocks = (seq_len + block_size - 1) // block_size
    if num_blocks <= 0 or num_blocks > int(block_table_row.shape[0]):
        return None

    cache_key = (
        int(seq_idx),
        int(seq_len),
        int(block_size),
        int(block_table_row.data_ptr()),
        int(key_cache.data_ptr()),
    )
    contig_cache = getattr(attn_metadata, "flash_v100_contig_dense_cache", None)
    if contig_cache is None:
        contig_cache = {}
        _as_flash_v100_metadata(
            attn_metadata
        ).flash_v100_contig_dense_cache = contig_cache

    start_block = contig_cache.get(cache_key)
    if start_block is None:
        blocks_cpu = block_table_row[:num_blocks].detach().cpu()
        if int(blocks_cpu[0].item()) < 0:
            contig_cache[cache_key] = -1
            return None
        if num_blocks > 1:
            expected = blocks_cpu[0] + torch.arange(
                num_blocks,
                dtype=blocks_cpu.dtype,
                device=blocks_cpu.device,
            )
            if not bool(torch.equal(blocks_cpu, expected)):
                contig_cache[cache_key] = -1
                return None
        start_block = int(blocks_cpu[0].item())
        contig_cache[cache_key] = start_block

    if start_block < 0:
        return None
    if start_block + num_blocks > int(key_cache.shape[0]):
        return None

    return start_block, num_blocks


def _contiguous_paged_kv_view(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table_row: torch.Tensor,
    seq_len: int,
    block_size: int,
    attn_metadata: TritonAttentionMetadata,
    seq_idx: int,
    allow_copy: bool,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return a dense [1, N, Hkv, D] K/V view for physically contiguous pages."""
    if key_cache.dtype != torch.float16 or value_cache.dtype != torch.float16:
        return None
    if key_cache.shape != value_cache.shape:
        return None
    if not allow_copy and (
        not key_cache.is_contiguous() or not value_cache.is_contiguous()
    ):
        return None

    start_info = _contiguous_paged_start_block(
        key_cache,
        block_table_row,
        seq_len,
        block_size,
        attn_metadata,
        seq_idx,
    )
    if start_info is None:
        return None
    start_block, num_blocks = start_info

    num_kv_heads = key_cache.shape[2]
    head_dim = key_cache.shape[3]
    end_block = start_block + num_blocks
    key_block_slice = key_cache[start_block:end_block]
    value_block_slice = value_cache[start_block:end_block]
    key_flat = key_block_slice.reshape(-1, num_kv_heads, head_dim)
    value_flat = value_block_slice.reshape(-1, num_kv_heads, head_dim)
    return (
        key_flat[:seq_len].unsqueeze(0),
        value_flat[:seq_len].unsqueeze(0),
    )


def _contiguous_paged_kv_bhmd(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table_row: torch.Tensor,
    seq_len: int,
    block_size: int,
    attn_metadata: TritonAttentionMetadata,
    seq_idx: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return dense [1, Hkv, N, D] K/V tensors for contiguous paged cache."""
    if key_cache.dtype != torch.float16 or value_cache.dtype != torch.float16:
        return None
    if key_cache.shape != value_cache.shape:
        return None

    start_info = _contiguous_paged_start_block(
        key_cache,
        block_table_row,
        seq_len,
        block_size,
        attn_metadata,
        seq_idx,
    )
    if start_info is None:
        return None
    start_block, num_blocks = start_info

    num_kv_heads = key_cache.shape[2]
    head_dim = key_cache.shape[3]
    end_block = start_block + num_blocks
    key_blocks = key_cache[start_block:end_block]
    value_blocks = value_cache[start_block:end_block]
    key_bhmd = (
        key_blocks.permute(2, 0, 1, 3)
        .reshape(1, num_kv_heads, -1, head_dim)[:, :, :seq_len, :]
        .contiguous()
    )
    value_bhmd = (
        value_blocks.permute(2, 0, 1, 3)
        .reshape(1, num_kv_heads, -1, head_dim)[:, :, :seq_len, :]
        .contiguous()
    )
    return key_bhmd, value_bhmd


def _cdiv_int(a: int, b: int) -> int:
    return (a + b - 1) // b


def _build_bfla_block_mask_for_seq(
    q_seq: torch.Tensor,
    key_cache: torch.Tensor,
    block_table_row: torch.Tensor,
    *,
    seq_len: int,
    block_size: int,
    mask_block_n: int,
    softmax_scale: float,
) -> torch.Tensor | None:
    """Build [1, Hkv, q_tiles, kv_tiles] sparse prefill mask."""
    if q_seq.ndim != 4 or q_seq.shape[0] != 1:
        return None
    if key_cache.dtype != torch.float16 or q_seq.dtype != torch.float16:
        return None
    if mask_block_n <= 0:
        return None

    pool_mode = envs.VLLM_FLASH_V100_BFLA_POOL.lower()
    flat_group_tokens = 64
    use_flat64 = pool_mode == "flat64"
    if use_flat64 and mask_block_n % flat_group_tokens != 0:
        return None

    q_len = int(q_seq.shape[1])
    num_query_heads = int(q_seq.shape[2])
    head_dim = int(q_seq.shape[3])
    num_kv_heads = int(key_cache.shape[2])
    if q_len <= 1 or seq_len < q_len:
        return None
    if num_query_heads % num_kv_heads != 0:
        return None

    q_blocks = _cdiv_int(q_len, mask_block_n)
    kv_tiles = _cdiv_int(seq_len, mask_block_n)
    if q_blocks <= 0 or kv_tiles <= 0:
        return None

    def pool_blocks(x: torch.Tensor) -> torch.Tensor:
        if use_flat64:
            groups = mask_block_n // flat_group_tokens
            return (
                x.view(
                    x.shape[0],
                    groups,
                    flat_group_tokens,
                    x.shape[2],
                    x.shape[3],
                )
                .permute(3, 0, 1, 2, 4)
                .reshape(
                    x.shape[2],
                    x.shape[0],
                    groups,
                    flat_group_tokens * x.shape[3],
                )
            )
        if pool_mode == "center":
            return x[:, min(mask_block_n // 2, x.shape[1] - 1)].permute(1, 0, 2)
        if pool_mode == "maxabs":
            idx = torch.argmax(x.abs(), dim=1, keepdim=True)
            return torch.gather(x, 1, idx).squeeze(1).permute(1, 0, 2)
        return x.mean(dim=1).permute(1, 0, 2)

    q_req = q_seq.squeeze(0)
    q_pad = torch.zeros(
        (q_blocks * mask_block_n, num_query_heads, head_dim),
        device=q_seq.device,
        dtype=q_seq.dtype,
    )
    q_pad[:q_len].copy_(q_req)
    q_low = pool_blocks(q_pad.view(q_blocks, mask_block_n, num_query_heads, head_dim))

    num_pages = _cdiv_int(seq_len, block_size)
    pages = block_table_row[:num_pages].to(torch.long)
    k_req = key_cache.index_select(0, pages).reshape(-1, num_kv_heads, head_dim)
    k_req = k_req[:seq_len]
    k_pad = torch.zeros(
        (kv_tiles * mask_block_n, num_kv_heads, head_dim),
        device=q_seq.device,
        dtype=key_cache.dtype,
    )
    k_pad[:seq_len].copy_(k_req)
    k_low = pool_blocks(k_pad.view(kv_tiles, mask_block_n, num_kv_heads, head_dim))

    num_queries_per_kv = num_query_heads // num_kv_heads
    keep_per_kv = torch.zeros(
        (num_kv_heads, q_blocks, kv_tiles),
        device=q_seq.device,
        dtype=torch.bool,
    )
    context_len = seq_len - q_len
    q_block_end = (
        context_len
        + (torch.arange(q_blocks, device=q_seq.device) + 1) * mask_block_n
        - 1
    )
    q_block_end = torch.clamp(q_block_end, max=seq_len - 1)
    k_block_start = torch.arange(kv_tiles, device=q_seq.device) * mask_block_n
    causal = k_block_start[None, :] <= q_block_end[:, None]

    threshold = float(envs.VLLM_FLASH_V100_BFLA_THRESHOLD)
    keep_mass = float(envs.VLLM_FLASH_V100_BFLA_KEEP_MASS)
    keep_ratio = float(envs.VLLM_FLASH_V100_BFLA_KEEP_RATIO)
    min_keep_blocks = int(envs.VLLM_FLASH_V100_BFLA_MIN_KEEP_BLOCKS)
    for kv_h in range(num_kv_heads):
        q_h0 = kv_h * num_queries_per_kv
        q_h1 = q_h0 + num_queries_per_kv
        if use_flat64:
            group_scores = torch.einsum(
                "hqgf,krf->hqkgr", q_low[q_h0:q_h1], k_low[kv_h]
            )
            scores = group_scores.amax(dim=(-1, -2))
        else:
            scores = torch.einsum("hqd,kd->hqk", q_low[q_h0:q_h1], k_low[kv_h])
        scores = scores.masked_fill(~causal[None, :, :], float("-inf"))
        probs = torch.softmax(scores.float() * softmax_scale, dim=-1)
        keep = (probs > threshold).any(dim=0)

        if keep_mass >= 1.0:
            keep |= causal
        elif keep_mass > 0:
            sorted_probs, sorted_idx = torch.sort(
                probs.float(), dim=-1, descending=True
            )
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            mass_keep_sorted = cumsum <= keep_mass
            mass_keep_sorted[..., 0] = True
            first_over = torch.argmax(
                (cumsum >= keep_mass).to(torch.int32), dim=-1, keepdim=True
            )
            mass_keep_sorted.scatter_(-1, first_over, True)
            mass_keep = torch.zeros_like(probs, dtype=torch.bool)
            mass_keep.scatter_(-1, sorted_idx, mass_keep_sorted)
            keep |= mass_keep.any(dim=0)

        if keep_ratio > 0 or min_keep_blocks > 0:
            topk = max(min_keep_blocks, int(kv_tiles * keep_ratio))
            topk = max(1, min(topk, kv_tiles))
            _, topk_idx = torch.topk(scores.float(), k=topk, dim=-1)
            topk_keep = torch.zeros_like(scores, dtype=torch.bool)
            topk_keep.scatter_(-1, topk_idx, True)
            keep |= topk_keep.any(dim=0)
        keep_per_kv[kv_h] = keep

    keep_per_kv &= causal[None, :, :]
    q_tile_abs = (
        context_len + torch.arange(q_blocks, device=q_seq.device) * mask_block_n
    ) // mask_block_n
    k_idx = torch.arange(kv_tiles, device=q_seq.device)
    local_blocks = max(0, int(envs.VLLM_FLASH_V100_BFLA_LOCAL_BLOCKS))
    local = (k_idx[None, :] <= q_tile_abs[:, None]) & (
        k_idx[None, :] >= q_tile_abs[:, None] - local_blocks
    )
    keep_per_kv |= local[None, :, :]
    keep_per_kv[:, :, 0] = True

    spec_stride = int(envs.VLLM_FLASH_V100_BFLA_SPEC_STRIDE)
    if spec_stride > 0:
        dropped = causal[None, :, :] & ~keep_per_kv
        q_idx = torch.arange(q_blocks, device=q_seq.device, dtype=torch.int64)[:, None]
        k_idx_i64 = torch.arange(kv_tiles, device=q_seq.device, dtype=torch.int64)[
            None, :
        ]
        stride_keep = (
            (q_idx * 131 + k_idx_i64 * 17 + int(envs.VLLM_FLASH_V100_BFLA_SPEC_SEED))
            % spec_stride
        ) == 0
        keep_per_kv |= dropped & stride_keep[None, :, :]

    spec_prob = float(envs.VLLM_FLASH_V100_BFLA_SPEC_PROB)
    if spec_prob > 0:
        prob = max(0.0, min(spec_prob, 1.0))
        dropped = causal[None, :, :] & ~keep_per_kv
        if prob >= 1.0:
            keep_per_kv |= dropped
        else:
            q_idx = torch.arange(q_blocks, device=q_seq.device, dtype=torch.int64)[
                None, :, None
            ]
            k_idx_i64 = torch.arange(kv_tiles, device=q_seq.device, dtype=torch.int64)[
                None, None, :
            ]
            h_idx = torch.arange(num_kv_heads, device=q_seq.device, dtype=torch.int64)[
                :, None, None
            ]
            hashed = (
                (q_idx + 1) * 1103515245
                + (k_idx_i64 + 1) * 12345
                + (h_idx + 1) * 2654435761
                + int(envs.VLLM_FLASH_V100_BFLA_SPEC_SEED)
            ) & 0x7FFFFFFF
            random_keep = (hashed % 1000000) < int(prob * 1000000)
            keep_per_kv |= dropped & random_keep

    return keep_per_kv.to(torch.int32).unsqueeze(0).contiguous()


def _torch_attention_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    causal: bool,
    window_size: tuple[int, int],
    softmax_scale: float,
) -> torch.Tensor:
    """Small debug-only fp32 attention reference for prefix/paged checks."""
    query_f = query.float()
    key_f = key.float()
    value_f = value.float()

    num_q_heads = query_f.shape[1]
    num_kv_heads = key_f.shape[1]
    if num_q_heads % num_kv_heads != 0:
        raise ValueError(
            "num attention heads must be divisible by num KV heads for debug "
            f"reference, got {num_q_heads=} {num_kv_heads=}"
        )
    if num_q_heads != num_kv_heads:
        repeat = num_q_heads // num_kv_heads
        key_f = key_f.repeat_interleave(repeat, dim=1)
        value_f = value_f.repeat_interleave(repeat, dim=1)

    # [H, M, N]
    scores = torch.einsum("mhd,nhd->hmn", query_f, key_f) * softmax_scale
    q_len = query_f.shape[0]
    k_len = key_f.shape[0]
    q_pos = torch.arange(q_len, device=query.device) + max(k_len - q_len, 0)
    k_pos = torch.arange(k_len, device=query.device)
    valid = torch.ones((q_len, k_len), device=query.device, dtype=torch.bool)
    if causal:
        valid &= k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)
    window_left, window_right = window_size
    if window_left >= 0:
        valid &= k_pos.unsqueeze(0) >= q_pos.unsqueeze(1) - window_left
    if window_right >= 0:
        valid &= k_pos.unsqueeze(0) <= q_pos.unsqueeze(1) + window_right
    scores = scores.masked_fill(~valid.unsqueeze(0), float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("hmn,nhd->mhd", probs, value_f)
    return out.to(dtype=query.dtype).unsqueeze(0)


def _ddtree_parent_ids_cpu(
    attn_metadata: TritonAttentionMetadata,
) -> torch.Tensor | None:
    parent_ids = getattr(attn_metadata, "ddtree_parent_ids", None)
    if parent_ids is None:
        return None
    parent_ids_cpu = getattr(attn_metadata, "ddtree_parent_ids_cpu", None)
    if parent_ids_cpu is None:
        parent_ids_cpu = parent_ids.detach().cpu()
        _as_flash_v100_metadata(attn_metadata).ddtree_parent_ids_cpu = parent_ids_cpu
    return parent_ids_cpu


def _ddtree_parent_metadata_requires_branch(
    attn_metadata: TritonAttentionMetadata,
    query_start_loc: torch.Tensor,
) -> bool:
    parent_ids = getattr(attn_metadata, "ddtree_parent_ids", None)
    num_tree_tokens_cpu = getattr(attn_metadata, "ddtree_num_tree_tokens_cpu", None)
    if parent_ids is None or num_tree_tokens_cpu is None:
        return False

    num_reqs = min(
        int(parent_ids.shape[0]),
        int(num_tree_tokens_cpu.numel()),
        max(0, len(query_start_loc) - 1),
    )
    if num_reqs <= 0:
        return False
    return bool(torch.any(num_tree_tokens_cpu[:num_reqs] > 0).item())


def _ddtree_triton_seq_lens_match(
    attn_metadata: TritonAttentionMetadata,
    seq_lens: torch.Tensor,
    num_reqs: int,
) -> bool:
    metadata_seq_lens = getattr(attn_metadata, "seq_lens", None)
    if metadata_seq_lens is None:
        return False
    if num_reqs <= 0:
        return True
    if metadata_seq_lens[:num_reqs].data_ptr() == seq_lens[:num_reqs].data_ptr():
        return True
    if _is_cuda_graph_capturing(metadata_seq_lens):
        return bool(
            getattr(attn_metadata, "ddtree_seq_lens_restored_for_triton", False)
        )
    return bool(
        torch.equal(
            metadata_seq_lens[:num_reqs].detach().cpu(),
            seq_lens[:num_reqs].detach().cpu(),
        )
    )


def _ddtree_triton_query_start_loc_match(
    attn_metadata: TritonAttentionMetadata,
    query_start_loc: torch.Tensor,
    num_reqs: int,
) -> bool:
    metadata_query_start_loc = getattr(attn_metadata, "query_start_loc", None)
    if metadata_query_start_loc is None:
        return False
    num_boundaries = num_reqs + 1
    if num_boundaries <= 1:
        return True
    if (
        metadata_query_start_loc[:num_boundaries].data_ptr()
        == query_start_loc[:num_boundaries].data_ptr()
    ):
        return True
    if _is_cuda_graph_capturing(metadata_query_start_loc):
        return bool(
            getattr(
                attn_metadata,
                "ddtree_query_start_loc_restored_for_triton",
                False,
            )
        )
    return bool(
        torch.equal(
            metadata_query_start_loc[:num_boundaries].detach().cpu(),
            query_start_loc[:num_boundaries].detach().cpu(),
        )
    )


def _ddtree_triton_parent_ids_for_query(
    parent_ids: torch.Tensor,
    num_tree_tokens_cpu: torch.Tensor | None,
    query_start_loc: torch.Tensor,
    *,
    is_capturing: bool,
) -> torch.Tensor | None:
    if num_tree_tokens_cpu is None or parent_ids.ndim != 2:
        return parent_ids

    max_q_len = int(parent_ids.shape[1])
    num_reqs = min(
        int(parent_ids.shape[0]),
        int(num_tree_tokens_cpu.numel()),
        max(0, len(query_start_loc) - 1),
    )
    if num_reqs <= 0 or max_q_len <= 0:
        return parent_ids

    query_lens = query_start_loc[1 : num_reqs + 1] - query_start_loc[:num_reqs]
    rows_needing_causal_parent: list[int] = []
    for req_idx in range(num_reqs):
        q_len = int(query_lens[req_idx].item())
        if q_len <= 0:
            continue
        if q_len > max_q_len:
            return None

        tree_len = int(num_tree_tokens_cpu[req_idx].item())
        if tree_len <= 0:
            rows_needing_causal_parent.append(req_idx)
            continue
        if q_len > tree_len + 1:
            return None

    if not rows_needing_causal_parent:
        return parent_ids
    if is_capturing:
        return None

    causal_parent_row = torch.arange(
        max_q_len,
        device=parent_ids.device,
        dtype=parent_ids.dtype,
    )
    causal_parent_row = torch.clamp(causal_parent_row - 1, min=0)
    triton_parent_ids = parent_ids.clone()
    triton_parent_ids[rows_needing_causal_parent, :] = causal_parent_row
    return triton_parent_ids


def _build_ddtree_visibility_mask(
    *,
    q_len: int,
    seq_len: int,
    prefix_len: int,
    tree_len: int,
    parent_row: torch.Tensor | None,
    device: torch.device,
    window_size: tuple[int, int],
) -> torch.Tensor:
    visible = torch.zeros((q_len, seq_len), dtype=torch.bool, device=device)
    if q_len <= 0 or seq_len <= 0:
        return visible

    for q_offset in range(q_len):
        logical_q_idx = prefix_len + q_offset
        if q_offset == 0 or tree_len <= 0:
            visible[q_offset, : min(logical_q_idx + 1, seq_len)] = True
            continue

        if q_offset > tree_len or parent_row is None:
            visible[q_offset, : min(logical_q_idx + 1, seq_len)] = True
            continue

        visible[q_offset, : min(prefix_len, seq_len)] = True
        if prefix_len < seq_len:
            visible[q_offset, prefix_len] = True
        if logical_q_idx < seq_len:
            visible[q_offset, logical_q_idx] = True

        ancestor = q_offset
        max_slots = int(parent_row.shape[0])
        for _ in range(max_slots):
            if ancestor < 0 or ancestor >= max_slots:
                break
            parent = int(parent_row[ancestor].item())
            parent = 0 if parent < 0 else parent
            parent_pos = prefix_len + parent
            if 0 <= parent_pos < seq_len:
                visible[q_offset, parent_pos] = True
            if parent <= 0:
                break
            ancestor = parent

    left, right = window_size
    if left >= 0 or right >= 0:
        q_pos = torch.arange(q_len, device=device) + prefix_len
        k_pos = torch.arange(seq_len, device=device)
        if left >= 0:
            visible &= k_pos.unsqueeze(0) >= q_pos.unsqueeze(1) - left
        if right >= 0:
            visible &= k_pos.unsqueeze(0) <= q_pos.unsqueeze(1) + right
    return visible


class FlashAttnV100MetadataBuilder(TritonAttentionMetadataBuilder):
    """Attach CPU metadata for the dense prefill path."""

    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        spec_config = getattr(self.vllm_config, "speculative_config", None)
        self._is_speculative_draft_model = (
            spec_config is not None
            and getattr(spec_config, "draft_model_config", None)
            is self.vllm_config.model_config
        )
        self._draft_block_table: torch.Tensor | None = None
        self._draft_seq_lens: torch.Tensor | None = None
        self._draft_query_start_loc: torch.Tensor | None = None
        self._flash_draft_buffer_shape: tuple[int, int] | None = None
        self._smallq_decode_block_table: torch.Tensor | None = None
        self._smallq_decode_seq_lens: torch.Tensor | None = None
        self._smallq_query_start_loc: torch.Tensor | None = None
        self._smallq_token_indices: torch.Tensor | None = None
        self._smallq_buffer_shape: tuple[int, int, int] | None = None
        self._decode_active_num_partitions: torch.Tensor | None = None

    def _attach_common_flash_metadata(
        self,
        attn_metadata: TritonAttentionMetadata,
        common_attn_metadata,
    ) -> None:
        flash_metadata = _as_flash_v100_metadata(attn_metadata)
        flash_metadata.query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        seq_lens_cpu = getattr(common_attn_metadata, "_seq_lens_cpu", None)
        if seq_lens_cpu is None:
            # Async speculative decode keeps device seq_lens authoritative and
            # may deliberately omit the exact CPU shadow. Flash-V100 only uses
            # this CPU view for route/partition hints; the kernel still consumes
            # attn_metadata.seq_lens, so an upper bound is preferable to the
            # deprecated lazy seq_lens.to("cpu") sync.
            seq_lens_cpu = getattr(
                common_attn_metadata,
                "seq_lens_cpu_upper_bound",
                None,
            )
        flash_metadata.seq_lens_cpu = (
            seq_lens_cpu
            if seq_lens_cpu is not None
            else common_attn_metadata.seq_lens_cpu
        )
        flash_metadata.causal = common_attn_metadata.causal
        flash_metadata.max_model_len = self.vllm_config.model_config.max_model_len
        flash_metadata.flash_v100_cudagraph_capture = False

    def _attach_ddtree_metadata(
        self,
        attn_metadata: TritonAttentionMetadata,
        *,
        ddtree_parent_ids: torch.Tensor | None,
        ddtree_num_tree_tokens_cpu: torch.Tensor | None,
    ) -> None:
        flash_metadata = _as_flash_v100_metadata(attn_metadata)
        flash_metadata.ddtree_parent_ids = None
        flash_metadata.ddtree_parent_ids_cpu = None
        flash_metadata.ddtree_num_tree_tokens_cpu = None
        flash_metadata.ddtree_seq_lens_restored_for_triton = False
        flash_metadata.ddtree_query_start_loc_restored_for_triton = False
        if ddtree_parent_ids is None:
            return
        if ddtree_num_tree_tokens_cpu is None:
            raise ValueError(
                "ddtree_num_tree_tokens_cpu is required with ddtree_parent_ids"
            )
        if ddtree_parent_ids.ndim != 2:
            raise ValueError("ddtree_parent_ids must have shape [batch, slots]")
        if ddtree_num_tree_tokens_cpu.ndim != 1:
            raise ValueError("ddtree_num_tree_tokens_cpu must be a 1D tensor")
        num_reqs = int(attn_metadata.query_start_loc.numel() - 1)
        if ddtree_parent_ids.shape[0] < num_reqs:
            raise ValueError(
                "ddtree_parent_ids must cover active requests: "
                f"{ddtree_parent_ids.shape[0]} < {num_reqs}"
            )
        if ddtree_num_tree_tokens_cpu.numel() < num_reqs:
            raise ValueError(
                "ddtree_num_tree_tokens_cpu must cover active requests: "
                f"{ddtree_num_tree_tokens_cpu.numel()} < {num_reqs}"
            )
        flash_metadata.ddtree_parent_ids = ddtree_parent_ids
        flash_metadata.ddtree_num_tree_tokens_cpu = ddtree_num_tree_tokens_cpu
        if bool(torch.any(ddtree_num_tree_tokens_cpu[:num_reqs] > 0).item()):
            seq_lens = getattr(attn_metadata, "seq_lens", None)
            seq_lens_cpu = getattr(attn_metadata, "seq_lens_cpu", None)
            if seq_lens is not None and seq_lens_cpu is not None:
                if _is_cuda_graph_capturing(seq_lens):
                    seq_lens[:num_reqs].copy_(
                        seq_lens_cpu[:num_reqs].to(
                            device=seq_lens.device,
                            dtype=seq_lens.dtype,
                        ),
                        non_blocking=True,
                    )
                flash_metadata.ddtree_seq_lens_restored_for_triton = True
            query_start_loc = getattr(attn_metadata, "query_start_loc", None)
            query_start_loc_cpu = getattr(attn_metadata, "query_start_loc_cpu", None)
            if query_start_loc is not None and query_start_loc_cpu is not None:
                num_boundaries = num_reqs + 1
                if _is_cuda_graph_capturing(query_start_loc):
                    query_start_loc[:num_boundaries].copy_(
                        query_start_loc_cpu[:num_boundaries].to(
                            device=query_start_loc.device,
                            dtype=query_start_loc.dtype,
                        ),
                        non_blocking=True,
                    )
                flash_metadata.ddtree_query_start_loc_restored_for_triton = True

    def _attach_decode_shape_hints(
        self,
        attn_metadata: TritonAttentionMetadata,
        common_attn_metadata,
        *,
        static_decode: bool = False,
    ) -> None:
        flash_metadata = _as_flash_v100_metadata(attn_metadata)
        flash_metadata.flash_v100_decode_max_seq_len_hint = None
        flash_metadata.flash_v100_decode_workspace_seq_capacity_hint = None
        flash_metadata.flash_v100_static_decode_seq_hint = None

        max_query_len = int(getattr(common_attn_metadata, "max_query_len", 0) or 0)
        if max_query_len != 1:
            return

        seq_lens_cpu = getattr(attn_metadata, "seq_lens_cpu", None)
        if seq_lens_cpu is not None and seq_lens_cpu.numel() > 0:
            max_seq_len_hint = int(seq_lens_cpu.max().item())
        else:
            max_seq_len_hint = int(getattr(common_attn_metadata, "max_seq_len", 0) or 0)
        if max_seq_len_hint <= 0:
            return

        flash_metadata.flash_v100_decode_max_seq_len_hint = max_seq_len_hint
        if not static_decode:
            return

        block_table = getattr(common_attn_metadata, "block_table_tensor", None)
        if block_table is None:
            block_table = getattr(attn_metadata, "block_table", None)
        raw_seq_capacity = (
            int(block_table.shape[1]) * int(self.block_size)
            if block_table is not None
            else max_seq_len_hint
        )
        static_seq_capacity = max(
            max_seq_len_hint,
            int(getattr(common_attn_metadata, "max_seq_len", 0) or 0),
        )
        workspace_seq_capacity = min(raw_seq_capacity, static_seq_capacity)
        if (
            raw_seq_capacity > max_seq_len_hint
            or workspace_seq_capacity > max_seq_len_hint
        ):
            flash_metadata.flash_v100_static_decode_seq_hint = workspace_seq_capacity
        flash_metadata.flash_v100_decode_workspace_seq_capacity_hint = (
            workspace_seq_capacity
        )

    def _ensure_decode_active_num_partitions(self) -> torch.Tensor:
        if self._decode_active_num_partitions is None:
            self._decode_active_num_partitions = torch.empty(
                (1,),
                dtype=torch.int32,
                device=self.device,
            )
        return self._decode_active_num_partitions

    def _update_decode_active_num_partitions(
        self,
        attn_metadata: TritonAttentionMetadata,
        *,
        stage: str,
    ) -> None:
        flash_metadata = _as_flash_v100_metadata(attn_metadata)
        flash_metadata.flash_v100_decode_active_num_partitions = None
        if not _decode_dynamic_partitions_enabled():
            return

        max_seq_len_hint = getattr(
            attn_metadata,
            "flash_v100_decode_max_seq_len_hint",
            None,
        )
        if max_seq_len_hint is None:
            return

        if (
            getattr(
                attn_metadata,
                "flash_v100_decode_workspace_seq_capacity_hint",
                None,
            )
            is None
            and self._decode_active_num_partitions is None
        ):
            return

        partition_size = _decode_partition_size_for_metadata(int(max_seq_len_hint))
        active = max(1, (int(max_seq_len_hint) + partition_size - 1) // partition_size)
        active_num_partitions = self._ensure_decode_active_num_partitions()
        active_num_partitions.fill_(active)
        flash_metadata.flash_v100_decode_active_num_partitions = active_num_partitions
        _trace_decode_active_metadata(
            stage=stage,
            max_seq_len_hint=int(max_seq_len_hint),
            workspace_seq_capacity_hint=getattr(
                attn_metadata,
                "flash_v100_decode_workspace_seq_capacity_hint",
                None,
            ),
            static_decode_seq_hint=getattr(
                attn_metadata,
                "flash_v100_static_decode_seq_hint",
                None,
            ),
            active=active,
            partition_size=partition_size,
        )

    def _debug_draft_metadata(
        self,
        stage: str,
        attn_metadata: TritonAttentionMetadata,
        common_attn_metadata,
    ) -> None:
        if not self._is_speculative_draft_model or not _draft_graph_debug_enabled():
            return
        _draft_graph_debug_log(
            f"builder:{stage}",
            "num_reqs=%s num_actual_tokens=%s max_query_len=%s max_seq_len=%s "
            "common_qsl_cpu=%s common_seq_cpu=%s %s %s %s %s %s %s %s",
            getattr(common_attn_metadata, "num_reqs", None),
            getattr(common_attn_metadata, "num_actual_tokens", None),
            getattr(common_attn_metadata, "max_query_len", None),
            getattr(common_attn_metadata, "max_seq_len", None),
            getattr(common_attn_metadata, "query_start_loc_cpu", None),
            getattr(common_attn_metadata, "seq_lens_cpu", None),
            _format_tensor_debug(
                getattr(common_attn_metadata, "query_start_loc", None),
                "common_qsl",
            ),
            _format_tensor_debug(
                getattr(common_attn_metadata, "seq_lens", None),
                "common_seq",
            ),
            _format_tensor_debug(
                getattr(common_attn_metadata, "block_table_tensor", None),
                "common_bt",
            ),
            _format_tensor_debug(
                getattr(attn_metadata, "query_start_loc", None),
                "attn_qsl",
            ),
            _format_tensor_debug(getattr(attn_metadata, "seq_lens", None), "attn_seq"),
            _format_tensor_debug(
                getattr(attn_metadata, "block_table", None),
                "attn_bt",
            ),
            _format_tensor_debug(
                getattr(attn_metadata, "smallq_decode_seq_lens", None),
                "smallq_seq",
            ),
        )

    def _ensure_flash_draft_graph_buffers(
        self,
        required_reqs: int,
        block_table: torch.Tensor,
    ) -> bool:
        req_capacity = max(
            int(self.vllm_config.scheduler_config.max_num_seqs),
            int(required_reqs),
            1,
        )
        block_cols = int(block_table.shape[1])
        shape = (req_capacity, block_cols)
        if self._flash_draft_buffer_shape == shape:
            return True

        if self._flash_draft_buffer_shape is not None:
            old_reqs, old_block_cols = self._flash_draft_buffer_shape
            return required_reqs <= old_reqs and block_cols == old_block_cols

        self._draft_block_table = torch.empty(
            (req_capacity, block_cols),
            dtype=torch.int32,
            device=self.device,
        )
        self._draft_seq_lens = torch.empty(
            (req_capacity,),
            dtype=torch.int32,
            device=self.device,
        )
        self._draft_query_start_loc = torch.empty(
            (req_capacity + 1,),
            dtype=torch.int32,
            device=self.device,
        )
        self._flash_draft_buffer_shape = shape
        return True

    def _stabilize_draft_graph_metadata(
        self,
        attn_metadata: TritonAttentionMetadata,
        common_attn_metadata,
    ) -> None:
        num_reqs = int(common_attn_metadata.num_reqs)
        if num_reqs <= 0:
            return

        block_table = attn_metadata.block_table[:num_reqs]
        if not self._ensure_flash_draft_graph_buffers(num_reqs, block_table):
            assert self._flash_draft_buffer_shape is not None
            req_capacity, block_cols = self._flash_draft_buffer_shape
            raise RuntimeError(
                "FLASH_ATTN_V100 draft CUDA graph metadata shape exceeds "
                "the captured persistent buffer capacity: "
                f"required_reqs={num_reqs}, "
                f"required_block_cols={int(block_table.shape[1])}, "
                f"capacity_reqs={req_capacity}, "
                f"capacity_block_cols={block_cols}. "
                "Replay would otherwise read stale draft metadata."
            )

        assert self._draft_block_table is not None
        assert self._draft_seq_lens is not None
        assert self._draft_query_start_loc is not None

        self._draft_block_table[:num_reqs].copy_(block_table, non_blocking=True)
        self._draft_seq_lens[:num_reqs].copy_(
            attn_metadata.seq_lens[:num_reqs],
            non_blocking=True,
        )
        self._draft_query_start_loc[: num_reqs + 1].copy_(
            attn_metadata.query_start_loc[: num_reqs + 1],
            non_blocking=True,
        )

        attn_metadata.block_table = self._draft_block_table[:num_reqs]
        attn_metadata.seq_lens = self._draft_seq_lens[:num_reqs]
        attn_metadata.query_start_loc = self._draft_query_start_loc[: num_reqs + 1]

    def _configured_smallq_max_query_len(self) -> int:
        return int(os.getenv("VLLM_FLASH_V100_SMALLQ_DECODE_MAX_Q", "16"))

    def _configured_smallq_max_model_len(self) -> int:
        return int(os.getenv("VLLM_FLASH_V100_SMALLQ_DECODE_MAX_MODEL_LEN", "0"))

    def _smallq_buffer_token_capacity(self, required_tokens: int) -> int:
        compilation_config = self.vllm_config.compilation_config
        graph_tokens = compilation_config.max_cudagraph_capture_size
        if graph_tokens is None and compilation_config.cudagraph_capture_sizes:
            graph_tokens = max(compilation_config.cudagraph_capture_sizes)
        if graph_tokens is None or graph_tokens <= 0:
            graph_tokens = required_tokens
        smallq_max_query_len = max(self._configured_smallq_max_query_len(), 0)
        max_num_seqs = max(int(self.vllm_config.scheduler_config.max_num_seqs), 1)
        # MTP verifier graph capture can bind a q=N branch before the runtime
        # request reaches the largest small-query shape. Keep the persistent
        # graph metadata buffers sized for the configured small-query envelope
        # instead of the first captured shape, otherwise replay would either
        # read stale metadata or trip the capacity guard at runtime.
        smallq_token_capacity = smallq_max_query_len * max_num_seqs
        return max(
            int(graph_tokens),
            int(required_tokens),
            int(smallq_token_capacity),
            1,
        )

    def _ensure_smallq_decode_buffers(
        self,
        required_tokens: int,
        required_reqs: int,
        block_table: torch.Tensor,
    ) -> bool:
        token_capacity = self._smallq_buffer_token_capacity(required_tokens)
        req_capacity = max(
            min(
                int(self.vllm_config.scheduler_config.max_num_seqs),
                token_capacity,
            ),
            int(required_reqs),
            1,
        )
        block_cols = int(block_table.shape[1])
        shape = (token_capacity, req_capacity, block_cols)
        if self._smallq_buffer_shape == shape:
            return True

        if self._smallq_buffer_shape is not None:
            old_tokens, old_reqs, old_block_cols = self._smallq_buffer_shape
            return (
                required_tokens <= old_tokens
                and required_reqs <= old_reqs
                and block_cols == old_block_cols
            )

        self._smallq_decode_block_table = torch.empty(
            (token_capacity, block_cols),
            dtype=torch.int32,
            device=self.device,
        )
        self._smallq_decode_seq_lens = torch.empty(
            (token_capacity,),
            dtype=torch.int32,
            device=self.device,
        )
        self._smallq_query_start_loc = torch.empty(
            (req_capacity + 1,),
            dtype=torch.int32,
            device=self.device,
        )
        self._smallq_token_indices = torch.arange(
            token_capacity,
            dtype=torch.int32,
            device=self.device,
        )
        self._smallq_buffer_shape = shape
        return True

    def _clear_smallq_decode_metadata(
        self,
        attn_metadata: TritonAttentionMetadata,
    ) -> None:
        flash_metadata = _as_flash_v100_metadata(attn_metadata)
        flash_metadata.smallq_decode_block_table = None
        flash_metadata.smallq_decode_seq_lens = None
        flash_metadata.smallq_query_start_loc = None
        flash_metadata.smallq_decode_max_seq_len_hint = None
        flash_metadata.smallq_decode_workspace_seq_capacity_hint = None
        flash_metadata.smallq_decode_partition_size_hint = None

    def _update_smallq_decode_metadata(
        self,
        attn_metadata: TritonAttentionMetadata,
        common_attn_metadata,
        *,
        force: bool = False,
        workspace_seq_capacity_cap: int | None = None,
        partition_size_hint: int | None = None,
    ) -> None:
        flash_metadata = _as_flash_v100_metadata(attn_metadata)
        profile_enabled = _dflash_ddtree_worker_profile_enabled()
        profile_t0 = time.perf_counter() if profile_enabled else 0.0
        profile_stage_t0 = profile_t0
        self._clear_smallq_decode_metadata(attn_metadata)
        clear_ms = (
            (time.perf_counter() - profile_stage_t0) * 1000.0
            if profile_enabled
            else 0.0
        )
        profile_stage_t0 = time.perf_counter() if profile_enabled else 0.0

        max_query_len = int(getattr(attn_metadata, "max_query_len", 1))
        smallq_max_query_len = self._configured_smallq_max_query_len()
        if (
            smallq_max_query_len <= 0
            or max_query_len <= 1
            or max_query_len > smallq_max_query_len
        ):
            return

        smallq_max_model_len = self._configured_smallq_max_model_len()
        max_model_len = int(self.vllm_config.model_config.max_model_len)
        if smallq_max_model_len > 0 and max_model_len > smallq_max_model_len:
            return

        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        seq_lens_cpu = getattr(common_attn_metadata, "_seq_lens_cpu", None)
        if seq_lens_cpu is None:
            # This metadata path is on the drafter hot loop. Async speculative
            # decode may omit the exact CPU shadow, but Flash-V100 only needs
            # a CPU value here for small-query route and workspace hints. Use
            # the scheduler-maintained upper bound to avoid an implicit
            # seq_lens.to("cpu") synchronization.
            seq_lens_cpu = getattr(
                common_attn_metadata,
                "seq_lens_cpu_upper_bound",
                None,
            )
        if seq_lens_cpu is None:
            seq_lens_cpu = common_attn_metadata.seq_lens_cpu
        query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        has_prefix_context = bool(torch.any(query_lens_cpu != seq_lens_cpu).item())
        if not force and not has_prefix_context and self._smallq_buffer_shape is None:
            return

        num_query_tokens = int(attn_metadata.num_actual_tokens)
        num_reqs = int(common_attn_metadata.num_reqs)
        if num_query_tokens <= 0 or num_reqs <= 0:
            return

        block_table = attn_metadata.block_table[:num_reqs]
        guard_ms = (
            (time.perf_counter() - profile_stage_t0) * 1000.0
            if profile_enabled
            else 0.0
        )
        profile_stage_t0 = time.perf_counter() if profile_enabled else 0.0
        if not self._ensure_smallq_decode_buffers(
            num_query_tokens,
            num_reqs,
            block_table,
        ):
            assert self._smallq_buffer_shape is not None
            token_capacity, req_capacity, block_cols = self._smallq_buffer_shape
            raise RuntimeError(
                "FLASH_ATTN_V100 small-query CUDA graph metadata shape exceeds "
                "the captured persistent buffer capacity: "
                f"required_tokens={num_query_tokens}, "
                f"required_reqs={num_reqs}, "
                f"required_block_cols={int(block_table.shape[1])}, "
                f"capacity_tokens={token_capacity}, "
                f"capacity_reqs={req_capacity}, "
                f"capacity_block_cols={block_cols}. "
                "Replay would otherwise use stale captured metadata."
            )
        ensure_ms = (
            (time.perf_counter() - profile_stage_t0) * 1000.0
            if profile_enabled
            else 0.0
        )
        assert self._smallq_decode_block_table is not None
        assert self._smallq_decode_seq_lens is not None
        assert self._smallq_query_start_loc is not None
        assert self._smallq_token_indices is not None

        profile_stage_t0 = time.perf_counter() if profile_enabled else 0.0
        query_start_loc = attn_metadata.query_start_loc[: num_reqs + 1]
        query_lens = query_start_loc[1:] - query_start_loc[:-1]
        real_query_lens = query_lens
        real_num_query_tokens = int(query_start_loc_cpu[-1].item())
        if real_num_query_tokens > num_query_tokens:
            return

        repeat_query_lens = query_lens
        padding_tokens = num_query_tokens - real_num_query_tokens
        if padding_tokens > 0:
            repeat_query_lens = query_lens.clone()
            repeat_query_lens[-1] += padding_tokens

        seq_lens = attn_metadata.seq_lens[:num_reqs]
        effective_seq_lens = torch.maximum(
            seq_lens,
            real_query_lens.to(dtype=seq_lens.dtype),
        )
        block_table = block_table.clamp_min(0)
        prep_ms = (
            (time.perf_counter() - profile_stage_t0) * 1000.0
            if profile_enabled
            else 0.0
        )
        profile_stage_t0 = time.perf_counter() if profile_enabled else 0.0
        decode_block_table = torch.repeat_interleave(
            block_table,
            repeat_query_lens,
            dim=0,
            output_size=num_query_tokens,
        ).contiguous()
        seq_lens_rep = torch.repeat_interleave(
            effective_seq_lens,
            repeat_query_lens,
            output_size=num_query_tokens,
        )
        query_lens_rep = torch.repeat_interleave(
            real_query_lens.to(dtype=seq_lens.dtype),
            repeat_query_lens,
            output_size=num_query_tokens,
        )
        start_locs_rep = torch.repeat_interleave(
            query_start_loc[:-1].to(dtype=seq_lens.dtype),
            repeat_query_lens,
            output_size=num_query_tokens,
        )
        token_indices = self._smallq_token_indices[:num_query_tokens].to(
            dtype=seq_lens.dtype
        )
        offsets = token_indices - start_locs_rep + 1
        decode_seq_lens = (seq_lens_rep - query_lens_rep + offsets).contiguous()
        if padding_tokens > 0:
            padding_mask = token_indices >= real_num_query_tokens
            decode_seq_lens = torch.where(
                padding_mask,
                torch.zeros_like(decode_seq_lens),
                decode_seq_lens,
            ).contiguous()
            decode_block_table = torch.where(
                padding_mask[:, None],
                torch.zeros_like(decode_block_table),
                decode_block_table,
            ).contiguous()
        expand_ms = (
            (time.perf_counter() - profile_stage_t0) * 1000.0
            if profile_enabled
            else 0.0
        )

        profile_stage_t0 = time.perf_counter() if profile_enabled else 0.0
        self._smallq_decode_block_table[:num_query_tokens].copy_(
            decode_block_table,
            non_blocking=True,
        )
        self._smallq_decode_seq_lens[:num_query_tokens].copy_(
            decode_seq_lens,
            non_blocking=True,
        )
        self._smallq_query_start_loc[: num_reqs + 1].copy_(
            query_start_loc,
            non_blocking=True,
        )
        copy_ms = (
            (time.perf_counter() - profile_stage_t0) * 1000.0
            if profile_enabled
            else 0.0
        )

        profile_stage_t0 = time.perf_counter() if profile_enabled else 0.0
        flash_metadata.smallq_decode_block_table = self._smallq_decode_block_table[
            :num_query_tokens
        ]
        flash_metadata.smallq_decode_seq_lens = self._smallq_decode_seq_lens[
            :num_query_tokens
        ]
        flash_metadata.smallq_query_start_loc = self._smallq_query_start_loc[
            : num_reqs + 1
        ]
        raw_seq_capacity = int(block_table.shape[1]) * int(self.block_size)
        max_seq_len_hint = int(seq_lens_cpu.max().item())
        if max_seq_len_hint > 0 and raw_seq_capacity > 0:
            # MTP verification reaches this backend as q>1 prefix prefill, but
            # the Flash-V100 long-context optimization still applies because
            # the actual compute is paged decode over each tiny query row.
            # Keep graph replay capacity fixed while letting kernels skip
            # inactive partitions for the current runtime sequence length.
            flash_metadata.smallq_decode_max_seq_len_hint = max_seq_len_hint
            if workspace_seq_capacity_cap is not None:
                # A distinct CUDA graph key guarantees replay only below this
                # bound. The block table remains full-width so runtime KV
                # addresses stay stable, while the captured workspace/grid is
                # reduced to the bounded context envelope.
                raw_seq_capacity = min(
                    raw_seq_capacity,
                    max(max_seq_len_hint, int(workspace_seq_capacity_cap)),
                )
            flash_metadata.smallq_decode_workspace_seq_capacity_hint = raw_seq_capacity
            flash_metadata.smallq_decode_partition_size_hint = partition_size_hint
        hint_ms = (
            (time.perf_counter() - profile_stage_t0) * 1000.0
            if profile_enabled
            else 0.0
        )
        if profile_enabled:
            logger.info(
                "FLASH_ATTN_V100 DDTREE_WORKER_PROFILE smallq_metadata "
                "total_ms=%.3f clear_ms=%.3f guard_ms=%.3f ensure_ms=%.3f "
                "prep_ms=%.3f expand_ms=%.3f copy_ms=%.3f hint_ms=%.3f "
                "num_reqs=%d num_query_tokens=%d real_query_tokens=%d "
                "padding_tokens=%d block_cols=%d",
                (time.perf_counter() - profile_t0) * 1000.0,
                clear_ms,
                guard_ms,
                ensure_ms,
                prep_ms,
                expand_ms,
                copy_ms,
                hint_ms,
                num_reqs,
                num_query_tokens,
                real_num_query_tokens,
                padding_tokens,
                int(block_table.shape[1]),
            )
        if _draft_graph_debug_enabled():
            _graph_metadata_debug_log(
                "smallq_update",
                "draft=%s force=%s num_reqs=%s num_query_tokens=%s "
                "real_num_query_tokens=%s padding_tokens=%s max_query_len=%s "
                "common_qsl_cpu=%s common_seq_cpu=%s %s %s %s %s %s %s",
                self._is_speculative_draft_model,
                force,
                num_reqs,
                num_query_tokens,
                real_num_query_tokens,
                padding_tokens,
                max_query_len,
                query_start_loc_cpu,
                seq_lens_cpu,
                _format_tensor_debug(attn_metadata.query_start_loc, "attn_qsl"),
                _format_tensor_debug(attn_metadata.seq_lens, "attn_seq"),
                _format_tensor_debug(attn_metadata.block_table, "attn_bt"),
                _format_tensor_debug(
                    flash_metadata.smallq_decode_block_table,
                    "smallq_bt",
                ),
                _format_tensor_debug(
                    flash_metadata.smallq_decode_seq_lens,
                    "smallq_seq",
                ),
                _format_tensor_debug(
                    flash_metadata.smallq_query_start_loc,
                    "smallq_qsl",
                ),
            )

    def build_for_cudagraph_capture(self, common_attn_metadata):
        capture_seq_lens_cpu = getattr(common_attn_metadata, "_seq_lens_cpu", None)
        capture_seq_lens_cpu = (
            capture_seq_lens_cpu.clone()
            if capture_seq_lens_cpu is not None
            else common_attn_metadata.seq_lens.detach().cpu().clone()
        )
        attn_metadata = super().build_for_cudagraph_capture(common_attn_metadata)
        self._attach_common_flash_metadata(attn_metadata, common_attn_metadata)
        flash_metadata = _as_flash_v100_metadata(attn_metadata)
        flash_metadata.seq_lens_cpu = capture_seq_lens_cpu

        # The Triton builder shortens capture seq_lens to 1 so full graph
        # capture stays cheap. That is valid for single-token decode, but the
        # FA2 small-query MTP verifier replays a tiny causal prefill as paged
        # decode. Capturing that branch with seq_len < query_len creates
        # negative per-token decode lengths and can poison long-context graph
        # replay. Keep capture cheap while preserving a valid verifier shape.
        max_query_len = getattr(attn_metadata, "max_query_len", 1)
        if max_query_len > 1:
            attn_metadata.seq_lens.fill_(max_query_len)
            workspace_seq_capacity_cap = (
                int(getattr(common_attn_metadata, "max_seq_len", 0) or 0) or None
            )
            partition_size_hint = None
            if (
                workspace_seq_capacity_cap is not None
                and workspace_seq_capacity_cap
                < int(self.vllm_config.model_config.max_model_len)
            ):
                partition_size_hint = _mtp_context_bucket_partition_size_hint()
            self._update_smallq_decode_metadata(
                attn_metadata,
                common_attn_metadata,
                force=True,
                workspace_seq_capacity_cap=workspace_seq_capacity_cap,
                partition_size_hint=partition_size_hint,
            )
        else:
            # PIECEWISE graph replay captures the q=1 decode kernel arguments
            # during metadata warmup. Runtime drafting updates the persistent
            # draft metadata buffers in build_for_drafting(), so capture must
            # bind the graph to the same buffers instead of transient dummy
            # capture tensors.
            self._stabilize_draft_graph_metadata(
                attn_metadata,
                common_attn_metadata,
            )
        self._debug_draft_metadata(
            "capture",
            attn_metadata,
            common_attn_metadata,
        )
        self._attach_decode_shape_hints(
            attn_metadata,
            common_attn_metadata,
            static_decode=True,
        )
        flash_metadata.flash_v100_cudagraph_capture = True
        self._update_decode_active_num_partitions(attn_metadata, stage="capture")

        return attn_metadata

    def build(
        self,
        common_prefix_len,
        common_attn_metadata,
        fast_build: bool = False,
        ddtree_parent_ids: torch.Tensor | None = None,
        ddtree_num_tree_tokens_cpu: torch.Tensor | None = None,
    ):
        attn_metadata = super().build(
            common_prefix_len, common_attn_metadata, fast_build
        )
        self._attach_common_flash_metadata(attn_metadata, common_attn_metadata)
        self._attach_ddtree_metadata(
            attn_metadata,
            ddtree_parent_ids=ddtree_parent_ids,
            ddtree_num_tree_tokens_cpu=ddtree_num_tree_tokens_cpu,
        )
        num_reqs = max(0, int(attn_metadata.query_start_loc.numel()) - 1)
        ddtree_tree_verify = (
            ddtree_parent_ids is not None
            and ddtree_num_tree_tokens_cpu is not None
            and getattr(attn_metadata, "max_query_len", 1) > 1
            and bool(torch.any(ddtree_num_tree_tokens_cpu[:num_reqs] > 0).item())
        )
        if (
            getattr(attn_metadata, "max_query_len", 1) == 1
            and self._flash_draft_buffer_shape is not None
        ):
            # FULL graph capture binds q=1 decode to these persistent buffers.
            # Refresh them on every runtime decode step so replay sees the
            # current request's block table and sequence metadata.
            self._stabilize_draft_graph_metadata(
                attn_metadata,
                common_attn_metadata,
            )
        if not ddtree_tree_verify:
            self._update_smallq_decode_metadata(
                attn_metadata,
                common_attn_metadata,
            )
        self._attach_decode_shape_hints(attn_metadata, common_attn_metadata)
        self._update_decode_active_num_partitions(attn_metadata, stage="build")
        self._debug_draft_metadata("build", attn_metadata, common_attn_metadata)
        return attn_metadata

    def build_for_drafting(self, common_attn_metadata, draft_index: int):
        profile_enabled = _dflash_ddtree_worker_profile_enabled()
        profile_t0 = time.perf_counter() if profile_enabled else 0.0
        profile_stage_t0 = profile_t0
        attn_metadata = super().build(
            common_prefix_len=0,
            common_attn_metadata=common_attn_metadata,
            fast_build=True,
        )
        super_build_ms = (
            (time.perf_counter() - profile_stage_t0) * 1000.0
            if profile_enabled
            else 0.0
        )
        profile_stage_t0 = time.perf_counter() if profile_enabled else 0.0
        self._attach_common_flash_metadata(attn_metadata, common_attn_metadata)
        attach_common_ms = (
            (time.perf_counter() - profile_stage_t0) * 1000.0
            if profile_enabled
            else 0.0
        )
        profile_stage_t0 = time.perf_counter() if profile_enabled else 0.0
        self._stabilize_draft_graph_metadata(attn_metadata, common_attn_metadata)
        stabilize_ms = (
            (time.perf_counter() - profile_stage_t0) * 1000.0
            if profile_enabled
            else 0.0
        )
        profile_stage_t0 = time.perf_counter() if profile_enabled else 0.0
        self._update_smallq_decode_metadata(attn_metadata, common_attn_metadata)
        smallq_ms = (
            (time.perf_counter() - profile_stage_t0) * 1000.0
            if profile_enabled
            else 0.0
        )
        profile_stage_t0 = time.perf_counter() if profile_enabled else 0.0
        self._attach_decode_shape_hints(attn_metadata, common_attn_metadata)
        shape_hints_ms = (
            (time.perf_counter() - profile_stage_t0) * 1000.0
            if profile_enabled
            else 0.0
        )
        profile_stage_t0 = time.perf_counter() if profile_enabled else 0.0
        self._update_decode_active_num_partitions(
            attn_metadata,
            stage=f"draft{draft_index}",
        )
        active_partitions_ms = (
            (time.perf_counter() - profile_stage_t0) * 1000.0
            if profile_enabled
            else 0.0
        )
        profile_stage_t0 = time.perf_counter() if profile_enabled else 0.0
        self._debug_draft_metadata(
            f"draft{draft_index}",
            attn_metadata,
            common_attn_metadata,
        )
        debug_ms = (
            (time.perf_counter() - profile_stage_t0) * 1000.0
            if profile_enabled
            else 0.0
        )
        if profile_enabled:
            logger.info(
                "FLASH_ATTN_V100 DDTREE_WORKER_PROFILE build_for_drafting "
                "draft_index=%d total_ms=%.3f super_build_ms=%.3f "
                "attach_common_ms=%.3f stabilize_ms=%.3f smallq_ms=%.3f "
                "shape_hints_ms=%.3f active_partitions_ms=%.3f debug_ms=%.3f "
                "max_query_len=%s num_actual_tokens=%s num_reqs=%s",
                draft_index,
                (time.perf_counter() - profile_t0) * 1000.0,
                super_build_ms,
                attach_common_ms,
                stabilize_ms,
                smallq_ms,
                shape_hints_ms,
                active_partitions_ms,
                debug_ms,
                getattr(common_attn_metadata, "max_query_len", None),
                getattr(common_attn_metadata, "num_actual_tokens", None),
                getattr(common_attn_metadata, "num_reqs", None),
            )
        return attn_metadata


class FlashAttnV100Impl(TritonAttentionImpl):
    """Flash Attention V100 implementation with explicit fallback policy."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        (
            self.flash_attn_func,
            self.flash_attn_bhmd_func,
            self.flash_attn_decode_paged,
            self.flash_attn_decode_paged_xqa,
            self.flash_attn_decode_paged_wmma,
            self.flash_attn_prefill_paged,
            self.flash_attn_prefill_paged_bhmd,
            self.flash_attn_prefill_paged_bfla,
            self.flash_attn_prefill_paged_splitkv,
        ) = _get_flash_ops()
        self.fp8_e5m2_paged_kv_to_fp16 = _get_fp8_e5m2_paged_kv_bridge_op()
        # V100 FA2 kernels consume fp16 Q. FP8 KV cache support is implemented
        # as storage compression only, with K/V dequantized inside FA2 kernels.
        self.supports_quant_query_input = False
        self.use_flash_v100 = self.flash_attn_func is not None
        self.use_flash_v100_decode = self.flash_attn_decode_paged is not None
        self._flash_decode_paged_kwargs = {
            name
            for name in (
                "window_size",
                "max_seq_len_hint",
                "workspace_seq_capacity_hint",
                "active_num_partitions",
                "partition_size_hint",
            )
            if self.flash_attn_decode_paged is not None
            and _callable_accepts_keyword(self.flash_attn_decode_paged, name)
        }
        paged_prefill_enable = os.getenv("VLLM_FLASH_V100_ENABLE_PAGED_PREFILL")
        paged_prefill_disable = (
            os.getenv("VLLM_FLASH_V100_DISABLE_PAGED_PREFILL", "0") == "1"
        )
        self.use_flash_v100_prefill_paged = (
            self.flash_attn_prefill_paged is not None
            and paged_prefill_enable != "0"
            and not paged_prefill_disable
        )
        self.use_fp8_prefill_bridge = (
            self.fp8_e5m2_paged_kv_to_fp16 is not None
            and os.getenv("VLLM_FLASH_V100_FP8_PREFILL_BRIDGE", "1") != "0"
        )
        self.use_flash_v100_prefill_splitkv = (
            self.flash_attn_prefill_paged_splitkv is not None
            and envs.VLLM_FLASH_V100_PREFILL_SPLIT_KV
            and self.use_flash_v100_prefill_paged
        )
        self.use_flash_v100_prefill_bfla = (
            self.flash_attn_prefill_paged_bfla is not None
            and envs.VLLM_FLASH_V100_BFLA_PREFILL
            and self.use_flash_v100_prefill_paged
        )
        self.use_flash_v100_prefill_contig_dense = (
            self.flash_attn_func is not None
            and self.use_flash_v100_prefill_paged
            and envs.VLLM_FLASH_V100_PREFILL_CONTIG_DENSE
        )
        self.prefill_contig_dense_min_q = (
            envs.VLLM_FLASH_V100_PREFILL_CONTIG_DENSE_MIN_Q
        )
        self.prefill_contig_dense_min_kv = (
            envs.VLLM_FLASH_V100_PREFILL_CONTIG_DENSE_MIN_KV
        )
        self.prefill_contig_dense_allow_copy = (
            envs.VLLM_FLASH_V100_PREFILL_CONTIG_DENSE_ALLOW_COPY
        )
        self.prefill_split_kv_tokens = envs.VLLM_FLASH_V100_PREFILL_SPLIT_KV_TOKENS
        self.prefill_split_kv_min_q = envs.VLLM_FLASH_V100_PREFILL_SPLIT_KV_MIN_Q
        self.prefill_split_kv_max_q = envs.VLLM_FLASH_V100_PREFILL_SPLIT_KV_MAX_Q
        self.prefill_split_kv_min_kv = envs.VLLM_FLASH_V100_PREFILL_SPLIT_KV_MIN_KV
        self.prefill_bfla_min_q = envs.VLLM_FLASH_V100_BFLA_MIN_Q
        self.prefill_bfla_min_kv = envs.VLLM_FLASH_V100_BFLA_MIN_KV
        self.prefill_bfla_mask_block_n = envs.VLLM_FLASH_V100_BFLA_MASK_BLOCK_N
        self.use_prefill_paged_cache = (
            os.getenv("VLLM_FLASH_V100_PREFILL_USE_PAGED_CACHE", "0") == "1"
        )
        # Explicit diagnostic fallback only. The production migration target is
        # a complete Flash-V100 backend, so selected Flash routes should not
        # hide Flash prefill issues behind Triton by default.
        self.use_triton_prefill = (
            os.getenv("VLLM_FLASH_V100_PREFILL_USE_TRITON", "0") != "0"
        )
        self.allow_triton_fallback = (
            os.getenv("VLLM_FLASH_V100_ALLOW_TRITON_FALLBACK", "0") == "1"
        )
        self.smallq_decode_max_query_len = int(
            os.getenv("VLLM_FLASH_V100_SMALLQ_DECODE_MAX_Q", "16")
        )
        self.smallq_decode_max_model_len = int(
            os.getenv("VLLM_FLASH_V100_SMALLQ_DECODE_MAX_MODEL_LEN", "0")
        )
        self.use_decode_dense_reference = (
            os.getenv("VLLM_FLASH_V100_DECODE_DENSE_REFERENCE", "0") == "1"
        )
        self.use_decode_dense_cache = (
            os.getenv("VLLM_FLASH_V100_DECODE_DENSE_CACHE", "0") == "1"
        )
        # Classified quality rule: long q=1 scalar paged decode is a Type-B
        # reduction-order path, not a Type-A layout bug. Keep it as the
        # production Flash decode default so an explicit FLASH_ATTN_V100
        # selection does not silently become Triton during CUDA graph capture.
        decode_paged_prefill_env = os.getenv("VLLM_FLASH_V100_DECODE_USE_PAGED_PREFILL")
        self.use_decode_paged_prefill = decode_paged_prefill_env == "1"
        decode_bhmd_out_env = os.getenv("VLLM_FLASH_V100_DECODE_USE_BHMD_OUT")
        self.use_decode_paged_prefill_bhmd_out = decode_bhmd_out_env != "0"
        self.use_decode_wmma_wrapper = (
            os.getenv("VLLM_FLASH_V100_DECODE_USE_WMMA_WRAPPER", "0") == "1"
        )
        self.use_decode_xqa = os.getenv("VLLM_FLASH_V100_DECODE_USE_XQA", "1") == "1"
        self.use_smallq_decode_xqa = (
            self.use_decode_xqa
            and os.getenv("VLLM_FLASH_V100_SMALLQ_DECODE_USE_XQA", "1") == "1"
        )
        decode_scalar_paged_env = os.getenv("VLLM_FLASH_V100_DECODE_USE_SCALAR_PAGED")
        self.use_decode_scalar_paged = decode_scalar_paged_env != "0"
        self.compare_bhmd_out_dir = os.getenv("VLLM_FLASH_V100_COMPARE_BHMD_OUT_DIR")
        self.compare_bhmd_out_max_calls = int(
            os.getenv("VLLM_FLASH_V100_COMPARE_BHMD_OUT_MAX_CALLS", "0")
        )
        self._compare_bhmd_out_calls = 0
        self.compare_triton_out_dir = os.getenv(
            "VLLM_FLASH_V100_COMPARE_TRITON_OUT_DIR"
        )
        self.compare_triton_out_max_calls = int(
            os.getenv("VLLM_FLASH_V100_COMPARE_TRITON_OUT_MAX_CALLS", "0")
        )
        self.compare_triton_tensor_dump_dir = os.getenv(
            "VLLM_FLASH_V100_COMPARE_TRITON_TENSOR_DUMP_DIR"
        )
        self.compare_triton_tensor_dump_max_tokens = int(
            os.getenv("VLLM_FLASH_V100_COMPARE_TRITON_TENSOR_DUMP_MAX_TOKENS", "64")
        )
        self._compare_triton_out_calls = 0
        self._decode_cache_k: torch.Tensor | None = None
        self._decode_cache_v: torch.Tensor | None = None
        self._decode_cache_len = 0
        self._decode_cache_capacity = 0

    def _reset_decode_cache(self) -> None:
        self._decode_cache_k = None
        self._decode_cache_v = None
        self._decode_cache_len = 0
        self._decode_cache_capacity = 0

    def _ensure_decode_cache_capacity(
        self,
        required_len: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        if (
            self._decode_cache_k is not None
            and self._decode_cache_v is not None
            and self._decode_cache_capacity >= required_len
            and self._decode_cache_k.shape[1] == num_kv_heads
            and self._decode_cache_k.shape[2] == head_dim
            and self._decode_cache_k.dtype == dtype
            and self._decode_cache_k.device == device
        ):
            return

        new_capacity = max(required_len, max(16, self._decode_cache_capacity * 2))
        new_k = torch.empty(
            (new_capacity, num_kv_heads, head_dim),
            dtype=dtype,
            device=device,
        )
        new_v = torch.empty(
            (new_capacity, num_kv_heads, head_dim),
            dtype=dtype,
            device=device,
        )

        if (
            self._decode_cache_k is not None
            and self._decode_cache_v is not None
            and self._decode_cache_len > 0
        ):
            new_k[: self._decode_cache_len].copy_(
                self._decode_cache_k[: self._decode_cache_len]
            )
            new_v[: self._decode_cache_len].copy_(
                self._decode_cache_v[: self._decode_cache_len]
            )

        self._decode_cache_k = new_k
        self._decode_cache_v = new_v
        self._decode_cache_capacity = new_capacity

    def _get_decode_kv_single_seq(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        seq_lens_cpu: torch.Tensor,
        block_size: int,
        head_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = int(seq_lens_cpu[0])
        q_len = int(attn_metadata.num_actual_tokens)
        num_kv_heads = key.shape[1]

        cache_hit = (
            self._decode_cache_k is not None
            and self._decode_cache_v is not None
            and seq_len > self._decode_cache_len
            and seq_len - q_len == self._decode_cache_len
        )

        if not cache_hit:
            k_cont, v_cont = _extract_contiguous_kv_from_paged_cache(
                kv_cache=kv_cache,
                block_table=attn_metadata.block_table,
                seq_lens=attn_metadata.seq_lens,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                block_size=block_size,
                total_tokens=seq_len,
            )
            self._ensure_decode_cache_capacity(
                seq_len,
                num_kv_heads,
                head_dim,
                k_cont.dtype,
                k_cont.device,
            )
            assert self._decode_cache_k is not None
            assert self._decode_cache_v is not None
            self._decode_cache_k[:seq_len].copy_(k_cont)
            self._decode_cache_v[:seq_len].copy_(v_cont)
            self._decode_cache_len = seq_len
            return (
                self._decode_cache_k[:seq_len],
                self._decode_cache_v[:seq_len],
            )

        self._ensure_decode_cache_capacity(
            seq_len,
            num_kv_heads,
            head_dim,
            key.dtype,
            key.device,
        )
        assert self._decode_cache_k is not None
        assert self._decode_cache_v is not None
        self._decode_cache_k[self._decode_cache_len : seq_len].copy_(key[:q_len])
        self._decode_cache_v[self._decode_cache_len : seq_len].copy_(value[:q_len])
        self._decode_cache_len = seq_len
        return (
            self._decode_cache_k[:seq_len],
            self._decode_cache_v[:seq_len],
        )

    def _maybe_compare_bhmd_out(
        self,
        layer: torch.nn.Module,
        q_bhmd: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        safe_bmhd: torch.Tensor,
    ) -> None:
        call_idx = self._reserve_bhmd_compare_call()
        if call_idx is None or self.flash_attn_prefill_paged_bhmd is None:
            return

        raw_bmhd = torch.empty_like(safe_bmhd)
        raw_bhmd = raw_bmhd.permute(0, 2, 1, 3)
        self.flash_attn_prefill_paged_bhmd(
            q_bhmd,
            key_cache,
            value_cache,
            block_table,
            seq_lens,
            softmax_scale=self.scale,
            out=raw_bhmd,
            kv_cache_dtype=self.kv_cache_dtype,
            k_scale=float(layer._k_scale_float),
            v_scale=float(layer._v_scale_float),
            causal=True,
        )
        self._write_bhmd_compare_report(
            raw_bmhd,
            safe_bmhd,
            call_idx,
            "scratch_raw_vs_safe",
            {
                "q_bhmd_stride": list(q_bhmd.stride()),
                "raw_bhmd_stride": list(raw_bhmd.stride()),
                "raw_bhmd_contiguous": raw_bhmd.is_contiguous(),
            },
        )

    def _reserve_bhmd_compare_call(self) -> int | None:
        if (
            not self.compare_bhmd_out_dir
            or self.compare_bhmd_out_max_calls <= 0
            or self._compare_bhmd_out_calls >= self.compare_bhmd_out_max_calls
        ):
            return None

        call_idx = self._compare_bhmd_out_calls
        self._compare_bhmd_out_calls += 1
        return call_idx

    def _reserve_triton_compare_call(self) -> int | None:
        if (
            not self.compare_triton_out_dir
            or self.compare_triton_out_max_calls <= 0
            or self._compare_triton_out_calls >= self.compare_triton_out_max_calls
        ):
            return None

        call_idx = self._compare_triton_out_calls
        self._compare_triton_out_calls += 1
        return call_idx

    def _write_bhmd_compare_report(
        self,
        candidate_bmhd: torch.Tensor,
        reference_bmhd: torch.Tensor,
        call_idx: int,
        mode: str,
        extra: dict[str, object],
    ) -> None:
        assert self.compare_bhmd_out_dir is not None
        diff = candidate_bmhd - reference_bmhd
        report = {
            "call_idx": call_idx,
            "mode": mode,
            "equal": bool(torch.equal(candidate_bmhd, reference_bmhd)),
            "max_diff": float(diff.abs().max().item()),
            "mean_diff": float(diff.abs().float().mean().item()),
            "num_different": int((candidate_bmhd != reference_bmhd).sum().item()),
            "shape_bmhd": list(reference_bmhd.shape),
            "pid": os.getpid(),
        }
        report.update(extra)

        os.makedirs(self.compare_bhmd_out_dir, exist_ok=True)
        file_name = (
            f"bhmd_compare_pid{os.getpid()}_call{call_idx}_{time.time_ns()}.json"
        )
        path = os.path.join(self.compare_bhmd_out_dir, file_name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True)
            f.write("\n")

    def _write_triton_compare_report(
        self,
        candidate: torch.Tensor,
        reference: torch.Tensor,
        call_idx: int,
        stage: str,
        extra: dict[str, object],
    ) -> None:
        assert self.compare_triton_out_dir is not None
        diff = candidate.float() - reference.float()
        abs_diff = diff.abs()
        report = {
            "call_idx": call_idx,
            "stage": stage,
            "equal": bool(torch.equal(candidate, reference)),
            "max_diff": float(abs_diff.max().item()) if abs_diff.numel() else 0.0,
            "mean_diff": float(abs_diff.mean().item()) if abs_diff.numel() else 0.0,
            "num_different": int((candidate != reference).sum().item()),
            "shape": list(candidate.shape),
            "dtype": str(candidate.dtype),
            "candidate_nan_count": int(torch.isnan(candidate).sum().item()),
            "reference_nan_count": int(torch.isnan(reference).sum().item()),
            "pid": os.getpid(),
        }
        report.update(extra)

        os.makedirs(self.compare_triton_out_dir, exist_ok=True)
        file_name = (
            f"triton_out_compare_pid{os.getpid()}_call{call_idx}_"
            f"{stage}_{time.time_ns()}.json"
        )
        path = os.path.join(self.compare_triton_out_dir, file_name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True)
            f.write("\n")

    def _maybe_write_triton_tensor_dump(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        candidate: torch.Tensor,
        reference: torch.Tensor,
        call_idx: int,
        stage: str,
        num_actual_tokens: int,
    ) -> dict[str, object]:
        if not self.compare_triton_tensor_dump_dir:
            return {}
        if num_actual_tokens > self.compare_triton_tensor_dump_max_tokens:
            return {
                "tensor_dump_skipped": "num_actual_tokens_exceeds_limit",
                "tensor_dump_max_tokens": self.compare_triton_tensor_dump_max_tokens,
            }

        payload: dict[str, object] = {
            "call_idx": call_idx,
            "stage": stage,
            "num_actual_tokens": num_actual_tokens,
            "scale": self.scale,
            "kv_cache_dtype": self.kv_cache_dtype,
            "layer": self._layer_debug_info(layer),
            "query": query[:num_actual_tokens].detach().cpu(),
            "raw_key": key[:num_actual_tokens].detach().cpu(),
            "raw_value": value[:num_actual_tokens].detach().cpu(),
            "candidate_output": candidate[:num_actual_tokens].detach().cpu(),
            "triton_reference_output": reference[:num_actual_tokens].detach().cpu(),
            "query_start_loc": attn_metadata.query_start_loc.detach().cpu(),
            "seq_lens": attn_metadata.seq_lens.detach().cpu(),
            "block_table": attn_metadata.block_table.detach().cpu(),
        }

        if stage in ("prefill_no_prefix", "prefill_no_prefix_paged_cache"):
            key_cache, _ = _split_paged_kv_cache(kv_cache)
            block_size = key_cache.shape[1]
            num_kv_heads = key_cache.shape[2]
            head_dim = key_cache.shape[3]
            query_start_loc_cpu = getattr(attn_metadata, "query_start_loc_cpu", None)
            query_start_loc = (
                query_start_loc_cpu
                if query_start_loc_cpu is not None
                else attn_metadata.query_start_loc
            )
            num_seqs = len(query_start_loc) - 1
            k_cont, v_cont = _extract_contiguous_kv_from_paged_cache(
                kv_cache=kv_cache,
                block_table=attn_metadata.block_table[:num_seqs],
                seq_lens=attn_metadata.seq_lens[:num_seqs],
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                block_size=block_size,
                total_tokens=num_actual_tokens,
            )
            k_cont, v_cont = _dequantize_fp8_contiguous_kv(
                k_cont,
                v_cont,
                self.kv_cache_dtype,
                float(layer._k_scale_float),
                float(layer._v_scale_float),
            )
            payload["cache_key"] = k_cont.detach().cpu()
            payload["cache_value"] = v_cont.detach().cpu()

        os.makedirs(self.compare_triton_tensor_dump_dir, exist_ok=True)
        file_name = (
            f"triton_tensor_dump_pid{os.getpid()}_call{call_idx}_"
            f"{stage}_{time.time_ns()}.pt"
        )
        path = os.path.join(self.compare_triton_tensor_dump_dir, file_name)
        torch.save(payload, path)
        return {"tensor_dump_path": path}

    @staticmethod
    def _small_tensor_list(
        tensor: torch.Tensor | None,
        limit: int = 32,
    ) -> list[int] | None:
        if tensor is None:
            return None
        flat = tensor.detach().cpu().reshape(-1)
        return [int(x) for x in flat[:limit].tolist()]

    @staticmethod
    def _layer_debug_info(layer: torch.nn.Module) -> dict[str, object]:
        return {
            "layer_name": getattr(layer, "layer_name", None),
            "is_dflash_draft_attn": getattr(layer, "is_dflash_draft_attn", False),
            "kv_sharing_target_layer_name": getattr(
                layer, "kv_sharing_target_layer_name", None
            ),
            "impl_kv_sharing_target_layer_name": getattr(
                getattr(layer, "impl", None), "kv_sharing_target_layer_name", None
            ),
        }

    @staticmethod
    def _tensor_compare_stats(
        candidate: torch.Tensor,
        reference: torch.Tensor,
    ) -> dict[str, object]:
        if candidate.shape != reference.shape:
            return {
                "shape_mismatch": True,
                "candidate_shape": list(candidate.shape),
                "reference_shape": list(reference.shape),
            }

        diff = candidate.float() - reference.float()
        abs_diff = diff.abs()
        return {
            "shape_mismatch": False,
            "equal": bool(torch.equal(candidate, reference)),
            "max_diff": float(abs_diff.max().item()) if abs_diff.numel() else 0.0,
            "mean_diff": float(abs_diff.mean().item()) if abs_diff.numel() else 0.0,
            "num_different": int((candidate != reference).sum().item()),
            "candidate_dtype": str(candidate.dtype),
            "reference_dtype": str(reference.dtype),
            "candidate_abs_max": float(candidate.float().abs().max().item())
            if candidate.numel()
            else 0.0,
            "reference_abs_max": float(reference.float().abs().max().item())
            if reference.numel()
            else 0.0,
            "candidate_mean": float(candidate.float().mean().item())
            if candidate.numel()
            else 0.0,
            "reference_mean": float(reference.float().mean().item())
            if reference.numel()
            else 0.0,
            "shape": list(candidate.shape),
        }

    def _prefill_raw_kv_cache_compare_stats(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        num_actual_tokens: int,
    ) -> dict[str, object]:
        key_cache, _ = _split_paged_kv_cache(kv_cache)

        block_size = key_cache.shape[1]
        num_kv_heads = key_cache.shape[2]
        head_dim = key_cache.shape[3]
        query_start_loc_cpu = getattr(attn_metadata, "query_start_loc_cpu", None)
        query_start_loc = (
            query_start_loc_cpu
            if query_start_loc_cpu is not None
            else attn_metadata.query_start_loc
        )
        num_seqs = len(query_start_loc) - 1
        k_cont, v_cont = _extract_contiguous_kv_from_paged_cache(
            kv_cache=kv_cache,
            block_table=attn_metadata.block_table[:num_seqs],
            seq_lens=attn_metadata.seq_lens[:num_seqs],
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            block_size=block_size,
            total_tokens=num_actual_tokens,
        )
        k_cont, v_cont = _dequantize_fp8_contiguous_kv(
            k_cont,
            v_cont,
            self.kv_cache_dtype,
            float(layer._k_scale_float),
            float(layer._v_scale_float),
        )
        query_start_loc_cpu = getattr(attn_metadata, "query_start_loc_cpu", None)
        seq_lens_cpu = getattr(attn_metadata, "seq_lens_cpu", None)
        return {
            "raw_key_vs_cache": self._tensor_compare_stats(
                key[:num_actual_tokens], k_cont
            ),
            "raw_value_vs_cache": self._tensor_compare_stats(
                value[:num_actual_tokens], v_cont
            ),
            "kv_cache_dtype": str(kv_cache.dtype),
            "kv_cache_shape": list(kv_cache.shape),
            "query_start_loc": self._small_tensor_list(attn_metadata.query_start_loc),
            "query_start_loc_cpu": self._small_tensor_list(query_start_loc_cpu),
            "seq_lens": self._small_tensor_list(attn_metadata.seq_lens),
            "seq_lens_cpu": self._small_tensor_list(seq_lens_cpu),
            "block_table_shape": list(attn_metadata.block_table.shape),
            "block_table_first_row": self._small_tensor_list(
                attn_metadata.block_table[:1]
            ),
        }

    def _maybe_compare_triton_output(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None,
        output_block_scale: torch.Tensor | None,
        stage: str,
    ) -> None:
        call_idx = self._reserve_triton_compare_call()
        if call_idx is None:
            return
        if query.is_cuda and torch.cuda.is_current_stream_capturing():
            return

        reference = torch.empty_like(output)
        super().forward(
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            reference,
            output_scale,
            output_block_scale,
        )
        if query.is_cuda:
            # Diagnostic-only: the Triton reference path may update KV cache
            # asynchronously before this hook extracts raw-vs-cache tensors.
            torch.accelerator.synchronize(query.device)
        num_actual_tokens = int(attn_metadata.num_actual_tokens)
        extra = {
            "num_actual_tokens": num_actual_tokens,
            "max_query_len": int(attn_metadata.max_query_len),
            "max_seq_len": int(attn_metadata.max_seq_len),
            "layer_type": type(layer).__name__,
        }
        extra.update(self._layer_debug_info(layer))
        if stage in ("prefill_no_prefix", "prefill_no_prefix_paged_cache"):
            extra.update(
                self._prefill_raw_kv_cache_compare_stats(
                    layer,
                    key,
                    value,
                    kv_cache,
                    attn_metadata,
                    num_actual_tokens,
                )
            )
        extra.update(
            self._maybe_write_triton_tensor_dump(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                reference,
                call_idx,
                stage,
                num_actual_tokens,
            )
        )
        self._write_triton_compare_report(
            output[:num_actual_tokens],
            reference[:num_actual_tokens],
            call_idx,
            stage,
            extra,
        )

    def _supports_flash_v100_path(self) -> bool:
        """Check whether current layer/config can run Flash V100 safely."""
        supported_kv_dtype = not _uses_fp8_kv_cache(
            self.kv_cache_dtype
        ) or self.kv_cache_dtype in ("fp8", "fp8_e4m3", "fp8_e5m2")
        return (
            self.use_flash_v100
            and self.attn_type == AttentionType.DECODER
            and self.alibi_slopes is None
            and self.logits_soft_cap == 0
            and self.sinks is None
            and supported_kv_dtype
        )

    def _flash_v100_has_sliding_window(self) -> bool:
        sliding_window = self.sliding_window
        if sliding_window is None:
            return False
        return tuple(sliding_window) != (-1, -1)

    def _flash_v100_window_size(self, causal: bool) -> tuple[int, int]:
        if not self._flash_v100_has_sliding_window():
            return (-1, -1)
        left, right = tuple(self.sliding_window)
        left = int(left)
        right = int(right)
        if not causal and left >= 0 and right == 0:
            right = left
        return (left, right)

    def _call_flash_attn_decode_paged(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        *,
        softmax_scale: float,
        out: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: float,
        v_scale: float,
        window_size: tuple[int, int] = (-1, -1),
        max_seq_len_hint: int | None = None,
        workspace_seq_capacity_hint: int | None = None,
        active_num_partitions: int | None = None,
        partition_size_hint: int | None = None,
    ) -> None:
        kwargs: dict[str, object] = {
            "softmax_scale": softmax_scale,
            "out": out,
            "kv_cache_dtype": kv_cache_dtype,
            "k_scale": k_scale,
            "v_scale": v_scale,
        }
        if "window_size" in self._flash_decode_paged_kwargs:
            kwargs["window_size"] = window_size
        elif tuple(window_size) != (-1, -1):
            raise RuntimeError(
                "FLASH_ATTN_V100 decode op does not support sliding-window "
                "attention with this extension build."
            )
        optional_kwargs = {
            "max_seq_len_hint": max_seq_len_hint,
            "workspace_seq_capacity_hint": workspace_seq_capacity_hint,
            "active_num_partitions": active_num_partitions,
            "partition_size_hint": partition_size_hint,
        }
        for name, value in optional_kwargs.items():
            if name in self._flash_decode_paged_kwargs:
                kwargs[name] = value
        self.flash_attn_decode_paged(
            query,
            key_cache,
            value_cache,
            block_table,
            seq_lens,
            **kwargs,
        )

    def _smallq_decode_xqa_allowed(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        seq_lens: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        *,
        window_size: tuple[int, int],
        max_seq_len_hint: int | None,
        workspace_seq_capacity_hint: int | None,
        partition_size_hint: int | None,
    ) -> bool:
        if (
            not self.use_smallq_decode_xqa
            or self.flash_attn_decode_paged_xqa is None
            or partition_size_hint is not None
            or window_size != (-1, -1)
            or query.shape[0] != seq_lens.shape[0]
            or query.shape[2] != 256
            or key_cache.shape[2] <= 0
            or query.shape[1] % key_cache.shape[2] != 0
        ):
            return False

        q_per_kv = query.shape[1] // key_cache.shape[2]
        if q_per_kv not in (6, 8):
            return False

        fp16_kv = (
            self.kv_cache_dtype in ("auto", "bfloat16")
            and key_cache.dtype == torch.float16
            and value_cache.dtype == torch.float16
        )
        fp8_e5m2_kv = (
            self.kv_cache_dtype == "fp8_e5m2"
            and key_cache.dtype == torch.uint8
            and value_cache.dtype == torch.uint8
        )
        if not (fp16_kv or fp8_e5m2_kv):
            return False

        graph_capture = bool(
            getattr(attn_metadata, "flash_v100_cudagraph_capture", False)
        ) or _is_cuda_graph_capturing(query)
        effective_seq_hint = max(
            int(max_seq_len_hint or 0),
            int(workspace_seq_capacity_hint or 0) if graph_capture else 0,
        )
        min_seq_len = int(
            os.getenv("VLLM_FLASH_V100_SMALLQ_DECODE_XQA_MIN_SEQ_LEN", "4096")
        )
        if fp8_e5m2_kv:
            min_seq_len = max(min_seq_len, _decode_fp8_xqa_min_seq_len())
        return effective_seq_hint >= max(1, min_seq_len)

    def _call_flash_attn_smallq_decode_paged(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        *,
        out: torch.Tensor,
        max_seq_len_hint: int | None,
        workspace_seq_capacity_hint: int | None,
        partition_size_hint: int | None,
    ) -> None:
        global _logged_prefill_smallq_decode_xqa
        window_size = self._flash_v100_window_size(causal=True)
        if self._smallq_decode_xqa_allowed(
            query,
            key_cache,
            value_cache,
            seq_lens,
            attn_metadata,
            window_size=window_size,
            max_seq_len_hint=max_seq_len_hint,
            workspace_seq_capacity_hint=workspace_seq_capacity_hint,
            partition_size_hint=partition_size_hint,
        ):
            verifier_partition_size_hint = (
                _mtp5_xqa_dual_cta_partition_size_hint()
                if (
                    query.shape[0] == 5
                    and query.shape[2] == 256
                    and key_cache.shape[1] == 1616
                    and key_cache.shape[2] > 0
                    and query.shape[1] == 6 * key_cache.shape[2]
                    and self.kv_cache_dtype == "fp8_e5m2"
                    and key_cache.dtype == torch.uint8
                    and value_cache.dtype == torch.uint8
                )
                else None
            )
            if not _logged_prefill_smallq_decode_xqa:
                logger.info(
                    "FLASH_ATTN_V100 MTP verifier XQA path active "
                    "(rows=%d, q_per_kv=%d, partition_hint=%s, "
                    "mtp5_dual_cta=%s).",
                    int(query.shape[0]),
                    int(query.shape[1] // key_cache.shape[2]),
                    verifier_partition_size_hint,
                    verifier_partition_size_hint is not None,
                )
                _logged_prefill_smallq_decode_xqa = True
            _log_fp8_kv_cache_route("decode", self.kv_cache_dtype, "xqa_paged")
            self.flash_attn_decode_paged_xqa(
                query,
                key_cache,
                value_cache,
                block_table,
                seq_lens,
                softmax_scale=self.scale,
                out=out,
                kv_cache_dtype=self.kv_cache_dtype,
                k_scale=float(layer._k_scale_float),
                v_scale=float(layer._v_scale_float),
                window_size=window_size,
                max_seq_len_hint=max_seq_len_hint,
                workspace_seq_capacity_hint=workspace_seq_capacity_hint,
                partition_size_hint=verifier_partition_size_hint,
            )
            _record_route("prefill_smallq_decode_xqa")
            return

        self._call_flash_attn_decode_paged(
            query,
            key_cache,
            value_cache,
            block_table,
            seq_lens,
            softmax_scale=self.scale,
            out=out,
            kv_cache_dtype=self.kv_cache_dtype,
            k_scale=float(layer._k_scale_float),
            v_scale=float(layer._v_scale_float),
            window_size=window_size,
            max_seq_len_hint=max_seq_len_hint,
            workspace_seq_capacity_hint=workspace_seq_capacity_hint,
            partition_size_hint=partition_size_hint,
        )
        _record_route("prefill_smallq_decode_scalar")

    def _small_query_decode_enabled(
        self,
        attn_metadata: TritonAttentionMetadata,
    ) -> bool:
        if (
            not getattr(attn_metadata, "causal", True)
            or not self.use_flash_v100_decode
            or self.smallq_decode_max_query_len <= 0
        ):
            return False

        query_start_loc_cpu = getattr(attn_metadata, "query_start_loc_cpu", None)
        query_start_loc = (
            query_start_loc_cpu
            if query_start_loc_cpu is not None
            else attn_metadata.query_start_loc
        )
        if len(query_start_loc) <= 1:
            return False

        query_lens = query_start_loc[1:] - query_start_loc[:-1]
        max_query_len = int(query_lens.max().item())
        max_model_len = getattr(attn_metadata, "max_model_len", 0)
        model_len_supported = (
            self.smallq_decode_max_model_len <= 0
            or max_model_len <= self.smallq_decode_max_model_len
        )
        return max_query_len <= self.smallq_decode_max_query_len and model_len_supported

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward path.

        - Prefill: use Flash-V100 by default. Triton prefill is an explicit
          diagnostic fallback only.
        - Decode: use scalar paged Flash-V100 by default, including CUDA graph
          capture/replay, so selecting this backend is not a no-op in
          production decode. Mixed Triton/Flash routes are never silent.
        """
        global _logged_decode_flash, _logged_prefill_flash
        global _logged_prefill_paged_cache
        global _logged_prefill_prefix_flash
        global _logged_prefill_triton_safe
        global _warned_decode_fallback
        global _warned_decode_strict_fallback, _warned_feature_fallback

        if attn_metadata is None:
            assert output is not None
            _record_route("metadata_none_zero_output")
            return output.fill_(0)

        if not self._supports_flash_v100_path():
            layer_info = self._layer_debug_info(layer)
            is_dflash_draft_attn = bool(layer_info.get("is_dflash_draft_attn"))
            message = (
                "FLASH_ATTN_V100 cannot run this layer/config because a required "
                "Flash op is unavailable or the attention features/KV cache dtype "
                "are unsupported. Select TRITON_ATTN for a full Triton route, or "
                "set VLLM_FLASH_V100_ALLOW_TRITON_FALLBACK=1 for explicit "
                "diagnostic fallback."
            )
            if not (self.allow_triton_fallback or is_dflash_draft_attn):
                raise RuntimeError(message)
            if self.use_flash_v100 and not _warned_feature_fallback:
                if is_dflash_draft_attn:
                    logger.warning(
                        "FLASH_ATTN_V100 falling back to Triton for D-Flash "
                        "draft attention layer %s because the SM70 Flash-V100 "
                        "backend does not yet support this layer/config.",
                        layer_info.get("layer_name"),
                    )
                else:
                    logger.warning("%s", message)
                _warned_feature_fallback = True
            _record_route(
                "dflash_draft_triton_fallback"
                if is_dflash_draft_attn
                else "unsupported_triton_fallback"
            )
            return super().forward(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
            )

        assert output is not None
        is_prefill = attn_metadata.max_query_len > 1
        is_capturing = _is_cuda_graph_capturing(query)
        layer_name = self._layer_debug_info(layer).get("layer_name")
        if _draft_graph_debug_enabled():
            _draft_graph_debug_log(
                "forward:enter",
                "layer=%s is_prefill=%s is_capturing=%s max_query_len=%s "
                "max_seq_len=%s num_actual_tokens=%s %s %s %s %s %s %s",
                layer_name,
                is_prefill,
                is_capturing,
                int(attn_metadata.max_query_len),
                int(attn_metadata.max_seq_len),
                int(attn_metadata.num_actual_tokens),
                _format_tensor_debug(query, "query"),
                _format_tensor_debug(output, "output"),
                _format_tensor_debug(
                    getattr(attn_metadata, "query_start_loc", None),
                    "attn_qsl",
                ),
                _format_tensor_debug(
                    getattr(attn_metadata, "seq_lens", None),
                    "attn_seq",
                ),
                _format_tensor_debug(
                    getattr(attn_metadata, "block_table", None),
                    "attn_bt",
                ),
                _format_tensor_debug(
                    getattr(attn_metadata, "smallq_decode_seq_lens", None),
                    "smallq_seq",
                ),
            )
        _sm70_profile_trace(
            "forward enter layer=%s q_shape=%s k_shape=%s v_shape=%s "
            "kv_shape=%s is_prefill=%s is_capturing=%s max_query_len=%s "
            "max_seq_len=%s num_actual_tokens=%s use_decode_scalar=%s "
            "use_decode_paged_prefill=%s use_prefill_paged=%s "
            "use_triton_prefill=%s",
            layer_name,
            tuple(query.shape),
            tuple(key.shape),
            tuple(value.shape),
            tuple(kv_cache.shape) if hasattr(kv_cache, "shape") else None,
            is_prefill,
            is_capturing,
            int(attn_metadata.max_query_len),
            int(attn_metadata.max_seq_len),
            int(attn_metadata.num_actual_tokens),
            self.use_decode_scalar_paged,
            self.use_decode_paged_prefill,
            self.use_flash_v100_prefill_paged,
            self.use_triton_prefill,
        )

        if is_prefill:
            available_query_tokens = min(
                int(query.shape[0]),
                int(key.shape[0]),
                int(value.shape[0]),
                int(output.shape[0]),
            )
            metadata_live_token_mismatch = (
                _metadata_expects_more_query_tokens_than_available(
                    attn_metadata,
                    available_query_tokens,
                )
            )
            if self.use_triton_prefill:
                if not _logged_prefill_triton_safe:
                    logger.info(
                        "FLASH_ATTN_V100 prefill uses explicit Triton diagnostic "
                        "fallback because VLLM_FLASH_V100_PREFILL_USE_TRITON=1; "
                        "this mixed route is not a final performance path."
                    )
                    _logged_prefill_triton_safe = True
                _sm70_profile_trace(
                    "forward branch=prefill_triton_safe layer=%s",
                    layer_name,
                )
                self._reset_decode_cache()
                _record_route("prefill_triton_safe")
                return super().forward(
                    layer,
                    query,
                    key,
                    value,
                    kv_cache,
                    attn_metadata,
                    output,
                    output_scale,
                    output_block_scale,
                )
            if is_capturing:
                # CUDA graph capture uses dummy metadata whose seq_lens can
                # look like no-prefix prefill, while replayed MTP verification
                # is a uniform small-query decode over an existing KV prefix.
                # Capture the same small-query kernel branch that replay needs.
                smallq_decode = self._small_query_decode_enabled(attn_metadata)
                if smallq_decode:
                    if _draft_graph_debug_enabled():
                        _draft_graph_debug_log(
                            "forward:prefill_capture_smallq",
                            "layer=%s %s %s %s",
                            layer_name,
                            _format_tensor_debug(
                                getattr(
                                    attn_metadata,
                                    "smallq_decode_block_table",
                                    None,
                                ),
                                "smallq_bt",
                            ),
                            _format_tensor_debug(
                                getattr(
                                    attn_metadata,
                                    "smallq_decode_seq_lens",
                                    None,
                                ),
                                "smallq_seq",
                            ),
                            _format_tensor_debug(
                                getattr(
                                    attn_metadata,
                                    "smallq_query_start_loc",
                                    None,
                                ),
                                "smallq_qsl",
                            ),
                        )
                    _sm70_profile_trace(
                        "forward branch=prefill_capture_smallq layer=%s",
                        layer_name,
                    )
                    _record_route("prefill_capture_smallq")
                    if getattr(attn_metadata, "ddtree_parent_ids", None) is None:
                        _record_route("prefill_capture_smallq_no_ddtree_metadata")
                    else:
                        _record_route("prefill_capture_smallq_ddtree_metadata")
                    return self._flash_v100_prefill_with_prefix(
                        layer,
                        query,
                        key,
                        value,
                        kv_cache,
                        attn_metadata,
                        output,
                    )
                _sm70_profile_trace(
                    "forward branch=prefill_capture_full_flash layer=%s",
                    layer_name,
                )
            has_prefix_context = metadata_live_token_mismatch or _has_prefix_context(
                attn_metadata
            )
            smallq_decode = has_prefix_context and self._small_query_decode_enabled(
                attn_metadata
            )
            if has_prefix_context:
                if _draft_graph_debug_enabled():
                    _draft_graph_debug_log(
                        "forward:prefill_prefix",
                        "layer=%s smallq=%s metadata_live_token_mismatch=%s %s %s %s",
                        layer_name,
                        smallq_decode,
                        metadata_live_token_mismatch,
                        _format_tensor_debug(
                            getattr(attn_metadata, "smallq_decode_block_table", None),
                            "smallq_bt",
                        ),
                        _format_tensor_debug(
                            getattr(attn_metadata, "smallq_decode_seq_lens", None),
                            "smallq_seq",
                        ),
                        _format_tensor_debug(
                            getattr(attn_metadata, "smallq_query_start_loc", None),
                            "smallq_qsl",
                        ),
                    )
                _sm70_profile_trace(
                    "forward branch=prefill_prefix layer=%s smallq=%s",
                    layer_name,
                    smallq_decode,
                )
                if not _logged_prefill_prefix_flash:
                    if smallq_decode:
                        logger.info(
                            "FLASH_ATTN_V100 prefill path active "
                            "(prefix/chunked via small-query paged decode)."
                        )
                    elif self.use_flash_v100_prefill_paged:
                        logger.info(
                            "FLASH_ATTN_V100 prefill path active "
                            "(prefix/chunked via direct paged prefill kernel)."
                        )
                    else:
                        logger.info(
                            "FLASH_ATTN_V100 prefill path active "
                            "(prefix/chunked via paged-KV gather)."
                        )
                    _logged_prefill_prefix_flash = True
                if metadata_live_token_mismatch:
                    logger.info(
                        "FLASH_ATTN_V100 prefill switched to prefix/live-token "
                        "path because layer QKV tokens (%d) are shorter than "
                        "query metadata span.",
                        available_query_tokens,
                    )
                _log_fp8_kv_cache_route("prefill", self.kv_cache_dtype, "prefix")
                self._reset_decode_cache()
                result = self._flash_v100_prefill_with_prefix(
                    layer,
                    query,
                    key,
                    value,
                    kv_cache,
                    attn_metadata,
                    output,
                )
                self._maybe_compare_triton_output(
                    layer,
                    query,
                    key,
                    value,
                    kv_cache,
                    attn_metadata,
                    output,
                    output_scale,
                    output_block_scale,
                    "prefill_prefix",
                )
                _record_route("prefill_prefix_flash")
                return result
            if not _logged_prefill_flash:
                logger.info(
                    "FLASH_ATTN_V100 prefill path active (no prefix/chunked context)."
                )
                _logged_prefill_flash = True
            self._reset_decode_cache()
            if self.use_prefill_paged_cache and self.use_flash_v100_prefill_paged:
                _sm70_profile_trace(
                    "forward branch=prefill_no_prefix_paged_cache layer=%s",
                    layer_name,
                )
                if not _logged_prefill_paged_cache:
                    logger.warning(
                        "FLASH_ATTN_V100 no-prefix prefill is reading paged "
                        "KV cache for strict input-source diagnostics. This "
                        "may be slower than dense raw-KV prefill."
                    )
                    _logged_prefill_paged_cache = True
                _log_fp8_kv_cache_route(
                    "prefill", self.kv_cache_dtype, "no_prefix_paged_cache"
                )
                result = self._flash_v100_prefill_with_prefix(
                    layer,
                    query,
                    key,
                    value,
                    kv_cache,
                    attn_metadata,
                    output,
                )
                self._maybe_compare_triton_output(
                    layer,
                    query,
                    key,
                    value,
                    kv_cache,
                    attn_metadata,
                    output,
                    output_scale,
                    output_block_scale,
                    "prefill_no_prefix_paged_cache",
                )
                _record_route("prefill_no_prefix_paged_cache_flash")
                return result
            _sm70_profile_trace(
                "forward branch=prefill_no_prefix_dense layer=%s",
                layer_name,
            )
            result = self._flash_v100_prefill(query, key, value, attn_metadata, output)
            self._maybe_compare_triton_output(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
                "prefill_no_prefix",
            )
            _record_route("prefill_no_prefix_dense_flash")
            return result

        if not self.use_flash_v100_decode:
            message = (
                "FLASH_ATTN_V100 decode cannot run because the paged decode op "
                "is unavailable. Select TRITON_ATTN for a full Triton route, or "
                "set VLLM_FLASH_V100_ALLOW_TRITON_FALLBACK=1 for explicit "
                "diagnostic fallback."
            )
            if not self.allow_triton_fallback:
                raise RuntimeError(message)
            if self.use_flash_v100 and not _warned_decode_fallback:
                logger.warning("%s", message)
                _warned_decode_fallback = True
            _sm70_profile_trace(
                "forward branch=decode_triton_no_flash_decode layer=%s",
                layer_name,
            )
            _record_route("decode_triton_no_flash_decode")
            return super().forward(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
            )

        if (
            self.use_decode_paged_prefill
            and self.use_flash_v100_prefill_paged
            and not is_capturing
        ):
            _log_fp8_kv_cache_route(
                "decode", self.kv_cache_dtype, "decode_as_paged_prefill"
            )
            _sm70_profile_trace(
                "forward branch=decode_paged_prefill layer=%s",
                layer_name,
            )
            result = self._flash_v100_decode_as_paged_prefill(
                layer,
                query,
                kv_cache,
                attn_metadata,
                output,
            )
            self._maybe_compare_triton_output(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
                "decode_paged_prefill",
            )
            _record_route("decode_paged_prefill")
            return result
        if self.use_decode_dense_cache and not is_capturing:
            _log_fp8_kv_cache_route("decode", self.kv_cache_dtype, "dense_cache_bridge")
            _sm70_profile_trace(
                "forward branch=decode_dense_cache layer=%s",
                layer_name,
            )
            result = self._flash_v100_decode_dense_cache(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
            )
            self._maybe_compare_triton_output(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
                "decode_dense_cache",
            )
            _record_route("decode_dense_cache")
            return result
        if self.use_decode_dense_reference and not is_capturing:
            _log_fp8_kv_cache_route(
                "decode", self.kv_cache_dtype, "dense_reference_bridge"
            )
            _sm70_profile_trace(
                "forward branch=decode_dense_reference layer=%s",
                layer_name,
            )
            result = self._flash_v100_decode_dense_reference(
                layer,
                query,
                kv_cache,
                attn_metadata,
                output,
            )
            self._maybe_compare_triton_output(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
                "decode_dense_reference",
            )
            _record_route("decode_dense_reference")
            return result
        if not self.use_decode_scalar_paged:
            message = (
                "FLASH_ATTN_V100 decode has no enabled Flash route: scalar "
                "paged decode is disabled and the strict paged-prefill bridge "
                "is unavailable or CUDA graph capture is active. Re-enable "
                "VLLM_FLASH_V100_DECODE_USE_SCALAR_PAGED=1, select TRITON_ATTN "
                "for a full Triton route, or set "
                "VLLM_FLASH_V100_ALLOW_TRITON_FALLBACK=1 for explicit "
                "diagnostic fallback."
            )
            if not self.allow_triton_fallback:
                raise RuntimeError(message)
            if not _warned_decode_strict_fallback:
                logger.warning("%s", message)
                _warned_decode_strict_fallback = True
            _sm70_profile_trace(
                "forward branch=decode_triton_scalar_disabled layer=%s",
                layer_name,
            )
            _record_route("decode_triton_scalar_disabled")
            return super().forward(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
            )

        if not _logged_decode_flash:
            logger.info(
                "FLASH_ATTN_V100 decode path active (paged KV, "
                "CUDA-graph safe; selected route is reported separately)."
            )
            _logged_decode_flash = True
        if _draft_graph_debug_enabled():
            _draft_graph_debug_log(
                "forward:decode",
                "layer=%s %s %s %s",
                layer_name,
                _format_tensor_debug(
                    getattr(attn_metadata, "query_start_loc", None),
                    "attn_qsl",
                ),
                _format_tensor_debug(
                    getattr(attn_metadata, "seq_lens", None),
                    "attn_seq",
                ),
                _format_tensor_debug(
                    getattr(attn_metadata, "block_table", None),
                    "attn_bt",
                ),
            )
        _sm70_profile_trace(
            "forward branch=decode_scalar_paged layer=%s",
            layer_name,
        )
        result = self._flash_v100_decode(
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
        )
        self._maybe_compare_triton_output(
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
            output_scale,
            output_block_scale,
            "decode_scalar_paged",
        )
        return result

    def _flash_v100_decode_as_paged_prefill(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Decode through the paged prefill WMMA kernel.

        This opt-in path keeps the paged KV layout but uses the same compute
        order as dense/paged prefill. It is a strictness bridge while the
        scalar paged decode kernel is brought to bitwise parity.
        """
        global _logged_decode_paged_prefill
        global _logged_decode_paged_prefill_bhmd
        global _logged_decode_paged_prefill_bhmd_q_clone
        global _logged_decode_wmma_wrapper
        if not _logged_decode_paged_prefill:
            logger.warning(
                "FLASH_ATTN_V100 decode-as-paged-prefill path active. This is "
                "for strict debugging and may be slower than paged decode."
            )
            _logged_decode_paged_prefill = True

        num_actual_tokens = attn_metadata.num_actual_tokens
        query = query[:num_actual_tokens]
        out_view = output[:num_actual_tokens]
        if query.shape[0] == 0:
            return output

        key_cache, value_cache = _split_paged_kv_cache(kv_cache)

        query_start_loc_cpu = getattr(attn_metadata, "query_start_loc_cpu", None)
        query_start_loc = (
            query_start_loc_cpu
            if query_start_loc_cpu is not None
            else attn_metadata.query_start_loc
        )
        seq_lens_cpu = getattr(attn_metadata, "seq_lens_cpu", None)
        seq_lens_host = (
            seq_lens_cpu if seq_lens_cpu is not None else attn_metadata.seq_lens
        )
        num_seqs = min(len(query_start_loc) - 1, len(seq_lens_host))
        if num_seqs > 0:
            query_lens = query_start_loc[1 : num_seqs + 1] - query_start_loc[:num_seqs]
            first_query_len = int(query_lens[0].item())
            total_query_tokens = first_query_len * num_seqs
            can_batch_decode = (
                first_query_len > 0
                and bool(torch.all(query_lens == first_query_len).item())
                and int(query_start_loc[0].item()) == 0
                and int(query_start_loc[num_seqs].item()) == total_query_tokens
                and total_query_tokens <= query.shape[0]
            )
            if can_batch_decode:
                q_batch = query[:total_query_tokens].reshape(
                    num_seqs,
                    first_query_len,
                    query.shape[1],
                    query.shape[2],
                )
                out_batch_view = out_view[:total_query_tokens].reshape(
                    num_seqs,
                    first_query_len,
                    query.shape[1],
                    query.shape[2],
                )
                q_bhmd = q_batch.permute(0, 2, 1, 3)
                out_bhmd = out_batch_view.permute(0, 2, 1, 3)
                if (
                    first_query_len == 1
                    and self.use_decode_wmma_wrapper
                    and self.flash_attn_decode_paged_wmma is not None
                ):
                    if not _logged_decode_wmma_wrapper:
                        logger.info(
                            "FLASH_ATTN_V100 decode WMMA wrapper path active "
                            "(experimental exactness bridge)."
                        )
                        _logged_decode_wmma_wrapper = True
                    q_wmma = q_batch[:, 0].contiguous()
                    out_wmma = out_batch_view[:, 0]
                    self.flash_attn_decode_paged_wmma(
                        q_wmma,
                        key_cache,
                        value_cache,
                        attn_metadata.block_table[:num_seqs],
                        attn_metadata.seq_lens[:num_seqs],
                        softmax_scale=self.scale,
                        out=out_wmma,
                        kv_cache_dtype=self.kv_cache_dtype,
                        k_scale=float(layer._k_scale_float),
                        v_scale=float(layer._v_scale_float),
                    )
                    return output
                if (
                    first_query_len == 1
                    and self.use_decode_paged_prefill_bhmd_out
                    and self.flash_attn_prefill_paged_bhmd is not None
                    and q_bhmd.is_contiguous()
                    and out_bhmd.is_contiguous()
                ):
                    if not _logged_decode_paged_prefill_bhmd:
                        logger.info(
                            "FLASH_ATTN_V100 decode-as-paged-prefill BHMD "
                            "out path active."
                        )
                        _logged_decode_paged_prefill_bhmd = True
                    compare_call_idx = self._reserve_bhmd_compare_call()
                    safe_bmhd = None
                    if compare_call_idx is not None:
                        safe_bmhd = self.flash_attn_prefill_paged(
                            q_batch,
                            key_cache,
                            value_cache,
                            attn_metadata.block_table[:num_seqs],
                            attn_metadata.seq_lens[:num_seqs],
                            softmax_scale=self.scale,
                            kv_cache_dtype=self.kv_cache_dtype,
                            k_scale=float(layer._k_scale_float),
                            v_scale=float(layer._v_scale_float),
                            causal=True,
                        )
                    raw_q_bhmd = q_bhmd
                    q_out_same_storage = _same_storage(raw_q_bhmd, out_bhmd)
                    if q_out_same_storage:
                        if not _logged_decode_paged_prefill_bhmd_q_clone:
                            logger.info(
                                "FLASH_ATTN_V100 BHMD out path cloned Q to "
                                "avoid input/output storage aliasing."
                            )
                            _logged_decode_paged_prefill_bhmd_q_clone = True
                        raw_q_bhmd = q_bhmd.clone()
                    self.flash_attn_prefill_paged_bhmd(
                        raw_q_bhmd,
                        key_cache,
                        value_cache,
                        attn_metadata.block_table[:num_seqs],
                        attn_metadata.seq_lens[:num_seqs],
                        softmax_scale=self.scale,
                        out=out_bhmd,
                        kv_cache_dtype=self.kv_cache_dtype,
                        k_scale=float(layer._k_scale_float),
                        v_scale=float(layer._v_scale_float),
                        causal=True,
                    )
                    if safe_bmhd is not None:
                        assert compare_call_idx is not None
                        self._write_bhmd_compare_report(
                            out_batch_view,
                            safe_bmhd,
                            compare_call_idx,
                            "direct_out_vs_safe",
                            {
                                "q_bhmd_stride": list(q_bhmd.stride()),
                                "out_bhmd_stride": list(out_bhmd.stride()),
                                "out_bhmd_contiguous": out_bhmd.is_contiguous(),
                                "q_out_same_storage": q_out_same_storage,
                            },
                        )
                    return output
                out_batch = self.flash_attn_prefill_paged(
                    q_batch,
                    key_cache,
                    value_cache,
                    attn_metadata.block_table[:num_seqs],
                    attn_metadata.seq_lens[:num_seqs],
                    softmax_scale=self.scale,
                    kv_cache_dtype=self.kv_cache_dtype,
                    k_scale=float(layer._k_scale_float),
                    v_scale=float(layer._v_scale_float),
                    causal=True,
                )
                if first_query_len == 1 and q_bhmd.is_contiguous():
                    self._maybe_compare_bhmd_out(
                        layer,
                        q_bhmd,
                        key_cache,
                        value_cache,
                        attn_metadata.block_table[:num_seqs],
                        attn_metadata.seq_lens[:num_seqs],
                        out_batch,
                    )
                out_view[:total_query_tokens].copy_(
                    out_batch.reshape(
                        total_query_tokens,
                        query.shape[1],
                        query.shape[2],
                    )
                )
                return output

        for i in range(num_seqs):
            start = int(query_start_loc[i].item())
            end = int(query_start_loc[i + 1].item())
            if end <= start:
                continue
            out_seq = self.flash_attn_prefill_paged(
                query[start:end].unsqueeze(0),
                key_cache,
                value_cache,
                attn_metadata.block_table[i : i + 1],
                attn_metadata.seq_lens[i : i + 1],
                softmax_scale=self.scale,
                kv_cache_dtype=self.kv_cache_dtype,
                k_scale=float(layer._k_scale_float),
                v_scale=float(layer._v_scale_float),
                causal=True,
            )
            out_view[start:end].copy_(out_seq.squeeze(0))

        return output

    def _flash_v100_decode_dense_cache(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Decode through dense Flash-V100 with an incremental single-seq KV cache.

        This is a strict single-concurrency bridge for no-MTP experiments. It
        avoids full paged-KV gather after the first step, but it is still an
        oracle path rather than the final paged decode kernel.
        """
        global _logged_decode_dense_cache
        if _uses_fp8_kv_cache(self.kv_cache_dtype):
            if self.use_flash_v100_prefill_paged:
                return self._flash_v100_decode_as_paged_prefill(
                    layer,
                    query,
                    kv_cache,
                    attn_metadata,
                    output,
                )
            return self._flash_v100_decode_dense_reference(
                layer,
                query,
                kv_cache,
                attn_metadata,
                output,
            )
        if not _logged_decode_dense_cache:
            logger.warning(
                "FLASH_ATTN_V100 decode dense-cache path active. This is "
                "single-sequence strict debugging and may be slower than paged decode."
            )
            _logged_decode_dense_cache = True

        num_actual_tokens = attn_metadata.num_actual_tokens
        query = query[:num_actual_tokens]
        out_view = output[:num_actual_tokens]
        if query.shape[0] == 0:
            return output

        query_start_loc_cpu = getattr(attn_metadata, "query_start_loc_cpu", None)
        query_start_loc = (
            query_start_loc_cpu
            if query_start_loc_cpu is not None
            else attn_metadata.query_start_loc
        )
        seq_lens_cpu = getattr(attn_metadata, "seq_lens_cpu", None)
        seq_lens_host = (
            seq_lens_cpu if seq_lens_cpu is not None else attn_metadata.seq_lens
        )
        num_seqs = min(len(query_start_loc) - 1, len(seq_lens_host))
        if num_seqs != 1:
            if self.use_flash_v100_prefill_paged:
                return self._flash_v100_decode_as_paged_prefill(
                    layer,
                    query,
                    kv_cache,
                    attn_metadata,
                    output,
                )
            return self._flash_v100_decode_dense_reference(
                layer,
                query,
                kv_cache,
                attn_metadata,
                output,
            )

        key_cache, _ = _split_paged_kv_cache(kv_cache)
        block_size = key_cache.shape[1]
        head_dim = key_cache.shape[3]
        seq_len = int(seq_lens_host[0].item())
        k_cont, v_cont = self._get_decode_kv_single_seq(
            key,
            value,
            kv_cache,
            attn_metadata,
            attn_metadata.seq_lens[:1],
            block_size,
            head_dim,
        )
        out_seq = self.flash_attn_func(
            query.unsqueeze(0),
            k_cont[:seq_len].unsqueeze(0),
            v_cont[:seq_len].unsqueeze(0),
            causal=True,
            softmax_scale=self.scale,
        )
        out_view.copy_(out_seq.squeeze(0))
        return output

    def _flash_v100_decode_dense_reference(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Decode through dense Flash-V100 over gathered KV.

        This is an opt-in strict-debug path, not a speed path. It gives us a
        dense Flash-V100 oracle while the paged decode kernel is brought to
        bitwise parity.
        """
        global _logged_decode_dense_reference
        if not _logged_decode_dense_reference:
            logger.warning(
                "FLASH_ATTN_V100 decode dense-reference path active. This is "
                "for strict debugging and is expected to be slower than paged decode."
            )
            _logged_decode_dense_reference = True

        num_actual_tokens = attn_metadata.num_actual_tokens
        query = query[:num_actual_tokens]
        out_view = output[:num_actual_tokens]
        if query.shape[0] == 0:
            return output

        key_cache, value_cache = _split_paged_kv_cache(kv_cache)
        block_size = key_cache.shape[1]
        num_kv_heads = key_cache.shape[2]
        head_dim = key_cache.shape[3]

        query_start_loc_cpu = getattr(attn_metadata, "query_start_loc_cpu", None)
        query_start_loc = (
            query_start_loc_cpu
            if query_start_loc_cpu is not None
            else attn_metadata.query_start_loc
        )
        seq_lens_cpu = getattr(attn_metadata, "seq_lens_cpu", None)
        seq_lens_host = (
            seq_lens_cpu if seq_lens_cpu is not None else attn_metadata.seq_lens
        )
        num_seqs = min(len(query_start_loc) - 1, len(seq_lens_host))

        for i in range(num_seqs):
            start = int(query_start_loc[i].item())
            end = int(query_start_loc[i + 1].item())
            if end <= start:
                continue
            seq_len = int(seq_lens_host[i].item())
            k_cont, v_cont = _extract_contiguous_kv_from_paged_cache(
                kv_cache=kv_cache,
                block_table=attn_metadata.block_table[i : i + 1],
                seq_lens=attn_metadata.seq_lens[i : i + 1],
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                block_size=block_size,
                total_tokens=seq_len,
            )
            k_cont, v_cont = _dequantize_fp8_contiguous_kv(
                k_cont,
                v_cont,
                self.kv_cache_dtype,
                float(layer._k_scale_float),
                float(layer._v_scale_float),
            )
            out_seq = self.flash_attn_func(
                query[start:end].unsqueeze(0),
                k_cont.unsqueeze(0),
                v_cont.unsqueeze(0),
                causal=True,
                softmax_scale=self.scale,
            )
            out_view[start:end].copy_(out_seq.squeeze(0))
        return output

    def _flash_v100_prefill(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Prefill path for no-prefix case (query_len == seq_len per sequence)."""
        causal = getattr(attn_metadata, "causal", True)
        window_size = self._flash_v100_window_size(causal)
        num_actual_tokens = attn_metadata.num_actual_tokens
        query_start_loc_cpu = getattr(attn_metadata, "query_start_loc_cpu", None)
        query_start_loc = (
            query_start_loc_cpu
            if query_start_loc_cpu is not None
            else attn_metadata.query_start_loc
        )
        return flash_v100_dense_prefill(
            query=query,
            key=key,
            value=value,
            output=output,
            query_start_loc=query_start_loc,
            num_actual_tokens=num_actual_tokens,
            softmax_scale=self.scale,
            causal=causal,
            window_size=window_size,
        )

    def _flash_v100_decode(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Decode path using Flash V100 directly over paged KV cache."""
        window_size = self._flash_v100_window_size(causal=True)
        num_actual_tokens = attn_metadata.num_actual_tokens
        query = query[:num_actual_tokens]
        out_view = output[:num_actual_tokens]

        if query.shape[0] == 0:
            return output

        key_cache, value_cache = _split_paged_kv_cache(kv_cache)
        q_per_kv = (
            query.shape[1] // key_cache.shape[2]
            if key_cache.shape[2] > 0 and query.shape[1] % key_cache.shape[2] == 0
            else 0
        )
        xqa_kv_supported = (
            self.kv_cache_dtype in ("auto", "bfloat16")
            and key_cache.dtype == torch.float16
            and value_cache.dtype == torch.float16
        ) or (
            self.kv_cache_dtype == "fp8_e5m2"
            and key_cache.dtype == torch.uint8
            and value_cache.dtype == torch.uint8
        )

        # FP8 G4 XQA had no end-to-end gain on 35B-A3B TP4 and has no accepted
        # sampled-quality advantage. Keep that shape on scalar decode.
        if (
            self.use_decode_xqa
            and self.flash_attn_decode_paged_xqa is not None
            and xqa_kv_supported
            and query.shape[0] == attn_metadata.seq_lens.shape[0]
            and query.shape[2] == 256
            and key_cache.shape[2] > 0
            and query.shape[1] % key_cache.shape[2] == 0
            and _decode_xqa_allowed_for_q_per_kv(q_per_kv, attn_metadata)
            and (
                self.kv_cache_dtype != "fp8_e5m2"
                or (q_per_kv != 4 and _decode_fp8_xqa_allowed(attn_metadata, query))
            )
            and window_size == (-1, -1)
        ):
            _log_fp8_kv_cache_route("decode", self.kv_cache_dtype, "xqa_paged")
            _trace_decode_active(
                route="decode_xqa_paged",
                query=query,
                key_cache=key_cache,
                seq_lens=attn_metadata.seq_lens,
                attn_metadata=attn_metadata,
                window_size=window_size,
            )
            self.flash_attn_decode_paged_xqa(
                query,
                key_cache,
                value_cache,
                attn_metadata.block_table,
                attn_metadata.seq_lens,
                softmax_scale=self.scale,
                out=out_view,
                kv_cache_dtype=self.kv_cache_dtype,
                k_scale=float(layer._k_scale_float),
                v_scale=float(layer._v_scale_float),
                window_size=window_size,
                max_seq_len_hint=getattr(
                    attn_metadata,
                    "flash_v100_decode_max_seq_len_hint",
                    None,
                ),
                workspace_seq_capacity_hint=getattr(
                    attn_metadata,
                    "flash_v100_decode_workspace_seq_capacity_hint",
                    None,
                ),
                active_num_partitions=getattr(
                    attn_metadata,
                    "flash_v100_decode_active_num_partitions",
                    None,
                ),
            )
            _record_route("decode_xqa_paged")
            return output

        _log_fp8_kv_cache_route("decode", self.kv_cache_dtype, "scalar_paged")
        _trace_decode_active(
            route="decode_scalar_paged",
            query=query,
            key_cache=key_cache,
            seq_lens=attn_metadata.seq_lens,
            attn_metadata=attn_metadata,
            window_size=window_size,
        )
        self._call_flash_attn_decode_paged(
            query,
            key_cache,
            value_cache,
            attn_metadata.block_table,
            attn_metadata.seq_lens,
            softmax_scale=self.scale,
            out=out_view,
            kv_cache_dtype=self.kv_cache_dtype,
            k_scale=float(layer._k_scale_float),
            v_scale=float(layer._v_scale_float),
            window_size=window_size,
            max_seq_len_hint=getattr(
                attn_metadata,
                "flash_v100_decode_max_seq_len_hint",
                None,
            ),
            workspace_seq_capacity_hint=getattr(
                attn_metadata,
                "flash_v100_decode_workspace_seq_capacity_hint",
                None,
            ),
            active_num_partitions=getattr(
                attn_metadata,
                "flash_v100_decode_active_num_partitions",
                None,
            ),
        )
        _record_route("decode_scalar_paged")
        return output

    def _flash_v100_ddtree_small_query_prefill_dense(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor,
        query_start_loc: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Correctness bridge for branched DDTree verifier attention."""
        global _logged_prefill_ddtree_dense
        global _logged_prefill_ddtree_triton
        global _logged_prefill_ddtree_triton_fallback

        is_capturing = _is_cuda_graph_capturing(query)
        parent_ids = getattr(attn_metadata, "ddtree_parent_ids", None)
        num_tree_tokens_cpu = getattr(attn_metadata, "ddtree_num_tree_tokens_cpu", None)
        num_reqs = min(
            max(0, len(query_start_loc) - 1),
            int(parent_ids.shape[0]) if parent_ids is not None else 0,
            int(num_tree_tokens_cpu.numel()) if num_tree_tokens_cpu is not None else 0,
        )
        window_size = self._flash_v100_window_size(causal=True)
        if (
            _dflash_ddtree_triton_branch_attn_enabled()
            and parent_ids is not None
            and _ddtree_triton_seq_lens_match(
                attn_metadata,
                seq_lens,
                num_reqs,
            )
            and _ddtree_triton_query_start_loc_match(
                attn_metadata,
                query_start_loc,
                num_reqs,
            )
        ):
            triton_parent_ids = _ddtree_triton_parent_ids_for_query(
                parent_ids,
                num_tree_tokens_cpu,
                query_start_loc,
                is_capturing=is_capturing,
            )
            if triton_parent_ids is not None:
                try:
                    from vllm.v1.attention.backends.ddtree_branch_triton import (
                        ddtree_branch_attention_correction,
                    )

                    ddtree_branch_attention_correction(
                        impl=self,
                        query=query,
                        key=key,
                        value=value,
                        key_cache=key_cache,
                        value_cache=value_cache,
                        output=output,
                        attn_metadata=attn_metadata,
                        parent_ids=triton_parent_ids,
                        window_size=window_size,
                    )
                except Exception:
                    if is_capturing or _dflash_ddtree_triton_branch_attn_strict():
                        raise
                    if _ddtree_trace_enabled():
                        _ddtree_trace_event(
                            "flash_ddtree_attention_route",
                            {
                                "route": "triton_exception_fallback",
                                "num_reqs": num_reqs,
                                "num_actual_tokens": int(
                                    getattr(attn_metadata, "num_actual_tokens", 0)
                                ),
                                "query_start_loc": query_start_loc.detach()
                                .cpu()
                                .tolist(),
                                "seq_lens": seq_lens.detach().cpu().tolist(),
                                "tree_tokens": (
                                    num_tree_tokens_cpu.detach().cpu().tolist()
                                    if num_tree_tokens_cpu is not None
                                    else None
                                ),
                            },
                        )
                    if not _logged_prefill_ddtree_triton_fallback:
                        logger.exception(
                            "FLASH_ATTN_V100 DDTree Triton verifier failed; "
                            "falling back to dense masked verifier."
                        )
                        _logged_prefill_ddtree_triton_fallback = True
                else:
                    if not _logged_prefill_ddtree_triton:
                        logger.info(
                            "FLASH_ATTN_V100 DDTree branched verifier path active "
                            "(Triton paged-KV ancestor mask)."
                        )
                    _logged_prefill_ddtree_triton = True
                    _record_route("prefill_ddtree_triton")
                    if _ddtree_trace_enabled():
                        _ddtree_trace_event(
                            "flash_ddtree_attention_route",
                            {
                                "route": "triton",
                                "num_reqs": num_reqs,
                                "num_actual_tokens": int(
                                    getattr(attn_metadata, "num_actual_tokens", 0)
                                ),
                                "query_start_loc": query_start_loc.detach()
                                .cpu()
                                .tolist(),
                                "seq_lens": seq_lens.detach().cpu().tolist(),
                                "tree_tokens": (
                                    num_tree_tokens_cpu.detach().cpu().tolist()
                                    if num_tree_tokens_cpu is not None
                                    else None
                                ),
                            },
                        )
                    return output

        if is_capturing:
            raise RuntimeError(
                "FLASH_ATTN_V100 DDTree dense verifier fallback is not "
                "CUDA-graph safe and the Triton branch verifier is disabled "
                "or unavailable."
            )

        parent_ids_cpu = _ddtree_parent_ids_cpu(attn_metadata)
        if parent_ids_cpu is None or num_tree_tokens_cpu is None:
            raise RuntimeError(
                "DDTree dense verifier fallback requires parent metadata"
            )

        if not _logged_prefill_ddtree_dense:
            logger.info(
                "FLASH_ATTN_V100 DDTree branched verifier path active "
                "(dense masked small-query fallback)."
            )
            _logged_prefill_ddtree_dense = True

        _record_route("prefill_ddtree_dense")
        if _ddtree_trace_enabled():
            _ddtree_trace_event(
                "flash_ddtree_attention_route",
                {
                    "route": "dense",
                    "num_reqs": num_reqs,
                    "num_actual_tokens": int(
                        getattr(attn_metadata, "num_actual_tokens", 0)
                    ),
                    "query_start_loc": query_start_loc.detach().cpu().tolist(),
                    "seq_lens": seq_lens.detach().cpu().tolist(),
                    "tree_tokens": num_tree_tokens_cpu.detach().cpu().tolist(),
                },
            )
        trace_kv_diff = os.getenv("VLLM_DFLASH_DDTREE_TRACE_KV_CACHE_DIFF", "0") == "1"
        profile_enabled = envs.VLLM_FLASH_V100_PREFILL_CHUNK_PROFILE
        profile_start: torch.cuda.Event | None = None
        profile_end: torch.cuda.Event | None = None
        if profile_enabled:
            profile_start = torch.cuda.Event(enable_timing=True)
            profile_end = torch.cuda.Event(enable_timing=True)
            profile_start.record()
        num_seqs = len(query_start_loc) - 1
        total_query_tokens = 0
        total_tree_tokens = 0
        max_seq_len = 0
        out_view = output[: attn_metadata.num_actual_tokens]
        for req_idx in range(num_seqs):
            start = int(query_start_loc[req_idx].item())
            end = int(query_start_loc[req_idx + 1].item())
            q_len = end - start
            if q_len <= 0:
                continue

            seq_len = int(seq_lens[req_idx].item())
            if seq_len <= 0:
                continue
            total_query_tokens += q_len
            max_seq_len = max(max_seq_len, seq_len)
            prefix_len = max(seq_len - q_len, 0)
            tree_len = (
                int(num_tree_tokens_cpu[req_idx].item())
                if req_idx < int(num_tree_tokens_cpu.numel())
                else 0
            )
            total_tree_tokens += max(tree_len, 0)
            parent_row = (
                parent_ids_cpu[req_idx]
                if req_idx < int(parent_ids_cpu.shape[0])
                else None
            )

            if trace_kv_diff:
                slot_mapping = getattr(attn_metadata, "slot_mapping", None)
                if (
                    slot_mapping is not None
                    and key is not None
                    and value is not None
                    and end <= int(slot_mapping.numel())
                ):
                    slot_slice = slot_mapping[start:end].to(torch.long)
                    valid_slots = slot_slice >= 0
                    if bool(valid_slots.all().item()):
                        slot_blocks = torch.div(
                            slot_slice,
                            key_cache.shape[1],
                            rounding_mode="floor",
                        )
                        slot_offsets = torch.remainder(slot_slice, key_cache.shape[1])
                        cache_k_by_slot = key_cache[slot_blocks, slot_offsets]
                        cache_v_by_slot = value_cache[slot_blocks, slot_offsets]
                        cache_k_by_slot, cache_v_by_slot = (
                            _dequantize_fp8_contiguous_kv(
                                cache_k_by_slot,
                                cache_v_by_slot,
                                self.kv_cache_dtype,
                                float(layer._k_scale_float),
                                float(layer._v_scale_float),
                            )
                        )
                        key_diff = (cache_k_by_slot - key[start:end]).abs()
                        value_diff = (cache_v_by_slot - value[start:end]).abs()
                        _ddtree_trace_event(
                            "flash_ddtree_kv_cache_diff",
                            {
                                "layer": str(
                                    self._layer_debug_info(layer).get("layer_name")
                                ),
                                "req_idx": req_idx,
                                "query_start": start,
                                "query_end": end,
                                "seq_len": seq_len,
                                "prefix_len": prefix_len,
                                "tree_len": tree_len,
                                "key_max_diff": float(key_diff.max().item()),
                                "key_mean_diff": float(key_diff.mean().item()),
                                "value_max_diff": float(value_diff.max().item()),
                                "value_mean_diff": float(value_diff.mean().item()),
                            },
                        )

            k_cont, v_cont = _extract_contiguous_kv_from_paged_cache(
                (key_cache, value_cache),
                attn_metadata.block_table[req_idx : req_idx + 1],
                attn_metadata.seq_lens[req_idx : req_idx + 1],
                key_cache.shape[2],
                key_cache.shape[3],
                key_cache.shape[1],
                total_tokens=seq_len,
            )
            k_cont, v_cont = _dequantize_fp8_contiguous_kv(
                k_cont,
                v_cont,
                self.kv_cache_dtype,
                float(layer._k_scale_float),
                float(layer._v_scale_float),
            )
            if prefix_len + q_len <= k_cont.shape[0]:
                k_cont[prefix_len : prefix_len + q_len].copy_(key[start:end])
                v_cont[prefix_len : prefix_len + q_len].copy_(value[start:end])

            q_seq = query[start:end]
            q_f = q_seq.float()
            k_f = k_cont.float()
            v_f = v_cont.float()
            if q_f.shape[1] % k_f.shape[1] != 0:
                raise ValueError(
                    "DDTree dense verifier requires Q heads divisible by KV heads, "
                    f"got {q_f.shape[1]} and {k_f.shape[1]}"
                )
            if q_f.shape[1] != k_f.shape[1]:
                repeat = q_f.shape[1] // k_f.shape[1]
                k_f = k_f.repeat_interleave(repeat, dim=1)
                v_f = v_f.repeat_interleave(repeat, dim=1)

            scores = torch.einsum("mhd,nhd->hmn", q_f, k_f) * self.scale
            visible = _build_ddtree_visibility_mask(
                q_len=q_len,
                seq_len=seq_len,
                prefix_len=prefix_len,
                tree_len=tree_len,
                parent_row=parent_row,
                device=query.device,
                window_size=window_size,
            )
            scores = scores.masked_fill(~visible.unsqueeze(0), float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            out_seq = torch.einsum("hmn,nhd->mhd", probs, v_f)
            out_view[start:end].copy_(out_seq.to(dtype=query.dtype))

        if profile_start is not None and profile_end is not None:
            profile_end.record()
            torch.accelerator.synchronize()
            logger.info(
                "FLASH_ATTN_V100 prefill chunk profile: route=%s layer=%s "
                "elapsed_ms=%.3f query_tokens=%d tree_tokens=%d max_seq_len=%d "
                "heads_q=%d heads_kv=%d head_dim=%d",
                "prefill_ddtree_dense",
                self._layer_debug_info(layer).get("layer_name"),
                float(profile_start.elapsed_time(profile_end)),
                total_query_tokens,
                total_tree_tokens,
                max_seq_len,
                int(query.shape[1]),
                int(key_cache.shape[2]),
                int(key_cache.shape[3]),
            )

        return output

    def _flash_v100_small_query_prefill_as_decode(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor,
        query_start_loc: torch.Tensor,
        _seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Run small causal prefix-prefill queries through paged decode.

        MTP verification presents a tiny query span over a long KV prefix. The
        paged prefill kernel is correct, but its work scheduling is much more
        expensive for this shape and exceeds SM70 shared-memory limits at very
        long contexts. Treating every query token as an independent decode row
        with an increasing seq_len preserves the causal mask without exposing
        future draft tokens.
        """
        device = attn_metadata.seq_lens.device
        dtype = attn_metadata.seq_lens.dtype

        num_query_tokens = min(
            int(attn_metadata.num_actual_tokens),
            int(query.shape[0]),
            int(output.shape[0]),
        )
        persistent_decode_block_table = getattr(
            attn_metadata,
            "smallq_decode_block_table",
            None,
        )
        persistent_decode_seq_lens = getattr(
            attn_metadata,
            "smallq_decode_seq_lens",
            None,
        )
        persistent_query_start_loc = getattr(
            attn_metadata,
            "smallq_query_start_loc",
            None,
        )
        if (
            persistent_decode_block_table is not None
            and persistent_decode_seq_lens is not None
            and persistent_query_start_loc is not None
            and int(persistent_decode_seq_lens.shape[0]) >= num_query_tokens
            and int(persistent_decode_block_table.shape[0]) >= num_query_tokens
            and not _metadata_expects_more_query_tokens_than_available(
                attn_metadata,
                num_query_tokens,
            )
        ):
            query = query[:num_query_tokens]
            out_view = output[:num_query_tokens]
            if _draft_graph_debug_enabled():
                _graph_metadata_debug_log(
                    "smallq_call",
                    "layer=%s num_query_tokens=%s %s %s %s %s %s",
                    self._layer_debug_info(layer).get("layer_name"),
                    num_query_tokens,
                    _format_tensor_debug(query, "query"),
                    _format_tensor_debug(out_view, "out"),
                    _format_tensor_debug(
                        persistent_decode_block_table[:num_query_tokens],
                        "smallq_bt",
                    ),
                    _format_tensor_debug(
                        persistent_decode_seq_lens[:num_query_tokens],
                        "smallq_seq",
                    ),
                    _format_tensor_debug(persistent_query_start_loc, "smallq_qsl"),
                )
            self._call_flash_attn_smallq_decode_paged(
                layer,
                query,
                key_cache,
                value_cache,
                persistent_decode_block_table[:num_query_tokens],
                persistent_decode_seq_lens[:num_query_tokens],
                attn_metadata,
                out=out_view,
                max_seq_len_hint=getattr(
                    attn_metadata,
                    "smallq_decode_max_seq_len_hint",
                    None,
                ),
                workspace_seq_capacity_hint=getattr(
                    attn_metadata,
                    "smallq_decode_workspace_seq_capacity_hint",
                    None,
                ),
                partition_size_hint=getattr(
                    attn_metadata,
                    "smallq_decode_partition_size_hint",
                    None,
                ),
            )
            return output

        if _is_cuda_graph_capturing(query):
            raise RuntimeError(
                "FLASH_ATTN_V100 small-query prefix prefill entered CUDA graph "
                "capture without persistent smallq decode metadata. The "
                "metadata builder must attach smallq_decode_block_table and "
                "smallq_decode_seq_lens so replay does not capture transient "
                "derived tensors."
            )

        query_start_loc_norm = _normalize_query_start_loc_for_available_tokens(
            query_start_loc,
            num_query_tokens,
        )
        query_start_loc_gpu = query_start_loc_norm.to(
            device=device,
            dtype=attn_metadata.query_start_loc.dtype,
        )
        query = query[:num_query_tokens]
        out_view = output[:num_query_tokens]
        query_lens_gpu = query_start_loc_gpu[1:] - query_start_loc_gpu[:-1]
        real_query_lens_gpu = query_lens_gpu
        real_num_query_tokens = query_start_loc_gpu[-1]
        num_seqs = query_lens_gpu.numel()
        if num_seqs > 0:
            # FULL CUDA graph replay may pad a 3-request MTP verifier batch
            # from 15 tokens to 20 tokens while query_start_loc still marks
            # only the 15 live tokens. Give the padded tail a dummy query span
            # so repeat_interleave keeps the captured graph shape. The padded
            # rows are masked below and must not read real KV cache entries.
            padding_tokens = torch.clamp(
                num_query_tokens - real_num_query_tokens,
                min=0,
            )
            query_lens_gpu = query_lens_gpu.clone()
            query_lens_gpu[-1] += padding_tokens

        seq_lens = _seq_lens[:num_seqs].to(
            device=device,
            dtype=attn_metadata.seq_lens.dtype,
        )
        effective_seq_lens = torch.maximum(
            seq_lens,
            real_query_lens_gpu.to(dtype=attn_metadata.seq_lens.dtype),
        )
        block_table = attn_metadata.block_table[:num_seqs].clamp_min(0)
        decode_block_table = torch.repeat_interleave(
            block_table,
            query_lens_gpu,
            dim=0,
            output_size=num_query_tokens,
        ).contiguous()
        seq_lens_rep = torch.repeat_interleave(
            effective_seq_lens,
            query_lens_gpu,
            output_size=num_query_tokens,
        )
        query_lens_rep = torch.repeat_interleave(
            real_query_lens_gpu.to(dtype=dtype),
            query_lens_gpu,
            output_size=num_query_tokens,
        )
        start_locs_rep = torch.repeat_interleave(
            query_start_loc_gpu[:-1].to(dtype=dtype),
            query_lens_gpu,
            output_size=num_query_tokens,
        )
        token_indices = torch.arange(
            num_query_tokens,
            device=device,
            dtype=dtype,
        )
        offsets = token_indices - start_locs_rep + 1
        decode_seq_lens = (seq_lens_rep - query_lens_rep + offsets).contiguous()
        padding_mask = token_indices >= real_num_query_tokens
        decode_seq_lens = torch.where(
            padding_mask,
            torch.zeros_like(decode_seq_lens),
            decode_seq_lens,
        ).contiguous()
        decode_block_table = torch.where(
            padding_mask[:, None],
            torch.zeros_like(decode_block_table),
            decode_block_table,
        ).contiguous()
        self._call_flash_attn_smallq_decode_paged(
            layer,
            query,
            key_cache,
            value_cache,
            decode_block_table,
            decode_seq_lens,
            attn_metadata,
            out=out_view,
            max_seq_len_hint=(int(seq_lens.max().item()) if num_seqs > 0 else None),
            workspace_seq_capacity_hint=(
                int(block_table.shape[1]) * int(key_cache.shape[1])
                if num_seqs > 0
                else None
            ),
            partition_size_hint=None,
        )
        return output

    def _should_use_fp8_prefill_bridge(
        self,
        *,
        q_len: int,
        head_dim: int,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        causal: bool,
        window_size: tuple[int, int],
    ) -> bool:
        return (
            self.use_fp8_prefill_bridge
            and self.use_flash_v100_prefill_paged
            and self.kv_cache_dtype == "fp8_e5m2"
            and key_cache.dtype == torch.uint8
            and value_cache.dtype == torch.uint8
            and key_cache.shape == value_cache.shape
            and head_dim == 256
            and q_len >= 32
            and causal
            and window_size == (-1, -1)
        )

    def _run_fp8_prefill_bridge(
        self,
        *,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        k_scale: float,
        v_scale: float,
        causal: bool,
        window_size: tuple[int, int],
    ) -> torch.Tensor | None:
        if block_table.shape[0] != 1:
            return None
        input_capacity = int(block_table.shape[1]) * int(key_cache.shape[1])
        required_blocks = _cdiv_int(
            input_capacity,
            _FP8_PREFILL_BRIDGE_PAGE_SIZE,
        )
        workspace = _get_fp8_prefill_bridge_workspace(
            key_cache,
            required_blocks,
        )
        if workspace is None:
            return None
        key_out, value_out, output_block_table = workspace
        self.fp8_e5m2_paged_kv_to_fp16(
            key_cache,
            value_cache,
            block_table,
            seq_lens,
            key_out,
            value_out,
            k_scale,
            v_scale,
        )
        return self.flash_attn_prefill_paged(
            query,
            key_out,
            value_out,
            output_block_table,
            seq_lens,
            softmax_scale=self.scale,
            kv_cache_dtype="auto",
            k_scale=1.0,
            v_scale=1.0,
            causal=causal,
            window_size=window_size,
        )

    def _should_use_prefill_splitkv(
        self,
        *,
        q_len: int,
        seq_len: int,
        head_dim: int,
        key_cache: torch.Tensor,
        causal: bool,
    ) -> bool:
        if not self.use_flash_v100_prefill_splitkv:
            return False
        if self.flash_attn_prefill_paged_splitkv is None:
            return False
        if not causal:
            return False
        if head_dim != 256:
            return False
        if key_cache.dtype != torch.float16:
            return False
        if q_len < self.prefill_split_kv_min_q:
            return False
        if self.prefill_split_kv_max_q > 0 and q_len > self.prefill_split_kv_max_q:
            return False
        if seq_len < self.prefill_split_kv_min_kv:
            return False
        return seq_len > self.prefill_split_kv_tokens

    def _should_use_prefill_bfla(
        self,
        *,
        q_len: int,
        seq_len: int,
        head_dim: int,
        key_cache: torch.Tensor,
        causal: bool,
        window_size: tuple[int, int],
    ) -> bool:
        if not self.use_flash_v100_prefill_bfla:
            return False
        if self.flash_attn_prefill_paged_bfla is None:
            return False
        if not causal or window_size != (-1, -1):
            return False
        if head_dim != 256:
            return False
        if key_cache.dtype != torch.float16:
            return False
        if q_len < self.prefill_bfla_min_q:
            return False
        if seq_len < self.prefill_bfla_min_kv:
            return False
        return self.prefill_bfla_mask_block_n > 0

    def _should_use_prefill_contig_dense(
        self,
        *,
        q_len: int,
        seq_len: int,
        head_dim: int,
        key_cache: torch.Tensor,
        causal: bool,
        window_size: tuple[int, int],
    ) -> bool:
        if not self.use_flash_v100_prefill_contig_dense:
            return False
        if not causal or window_size != (-1, -1):
            return False
        if head_dim != 256:
            return False
        if key_cache.dtype != torch.float16:
            return False
        if q_len < self.prefill_contig_dense_min_q:
            return False
        return seq_len >= self.prefill_contig_dense_min_kv

    def _run_prefill_paged_call(
        self,
        *,
        route: str,
        q_len: int,
        seq_len: int,
        heads_q: int,
        heads_kv: int,
        head_dim: int,
        block_size: int,
        fn: Callable[[], torch.Tensor],
    ) -> torch.Tensor:
        if not envs.VLLM_FLASH_V100_PREFILL_CHUNK_PROFILE:
            return fn()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        out = fn()
        end_event.record()
        torch.accelerator.synchronize()
        logger.info(
            "FLASH_ATTN_V100 prefill chunk profile: route=%s q_len=%d "
            "seq_len=%d heads_q=%d heads_kv=%d head_dim=%d block_size=%d "
            "elapsed_ms=%.3f",
            route,
            q_len,
            seq_len,
            heads_q,
            heads_kv,
            head_dim,
            block_size,
            float(start_event.elapsed_time(end_event)),
        )
        return out

    def _flash_v100_prefill_with_prefix(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor | None,
        value: torch.Tensor | None,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Prefill path for prefix/chunked context via gathered contiguous KV."""
        global _logged_dflash_prefix_dump
        global _logged_prefill_prefix_bfla
        global _logged_prefill_prefix_contig_dense
        global _logged_prefill_prefix_splitkv
        global _logged_fp8_prefill_bridge
        global _logged_prefill_compare, _logged_prefill_smallq_decode
        causal = getattr(attn_metadata, "causal", True)
        window_size = self._flash_v100_window_size(causal)
        num_actual_tokens = attn_metadata.num_actual_tokens
        query = query[:num_actual_tokens]
        out_view = output[:num_actual_tokens]

        query_start_loc_cpu = getattr(attn_metadata, "query_start_loc_cpu", None)
        query_start_loc = (
            query_start_loc_cpu
            if query_start_loc_cpu is not None
            else attn_metadata.query_start_loc
        )
        query_start_loc = _normalize_query_start_loc_for_available_tokens(
            query_start_loc,
            int(query.shape[0]),
        )
        seq_lens_cpu = getattr(attn_metadata, "seq_lens_cpu", None)
        seq_lens = seq_lens_cpu if seq_lens_cpu is not None else attn_metadata.seq_lens
        num_seqs = len(query_start_loc) - 1

        key_cache, value_cache = _split_paged_kv_cache(kv_cache)
        block_size = key_cache.shape[1]
        num_kv_heads = key_cache.shape[2]
        head_dim = key_cache.shape[3]
        debug_compare = os.getenv("VLLM_FLASH_V100_DEBUG_PREFILL_COMPARE", "0") == "1"
        dflash_dump = (
            _dflash_prefix_dump_enabled()
            and not _logged_dflash_prefix_dump
            and bool(getattr(layer, "is_dflash_draft_attn", False))
        )

        query_lens = query_start_loc[1:] - query_start_loc[:-1]
        max_query_len = int(query_lens.max().item()) if num_seqs > 0 else 0
        if causal and _ddtree_parent_metadata_requires_branch(
            attn_metadata,
            query_start_loc,
        ):
            return self._flash_v100_ddtree_small_query_prefill_dense(
                layer,
                query,
                key,
                value,
                key_cache,
                value_cache,
                attn_metadata,
                output,
                query_start_loc,
                seq_lens,
            )

        if (
            causal
            and self.use_flash_v100_decode
            and self.smallq_decode_max_query_len > 0
            and max_query_len <= self.smallq_decode_max_query_len
            and (
                self.smallq_decode_max_model_len <= 0
                or getattr(attn_metadata, "max_model_len", 0)
                <= self.smallq_decode_max_model_len
            )
            and not self.use_decode_paged_prefill
        ):
            if not _logged_prefill_smallq_decode:
                logger.info(
                    "FLASH_ATTN_V100 prefix prefill small-query path active "
                    "(paged decode verifier, max_query_len<=%d).",
                    self.smallq_decode_max_query_len,
                )
                _logged_prefill_smallq_decode = True
            return self._flash_v100_small_query_prefill_as_decode(
                layer,
                query,
                key_cache,
                value_cache,
                attn_metadata,
                output,
                query_start_loc,
                seq_lens,
            )

        for i in range(num_seqs):
            start = int(query_start_loc[i].item())
            end = int(query_start_loc[i + 1].item())
            if end <= start:
                continue

            if self.use_flash_v100_prefill_paged:
                q_len = end - start
                seq_len = int(seq_lens[i].item())
                q_seq = query[start:end].unsqueeze(0)
                bfla_block_mask = None
                use_bfla = self._should_use_prefill_bfla(
                    q_len=q_len,
                    seq_len=seq_len,
                    head_dim=head_dim,
                    key_cache=key_cache,
                    causal=causal,
                    window_size=window_size,
                )
                if use_bfla:
                    bfla_block_mask = _build_bfla_block_mask_for_seq(
                        q_seq,
                        key_cache,
                        attn_metadata.block_table[i],
                        seq_len=seq_len,
                        block_size=block_size,
                        mask_block_n=self.prefill_bfla_mask_block_n,
                        softmax_scale=self.scale,
                    )
                contig_dense_kv = None
                contig_dense_kv_bhmd = None
                if bfla_block_mask is None and self._should_use_prefill_contig_dense(
                    q_len=q_len,
                    seq_len=seq_len,
                    head_dim=head_dim,
                    key_cache=key_cache,
                    causal=causal,
                    window_size=window_size,
                ):
                    if (
                        self.prefill_contig_dense_allow_copy
                        and self.flash_attn_bhmd_func is not None
                    ):
                        contig_dense_kv_bhmd = _contiguous_paged_kv_bhmd(
                            key_cache,
                            value_cache,
                            attn_metadata.block_table[i],
                            seq_len,
                            block_size,
                            attn_metadata,
                            i,
                        )
                    if contig_dense_kv_bhmd is None:
                        contig_dense_kv = _contiguous_paged_kv_view(
                            key_cache,
                            value_cache,
                            attn_metadata.block_table[i],
                            seq_len,
                            block_size,
                            attn_metadata,
                            i,
                            self.prefill_contig_dense_allow_copy,
                        )
                use_splitkv = self._should_use_prefill_splitkv(
                    q_len=q_len,
                    seq_len=seq_len,
                    head_dim=head_dim,
                    key_cache=key_cache,
                    causal=causal,
                )
                use_fp8_bridge = self._should_use_fp8_prefill_bridge(
                    q_len=q_len,
                    head_dim=head_dim,
                    key_cache=key_cache,
                    value_cache=value_cache,
                    causal=causal,
                    window_size=window_size,
                )
                if bfla_block_mask is not None:
                    if not _logged_prefill_prefix_bfla:
                        logger.info(
                            "FLASH_ATTN_V100 prefix prefill BFLA sparse path "
                            "active (min_q=%d min_kv=%d mask_block_n=%d "
                            "keep_mass=%.4f local_blocks=%d pool=%s).",
                            self.prefill_bfla_min_q,
                            self.prefill_bfla_min_kv,
                            self.prefill_bfla_mask_block_n,
                            envs.VLLM_FLASH_V100_BFLA_KEEP_MASS,
                            envs.VLLM_FLASH_V100_BFLA_LOCAL_BLOCKS,
                            envs.VLLM_FLASH_V100_BFLA_POOL,
                        )
                        _logged_prefill_prefix_bfla = True
                    _record_route("prefill_prefix_bfla")
                    out_seq = self._run_prefill_paged_call(
                        route="prefill_prefix_bfla",
                        q_len=q_len,
                        seq_len=seq_len,
                        heads_q=query.shape[1],
                        heads_kv=num_kv_heads,
                        head_dim=head_dim,
                        block_size=block_size,
                        fn=partial(
                            self.flash_attn_prefill_paged_bfla,
                            q_seq,
                            key_cache,
                            value_cache,
                            attn_metadata.block_table[i : i + 1],
                            attn_metadata.seq_lens[i : i + 1],
                            bfla_block_mask,
                            self.prefill_bfla_mask_block_n,
                            softmax_scale=self.scale,
                            kv_cache_dtype=self.kv_cache_dtype,
                            k_scale=float(layer._k_scale_float),
                            v_scale=float(layer._v_scale_float),
                            causal=causal,
                            window_size=window_size,
                        ),
                    )
                elif contig_dense_kv_bhmd is not None:
                    if not _logged_prefill_prefix_contig_dense:
                        logger.info(
                            "FLASH_ATTN_V100 prefix prefill contiguous dense "
                            "BHMD path active (min_q=%d min_kv=%d allow_copy=%s).",
                            self.prefill_contig_dense_min_q,
                            self.prefill_contig_dense_min_kv,
                            str(self.prefill_contig_dense_allow_copy),
                        )
                        _logged_prefill_prefix_contig_dense = True
                    k_bhmd, v_bhmd = contig_dense_kv_bhmd
                    q_bhmd = q_seq.permute(0, 2, 1, 3).contiguous()
                    _record_route("prefill_prefix_contig_dense_bhmd")
                    out_bhmd = self._run_prefill_paged_call(
                        route="prefill_prefix_contig_dense_bhmd",
                        q_len=q_len,
                        seq_len=seq_len,
                        heads_q=query.shape[1],
                        heads_kv=num_kv_heads,
                        head_dim=head_dim,
                        block_size=block_size,
                        fn=lambda q_bhmd=q_bhmd,  # type: ignore[misc]
                        k_bhmd=k_bhmd,
                        v_bhmd=v_bhmd: self.flash_attn_bhmd_func(
                            q_bhmd,
                            k_bhmd,
                            v_bhmd,
                            causal=causal,
                            softmax_scale=self.scale,
                            window_size=window_size,
                        ),
                    )
                    out_view[start:end].copy_(out_bhmd.squeeze(0).permute(1, 0, 2))
                    continue
                elif contig_dense_kv is not None:
                    if not _logged_prefill_prefix_contig_dense:
                        logger.info(
                            "FLASH_ATTN_V100 prefix prefill contiguous dense "
                            "path active (min_q=%d min_kv=%d).",
                            self.prefill_contig_dense_min_q,
                            self.prefill_contig_dense_min_kv,
                        )
                        _logged_prefill_prefix_contig_dense = True
                    k_dense, v_dense = contig_dense_kv
                    _record_route("prefill_prefix_contig_dense")
                    out_seq = self._run_prefill_paged_call(
                        route="prefill_prefix_contig_dense",
                        q_len=q_len,
                        seq_len=seq_len,
                        heads_q=query.shape[1],
                        heads_kv=num_kv_heads,
                        head_dim=head_dim,
                        block_size=block_size,
                        fn=lambda q_seq=q_seq,  # type: ignore[misc]
                        k_dense=k_dense,
                        v_dense=v_dense: self.flash_attn_func(
                            q_seq,
                            k_dense,
                            v_dense,
                            causal=causal,
                            softmax_scale=self.scale,
                            window_size=window_size,
                        ),
                    )
                elif use_fp8_bridge:
                    out_seq = self._run_fp8_prefill_bridge(
                        query=q_seq,
                        key_cache=key_cache,
                        value_cache=value_cache,
                        block_table=attn_metadata.block_table[i : i + 1],
                        seq_lens=attn_metadata.seq_lens[i : i + 1],
                        k_scale=float(layer._k_scale_float),
                        v_scale=float(layer._v_scale_float),
                        causal=causal,
                        window_size=window_size,
                    )
                    if out_seq is not None:
                        if not _logged_fp8_prefill_bridge:
                            logger.info(
                                "FLASH_ATTN_V100 FP8 E5M2 prefill bridge "
                                "active (one-pass dequant, shared FP16 page-%d "
                                "workspace).",
                                _FP8_PREFILL_BRIDGE_PAGE_SIZE,
                            )
                            _logged_fp8_prefill_bridge = True
                        _record_route("prefill_prefix_fp8_e5m2_bridge")
                    else:
                        out_seq = self.flash_attn_prefill_paged(
                            q_seq,
                            key_cache,
                            value_cache,
                            attn_metadata.block_table[i : i + 1],
                            attn_metadata.seq_lens[i : i + 1],
                            softmax_scale=self.scale,
                            kv_cache_dtype=self.kv_cache_dtype,
                            k_scale=float(layer._k_scale_float),
                            v_scale=float(layer._v_scale_float),
                            causal=causal,
                            window_size=window_size,
                        )
                elif use_splitkv:
                    if not _logged_prefill_prefix_splitkv:
                        logger.info(
                            "FLASH_ATTN_V100 prefix prefill split-KV path active "
                            "(split_kv_tokens=%d min_q=%d max_q=%d min_kv=%d).",
                            self.prefill_split_kv_tokens,
                            self.prefill_split_kv_min_q,
                            self.prefill_split_kv_max_q,
                            self.prefill_split_kv_min_kv,
                        )
                        _logged_prefill_prefix_splitkv = True
                    _record_route("prefill_prefix_splitkv")
                    out_seq = self._run_prefill_paged_call(
                        route="prefill_prefix_splitkv",
                        q_len=q_len,
                        seq_len=seq_len,
                        heads_q=query.shape[1],
                        heads_kv=num_kv_heads,
                        head_dim=head_dim,
                        block_size=block_size,
                        fn=lambda q_seq=q_seq,  # type: ignore[misc]
                        i=i,
                        seq_len=seq_len: self.flash_attn_prefill_paged_splitkv(
                            q_seq,
                            key_cache,
                            value_cache,
                            attn_metadata.block_table[i : i + 1],
                            attn_metadata.seq_lens[i : i + 1],
                            softmax_scale=self.scale,
                            kv_cache_dtype=self.kv_cache_dtype,
                            k_scale=float(layer._k_scale_float),
                            v_scale=float(layer._v_scale_float),
                            causal=causal,
                            window_size=window_size,
                            split_kv_tokens=self.prefill_split_kv_tokens,
                            max_seq_len_hint=seq_len,
                        ),
                    )
                else:
                    out_seq = self._run_prefill_paged_call(
                        route="prefill_prefix_paged",
                        q_len=q_len,
                        seq_len=seq_len,
                        heads_q=query.shape[1],
                        heads_kv=num_kv_heads,
                        head_dim=head_dim,
                        block_size=block_size,
                        fn=lambda q_seq=q_seq, i=i: self.flash_attn_prefill_paged(  # type: ignore[misc]
                            q_seq,
                            key_cache,
                            value_cache,
                            attn_metadata.block_table[i : i + 1],
                            attn_metadata.seq_lens[i : i + 1],
                            softmax_scale=self.scale,
                            kv_cache_dtype=self.kv_cache_dtype,
                            k_scale=float(layer._k_scale_float),
                            v_scale=float(layer._v_scale_float),
                            causal=causal,
                            window_size=window_size,
                        ),
                    )
                need_dense_debug = (
                    debug_compare and not _logged_prefill_compare
                ) or dflash_dump
                if need_dense_debug:
                    k_cont, v_cont = _extract_contiguous_kv_from_paged_cache(
                        kv_cache=kv_cache,
                        block_table=attn_metadata.block_table[i : i + 1],
                        seq_lens=attn_metadata.seq_lens[i : i + 1],
                        num_kv_heads=num_kv_heads,
                        head_dim=head_dim,
                        block_size=block_size,
                        total_tokens=seq_len,
                    )
                    k_cont, v_cont = _dequantize_fp8_contiguous_kv(
                        k_cont,
                        v_cont,
                        self.kv_cache_dtype,
                        float(layer._k_scale_float),
                        float(layer._v_scale_float),
                    )
                    if bool(getattr(layer, "is_dflash_draft_attn", False)):
                        ref_out = _torch_attention_reference(
                            query[start:end],
                            k_cont,
                            v_cont,
                            causal=causal,
                            softmax_scale=self.scale,
                            window_size=window_size,
                        )
                    else:
                        ref_out = self.flash_attn_func(
                            query[start:end].unsqueeze(0),
                            k_cont.unsqueeze(0),
                            v_cont.unsqueeze(0),
                            causal=causal,
                            softmax_scale=self.scale,
                            window_size=window_size,
                        )
                    diff = (out_seq - ref_out).abs()
                    nan_count = int(torch.isnan(out_seq).sum().item())
                    if debug_compare and not _logged_prefill_compare:
                        logger.warning(
                            "FLASH_ATTN_V100 debug prefix compare: "
                            "query_len=%d seq_len=%d max_diff=%.8f mean_diff=%.8f "
                            "nan_count=%d q_absmax=%.6f k_absmax=%.6f "
                            "v_absmax=%.6f kv_cache_shape=%s key_shape=%s "
                            "key_stride=%s value_stride=%s key_contig=%s "
                            "value_contig=%s",
                            end - start,
                            seq_len,
                            float(diff.max().item()),
                            float(diff.mean().item()),
                            nan_count,
                            float(query[start:end].abs().max().item()),
                            float(k_cont.abs().max().item()),
                            float(v_cont.abs().max().item()),
                            tuple(kv_cache.shape),
                            tuple(key_cache.shape),
                            tuple(key_cache.stride()),
                            tuple(value_cache.stride()),
                            str(key_cache.is_contiguous()),
                            str(value_cache.is_contiguous()),
                        )
                    if dflash_dump:
                        slot_mapping = getattr(attn_metadata, "slot_mapping", None)
                        slot_slice = None
                        cache_k_by_slot = None
                        cache_v_by_slot = None
                        key_input = None
                        value_input = None
                        slot_k_diff = None
                        slot_v_diff = None
                        tail_k_diff = None
                        tail_v_diff = None
                        if (
                            slot_mapping is not None
                            and key is not None
                            and value is not None
                            and key_cache.dtype != torch.uint8
                        ):
                            slot_slice = slot_mapping[start:end].to(torch.long)
                            valid_slots = slot_slice >= 0
                            if bool(valid_slots.all().item()):
                                slot_blocks = torch.div(
                                    slot_slice,
                                    block_size,
                                    rounding_mode="floor",
                                )
                                slot_offsets = torch.remainder(slot_slice, block_size)
                                cache_k_by_slot = key_cache[slot_blocks, slot_offsets]
                                cache_v_by_slot = value_cache[
                                    slot_blocks,
                                    slot_offsets,
                                ]
                                cache_k_by_slot, cache_v_by_slot = (
                                    _dequantize_fp8_contiguous_kv(
                                        cache_k_by_slot,
                                        cache_v_by_slot,
                                        self.kv_cache_dtype,
                                        float(layer._k_scale_float),
                                        float(layer._v_scale_float),
                                    )
                                )
                                key_input = key[start:end]
                                value_input = value[start:end]
                                slot_k_diff = (cache_k_by_slot - key_input).abs()
                                slot_v_diff = (cache_v_by_slot - value_input).abs()
                                tail_start = max(0, seq_len - (end - start))
                                tail_k = k_cont[tail_start:seq_len]
                                tail_v = v_cont[tail_start:seq_len]
                                if tail_k.shape == key_input.shape:
                                    tail_k_diff = (tail_k - key_input).abs()
                                    tail_v_diff = (tail_v - value_input).abs()

                        dump_path = (
                            f"/tmp/flash_v100_dflash_prefix_dump_pid{os.getpid()}"
                            f"_seq{i}.pt"
                        )
                        torch.save(
                            {
                                "layer_name": self._layer_debug_info(layer).get(
                                    "layer_name"
                                ),
                                "causal": causal,
                                "window_size": window_size,
                                "query_start_loc": query_start_loc.detach().cpu(),
                                "seq_lens": seq_lens.detach().cpu(),
                                "attn_seq_lens": attn_metadata.seq_lens.detach().cpu(),
                                "block_table": attn_metadata.block_table[i : i + 1]
                                .detach()
                                .cpu(),
                                "slot_mapping": None
                                if slot_slice is None
                                else slot_slice.detach().cpu(),
                                "query": query[start:end].detach().cpu(),
                                "key_input": None
                                if key_input is None
                                else key_input.detach().cpu(),
                                "value_input": None
                                if value_input is None
                                else value_input.detach().cpu(),
                                "cache_k_by_slot": None
                                if cache_k_by_slot is None
                                else cache_k_by_slot.detach().cpu(),
                                "cache_v_by_slot": None
                                if cache_v_by_slot is None
                                else cache_v_by_slot.detach().cpu(),
                                "k_cont_tail": k_cont[
                                    max(0, seq_len - (end - start)) : seq_len
                                ]
                                .detach()
                                .cpu(),
                                "v_cont_tail": v_cont[
                                    max(0, seq_len - (end - start)) : seq_len
                                ]
                                .detach()
                                .cpu(),
                                "k_cont": k_cont.detach().cpu(),
                                "v_cont": v_cont.detach().cpu(),
                                "out_seq": out_seq.detach().cpu(),
                                "ref_out": ref_out.detach().cpu(),
                                "paged_vs_dense_max": float(diff.max().item()),
                                "paged_vs_dense_mean": float(diff.mean().item()),
                                "slot_k_max": None
                                if slot_k_diff is None
                                else float(slot_k_diff.max().item()),
                                "slot_v_max": None
                                if slot_v_diff is None
                                else float(slot_v_diff.max().item()),
                                "tail_k_max": None
                                if tail_k_diff is None
                                else float(tail_k_diff.max().item()),
                                "tail_v_max": None
                                if tail_v_diff is None
                                else float(tail_v_diff.max().item()),
                                "kv_cache_shape": tuple(kv_cache.shape),
                                "key_cache_shape": tuple(key_cache.shape),
                                "key_cache_stride": tuple(key_cache.stride()),
                                "value_cache_stride": tuple(value_cache.stride()),
                            },
                            dump_path,
                        )
                        logger.warning(
                            "FLASH_ATTN_V100 saved DFlash prefix dump to %s "
                            "(paged_vs_dense_max=%.8f slot_k_max=%s tail_k_max=%s)",
                            dump_path,
                            float(diff.max().item()),
                            "n/a"
                            if slot_k_diff is None
                            else f"{float(slot_k_diff.max().item()):.8f}",
                            "n/a"
                            if tail_k_diff is None
                            else f"{float(tail_k_diff.max().item()):.8f}",
                        )
                        _logged_dflash_prefix_dump = True
                    if debug_compare and not _logged_prefill_compare and nan_count > 0:
                        dump_path = (
                            f"/tmp/flash_v100_prefill_nan_dump_pid{os.getpid()}.pt"
                        )
                        torch.save(
                            {
                                "query": query[start:end].detach().cpu(),
                                "key_cache": key_cache.detach().cpu(),
                                "value_cache": value_cache.detach().cpu(),
                                "block_table": attn_metadata.block_table[i : i + 1]
                                .detach()
                                .cpu(),
                                "seq_lens": attn_metadata.seq_lens[i : i + 1]
                                .detach()
                                .cpu(),
                                "k_cont": k_cont.detach().cpu(),
                                "v_cont": v_cont.detach().cpu(),
                                "out_seq": out_seq.detach().cpu(),
                                "ref_out": ref_out.detach().cpu(),
                            },
                            dump_path,
                        )
                        logger.warning(
                            "FLASH_ATTN_V100 saved failing prefix prefill dump to %s",
                            dump_path,
                        )
                    if debug_compare and not _logged_prefill_compare:
                        _logged_prefill_compare = True
            else:
                seq_len = int(seq_lens[i].item())
                k_cont, v_cont = _extract_contiguous_kv_from_paged_cache(
                    kv_cache=kv_cache,
                    block_table=attn_metadata.block_table[i : i + 1],
                    seq_lens=attn_metadata.seq_lens[i : i + 1],
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    block_size=block_size,
                    total_tokens=seq_len,
                )
                k_cont, v_cont = _dequantize_fp8_contiguous_kv(
                    k_cont,
                    v_cont,
                    self.kv_cache_dtype,
                    float(layer._k_scale_float),
                    float(layer._v_scale_float),
                )

                out_seq = self.flash_attn_func(
                    query[start:end].unsqueeze(0),
                    k_cont.unsqueeze(0),
                    v_cont.unsqueeze(0),
                    causal=causal,
                    softmax_scale=self.scale,
                    window_size=window_size,
                )
            out_view[start:end].copy_(out_seq.squeeze(0))

        return output


class FlashAttnV100Backend(TritonAttentionBackend):
    """Flash Attention V100 Backend."""

    # Keep vLLM unified KV cache update path.
    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_impl_cls():
        return FlashAttnV100Impl

    @staticmethod
    def get_builder_cls():
        return FlashAttnV100MetadataBuilder

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN_V100"

    @staticmethod
    def get_supported_kernel_block_sizes():
        if envs.VLLM_FLASH_V100_KERNEL_BLOCK_SIZE16:
            return [16]
        return TritonAttentionBackend.get_supported_kernel_block_sizes()

    @classmethod
    def supports_non_causal(cls) -> bool:
        # D-Flash uses non-causal decoder attention over the draft query
        # tokens. The V100 backend handles this in the prefill paths by
        # forwarding attn_metadata.causal to FA2/Triton-compatible kernels.
        return True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        # Keep this aligned with the dense prefill kernel dispatch table.
        return [64, 128, 256]
