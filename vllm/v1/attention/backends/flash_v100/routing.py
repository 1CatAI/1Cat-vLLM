# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Route accounting, tracing and decode partition/XQA admission policy."""

from __future__ import annotations

import atexit
import json
import os

import torch

import vllm.envs as envs
from vllm.forward_context import CUDAGRAPH_VARIANT_LONG_CONTEXT
from vllm.logger import init_logger
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionMetadata,
)
from vllm.v1.attention.kv_codecs import (
    FP8_E4M3,
    FP8_E5M2,
    FP16,
    canonical_kv_cache_dtype,
    resolve_kv_codec,
)

logger = init_logger("vllm.v1.attention.backends.flash_attn_v100")


def _batch_context_routing_for_graph_variant(
    routing_enabled: bool,
    graph_variant: int | None,
) -> bool:
    if not routing_enabled:
        return False
    if graph_variant is None:
        # Eager execution can route directly from the live batch and context.
        return True
    return graph_variant == CUDAGRAPH_VARIANT_LONG_CONTEXT


def _batch_context_routing_cache_dtype_supported(cache_dtype: str | None) -> bool:
    """Admit the exact FP8 XQA formats implemented by Flash-V100."""
    codec = resolve_kv_codec(cache_dtype)
    return codec is FP8_E5M2 or (
        codec is FP8_E4M3 and envs.VLLM_FLASH_V100_E4M3_BATCH_XQA
    )


_logged_fp8_kv_prefill = False
_logged_fp8_kv_decode = False
_logged_kv_dtype_contracts: set[str] = set()
_route_summary_registered = False
_route_counts: dict[str, int] = {}
_decode_active_trace_signatures: set[tuple[object, ...]] = set()
_DEFAULT_DECODE_PARTITION_SIZE = 256
_VALID_DECODE_PARTITION_SIZES = (256, 512, 1024)
_DEFAULT_Q4_XQA_MIN_SEQ_LEN = 32768
_DEFAULT_FP8_XQA_MIN_SEQ_LEN = 16384


def _normalize_flash_v100_kv_cache_dtype(kv_cache_dtype: str) -> str:
    # One spelling per codec: "float16" is the extension's "auto" layout and
    # the `fp8` shorthand is E4M3, so every route sees the same format name.
    return canonical_kv_cache_dtype(kv_cache_dtype)


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


def _g6_aligned_page_partition_size_hint(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    kv_cache_dtype: str,
) -> int | None:
    if os.getenv("VLLM_FLASH_V100_DECODE_PARTITION_SIZE") is not None:
        return None
    if os.getenv("VLLM_FLASH_V100_XQA_G6_P1024_SAWTOOTH", "1") == "0":
        return None
    if not (
        query.ndim == 3
        and query.shape[0] == 1
        and query.shape[2] == 256
        and key_cache.ndim == 4
        and key_cache.shape[1] >= _DEFAULT_DECODE_PARTITION_SIZE
        and key_cache.shape[1] % 16 == 0
        and key_cache.shape[2] > 0
        and key_cache.shape[3] == 256
        and query.shape[1] == 6 * key_cache.shape[2]
        and value_cache.shape == key_cache.shape
        and value_cache.dtype == key_cache.dtype
    ):
        return None
    codec = resolve_kv_codec(kv_cache_dtype)
    if codec is None or not codec.stores(key_cache, value_cache):
        return None
    if codec is FP16 and key_cache.shape[1] == 784:
        # The exact FP16 page-784 graph contains p256 and p1024 nodes and
        # selects between them from device seq_lens. Plan the p256 workspace
        # envelope once.
        return 256
    if codec is FP8_E4M3:
        # Plan a p64 workspace envelope. The native G6 path keeps this captured
        # shape while selecting p64/p256 and long wave partitions from device
        # sequence lengths.
        return 64
    if codec is FP8_E5M2:
        # Plan the largest p256 workspace once. The extension selects p256 or
        # p1024 from device seq_lens, so CUDA graph replay keeps one
        # captured shape while short and long contexts use different kernels.
        # Keep this layout-driven rather than using model-name allowlists.
        return 256
    return None


def _log_kv_dtype_contract(kv_cache_dtype: str) -> None:
    if kv_cache_dtype in _logged_kv_dtype_contracts:
        return
    _logged_kv_dtype_contracts.add(kv_cache_dtype)
    if kv_cache_dtype == "fp8":
        logger.warning(
            "SM70 Flash-V100 received an unresolved `fp8` KV-cache dtype and "
            "will interpret it as upstream E4M3. Normal EngineArgs processing "
            "rewrites the SM70 `fp8` shorthand to `fp8_e4m3`; this warning "
            "usually means the backend was constructed directly. KV-cache "
            "dtype is independent of model weight quantization."
        )
    elif kv_cache_dtype == "fp8_e4m3":
        logger.info(
            "SM70 Flash-V100 is using explicitly requested E4M3 KV cache. "
            "The decode route depends on the native extension and tensor "
            "layout. KV-cache dtype is independent of model weight quantization."
        )
    elif kv_cache_dtype == "fp8_e5m2":
        logger.info(
            "SM70 Flash-V100 is using explicit E5M2 KV cache. This controls "
            "KV storage only; model weight quantization is configured "
            "separately."
        )


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
    graph_capture = bool(
        getattr(attn_metadata, "flash_v100_cudagraph_capture", False)
    ) or _is_cuda_graph_capturing(query)
    if graph_capture:
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


def _e4m3_batch_xqa_allowed(query: torch.Tensor) -> bool:
    """Admit GQA6 batches independently of the number of local KV heads."""
    return (
        envs.VLLM_FLASH_V100_E4M3_BATCH_XQA
        and query.ndim == 3
        and query.shape[0] > 1
        and query.shape[1] > 0
        and query.shape[1] % 6 == 0
        and query.shape[2] == 256
    )


def _same_storage(left: torch.Tensor, right: torch.Tensor) -> bool:
    return left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()


def _is_cuda_graph_capturing(tensor: torch.Tensor) -> bool:
    return bool(tensor.is_cuda and torch.cuda.is_current_stream_capturing())


def _route_summary_enabled() -> bool:
    if "VLLM_SM70_DEBUG" in os.environ:
        return "routing" in envs.VLLM_SM70_DEBUG
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
        _route_counts.clear()


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
