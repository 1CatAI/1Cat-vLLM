# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Existing decode workspace and partition policy, extracted without retuning.

The format-specific G6 workspace envelopes are legacy contracts. Unifying their
schedules is a later measured change; relocation does not imply that migration.
"""

from __future__ import annotations

import os

import torch

_DEFAULT_DECODE_PARTITION_SIZE = 256

_VALID_DECODE_PARTITION_SIZES = (256, 512, 1024)


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
    if (
        kv_cache_dtype in ("auto", "float16", "bfloat16")
        and key_cache.dtype == torch.float16
        and key_cache.shape[1] == 784
    ):
        # The exact FP16 page-784 graph contains p256 and p1024 nodes and
        # selects between them from device seq_lens. Plan the p256 workspace
        # envelope once.
        return 256
    if kv_cache_dtype in ("fp8", "fp8_e4m3") and key_cache.dtype == torch.uint8:
        # Plan a p64 workspace envelope. The native G6 path keeps this captured
        # shape while selecting p64/p256 and long wave partitions from device
        # sequence lengths.
        return 64
    if kv_cache_dtype == "fp8_e5m2" and key_cache.dtype == torch.uint8:
        # Plan the largest p256 workspace once. The extension selects p256 or
        # p1024 from device seq_lens, so CUDA graph replay keeps one
        # captured shape while short and long contexts use different kernels.
        # Keep this layout-driven rather than using model-name allowlists.
        return 256
    return None


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
