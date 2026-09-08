# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Admission for the single-request E4M3 FP32 small-Q route."""

import os

import torch


def load_grouped_e4m3_fp32():
    try:
        from flash_attn_v100 import (
            flash_attn_grouped_e4m3_fp32_available,
            flash_attn_grouped_e4m3_fp32_paged,
        )
    except ImportError:
        return None
    if not flash_attn_grouped_e4m3_fp32_available():
        return None
    return flash_attn_grouped_e4m3_fp32_paged


def grouped_e4m3_fp32_allowed(
    instance, query, k, v, table, lengths, metadata, *, out, partition_size_hint
):
    parent_table = getattr(metadata, "block_table", None)
    parent_seq = getattr(metadata, "seq_lens", None)
    if not (
        getattr(instance, "flash_attn_grouped_e4m3_fp32_paged", None) is not None
        and instance.kv_cache_dtype == "fp8_e4m3"
        and instance.use_smallq_decode_xqa
        and partition_size_hint is None
        and not os.environ.get("VLLM_FLASH_V100_DECODE_PARTITION_SIZE")
        and getattr(metadata, "causal", True)
        and instance._flash_v100_window_size(causal=True) == (-1, -1)
        and query.ndim == 3
        and 2 <= query.shape[0] <= 8
        and query.shape[1:] == (6, 256)
        and query.dtype == torch.float16
        and query.is_contiguous()
        and out.shape == query.shape
        and out.dtype == query.dtype
        and out.device == query.device
        and out.is_contiguous()
        and k.ndim == 4
        and k.shape[1] in (800, 848, 1616, 1648, 1728, 3296, 3456)
        and k.shape[2:] == (1, 256)
        and v.shape == k.shape
        and k.dtype == torch.uint8
        and v.dtype == torch.uint8
        and parent_seq is not None
        and parent_seq.shape == (1,)
        and parent_table is not None
        and parent_table.ndim == 2
        and parent_table.shape[0] == 1
        and 0 < parent_table.shape[1] * k.shape[1] <= 266240
        and lengths.shape == (query.shape[0],)
        and table.shape == (query.shape[0], parent_table.shape[1])
    ):
        return False
    # The parent metadata proves that every real query belongs to one request.
    # Device row lengths, including graph padding, define the visible KV prefix.
    return all(
        t.device == query.device and t.dtype == torch.int32 and t.is_contiguous()
        for t in (table, lengths, parent_table, parent_seq)
    ) and all(
        t.device == query.device
        and t.stride(-1) == 1
        and t.data_ptr() % 16 == 0
        and all(s % 8 == 0 for s in t.stride()[:3])
        for t in (k, v)
    )
