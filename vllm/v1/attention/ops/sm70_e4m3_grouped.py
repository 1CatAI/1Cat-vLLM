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
    from vllm.v1.attention.ops.sm70_e4m3_long import wrap_long_attention

    return wrap_long_attention(flash_attn_grouped_e4m3_fp32_paged)


GROUP_ROWS = 8
MAX_GROUPS_PER_CALL = 16


def grouped_e4m3_fp32_groups_allowed(
    instance, query, k, v, group_table, row_lengths, *, causal, out
):
    """Admit explicit request-major groups of ``GROUP_ROWS`` query rows.

    A mixed prefill/decode batch cannot satisfy ``grouped_e4m3_fp32_allowed``
    because the parent metadata also describes the prefill requests. The caller
    instead gathers the verification/decode requests into compact groups: row
    ``8 * g + r`` belongs to ``group_table[g]`` and its visible KV prefix is
    ``row_lengths[8 * g + r]`` (zero for padding). One group is one request, or
    one eight-token slice of a longer verification span; the causal boundary of
    every row is carried by its own length, so slicing a span is exact.
    """
    groups = group_table.shape[0] if group_table.ndim == 2 else 0
    rows = query.shape[0] if query.ndim == 3 else 0
    if not 1 <= groups <= MAX_GROUPS_PER_CALL or rows != groups * GROUP_ROWS:
        return False
    if groups > 1:
        from flash_attn_v100 import flash_attn_grouped_e4m3_fp32_available

        if not flash_attn_grouped_e4m3_fp32_available(6):
            return False
    return bool(
        getattr(instance, "flash_attn_grouped_e4m3_fp32_paged", None) is not None
        and instance.kv_cache_dtype == "fp8_e4m3"
        and instance.use_smallq_decode_xqa
        and not os.environ.get("VLLM_FLASH_V100_DECODE_PARTITION_SIZE")
        and causal
        and instance._flash_v100_window_size(causal=True) == (-1, -1)
        and query.shape[1] > 0
        and query.shape[2] == 256
        and query.dtype == torch.float16
        and query.is_contiguous()
        and out.shape == query.shape
        and out.dtype == query.dtype
        and out.device == query.device
        and out.is_contiguous()
        and k.ndim == 4
        and k.shape[1] > 0
        and k.shape[1] % 16 == 0
        and k.shape[2] * 6 == query.shape[1]
        and k.shape[3] == 256
        and v.shape == k.shape
        and k.dtype == torch.uint8
        and v.dtype == torch.uint8
        and 0 < group_table.shape[1] * k.shape[1] <= 266240
        and row_lengths.shape == (rows,)
        and all(
            t.device == query.device and t.dtype == torch.int32 and t.is_contiguous()
            for t in (group_table, row_lengths)
        )
        and all(
            t.device == query.device
            and t.stride(-1) == 1
            and t.data_ptr() % 16 == 0
            and all(s % 8 == 0 for s in t.stride()[:3])
            for t in (k, v)
        )
    )


def grouped_e4m3_fp32_allowed(
    instance, query, k, v, table, lengths, metadata, *, out, partition_size_hint
):
    parent_table = getattr(metadata, "block_table", None)
    parent_seq = getattr(metadata, "seq_lens", None)
    batch_size = (
        parent_table.shape[0]
        if parent_table is not None and parent_table.ndim == 2
        else 0
    )
    query_rows = query.shape[0] if query.ndim == 3 else 0
    if batch_size > 1:
        from flash_attn_v100 import flash_attn_grouped_e4m3_fp32_available

        batch_supported = flash_attn_grouped_e4m3_fp32_available(6)
    else:
        batch_supported = True
    if not (
        getattr(instance, "flash_attn_grouped_e4m3_fp32_paged", None) is not None
        and batch_supported
        and instance.kv_cache_dtype == "fp8_e4m3"
        and instance.use_smallq_decode_xqa
        and partition_size_hint is None
        and not os.environ.get("VLLM_FLASH_V100_DECODE_PARTITION_SIZE")
        and getattr(metadata, "causal", True)
        and instance._flash_v100_window_size(causal=True) == (-1, -1)
        and query.ndim == 3
        and (
            (batch_size == 1 and 2 <= query_rows <= 8)
            or (2 <= batch_size <= 16 and query_rows == batch_size * 8)
        )
        and query.shape[1] > 0
        and query.shape[2] == 256
        and query.dtype == torch.float16
        and query.is_contiguous()
        and out.shape == query.shape
        and out.dtype == query.dtype
        and out.device == query.device
        and out.is_contiguous()
        and k.ndim == 4
        and k.shape[1] > 0
        and k.shape[1] % 16 == 0
        and k.shape[2] * 6 == query.shape[1]
        and k.shape[3] == 256
        and v.shape == k.shape
        and k.dtype == torch.uint8
        and v.dtype == torch.uint8
        and parent_seq is not None
        and parent_seq.shape == (batch_size,)
        and parent_table is not None
        and parent_table.ndim == 2
        and parent_table.shape[0] == batch_size
        and 0 < parent_table.shape[1] * k.shape[1] <= 266240
        and lengths.shape == (query_rows,)
        and table.shape == (query_rows, parent_table.shape[1])
    ):
        return False
    # Parent metadata maps each eight-row group to one request. Device row
    # lengths, including graph padding, define each query's visible KV prefix.
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
