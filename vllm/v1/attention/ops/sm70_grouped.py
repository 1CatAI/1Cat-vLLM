# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Codec-parameterized admission for the SM70 grouped attention family."""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from typing import Any, Literal

import torch

from vllm import envs
from vllm.v1.attention.kv_codecs import FP8_E4M3, FP16, KVCodec, resolve_kv_codec

GROUP_ROWS = 8
MAX_CONTEXT = 266240


@dataclass(frozen=True)
class GroupedContract:
    codec: KVCodec
    operator_attr: str
    operator_name: str
    loader_module: str
    loader_name: str
    batch_revision: tuple[str, str, int] | None
    max_groups: int
    explicit_groups: bool
    page_sizes: tuple[int, ...] | None
    page_alignment: int
    query_heads: int | None
    require_cuda: bool
    query_alignment: int
    raw_partition_policy: bool
    early_query_layout: bool
    early_cache_dtype: bool
    metadata_before_cache_layout: bool


GROUPED_CONTRACTS = {
    FP16: GroupedContract(
        codec=FP16,
        operator_attr="flash_attn_grouped_fp16_fp32_paged",
        operator_name="sm70_grouped_fp16_fwd",
        loader_module="vllm.v1.attention.ops.sm70_fp16_grouped",
        loader_name="load_grouped_fp16_fp32",
        batch_revision=None,
        max_groups=4,
        explicit_groups=False,
        page_sizes=(832,),
        page_alignment=1,
        query_heads=6,
        require_cuda=True,
        query_alignment=16,
        raw_partition_policy=False,
        early_query_layout=False,
        early_cache_dtype=False,
        metadata_before_cache_layout=False,
    ),
    FP8_E4M3: GroupedContract(
        codec=FP8_E4M3,
        operator_attr="flash_attn_grouped_e4m3_fp32_paged",
        operator_name="grouped_e4m3_fp32",
        loader_module="vllm.v1.attention.ops.sm70_e4m3_grouped",
        loader_name="load_grouped_e4m3_fp32",
        batch_revision=("flash_attn_v100", "flash_attn_grouped_e4m3_fp32_available", 6),
        max_groups=16,
        explicit_groups=True,
        page_sizes=None,
        page_alignment=16,
        query_heads=None,
        require_cuda=False,
        query_alignment=0,
        raw_partition_policy=True,
        early_query_layout=True,
        early_cache_dtype=True,
        metadata_before_cache_layout=True,
    ),
}


def _query_layout_reason(
    contract: GroupedContract, q: torch.Tensor, out: torch.Tensor
) -> str | None:
    tensors: tuple[torch.Tensor, ...] = (q, out)
    # The legacy E4M3 admission validates query layout before output shape;
    # FP16 reports output shape before either tensor layout.
    if contract.early_query_layout:
        tensors = (out,)
        if q.dtype != torch.float16 or not q.is_contiguous():
            return "query_or_output_layout"
    if out.shape != q.shape:
        return "output_shape"
    for tensor in tensors:
        if (
            tensor.dtype != torch.float16
            or tensor.device != q.device
            or not tensor.is_contiguous()
            or (
                contract.query_alignment
                and tensor.data_ptr() % contract.query_alignment
            )
        ):
            return "query_or_output_layout"
    return None


def _metadata_layout_valid(q, tensors):
    return all(
        t.dtype == torch.int32 and t.device == q.device and t.is_contiguous()
        for t in tensors
    )


def _cache_layout_valid(contract, q, k, v):
    return all(
        (contract.early_cache_dtype or t.dtype == contract.codec.storage_dtype)
        and t.device == q.device
        and t.stride(-1) == 1
        and t.data_ptr() % 16 == 0
        and all(s % 8 == 0 for s in t.stride()[:3])
        for t in (k, v)
    )


def _import_symbol(module_name: str, name: str):
    module = importlib.import_module(module_name)
    try:
        return getattr(module, name)
    except AttributeError as exc:
        # Keep the exception category of the former from-import capability query.
        raise ImportError(f"cannot import name {name!r} from {module_name!r}") from exc


def grouped_fp32_reason(
    codec: KVCodec | None,
    instance: Any,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    table: torch.Tensor,
    lengths: torch.Tensor,
    metadata: Any = None,
    *,
    out: torch.Tensor,
    partition_size_hint: int | None = None,
    layout: Literal["request_rows", "explicit_groups"] = "request_rows",
    causal: bool = True,
) -> str | None:
    """Return the first rejected contract, or None for the admitted native route.

    Request rows have one table row per query row plus parent request metadata.
    Explicit groups already carry one table row per padded eight-query group.
    Storage/layout limits and validation order preserve each original admission.
    """
    if codec is None:
        return "codec_unsupported"
    contract = GROUPED_CONTRACTS.get(codec)
    if contract is None:
        return "codec_unsupported"
    codec = contract.codec
    explicit = layout == "explicit_groups"
    if explicit and not contract.explicit_groups:
        return "layout_unsupported"
    parent = table if explicit else getattr(metadata, "block_table", None)
    parent_seq = None if explicit else getattr(metadata, "seq_lens", None)
    groups = parent.shape[0] if parent is not None and parent.ndim == 2 else 0
    rows = q.shape[0] if q.ndim == 3 else 0
    if explicit and (
        not 1 <= groups <= contract.max_groups or rows != groups * GROUP_ROWS
    ):
        return "query_rows"
    # Preserve the native revision query, including its legacy eager evaluation
    # for batched request metadata before the operator/policy guards.
    batch_supported = True
    if contract.batch_revision is not None and groups > 1:
        module_name, function_name, revision = contract.batch_revision
        capability = _import_symbol(module_name, function_name)
        batch_supported = capability(revision)
    if getattr(instance, contract.operator_attr, None) is None:
        return f"operator_missing:{contract.operator_name}"
    if not batch_supported:
        return "operator_batch_revision"
    if resolve_kv_codec(instance.kv_cache_dtype) is not codec:
        return "kv_dtype"
    if (
        not instance.use_smallq_decode_xqa
        or partition_size_hint is not None
        or (
            os.environ.get("VLLM_FLASH_V100_DECODE_PARTITION_SIZE")
            if contract.raw_partition_policy
            else envs.VLLM_FLASH_V100_DECODE_PARTITION_SIZE
        )
    ):
        return "decode_policy"
    if not (causal if explicit else getattr(metadata, "causal", True)) or (
        instance._flash_v100_window_size(causal=True) != (-1, -1)
    ):
        return "causal_or_window"
    if parent is None or parent.ndim != 2:
        return "request_metadata"
    if not 1 <= groups <= contract.max_groups:
        return "request_group_count"
    if not (
        (groups == 1 and 2 <= rows <= GROUP_ROWS)
        or (groups > 1 and rows == groups * GROUP_ROWS)
    ):
        return "query_rows"
    if (
        q.shape[1:] != (contract.query_heads, 256)
        if contract.query_heads is not None
        else (q.shape[1] <= 0 or q.shape[2] != 256)
    ):
        return "head_shape"
    if contract.early_query_layout:
        reason = _query_layout_reason(contract, q, out)
        if reason is not None:
            return reason
    if (
        k.ndim != 4
        or k.shape[1] <= 0
        or k.shape[1] % contract.page_alignment
        or (contract.page_sizes is not None and k.shape[1] not in contract.page_sizes)
        or k.shape[2] * 6 != q.shape[1]
        or k.shape[3] != 256
        or v.shape != k.shape
    ):
        return "kv_shape_or_unmeasured_page"
    if contract.early_cache_dtype and not codec.stores(k, v):
        return "kv_layout"
    if not 0 < parent.shape[1] * k.shape[1] <= MAX_CONTEXT:
        return "context_capacity"
    if not explicit and (parent_seq is None or parent_seq.shape != (groups,)):
        return "request_metadata"
    if lengths.shape != (rows,) or (
        not explicit and table.shape != (rows, parent.shape[1])
    ):
        return "row_metadata"
    if contract.require_cuda and q.device.type != "cuda":
        return "device_not_cuda"
    if not contract.early_query_layout:
        reason = _query_layout_reason(contract, q, out)
        if reason is not None:
            return reason
    metadata_tensors = (
        (table, lengths) if explicit else (table, lengths, parent, parent_seq)
    )
    if contract.metadata_before_cache_layout and not _metadata_layout_valid(
        q, metadata_tensors
    ):
        return "metadata_layout"
    if not _cache_layout_valid(contract, q, k, v):
        return "kv_layout"
    if not contract.metadata_before_cache_layout and not _metadata_layout_valid(
        q, metadata_tensors
    ):
        return "metadata_layout"
    return None


# Compatibility call signatures; internal users import these from this owner.
def grouped_fp16_fp32_reason(
    instance, q, k, v, table, lengths, metadata, *, out, partition_size_hint
):
    return grouped_fp32_reason(
        FP16,
        instance,
        q,
        k,
        v,
        table,
        lengths,
        metadata,
        out=out,
        partition_size_hint=partition_size_hint,
    )


def grouped_e4m3_fp32_allowed(
    instance, query, k, v, table, lengths, metadata, *, out, partition_size_hint
):
    return (
        grouped_fp32_reason(
            FP8_E4M3,
            instance,
            query,
            k,
            v,
            table,
            lengths,
            metadata,
            out=out,
            partition_size_hint=partition_size_hint,
        )
        is None
    )


def grouped_e4m3_fp32_groups_allowed(
    instance, query, k, v, group_table, row_lengths, *, causal, out
):
    return (
        grouped_fp32_reason(
            FP8_E4M3,
            instance,
            query,
            k,
            v,
            group_table,
            row_lengths,
            layout="explicit_groups",
            causal=causal,
            out=out,
        )
        is None
    )


MAX_GROUPS_PER_CALL = GROUPED_CONTRACTS[FP8_E4M3].max_groups
FP16_MAX_GROUPS = GROUPED_CONTRACTS[FP16].max_groups


def load_grouped_fp32(codec: KVCodec):
    contract = GROUPED_CONTRACTS.get(codec)
    if contract is None:
        return None
    module = importlib.import_module(contract.loader_module)
    return getattr(module, contract.loader_name)()


def load_grouped_fp16_fp32():
    return load_grouped_fp32(FP16)


def load_grouped_e4m3_fp32():
    return load_grouped_fp32(FP8_E4M3)


def clear_grouped_fp16_workspaces():
    from vllm.v1.attention.ops import sm70_fp16_grouped

    sm70_fp16_grouped.clear_grouped_fp16_workspaces()
