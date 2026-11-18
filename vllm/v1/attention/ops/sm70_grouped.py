# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Codec-parameterized admission for the SM70 grouped attention family."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Literal

import torch

from vllm.config import get_current_vllm_config_or_none
from vllm.config.execution_policy import graph_policy
from vllm.v1.attention.kv_codecs import (
    FP8_E4M3,
    FP16,
    INT8_PER_TOKEN_HEAD,
    KVCodec,
    resolve_kv_codec,
)

GROUP_ROWS = 8
MAX_CONTEXT = 266240


def model_context_limit(capacity: int | None = None, *, fallback: int) -> int:
    """Resolve a served window within the native int32 page-arithmetic bound."""
    if capacity is None:
        config = get_current_vllm_config_or_none()
        capacity = (
            config.model_config.max_model_len
            if config is not None and config.model_config is not None
            else fallback
        )
    if type(capacity) is not int or not 0 < capacity <= 2**31 - 4096:
        raise ValueError("SM70 grouped context must fit positive int32 page arithmetic")
    return capacity


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
        page_sizes=(800, 832),
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
    INT8_PER_TOKEN_HEAD: GroupedContract(
        codec=INT8_PER_TOKEN_HEAD,
        operator_attr="flash_attn_grouped_e4m3_fp32_paged",
        operator_name="sm70_grouped_int8_pth_fwd",
        loader_module="vllm.v1.attention.ops.sm70_int8_grouped",
        loader_name="load_grouped_int8_pth_fp32",
        batch_revision=None,
        max_groups=16,
        explicit_groups=True,
        page_sizes=None,
        page_alignment=16,
        query_heads=None,
        require_cuda=True,
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
    operator = getattr(instance, contract.operator_attr, None)
    request_query_rows = getattr(operator, "request_query_rows", (GROUP_ROWS,))
    max_groups = getattr(operator, "max_batch_size", contract.max_groups)
    if explicit and (
        not 1 <= groups <= contract.max_groups or rows != groups * GROUP_ROWS
    ):
        return "query_rows"
    # Preserve the native revision query, including its legacy eager evaluation
    # for batched request metadata before the operator/policy guards.
    batch_supported = True
    if (
        contract.batch_revision is not None
        and groups > 1
        and not hasattr(operator, "request_query_rows")
    ):
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
        or graph_policy().decode_partition_size
    ):
        return "decode_policy"
    if not (causal if explicit else getattr(metadata, "causal", True)) or (
        instance._flash_v100_window_size(causal=True) != (-1, -1)
    ):
        return "causal_or_window"
    if parent is None or parent.ndim != 2:
        return "request_metadata"
    if not 1 <= groups <= max_groups:
        return "request_group_count"
    if not (
        (groups == 1 and 2 <= rows <= GROUP_ROWS)
        or (groups > 1 and rows % groups == 0 and rows // groups in request_query_rows)
    ):
        return "query_rows"
    if (
        not explicit
        and groups > 1
        and (hasattr(operator, "request_query_rows") or rows != groups * GROUP_ROWS)
    ):
        starts = getattr(metadata, "query_start_loc_cpu", None)
        # Runtime-packed batches must retain request boundaries. CUDA graph
        # capture supplies these CPU starts before replay; ragged eager input
        # keeps the existing general fallback until explicitly packed.
        # Explicit groups carry one table row per group and no request
        # metadata: their caller has already packed them.
        if (
            starts is None
            or starts.device.type != "cpu"
            or starts.numel() < groups + 1
            or int(starts[0]) != 0
            or int(starts[groups]) != rows
            or not bool(
                (starts[1 : groups + 1] - starts[:groups] == rows // groups).all()
            )
        ):
            return "request_query_layout"
    multi_kv = getattr(operator, "multi_kv_heads", False)
    runtime_pages = getattr(operator, "runtime_pages", False)
    if (
        q.shape[1:] != (contract.query_heads, 256)
        if (contract.query_heads is not None and not multi_kv)
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
        or (
            contract.page_sizes is not None
            and not runtime_pages
            and k.shape[1] not in contract.page_sizes
        )
        or k.shape[2] * 6 != q.shape[1]
        or k.shape[3] != 256
        or v.shape != k.shape
    ):
        return "kv_shape_or_unmeasured_page"
    if contract.early_cache_dtype and not codec.stores(k, v):
        return "kv_layout"
    context_limit = MAX_CONTEXT
    if parent.shape[1] * k.shape[1] > MAX_CONTEXT:
        capacity = _import_symbol(contract.loader_module, "grouped_context_limit")()
        if capacity > MAX_CONTEXT:
            context_limit = min(
                ((capacity + k.shape[1] - 1) // k.shape[1]) * k.shape[1],
                ((2**31 - 4096) // k.shape[1]) * k.shape[1],
            )
    if not 0 < parent.shape[1] * k.shape[1] <= context_limit:
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
    target_codec = (
        INT8_PER_TOKEN_HEAD
        if resolve_kv_codec(getattr(instance, "kv_cache_dtype", None))
        is INT8_PER_TOKEN_HEAD
        else FP8_E4M3
    )
    return (
        grouped_fp32_reason(
            target_codec,
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
    target_codec = (
        INT8_PER_TOKEN_HEAD
        if resolve_kv_codec(getattr(instance, "kv_cache_dtype", None))
        is INT8_PER_TOKEN_HEAD
        else FP8_E4M3
    )
    return (
        grouped_fp32_reason(
            target_codec,
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


def load_grouped_int8_pth_fp32():
    return load_grouped_fp32(INT8_PER_TOKEN_HEAD)


def clear_grouped_fp16_workspaces():
    from vllm.v1.attention.ops import sm70_fp16_grouped

    sm70_fp16_grouped.clear_grouped_fp16_workspaces()


def try_eager_grouped_fp32(
    grouped_admission: Any,
    grouped_op: Any,
    scale: float,
    layer: torch.nn.Module,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    attn_metadata: Any,
    *,
    out: torch.Tensor,
    max_seq_len_hint: int | None,
    partition_size_hint: int | None,
) -> bool:
    from contextlib import contextmanager, suppress
    from dataclasses import replace
    from types import SimpleNamespace

    from vllm.forward_context import (
        BatchDescriptor,
        get_forward_context,
        is_forward_context_available,
    )

    parent_bt = getattr(attn_metadata, "block_table", None)
    parent_seq = getattr(attn_metadata, "seq_lens", None)
    if (
        not query.is_cuda
        or not getattr(grouped_admission, "use_smallq_decode_xqa", False)
        or not bool(getattr(attn_metadata, "causal", True))
        or grouped_admission._flash_v100_window_size(causal=True) != (-1, -1)
        or parent_bt is None
        or parent_bt.ndim != 2
        or key_cache.ndim != 4
        or key_cache.shape[1] <= 0
        or query.ndim != 3
        or query.shape[2] != 256
        or key_cache.shape[2] * 6 != query.shape[1]
    ):
        return False
    page = int(key_cache.shape[1])
    eff_max_seq = int(max_seq_len_hint or getattr(attn_metadata, "max_seq_len", 0) or 0)
    table_for_group = parent_bt
    row_bt = block_table
    eager_bucket: int | None = None
    if eff_max_seq > 0:
        cols = min(parent_bt.shape[1], max(1, (eff_max_seq + page - 1) // page))
        if cols * page > MAX_CONTEXT:
            eager_bucket = cols * page
        if cols < parent_bt.shape[1]:
            meta_any: Any = attn_metadata
            cached_bt = getattr(meta_any, "_flash_v100_trimmed_bt", None)
            if cached_bt is None or cached_bt[0] != cols:
                trimmed_parent = parent_bt[:, :cols].contiguous()
                trimmed_rows = (
                    block_table[:, :cols].contiguous()
                    if block_table is not None and block_table.ndim == 2
                    else block_table
                )
                cached_bt = (cols, trimmed_parent, trimmed_rows)
                with suppress(AttributeError):
                    meta_any._flash_v100_trimmed_bt = cached_bt
            _, table_for_group, row_bt = cached_bt

    @contextmanager
    def _eager_bucket_scope(tokens: int):
        if eager_bucket is None or not is_forward_context_available():
            yield
            return
        ctx = get_forward_context()
        prev = ctx.batch_descriptor
        if prev is not None and prev.attention_context_bucket is not None:
            yield
            return
        ctx.batch_descriptor = (
            replace(prev, attention_context_bucket=eager_bucket)
            if prev is not None
            else BatchDescriptor(
                num_tokens=tokens, attention_context_bucket=eager_bucket
            )
        )
        try:
            yield
        finally:
            ctx.batch_descriptor = prev

    if table_for_group is not parent_bt or eager_bucket is not None:
        meta_proxy = SimpleNamespace(
            block_table=table_for_group,
            seq_lens=parent_seq,
            causal=getattr(attn_metadata, "causal", True),
            query_start_loc_cpu=getattr(attn_metadata, "query_start_loc_cpu", None),
        )
        with _eager_bucket_scope(int(query.shape[0])):
            if grouped_e4m3_fp32_allowed(
                grouped_admission,
                query,
                key_cache,
                value_cache,
                row_bt,
                seq_lens,
                meta_proxy,
                out=out,
                partition_size_hint=partition_size_hint,
            ):
                grouped_op(
                    query,
                    key_cache,
                    value_cache,
                    table_for_group,
                    seq_lens,
                    out=out,
                    softmax_scale=scale,
                    k_scale=float(layer._k_scale_float),
                    v_scale=float(layer._v_scale_float),
                )
                return True

    if partition_size_hint is not None:
        return False
    groups = int(table_for_group.shape[0])
    rows = int(query.shape[0])
    qsl_cpu = getattr(attn_metadata, "query_start_loc_cpu", None)
    if (
        groups < 1
        or qsl_cpu is None
        or qsl_cpu.device.type != "cpu"
        or qsl_cpu.numel() < groups + 1
        or seq_lens.shape != (rows,)
    ):
        return False
    meta_plan_holder: Any = attn_metadata
    cached_plan: tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] | None
    if hasattr(meta_plan_holder, "_flash_v100_ragged_smallq_plan"):
        cached_plan = meta_plan_holder._flash_v100_ragged_smallq_plan
    else:
        qsl = qsl_cpu[: groups + 1].tolist()
        if qsl[0] != 0 or qsl[groups] != rows:
            cached_plan = None
        else:
            q_lens = [qsl[i + 1] - qsl[i] for i in range(groups)]
            if not q_lens or min(q_lens) < 1:
                cached_plan = None
            else:
                dst: list[int] = []
                group_req: list[int] = []
                for i, q_len in enumerate(q_lens):
                    base = len(group_req) * GROUP_ROWS
                    group_req.extend([i] * -(-q_len // GROUP_ROWS))
                    dst.extend(range(base, base + q_len))
                dst_idx = torch.tensor(
                    dst,
                    dtype=torch.int64,
                    device="cpu",
                    pin_memory=query.device.type == "cuda",
                ).to(query.device, non_blocking=True)
                group_req_idx = (
                    torch.tensor(
                        group_req,
                        dtype=torch.int64,
                        device="cpu",
                        pin_memory=query.device.type == "cuda",
                    ).to(query.device, non_blocking=True)
                    if len(group_req) != groups
                    else None
                )
                lengths_pad = torch.zeros(
                    len(group_req) * GROUP_ROWS,
                    dtype=torch.int32,
                    device=seq_lens.device,
                )
                lengths_pad.index_copy_(0, dst_idx, seq_lens)
                cached_plan = (dst_idx, lengths_pad, group_req_idx)
        with suppress(AttributeError):
            meta_plan_holder._flash_v100_ragged_smallq_plan = cached_plan
    if cached_plan is None:
        return False
    dst_idx, lengths_pad, group_req_idx = cached_plan
    explicit_table = (
        table_for_group[:groups]
        if group_req_idx is None
        else table_for_group.index_select(0, group_req_idx)
    )
    num_explicit_groups = int(explicit_table.shape[0])
    total_rows = num_explicit_groups * GROUP_ROWS
    q_pad = query.new_zeros((total_rows, query.shape[1], query.shape[2]))
    out_pad = torch.empty_like(q_pad)
    causal = bool(getattr(attn_metadata, "causal", True))
    chunks = [
        (g0, min(g0 + MAX_GROUPS_PER_CALL, num_explicit_groups))
        for g0 in range(0, num_explicit_groups, MAX_GROUPS_PER_CALL)
    ]
    with _eager_bucket_scope(total_rows):
        for g0, g1 in chunks:
            r0, r1 = g0 * GROUP_ROWS, g1 * GROUP_ROWS
            if not grouped_e4m3_fp32_groups_allowed(
                grouped_admission,
                q_pad[r0:r1],
                key_cache,
                value_cache,
                explicit_table[g0:g1],
                lengths_pad[r0:r1],
                causal=causal,
                out=out_pad[r0:r1],
            ):
                return False
        q_pad.index_copy_(0, dst_idx, query)
        k_scale = float(layer._k_scale_float)
        v_scale = float(layer._v_scale_float)
        for g0, g1 in chunks:
            r0, r1 = g0 * GROUP_ROWS, g1 * GROUP_ROWS
            grouped_op(
                q_pad[r0:r1],
                key_cache,
                value_cache,
                explicit_table[g0:g1],
                lengths_pad[r0:r1],
                out=out_pad[r0:r1],
                softmax_scale=scale,
                k_scale=k_scale,
                v_scale=v_scale,
            )
    out.copy_(out_pad.index_select(0, dst_idx))
    return True
