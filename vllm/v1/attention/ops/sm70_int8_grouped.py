# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native SM70 per-token-head INT8 grouped and scalar-tail attention loaders."""

from __future__ import annotations

import torch

from vllm.config.execution_policy import flash_v100_options
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.v1.attention.kv_codecs import (
    INT8_PER_TOKEN_HEAD,
    per_token_head_scales_for_kv_views,
    resolve_kv_codec,
)
from vllm.v1.attention.ops.sm70_grouped import MAX_CONTEXT
from vllm.v1.attention.ops.sm70_grouped_long import (
    _WORKSPACES,
    BUILTIN_MANIFEST,
    BUILTIN_QUERY_ROWS,
    builtin_context_limit,
    builtin_dynamic_context,
    context_table_supported,
    long_attention_graph_contract,
    long_attention_max_batch_size,
)

logger = init_logger("vllm.v1.attention.ops.sm70_int8_grouped")

BUILTIN_INT8_OP = "sm70_grouped_int8_pth_fwd"
BUILTIN_INT8_SCALAR_OP = "sm70_scalar_int8_pth_fwd"
INT8_SCALAR_MIN_PAGE_SIZE = 16

_builtin_int8_cache: dict[str, object] = {}
_int8_scalar_tail_cache: dict[tuple[torch.device, int], object] = {}


def grouped_context_limit() -> int:
    """Resolve the served context window for per-token-head INT8 grouped graphs."""
    descriptor = (
        get_forward_context().batch_descriptor
        if is_forward_context_available()
        else None
    )
    bucket = getattr(descriptor, "attention_context_bucket", None)
    if bucket is not None and bucket > 0 and builtin_dynamic_context():
        capacity, _ = long_attention_graph_contract(bucket)
        if capacity is not None and bucket <= capacity:
            return capacity
    return MAX_CONTEXT


def builtin_int8_pth_attention():
    if "grouped" not in _builtin_int8_cache:
        try:
            from vllm.vllm_flash_attn.flash_attn_interface import (
                ensure_fa2_library_loaded,
            )

            ensure_fa2_library_loaded()
            _builtin_int8_cache["grouped"] = getattr(
                torch.ops._vllm_fa2_C, BUILTIN_INT8_OP
            )
        except (AttributeError, ImportError):
            _builtin_int8_cache["grouped"] = None
    return _builtin_int8_cache["grouped"]


def builtin_int8_scalar_tail_attention():
    if "scalar" not in _builtin_int8_cache:
        try:
            from vllm.vllm_flash_attn.flash_attn_interface import (
                ensure_fa2_library_loaded,
            )

            ensure_fa2_library_loaded()
            _builtin_int8_cache["scalar"] = getattr(
                torch.ops._vllm_fa2_C, BUILTIN_INT8_SCALAR_OP
            )
        except (AttributeError, ImportError):
            _builtin_int8_cache["scalar"] = None
    return _builtin_int8_cache["scalar"]


def int8_scalar_tail_available() -> bool:
    return builtin_int8_scalar_tail_attention() is not None


def load_grouped_int8_pth_fp32():
    from vllm.config import get_current_vllm_config_or_none

    operator = builtin_int8_pth_attention()
    if operator is None or not flash_v100_options().value("e4m3_long_enabled"):
        return None
    config = get_current_vllm_config_or_none()
    model = getattr(config, "model_config", None)
    model_capacity = getattr(model, "max_model_len", None)
    capability = builtin_context_limit(model_capacity)
    max_batch_size = long_attention_max_batch_size(
        dict(BUILTIN_MANIFEST, max_context=capability)
    )
    request_query_rows = BUILTIN_QUERY_ROWS

    def run(
        q, k, v, table, row_lengths, *, out, softmax_scale, k_scale=1.0, v_scale=1.0
    ):
        del k_scale, v_scale
        descriptor = (
            get_forward_context().batch_descriptor
            if is_forward_context_available()
            else None
        )
        bucket = getattr(descriptor, "attention_context_bucket", None)
        batch_size = int(table.shape[0]) if table.ndim == 2 else 0
        if bucket is None and table.ndim == 2 and k.ndim == 4:
            bucket = model_capacity or int(table.shape[1]) * int(k.shape[1])
        stream = torch.cuda.current_stream(q.device).cuda_stream
        key = ("int8_pth_v1", bucket, 80, q.device, stream)
        bank = _WORKSPACES.setdefault(key, [])
        panels = batch_size * int(k.shape[2])
        for partial, lse in bank:
            if partial.shape[0] >= panels:
                break
        else:
            partial = torch.empty(
                (panels, 80, 8, 6, 256), dtype=torch.float32, device=q.device
            )
            lse = torch.empty(
                (panels, 80, 8, 6, 2), dtype=torch.float32, device=q.device
            )
            bank.append((partial, lse))
        if panels == 1:
            partial_view, lse_view = partial[0], lse[0]
        else:
            partial_view, lse_view = partial[:panels], lse[:panels]
        k_scales, v_scales = per_token_head_scales_for_kv_views(k, v)
        return operator(
            q,
            k,
            v,
            k_scales,
            v_scales,
            out,
            table,
            row_lengths,
            partial_view,
            lse_view,
            float(softmax_scale),
        )

    run.__dict__.update(
        request_query_rows=request_query_rows,
        max_batch_size=max_batch_size,
        multi_kv_heads=True,
        runtime_pages=True,
    )
    return run


def load_int8_scalar_tail_attention(device: torch.device):
    from vllm.config import get_current_vllm_config_or_none

    operator = builtin_int8_scalar_tail_attention()
    grouped_operator = builtin_int8_pth_attention()
    if operator is None and grouped_operator is None:
        return None
    config = get_current_vllm_config_or_none()
    model = getattr(config, "model_config", None)
    model_capacity = getattr(model, "max_model_len", None)
    capability = builtin_context_limit(model_capacity, scalar=True)
    cache_key = (device, capability)
    if cache_key in _int8_scalar_tail_cache:
        return _int8_scalar_tail_cache[cache_key]
    partitions = (capability + 1023) // 1024
    buffers = (
        (
            torch.empty((1, 6, partitions, 256), dtype=torch.float32, device=device),
            torch.empty((1, 6, partitions), dtype=torch.float32, device=device),
            torch.empty((1, 6, partitions), dtype=torch.float32, device=device),
            torch.full((1,), partitions, dtype=torch.int32, device=device),
        )
        if operator is not None
        else None
    )
    logger.info_once(
        "Loaded SM70 INT8 PTH scalar tail attention (%s): %d partitions, "
        "FP32 numerator/max/sum (2CTA Tensor Core fastpath=%s).",
        BUILTIN_INT8_SCALAR_OP,
        partitions,
        grouped_operator is not None,
        scope="process",
    )

    def run(
        q,
        k,
        v,
        table,
        lengths,
        *,
        out,
        softmax_scale,
        k_scale,
        v_scale,
        kv_cache_dtype,
        window_size,
        max_seq_len_hint,
        partition_size_hint,
        anchor_lens,
        anchored_window,
    ) -> bool:
        del k_scale, v_scale
        descriptor = (
            get_forward_context().batch_descriptor
            if is_forward_context_available()
            else None
        )
        bucket = getattr(descriptor, "attention_context_bucket", None)
        if bucket is None and table.ndim == 2 and k.ndim == 4:
            bucket = model_capacity or int(table.shape[1]) * int(k.shape[1])
        if not (
            bucket is not None
            and 0 < bucket <= capability
            and q.ndim == 3
            and q.shape[0] == 1
            and q.shape[1] > 0
            and q.shape[2] == 256
            and q.dtype == torch.float16
            and q.device == device
            and q.is_contiguous()
            and k.ndim == 4
            and context_table_supported(table, k.shape[1], capability)
            and k.shape[1] >= INT8_SCALAR_MIN_PAGE_SIZE
            and k.shape[2] * 6 == q.shape[1]
            and k.shape[3] == 256
            and k.dtype == v.dtype == torch.int8
            and v.shape == k.shape
            and resolve_kv_codec(kv_cache_dtype) is INT8_PER_TOKEN_HEAD
            and window_size == (-1, -1)
            and anchor_lens is None
            and anchored_window == 0
            and partition_size_hint in (None, 1024)
            and type(max_seq_len_hint) is int
            and 0 < max_seq_len_hint <= capability
        ):
            return False
        k_scales, v_scales = per_token_head_scales_for_kv_views(k, v)
        if (
            grouped_operator is not None
            and k.shape[1] % 16 == 0
            and k.stride(3) == 1
            and v.stride(3) == 1
            and k.data_ptr() % 16 == 0
            and v.data_ptr() % 16 == 0
            and all(
                tensor.stride(dim) % 8 == 0 for tensor in (k, v) for dim in range(3)
            )
            and out.shape == q.shape
            and out.device == device
            and out.dtype == torch.float16
            and out.is_contiguous()
            and table.device == device
            and table.dtype == torch.int32
            and table.is_contiguous()
            and lengths.ndim == 1
            and lengths.shape[0] == 1
            and lengths.device == device
            and lengths.dtype == torch.int32
            and lengths.is_contiguous()
        ):
            stream = torch.cuda.current_stream(device).cuda_stream
            key = ("int8_pth_v1", bucket, 80, device, stream)
            bank = _WORKSPACES.setdefault(key, [])
            panels = int(k.shape[2])
            for partial, lse in bank:
                if partial.shape[0] >= panels:
                    break
            else:
                partial = torch.empty(
                    (panels, 80, 8, 6, 256), dtype=torch.float32, device=device
                )
                lse = torch.empty(
                    (panels, 80, 8, 6, 2), dtype=torch.float32, device=device
                )
                bank.append((partial, lse))
            if panels == 1:
                partial_view, lse_view = partial[0], lse[0]
            else:
                partial_view, lse_view = partial[:panels], lse[:panels]
            grouped_operator(
                q,
                k,
                v,
                k_scales,
                v_scales,
                out,
                table,
                lengths,
                partial_view,
                lse_view,
                float(softmax_scale),
            )
            return True
        if operator is None or buffers is None:
            return False
        if k.shape[2] == 1:
            operator(
                q,
                k,
                v,
                k_scales,
                v_scales,
                out,
                table,
                lengths,
                *buffers,
                float(softmax_scale),
            )
        else:
            for head in range(k.shape[2]):
                group_q = q[:, head * 6 : (head + 1) * 6].contiguous()
                group_out = torch.empty_like(group_q)
                operator(
                    group_q,
                    k[:, :, head : head + 1],
                    v[:, :, head : head + 1],
                    k_scales[:, :, head : head + 1],
                    v_scales[:, :, head : head + 1],
                    group_out,
                    table,
                    lengths,
                    *buffers,
                    float(softmax_scale),
                )
                out[:, head * 6 : (head + 1) * 6].copy_(group_out)
        return True

    _int8_scalar_tail_cache[cache_key] = run
    return run
