# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compensated SM70 E4M3 single-query decode attention wrapper."""

from __future__ import annotations

import functools
import inspect
import math
from contextlib import suppress

import torch

from vllm.config import get_current_vllm_config_or_none
from vllm.config.execution_policy import graph_policy
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.platforms import current_platform
from vllm.v1.attention.kv_codecs import (
    FP8_E4M3,
    INT8_PER_TOKEN_HEAD,
    per_token_head_scales_for_kv_views,
    resolve_kv_codec,
)
from vllm.v1.attention.ops.sm70_grouped_long import (
    _WORKSPACES,
    context_table_supported,
    long_attention_capability,
    long_attention_max_batch_size,
    long_attention_page_supported,
    resolve_long_attention,
    run_six_head_groups,
)
from vllm.v1.attention.ops.sm70_int8_grouped import builtin_int8_pth_attention


def _kv_layout_supported(
    k: torch.Tensor, v: torch.Tensor, expected_dtype: torch.dtype = torch.uint8
) -> bool:
    if (
        k.dtype != expected_dtype
        or v.dtype != expected_dtype
        or k.stride(3) != 1
        or v.stride(3) != 1
        or k.data_ptr() % 16 != 0
        or v.data_ptr() % 16 != 0
    ):
        return False
    return all(tensor.stride(dim) % 8 == 0 for tensor in (k, v) for dim in range(3))


def triton_int8_pth_decode_fallback(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    table: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    softmax_scale: float,
    out: torch.Tensor,
    window_size: tuple[int, int] = (-1, -1),
    max_seq_len_hint: int | None = None,
) -> torch.Tensor:
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention
    from vllm.v1.kv_cache_interface import KVQuantMode

    k_scales, v_scales = per_token_head_scales_for_kv_views(k, v)
    if q.ndim == 4:
        b_sz, q_len = int(q.shape[0]), int(q.shape[1])
        q_flat = q.reshape(b_sz * q_len, q.shape[2], q.shape[3])
        out_flat = out.reshape(b_sz * q_len, out.shape[2], out.shape[3])
    else:
        b_sz, q_len = int(q.shape[0]), 1
        q_flat = q
        out_flat = out
    cu_q = torch.arange(b_sz + 1, dtype=torch.int32, device=q.device) * q_len
    max_k = int(max_seq_len_hint or (int(table.shape[1]) * int(k.shape[1])))
    unified_attention(
        q=q_flat,
        k=k,
        v=v,
        out=out_flat,
        cu_seqlens_q=cu_q,
        max_seqlen_q=q_len,
        seqused_k=seq_lens,
        max_seqlen_k=max_k,
        softmax_scale=softmax_scale,
        causal=True,
        window_size=window_size,
        block_table=table,
        softcap=0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        kv_quant_mode=KVQuantMode.INT8_PER_TOKEN_HEAD,
        k_scale_cache=k_scales,
        v_scale_cache=v_scales,
    )
    return out


def wrap_e4m3_decode_attention(fallback):
    if fallback is None or not callable(fallback):
        return fallback
    target = fallback.func if isinstance(fallback, functools.partial) else fallback
    if not getattr(target, "__module__", "").startswith("flash_attn_v100"):
        return fallback
    if not current_platform.is_cuda() or not current_platform.is_device_capability(70):
        return fallback
    operator, manifest = resolve_long_attention()
    int8_operator = builtin_int8_pth_attention()
    if (operator is None or manifest is None) and int8_operator is None:
        return fallback
    if manifest is None:
        return fallback

    capability = long_attention_capability(manifest)
    max_batch_size = long_attention_max_batch_size(manifest)
    parallel_kv_heads = manifest.get("kv_heads_revision", 0) >= 1
    variable_rows = manifest.get("query_rows_revision", 1) >= 2
    if not variable_rows:
        return fallback

    config = get_current_vllm_config_or_none()
    model = getattr(config, "model_config", None)
    model_capacity = getattr(model, "max_model_len", None)
    explicit_partition_size = bool(graph_policy().decode_partition_size)

    def run(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        table: torch.Tensor,
        seq_lens: torch.Tensor,
        softmax_scale: float | None = None,
        out: torch.Tensor | None = None,
        kv_cache_dtype: str = "auto",
        k_scale: float = 1.0,
        v_scale: float = 1.0,
        window_size: tuple[int, int] = (-1, -1),
        *extra_args: object,
        **kwargs: object,
    ):
        anchor_lens = kwargs.get("anchor_lens")
        anchored_window = kwargs.get("anchored_window", 0)
        batch_size = int(table.shape[0]) if table.ndim == 2 else 0
        scale = (
            float(q.shape[-1] ** -0.5)
            if softmax_scale is None
            else float(softmax_scale)
        )
        descriptor = (
            get_forward_context().batch_descriptor
            if is_forward_context_available()
            else None
        )
        bucket = getattr(descriptor, "attention_context_bucket", None)
        if bucket is None and table.ndim == 2 and k.ndim == 4:
            bucket = model_capacity or int(table.shape[1]) * int(k.shape[1])

        codec = resolve_kv_codec(kv_cache_dtype)
        is_int8_pth = codec is INT8_PER_TOKEN_HEAD and int8_operator is not None
        is_e4m3 = codec is FP8_E4M3 and operator is not None
        expected_storage = torch.int8 if is_int8_pth else torch.uint8

        if not (
            not extra_args
            and not explicit_partition_size
            and out is not None
            and (is_e4m3 or is_int8_pth)
            and tuple(window_size) == (-1, -1)
            and anchor_lens is None
            and anchored_window == 0
            and bucket is not None
            and bucket <= capability
            and 1 <= batch_size <= max_batch_size
            and q.ndim == 3
            and q.shape[0] == batch_size
            and q.shape[2] == 256
            and k.ndim == 4
            and k.shape[2] >= 1
            and k.shape[2] * 6 == q.shape[1]
            and k.shape[3] == 256
            and v.shape == k.shape
            and q.is_cuda
            and q.dtype == torch.float16
            and q.is_contiguous()
            and out.shape == q.shape
            and out.device == q.device
            and out.dtype == torch.float16
            and out.is_contiguous()
            and table.device == q.device
            and table.dtype == torch.int32
            and table.is_contiguous()
            and seq_lens.ndim == 1
            and seq_lens.shape[0] == batch_size
            and seq_lens.device == q.device
            and seq_lens.dtype == torch.int32
            and seq_lens.is_contiguous()
            and k.device == q.device
            and v.device == q.device
            and math.isfinite(scale)
            and (
                is_int8_pth
                or (
                    math.isfinite(k_scale)
                    and math.isfinite(v_scale)
                    and k_scale > 0
                    and v_scale > 0
                )
            )
            and long_attention_page_supported(int(k.shape[1]), manifest)
            and context_table_supported(table, int(k.shape[1]), capability)
            and _kv_layout_supported(k, v, expected_storage)
        ):
            if codec is INT8_PER_TOKEN_HEAD and out is not None and q.ndim in (3, 4):
                hint = kwargs.get("max_seq_len_hint")
                return triton_int8_pth_decode_fallback(
                    q,
                    k,
                    v,
                    table,
                    seq_lens,
                    softmax_scale=scale,
                    out=out,
                    window_size=window_size,
                    max_seq_len_hint=hint if isinstance(hint, int) else None,
                )
            return fallback(
                q,
                k,
                v,
                table,
                seq_lens,
                softmax_scale,
                out,
                kv_cache_dtype,
                k_scale,
                v_scale,
                window_size,
                *extra_args,
                **kwargs,
            )

        stream = torch.cuda.current_stream(q.device).cuda_stream
        ws_tag = "int8_pth_v1" if is_int8_pth else manifest["source_sha256"]
        key = (ws_tag, bucket, 80, q.device, stream)
        bank = _WORKSPACES.setdefault(key, [])
        panels = batch_size * (
            int(k.shape[2]) if (is_int8_pth or parallel_kv_heads) else 1
        )
        for partial_buf, lse_buf in bank:
            if partial_buf.shape[0] >= panels:
                break
        else:
            partial_buf = torch.empty(
                (panels, 80, 8, 6, 256), dtype=torch.float32, device=q.device
            )
            lse_buf = torch.empty(
                (panels, 80, 8, 6, 2), dtype=torch.float32, device=q.device
            )
            bank.append((partial_buf, lse_buf))
        if panels == 1:
            partial_view, lse_view = partial_buf[0], lse_buf[0]
        else:
            partial_view, lse_view = partial_buf[:panels], lse_buf[:panels]
        if is_int8_pth:
            k_scales, v_scales = per_token_head_scales_for_kv_views(k, v)
            return int8_operator(
                q,
                k,
                v,
                k_scales,
                v_scales,
                out,
                table,
                seq_lens,
                partial_view,
                lse_view,
                scale,
            )
        return run_six_head_groups(
            operator,
            q,
            k,
            v,
            out,
            table,
            seq_lens,
            partial_view,
            lse_view,
            scale,
            float(k_scale),
            float(v_scale),
            parallel_kv_heads=parallel_kv_heads,
        )

    run.__dict__.update(getattr(fallback, "__dict__", {}))
    run._handles_int8_pth = True  # type: ignore[attr-defined]
    with suppress(TypeError, ValueError):
        run.__signature__ = inspect.signature(fallback)  # type: ignore[attr-defined]
    return run
