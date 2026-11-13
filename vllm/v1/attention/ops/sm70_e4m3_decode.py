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
from vllm.v1.attention.kv_codecs import FP8_E4M3, resolve_kv_codec
from vllm.v1.attention.ops.sm70_grouped_long import (
    _WORKSPACES,
    context_table_supported,
    long_attention_capability,
    long_attention_max_batch_size,
    long_attention_page_supported,
    resolve_long_attention,
    run_six_head_groups,
)


def _kv_layout_supported(k: torch.Tensor, v: torch.Tensor) -> bool:
    if (
        k.dtype != torch.uint8
        or v.dtype != torch.uint8
        or k.stride(3) != 1
        or v.stride(3) != 1
        or k.data_ptr() % 16 != 0
        or v.data_ptr() % 16 != 0
    ):
        return False
    return all(tensor.stride(dim) % 8 == 0 for tensor in (k, v) for dim in range(3))


def wrap_e4m3_decode_attention(fallback):
    if fallback is None or not callable(fallback):
        return fallback
    target = fallback.func if isinstance(fallback, functools.partial) else fallback
    if not getattr(target, "__module__", "").startswith("flash_attn_v100"):
        return fallback
    if not current_platform.is_cuda() or not current_platform.is_device_capability(70):
        return fallback
    operator, manifest = resolve_long_attention()
    if operator is None or manifest is None:
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

        if not (
            not extra_args
            and not explicit_partition_size
            and out is not None
            and resolve_kv_codec(kv_cache_dtype) is FP8_E4M3
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
            and math.isfinite(k_scale)
            and math.isfinite(v_scale)
            and k_scale > 0
            and v_scale > 0
            and long_attention_page_supported(int(k.shape[1]), manifest)
            and context_table_supported(table, int(k.shape[1]), capability)
            and _kv_layout_supported(k, v)
        ):
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
        key = (manifest["source_sha256"], bucket, 80, q.device, stream)
        bank = _WORKSPACES.setdefault(key, [])
        panels = batch_size * (int(k.shape[2]) if parallel_kv_heads else 1)
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
    with suppress(TypeError, ValueError):
        run.__signature__ = inspect.signature(fallback)  # type: ignore[attr-defined]
    return run
