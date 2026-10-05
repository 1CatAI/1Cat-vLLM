# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP16 paged-KV verification with FP32 probability and PV arithmetic."""

import torch

from vllm import envs
from vllm.platforms import current_platform

OPERATOR = "sm70_grouped_fp16_fwd"
MAX_GROUPS = 4
MAX_CONTEXT = 266240
_WORKSPACES: dict[tuple, list[tuple[torch.Tensor, torch.Tensor]]] = {}


def clear_grouped_fp16_workspaces():
    _WORKSPACES.clear()


def load_grouped_fp16_fp32():
    if not current_platform.is_device_capability(70):
        return None
    from vllm.vllm_flash_attn.flash_attn_interface import ensure_fa2_library_loaded

    ensure_fa2_library_loaded()
    operator = getattr(torch.ops._vllm_fa2_C, OPERATOR, None)
    return _run if operator is not None else None


def _run(q, k, v, table, row_lengths, *, out, softmax_scale):
    groups = table.shape[0]
    key = (q.device, torch.cuda.current_stream(q.device).cuda_stream)
    bank = _WORKSPACES.setdefault(key, [])
    for partial, lse in bank:
        if partial.shape[0] >= groups:
            break
    else:
        partial = torch.empty(
            (groups, 80, 8, 6, 256), dtype=torch.float32, device=q.device
        )
        lse = torch.empty((groups, 80, 8, 6, 2), dtype=torch.float32, device=q.device)
        bank.append((partial, lse))
    partial, lse = (
        (partial[0], lse[0]) if groups == 1 else (partial[:groups], lse[:groups])
    )
    torch.ops._vllm_fa2_C.sm70_grouped_fp16_fwd(
        q, k, v, out, table, row_lengths, partial, lse, softmax_scale
    )
    return out


def grouped_fp16_fp32_reason(
    instance, q, k, v, table, lengths, metadata, *, out, partition_size_hint
):
    if getattr(instance, "flash_attn_grouped_fp16_fp32_paged", None) is None:
        return "operator_missing:sm70_grouped_fp16_fwd"
    if instance.kv_cache_dtype not in ("auto", "float16", "bfloat16"):
        return "kv_dtype"
    if (
        not instance.use_smallq_decode_xqa
        or partition_size_hint is not None
        or envs.VLLM_FLASH_V100_DECODE_PARTITION_SIZE
    ):
        return "decode_policy"
    if not getattr(metadata, "causal", True) or instance._flash_v100_window_size(
        causal=True
    ) != (-1, -1):
        return "causal_or_window"
    parent = getattr(metadata, "block_table", None)
    parent_seq = getattr(metadata, "seq_lens", None)
    if parent is None or parent.ndim != 2:
        return "request_metadata"
    groups = parent.shape[0]
    if not 1 <= groups <= MAX_GROUPS:
        return "request_group_count"
    rows = q.shape[0] if q.ndim == 3 else 0
    if not ((groups == 1 and 2 <= rows <= 8) or (groups > 1 and rows == groups * 8)):
        return "query_rows"
    if q.shape[1:] != (6, 256):
        return "head_shape"
    if k.ndim != 4 or k.shape[1:] != (832, 1, 256) or v.shape != k.shape:
        return "kv_shape_or_unmeasured_page"
    if not 0 < parent.shape[1] * k.shape[1] <= MAX_CONTEXT:
        return "context_capacity"
    if parent_seq is None or parent_seq.shape != (groups,):
        return "request_metadata"
    if lengths.shape != (rows,) or table.shape != (rows, parent.shape[1]):
        return "row_metadata"
    if q.device.type != "cuda":
        return "device_not_cuda"
    if out.shape != q.shape:
        return "output_shape"
    for tensor in (q, out):
        if (
            tensor.dtype != torch.float16
            or tensor.device != q.device
            or not tensor.is_contiguous()
            or tensor.data_ptr() % 16
        ):
            return "query_or_output_layout"
    for tensor in (k, v):
        if (
            tensor.dtype != torch.float16
            or tensor.device != q.device
            or tensor.stride(-1) != 1
            or tensor.data_ptr() % 16
            or any(s % 8 for s in tensor.stride()[:3])
        ):
            return "kv_layout"
    for tensor in (table, lengths, parent, parent_seq):
        if (
            tensor.dtype != torch.int32
            or tensor.device != q.device
            or not tensor.is_contiguous()
        ):
            return "metadata_layout"
    return None
