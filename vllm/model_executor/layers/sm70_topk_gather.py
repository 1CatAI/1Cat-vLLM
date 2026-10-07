# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gather compact vocabulary candidates with one lossless TP message."""

import torch

from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton


@triton.jit
def _pack(
    values, ids, wire, COUNT: tl.constexpr, HALF: tl.constexpr, BLOCK: tl.constexpr
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = index < COUNT
    value = tl.load(values + index, mask, 0)
    bits = (
        value.to(tl.int16, bitcast=True).to(tl.int32)
        if HALF
        else value.to(tl.int32, bitcast=True)
    )
    token = tl.load(ids + index, mask, 0).to(tl.int32)
    tl.store(wire + index * 2, bits, mask)
    tl.store(wire + index * 2 + 1, token, mask)


@triton.jit
def _unpack(
    wire, values, ids, COUNT: tl.constexpr, HALF: tl.constexpr, BLOCK: tl.constexpr
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = index < COUNT
    bits = tl.load(wire + index * 2, mask, 0)
    token = tl.load(wire + index * 2 + 1, mask, 0)
    value = (
        bits.to(tl.int16).to(tl.float16, bitcast=True)
        if HALF
        else bits.to(tl.float32, bitcast=True)
    )
    tl.store(values + index, value, mask)
    tl.store(ids + index, token.to(tl.int64), mask)


def packed_topk_reason(
    values, ids, *, vocab_size: int, tp_size: int, enabled: bool = True
):
    if not enabled:
        return "disabled_by_policy"
    if not values.is_cuda or not current_platform.is_device_capability(
        70, device_id=values.device.index
    ):
        return "requires_sm70_cuda"
    if tp_size not in (2, 4):
        return "unmeasured_tp_size"
    if values.ndim != 2 or values.shape[0] not in (1, 7, 8, 28, 32):
        return "unmeasured_row_count"
    if values.shape[1] not in (20, 64):
        return "unmeasured_candidate_count"
    if values.dtype not in (torch.float16, torch.float32):
        return "value_dtype"
    if ids.shape != values.shape or ids.dtype != torch.int64:
        return "token_id_layout"
    if (
        not values.is_contiguous()
        or not ids.is_contiguous()
        or ids.device != values.device
    ):
        return "candidate_layout"
    if not 0 < vocab_size < 2**31:
        return "vocabulary_exceeds_int32"
    return None


def gather_topk_pairs(
    values: torch.Tensor,
    ids: torch.Tensor,
    *,
    vocab_size: int,
    enabled: bool = True,
    selections: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return rank-major candidates; preserve the caller's global TopK order.

    IDs must be valid vocabulary indices or the -1 sentinel. Float values are
    transported as their original FP16/FP32 bits, never rounded. The returned
    dtype stays unchanged so the caller retains its TopK tie implementation.
    """
    tp_size = get_tensor_model_parallel_world_size()
    reason = packed_topk_reason(
        values, ids, vocab_size=vocab_size, tp_size=tp_size, enabled=enabled
    )
    if selections is not None and not torch.compiler.is_compiling():
        selections["compact_topk_pairs"] = {
            "enabled": reason is None,
            "reason": reason,
            "scope": "collective_capability",
            "rows": values.shape[0],
            "candidates": values.shape[-1],
            "tp_size": tp_size,
        }
    if reason is not None:
        return None
    rows, local_k = values.shape
    wire = torch.empty((rows, local_k * 2), dtype=torch.int32, device=values.device)
    _pack[(triton.cdiv(values.numel(), 256),)](
        values, ids, wire, values.numel(), values.dtype == torch.float16, 256
    )
    gathered = tensor_model_parallel_all_gather(wire, dim=-1)
    result_values = torch.empty(
        (rows, local_k * tp_size), dtype=values.dtype, device=values.device
    )
    result_ids = torch.empty_like(result_values, dtype=torch.int64)
    _unpack[(triton.cdiv(result_values.numel(), 256),)](
        gathered,
        result_values,
        result_ids,
        result_values.numel(),
        values.dtype == torch.float16,
        256,
    )
    return result_values, result_ids
