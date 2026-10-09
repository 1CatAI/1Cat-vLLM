# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lossless FP16/FP32-source selection for a guarded DFlash2 target probe."""

import torch

from vllm.platforms import current_platform


def compact_half_topk(
    logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return a sorted top-64 probe; the caller must retain the tie guard.

    This does not implement the draft selector's tie contract. Never cast
    FP32 logits to FP16 to qualify: ranking uses the original source values.
    """
    if (
        not logits.is_cuda
        or logits.dtype not in (torch.float16, torch.float32)
        or logits.ndim != 2
        or not 1 <= logits.shape[0] <= 32
        or not 64 <= logits.shape[1] <= 65535
        or logits.stride(1) != 1
        or logits.stride(0) > (2**31 - 1) // logits.shape[0]
        or not current_platform.is_device_capability(70, device_id=logits.device.index)
        or not hasattr(torch.ops._C, "sm70_compact_half_topk_out")
    ):
        return None
    rows, width = logits.shape
    scratch_width = ((width + 1023) // 1024) * 64
    if logits.dtype == torch.float32 and scratch_width > 1024:
        scratch_width += ((scratch_width + 511) // 512) * 64
    partial = torch.empty(
        (rows, scratch_width),
        dtype=torch.uint32 if logits.dtype == torch.float16 else torch.uint64,
        device=logits.device,
    )
    values = torch.empty((rows, 64), dtype=torch.float32, device=logits.device)
    ids = torch.empty((rows, 64), dtype=torch.int64, device=logits.device)
    torch.ops._C.sm70_compact_half_topk_out(logits, partial, values, ids, 64)
    return values, ids


if hasattr(torch.ops._C, "sm70_compact_half_topk_out"):

    @torch.library.register_fake("_C::sm70_compact_half_topk_out")
    def _fake(logits, partial, values, ids, k) -> None:
        return None
