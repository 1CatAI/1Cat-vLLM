# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fuse the SM70 DFlash2 verifier's channel FP8 and FP16 projections."""

import torch

from vllm.model_executor.layers.linear import LinearBase
from vllm.platforms import current_platform


def apply_gdn_ba_verify(layer, x, outputs=None):
    qkvz = layer.in_proj_qkvz
    ba = layer.in_proj_ba
    if (
        not getattr(layer, "enable_sm70_dflash2_fused_gdn_verify", False)
        or not getattr(layer, "enable_sm70_gdn_ba_verify", False)
        or not isinstance(qkvz, LinearBase)
        or not isinstance(ba, LinearBase)
        or not current_platform.is_device_capability(70)
        or not getattr(qkvz, "sm70_fp8_qpn8", False)
        or x.ndim != 2
        or x.shape[1] != 5120
        or not 2 <= x.shape[0] <= 32
        or x.dtype != torch.float16
        or not x.is_contiguous()
        or qkvz.weight.shape != (5120, 4096)
        or qkvz.weight_scale_inv.shape != (1, 4096)
        or ba.weight.shape != (24, 5120)
        or ba.weight.dtype != torch.float16
        or not ba.weight.is_contiguous()
        or layer.gqa_interleaved_layout
        or layer.disable_tp_for_ba_proj
        or not hasattr(torch.ops._C, "fp8_qpn8_dispatch_ba_split_sm70_out")
    ):
        return None
    rows = x.shape[0]
    masked_rows = hasattr(torch.ops._C, "fp8_qpn8_dispatch_ba_masked_sm70_out")
    if rows < 8 and not masked_rows:
        # MTP verifies 1 + drafts < 8 rows. Below eight the native route
        # splits into QPN8 plus a cuBLAS b/a GEMM and a copy; zero rows pad
        # the block to the fused M8 launch, which reads the weights once.
        padded = x.new_zeros((8, x.shape[1]))
        padded[:rows] = x
        fused = apply_gdn_ba_verify(layer, padded)
        if fused is None:
            return None
        qkv, z, b, a = (part[:rows] for part in fused)
        if outputs is None:
            return qkv, z, b, a
        for out, part in zip(outputs, (qkv, z, b, a)):
            out.copy_(part)
        return outputs
    if outputs is None:
        qkv = x.new_empty((rows, 2560))
        z = x.new_empty((rows, 1536))
        b = x.new_empty((rows, 12))
        a = torch.empty_like(b)
    else:
        qkv, z, b, a = outputs
        if any(
            tensor.shape != shape
            or tensor.dtype != x.dtype
            or tensor.device != x.device
            or not tensor.is_contiguous()
            for tensor, shape in zip(
                outputs, ((rows, 2560), (rows, 1536), (rows, 12), (rows, 12))
            )
        ):
            return None
    op = (
        torch.ops._C.fp8_qpn8_dispatch_ba_masked_sm70_out
        if rows < 8
        else torch.ops._C.fp8_qpn8_dispatch_ba_split_sm70_out
    )
    op(
        qkv,
        z,
        b,
        a,
        x.new_empty((rows, 4096)),
        x.new_empty((rows, 24)),
        0,
        x,
        qkvz.weight,
        qkvz.weight_scale_inv,
        ba.weight,
    )
    return qkv, z, b, a
