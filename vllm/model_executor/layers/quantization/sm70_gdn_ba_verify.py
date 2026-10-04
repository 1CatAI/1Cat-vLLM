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
        or x.shape != (8, 5120)
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
    if outputs is None:
        qkv = x.new_empty((8, 2560))
        z = x.new_empty((8, 1536))
        b = x.new_empty((8, 12))
        a = torch.empty_like(b)
    else:
        qkv, z, b, a = outputs
        if any(
            tensor.shape != shape
            or tensor.dtype != x.dtype
            or tensor.device != x.device
            or not tensor.is_contiguous()
            for tensor, shape in zip(outputs, ((8, 2560), (8, 1536), (8, 12), (8, 12)))
        ):
            return None
    torch.ops._C.fp8_qpn8_dispatch_ba_split_sm70_out(
        qkv,
        z,
        b,
        a,
        x.new_empty((8, 4096)),
        x.new_empty((8, 24)),
        0,
        x,
        qkvz.weight,
        qkvz.weight_scale_inv,
        ba.weight,
    )
    return qkv, z, b, a
