# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact FP16 storage for the small DFlash2 verifier's dense projections."""

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

# Each geometry was measured at M8 with unchanged loaded FP16 weight shards.
_GEOMETRY = {
    (1280, 5120): (16, 16),
    (1536, 5120): (32, 16),
    (5120, 1024): (32, 8),
    (8704, 5120): (32, 16),
    (5120, 4352): (32, 8),
}


def prepare_dflash2_fp16_m8(layer: torch.nn.Module) -> bool:
    if not getattr(layer, "_sm70_dflash2_fp16_m8", False):
        return False
    if envs.VLLM_BATCH_INVARIANT:
        return False
    if not hasattr(torch.ops._C, "sm70_dflash2_fp16_dispatch_out"):
        return False
    weight = layer.weight
    if (
        weight.ndim != 2
        or weight.dtype != torch.float16
        or not weight.is_cuda
        or not weight.is_contiguous()
        or getattr(layer, "bias", None) is not None
        or not current_platform.is_device_capability(70, device_id=weight.device.index)
    ):
        return False
    geometry = _GEOMETRY.get(tuple(weight.shape))
    if geometry is None:
        return False
    tile, warps = geometry
    n, k = weight.shape
    slots = torch.arange(tile, device=weight.device)
    columns = (
        (slots & 3) | ((slots & 12) << 1) | ((slots & 16) >> 2)
        if tile == 32
        else (slots & 3) | ((slots & 4) << 1) | ((slots & 8) >> 1)
    )
    packed = (
        weight.view(n // tile, tile, k // 16, 16)
        .permute(0, 2, 1, 3)
        .index_select(2, columns)
        .contiguous()
    )
    # Retain the original matrix for prefill, batched drafting, and context KV.
    layer.register_buffer("_sm70_dflash2_fp16_packed", packed, persistent=False)
    layer._sm70_dflash2_fp16_geometry = geometry
    logger.info_once(
        "SM70 DFlash2 FP16 M8 layout prepared: N%d/K%d, tile=%d, warps=%d.",
        n,
        k,
        tile,
        warps,
    )
    return True


def apply_dflash2_fp16_m8(
    layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None
) -> torch.Tensor | None:
    packed = getattr(layer, "_sm70_dflash2_fp16_packed", None)
    if (
        packed is None
        or bias is not None
        or getattr(layer, "_sm70_f16_prepared", False)
        or envs.VLLM_BATCH_INVARIANT
        or x.ndim != 2
        or x.shape[1] != layer.weight.shape[1]
        or x.dtype != torch.float16
        or not x.is_contiguous()
    ):
        return None
    if not torch.compiler.is_compiling() and x.shape[0] != 8:
        return None
    output = x.new_empty((x.shape[0], layer.weight.shape[0]))
    tile, warps = layer._sm70_dflash2_fp16_geometry
    torch.ops._C.sm70_dflash2_fp16_dispatch_out(
        output, x, packed, layer.weight, tile, warps
    )
    return output


if hasattr(torch.ops._C, "sm70_dflash2_fp16_m8_out"):

    @torch.library.register_fake("_C::sm70_dflash2_fp16_m8_out")
    def _fake(output, input, packed, tile, warps) -> None:
        return None


if hasattr(torch.ops._C, "sm70_dflash2_fp16_dispatch_out"):

    @torch.library.register_fake("_C::sm70_dflash2_fp16_dispatch_out")
    def _fake_dispatch(output, input, packed, weight, tile, warps) -> None:
        return None
