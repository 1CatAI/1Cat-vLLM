# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Channel-FP8 M8 projection route for the SM70 DFlash2 decoder.

Only the four large decoder projections are quantized. The existing FP16
matrices remain available for context preparation and other batch sizes. The
FP8 layout replaces the FP16 M8 packing. Norms and the LM head are outside this
route.
"""

import torch

import vllm.envs as envs
from vllm import _sm70_ops as ops
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

_SHAPES = {
    (1536, 5120): 16,
    (5120, 1024): 16,
    (8704, 5120): 8,
    (5120, 4352): 16,
}


def _dispatch(
    x: torch.Tensor,
    weight: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    split: int,
) -> torch.Tensor:
    output = x.new_empty((x.shape[0], weight.shape[0]))
    if x.shape[0] == 8:
        ops.fp8_qpn8_gemm_sm70_out(output, x, codes, scales, split, 2, True, False)
    else:
        # This is the same mm_out used by the old FP16 M8 dispatch fallback.
        torch.mm(x, weight.t(), out=output)
    return output


def _dispatch_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    split: int,
) -> torch.Tensor:
    return x.new_empty((x.shape[0], weight.shape[0]))


direct_register_custom_op(
    op_name="sm70_dflash2_fp8_m8_dispatch",
    op_func=_dispatch,
    fake_impl=_dispatch_fake,
)


@torch.no_grad()
def prepare_dflash2_fp8_m8(layer: torch.nn.Module) -> bool:
    if getattr(layer, "_sm70_dflash2_fp8_codes", None) is not None:
        return True
    if (
        not getattr(layer, "_sm70_dflash2_fp8_m8", False)
        or envs.VLLM_BATCH_INVARIANT
        or getattr(layer, "bias", None) is not None
    ):
        return False
    weight = layer.weight
    if (
        weight.dtype != torch.float16
        or not weight.is_cuda
        or weight.ndim != 2
        or not weight.is_contiguous()
        or not current_platform.is_device_capability(70, device_id=weight.device.index)
        or not hasattr(torch.ops._C, "fp8_qpn8_prepare_sm70")
        or not hasattr(torch.ops._C, "fp8_qpn8_gemm_sm70_out")
    ):
        return False
    split = _SHAPES.get(tuple(weight.shape))
    if split is None:
        return False
    dense = weight.float()
    scales = dense.abs().amax(1, keepdim=True) / 448.0
    # Zero channels retain exact zero codes. Keep the smallest nonzero FP16
    # scale representable by the QPN8 channel-scale storage.
    scales = torch.where(scales == 0, torch.ones_like(scales), scales)
    scales = scales.clamp_min(2.0**-24)
    quantized = (dense / scales).to(torch.float8_e4m3fn)
    codes, packed_scales = ops.fp8_qpn8_prepare_sm70(quantized, scales)
    layer.register_buffer("_sm70_dflash2_fp8_codes", codes, persistent=False)
    layer.register_buffer("_sm70_dflash2_fp8_scales", packed_scales, persistent=False)
    layer._sm70_dflash2_fp8_split = split
    logger.info_once("SM70 DFlash2 channel-FP8 M8 decoder projections prepared.")
    return True


def apply_dflash2_fp8_m8(
    layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None
) -> torch.Tensor | None:
    codes = getattr(layer, "_sm70_dflash2_fp8_codes", None)
    if (
        codes is None
        or bias is not None
        or envs.VLLM_BATCH_INVARIANT
        or getattr(layer, "_sm70_f16_prepared", False)
        or x.ndim != 2
        or x.dtype != torch.float16
        or x.shape[1] != layer.weight.shape[1]
        or not x.is_contiguous()
    ):
        return None
    if not torch.compiler.is_compiling() and x.shape[0] != 8:
        return None
    return torch.ops.vllm.sm70_dflash2_fp8_m8_dispatch(
        x,
        layer.weight,
        codes,
        layer._sm70_dflash2_fp8_scales,
        layer._sm70_dflash2_fp8_split,
    )
