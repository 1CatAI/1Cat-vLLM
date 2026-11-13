# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Channel-QPN8 projections for the SM70 Qwen3.5/3.8 native MTP drafter.

The drafter's BF16 checkpoint layer runs as FP16 TurboMind GEMMs, re-reading
every weight byte at each of its draft steps. Its four decoder projections
have the target's local shapes, so they are requantized once to channel FP8
and use the target's QPN8 dispatch (row blocks up to 128, then the shared
dense prefill workspace). Only drafts change; verification is untouched.
"""

from __future__ import annotations

import torch
from torch import nn

from vllm.logger import init_logger
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform

logger = init_logger(__name__)

_MARK = "_sm70_mtp_draft_qpn8"
_READY = "_sm70_mtp_draft_qpn8_ready"
_SUFFIXES = frozenset({"fc", "qkv_proj", "o_proj", "gate_up_proj", "down_proj"})
_REQUIRED_OPS = ("fp8_qpn8_prepare_sm70", "fp8_qpn8_dispatch_sm70_out")


def mark_mtp_draft_qpn8(model: nn.Module) -> int:
    """Admit the drafter's unquantized decoder projections before loading."""
    from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod

    count = 0
    for module in model.modules():
        if (
            isinstance(module, LinearBase)
            and isinstance(module.quant_method, UnquantizedLinearMethod)
            and getattr(module, "prefix", "").rsplit(".", 1)[-1] in _SUFFIXES
            and getattr(module, "bias", None) is None
        ):
            setattr(module, _MARK, True)
            count += 1
    return count


@torch.no_grad()
def prepare_mtp_draft_qpn8(layer: nn.Module) -> bool:
    if getattr(layer, _READY, False):
        return True
    if not getattr(layer, _MARK, False):
        return False
    weight = getattr(layer, "weight", None)
    if (
        weight is None
        or weight.dtype != torch.float16
        or not weight.is_cuda
        or weight.ndim != 2
        or not current_platform.is_device_capability(70)
        or any(not hasattr(torch.ops._C, name) for name in _REQUIRED_OPS)
    ):
        return False
    n, k = (int(dim) for dim in weight.shape)
    if n % 32 or k % 128:
        return False
    from vllm.model_executor.kernels.linear.qpn.fp8 import (
        _get_sm70_fp8_prefill_exact_dense_workspace,
    )
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a16_fp8 import (  # noqa: E501
        _SM70_CHANNEL_FP8_QPN8_CONFIGS,
    )
    from vllm.model_executor.layers.quantization.sm70_online_qpn8 import (
        prepare_channel_qpn8_weight,
    )

    codes, scales = prepare_channel_qpn8_weight(weight)
    if _get_sm70_fp8_prefill_exact_dense_workspace(codes) is None:
        return False
    split_k, nacc, prefetch = _SM70_CHANNEL_FP8_QPN8_CONFIGS.get(
        (k, n), (8 if k % 256 else 16, 2, False)
    )
    replace_parameter(layer, "weight", codes)
    layer.register_buffer("_sm70_mtp_draft_qpn8_scales", scales, persistent=False)
    layer._sm70_mtp_draft_qpn8_shape = (n, k)
    layer._sm70_mtp_draft_qpn8_config = (split_k, nacc, prefetch)
    layer._unquantized_fused_silu_and_mul = apply_mtp_draft_qpn8_fused_silu
    setattr(layer, _READY, True)
    logger.info_once(
        "SM70 MTP drafter projections use channel-QPN8 (FP16 weights released)."
    )
    return True


def apply_mtp_draft_qpn8(
    layer: nn.Module, x: torch.Tensor, bias: torch.Tensor | None
) -> torch.Tensor | None:
    if not getattr(layer, _READY, False):
        return None
    n, k = layer._sm70_mtp_draft_qpn8_shape
    split_k, nacc, prefetch = layer._sm70_mtp_draft_qpn8_config
    x_2d = x.reshape(-1, k)
    if not x_2d.is_contiguous():
        x_2d = x_2d.contiguous()
    out = x_2d.new_empty((x_2d.shape[0], n))
    torch.ops.vllm.sm70_ct_fp8_qpn8_dispatch(
        out,
        x_2d,
        layer.weight,
        layer._sm70_mtp_draft_qpn8_scales,
        split_k,
        nacc,
        prefetch,
        False,
    )
    if bias is not None:
        out.add_(bias)
    return out.reshape(*x.shape[:-1], n)


def apply_mtp_draft_qpn8_fused_silu(
    layer: nn.Module, x: torch.Tensor
) -> torch.Tensor | None:
    if not getattr(layer, _READY, False):
        return None
    if not getattr(layer, "prefix", "").endswith(".gate_up_proj"):
        return None
    if x.dtype != torch.float16:
        return None
    n, k = layer._sm70_mtp_draft_qpn8_shape
    if n % 64 != 0 or k % 128 != 0:
        return None
    x_2d = x.reshape(-1, k)
    if not x_2d.is_contiguous():
        x_2d = x_2d.contiguous()
    out_features = n // 2
    out = x_2d.new_empty((x_2d.shape[0], out_features))
    if x_2d.shape[0] == 0:
        return out.reshape(*x.shape[:-1], out_features)
    torch.ops.vllm.sm70_ct_fp8_qpn8_dispatch(
        out,
        x_2d,
        layer.weight,
        layer._sm70_mtp_draft_qpn8_scales,
        8,
        2,
        False,
        True,
    )
    return out.reshape(*x.shape[:-1], out_features)
