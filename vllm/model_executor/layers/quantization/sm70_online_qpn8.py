# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Narrow online QPN8 route for Qwen4Exp FP16 decode projections on SM70."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import vllm.envs as envs
from vllm import _sm70_ops as sm70_ops
from vllm.config import get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from torch import nn


logger = init_logger(__name__)

_STATE_ATTR = "_sm70_qwen4_exp_online_qpn8"
_CODES_ATTR = "_sm70_qwen4_exp_online_qpn8_codes"
_SCALES_ATTR = "_sm70_qwen4_exp_online_qpn8_scales"
_HC_PARTIALS_ATTR = "_sm70_qwen4_exp_online_qpn8_hc_partials"
_WORKSPACE_ELEMENTS = 4096 * 2560
_workspaces: dict[tuple[int, torch.dtype], torch.Tensor] = {}
_hc_partials: dict[int, torch.Tensor] = {}

_HC_DOWN_SUFFIX = ".input_mix_weight_down_block_inject"
_HC_DOWN_LOGICAL_N = 336
_HC_DOWN_PADDED_N = 384

_REQUIRED_OPS = (
    "fp8_qpn8_prepare_sm70",
    "fp8_qpn8_dispatch_sm70_out",
)


def _shape_config(prefix: str, k: int, n: int) -> tuple[int, int, bool] | None:
    """Return the measured ``split_k, nacc, prefetch`` for one local weight."""
    if prefix.endswith(_HC_DOWN_SUFFIX) and (k, n) == (
        10240,
        _HC_DOWN_LOGICAL_N,
    ):
        return 32, 1, False
    if prefix.endswith(".input_mix_weight_up") and (k, n) == (320, 10240):
        return 4, 2, False
    if prefix.endswith(".linear_attn.in_proj_qkvz") and (k, n) == (2560, 4096):
        return 16, 1, False
    if prefix.endswith(".linear_attn.out_proj") and (k, n) == (1536, 2560):
        return 12, 2, False
    if prefix.endswith(".self_attn.qkv_proj") and (k, n) == (2560, 3584):
        return 16, 1, False
    if prefix.endswith(".self_attn.o_proj") and (k, n) == (1536, 2560):
        return 12, 2, False
    return None


def _exact_runtime_contract() -> bool:
    try:
        config = get_current_vllm_config()
        text_config = config.model_config.hf_text_config
        tp_size = get_tensor_model_parallel_world_size()
    except (AssertionError, AttributeError, RuntimeError):
        return False
    return bool(
        tp_size == 4
        and config.speculative_config is None
        and int(getattr(text_config, "hidden_size", 0)) == 2560
        and int(getattr(text_config, "num_hidden_layers", 0)) == 48
        and int(getattr(text_config, "num_experts", 0)) == 512
        and int(getattr(text_config, "num_experts_per_tok", 0)) == 10
        and int(getattr(text_config, "moe_intermediate_size", 0)) == 640
        and int(getattr(text_config, "hc_count", 0)) == 4
        and int(getattr(text_config, "hc_lowrank", 0)) == 320
        and int(getattr(text_config, "num_attention_heads", 0)) == 24
        and int(getattr(text_config, "num_key_value_heads", 0)) == 2
        and int(getattr(text_config, "indexer_head_dim", 0)) == 128
        and int(getattr(text_config, "indexer_budget", 0)) == 2048
        and int(getattr(text_config, "indexer_compress_ratio", 0)) == 4
    )


def _workspace(weight: torch.Tensor) -> torch.Tensor:
    device_index = weight.device.index
    if device_index is None:
        device_index = torch.accelerator.current_device_index()
    key = (device_index, weight.dtype)
    workspace = _workspaces.get(key)
    if workspace is None:
        workspace = torch.empty(
            (_WORKSPACE_ELEMENTS,), dtype=torch.float16, device=weight.device
        )
        _workspaces[key] = workspace
    return workspace


def _hc_workspace(weight: torch.Tensor) -> torch.Tensor:
    device_index = weight.device.index
    if device_index is None:
        device_index = torch.accelerator.current_device_index()
    partials = _hc_partials.get(device_index)
    if partials is None:
        partials = torch.empty(
            (32 * _HC_DOWN_PADDED_N,),
            dtype=torch.float32,
            device=weight.device,
        )
        _hc_partials[device_index] = partials
    return partials


def maybe_prepare_online_qpn8(layer: nn.Module) -> bool:
    """Quantize one admitted FP16 weight and retain only its QPN8 layout."""
    if getattr(layer, _STATE_ATTR, False):
        return True
    if not envs.VLLM_SM70_QWEN4_EXP_ONLINE_QPN8 or not _exact_runtime_contract():
        return False
    if any(not hasattr(torch.ops._C, name) for name in _REQUIRED_OPS):
        raise RuntimeError(
            "VLLM_SM70_QWEN4_EXP_ONLINE_QPN8=1 requires the SM70 QPN8 ops"
        )
    weight = getattr(layer, "weight", None)
    if (
        weight is None
        or weight.dtype != torch.float16
        or not weight.is_cuda
        or weight.ndim != 2
        or not current_platform.is_device_capability(70)
    ):
        return False

    logical_n, k = (int(dim) for dim in weight.shape)
    prefix = str(getattr(layer, "prefix", ""))
    config = _shape_config(prefix, k, logical_n)
    if config is None:
        return False
    padded_n = _HC_DOWN_PADDED_N if prefix.endswith(_HC_DOWN_SUFFIX) else logical_n
    if padded_n % 32 or k % 16:
        raise RuntimeError(f"online QPN8 received an unaligned weight {(padded_n, k)}")

    if padded_n != logical_n:
        padding = weight.new_zeros((padded_n - logical_n, k))
        weight_for_quant = torch.cat((weight, padding), dim=0)
    else:
        weight_for_quant = weight

    weight_f32 = weight_for_quant.float()
    channel_scales = weight_f32.abs().amax(dim=1, keepdim=True).div_(448.0)
    channel_scales = torch.where(
        channel_scales == 0, torch.ones_like(channel_scales), channel_scales
    )
    qweight = (weight_f32 / channel_scales).to(torch.float8_e4m3fn)
    codes, scales = sm70_ops.fp8_qpn8_prepare_sm70(
        qweight.contiguous(), channel_scales.contiguous()
    )
    workspace = _workspace(weight)

    replace_parameter(layer, "weight", codes)
    replace_parameter(layer, "weight_scale_inv", scales)
    setattr(layer, _CODES_ATTR, layer.weight)
    setattr(layer, _SCALES_ATTR, layer.weight_scale_inv)
    layer._sm70_qwen4_exp_online_qpn8_k = k
    layer._sm70_qwen4_exp_online_qpn8_n = padded_n
    layer._sm70_qwen4_exp_online_qpn8_logical_n = logical_n
    layer._sm70_qwen4_exp_online_qpn8_split_k = config[0]
    layer._sm70_qwen4_exp_online_qpn8_nacc = config[1]
    layer._sm70_qwen4_exp_online_qpn8_prefetch = config[2]
    layer._sm70_qwen4_exp_online_qpn8_workspace = workspace
    layer._sm70_qwen4_exp_online_qpn8_workspace_ptr = workspace.data_ptr()
    if prefix.endswith(_HC_DOWN_SUFFIX):
        # Resolve the shared scratch tensor while weights are prepared. Looking
        # it up through the process-global dictionary from forward makes
        # TorchDynamo emit a dictionary clear/update, which cudagraph rejects
        # as a module-state mutation.
        layer.register_buffer(
            _HC_PARTIALS_ATTR,
            _hc_workspace(weight),
            persistent=False,
        )
    layer.sm70_fp8_qpn8 = True
    layer.sm70_fp8_prefill_exact_dense_workspace_ptr = workspace.data_ptr()
    setattr(layer, _STATE_ATTR, True)
    logger.info_once(
        "SM70 Qwen4Exp online channel-QPN8 enabled for selected FP16 decode "
        "projections."
    )
    return True


def maybe_apply_online_qpn8(
    layer: nn.Module,
    x: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor | None:
    """Run a prepared channel-QPN8 projection, or return ``None``."""
    if not getattr(layer, _STATE_ATTR, False):
        return None
    k = int(layer._sm70_qwen4_exp_online_qpn8_k)
    n = int(layer._sm70_qwen4_exp_online_qpn8_n)
    logical_n = int(layer._sm70_qwen4_exp_online_qpn8_logical_n)
    if x.dtype != torch.float16 or x.shape[-1] != k:
        raise RuntimeError(
            "SM70 Qwen4Exp online QPN8 requires FP16 inputs with "
            f"K={k}, got dtype={x.dtype} shape={tuple(x.shape)}"
        )
    x_2d = x.reshape(-1, k)
    if not x_2d.is_contiguous():
        x_2d = x_2d.contiguous()
    out = torch.empty((x_2d.size(0), n), dtype=x.dtype, device=x.device)
    sm70_ops.fp8_qpn8_dispatch_sm70_out(
        out,
        int(layer._sm70_qwen4_exp_online_qpn8_workspace_ptr),
        x_2d,
        getattr(layer, _CODES_ATTR),
        getattr(layer, _SCALES_ATTR),
        int(layer._sm70_qwen4_exp_online_qpn8_split_k),
        int(layer._sm70_qwen4_exp_online_qpn8_nacc),
        bool(layer._sm70_qwen4_exp_online_qpn8_prefetch),
        False,
    )
    logical_out = out[:, :logical_n]
    if bias is not None:
        logical_out.add_(bias)
    return logical_out.reshape(*x.shape[:-1], logical_n)


def maybe_apply_fused_hc(
    down_layer: nn.Module,
    up_layer: nn.Module,
    xn: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Run the exact Qwen4Exp HC projection pair through one opaque route."""
    if not (
        getattr(down_layer, _STATE_ATTR, False)
        and getattr(up_layer, _STATE_ATTR, False)
    ):
        return None
    if not sm70_ops.has_fp8_qpn8_hc_dispatch():
        raise RuntimeError("prepared Qwen4Exp HC requires the fused QPN8 op")
    if xn.dtype != torch.float16 or xn.ndim != 2 or xn.size(1) != 10240:
        raise RuntimeError(
            "fused Qwen4Exp HC requires a contiguous FP16 [M, 10240] input"
        )
    if not xn.is_contiguous():
        xn = xn.contiguous()
    if (
        int(down_layer._sm70_qwen4_exp_online_qpn8_k) != 10240
        or int(down_layer._sm70_qwen4_exp_online_qpn8_n) != _HC_DOWN_PADDED_N
        or int(up_layer._sm70_qwen4_exp_online_qpn8_k) != 320
        or int(up_layer._sm70_qwen4_exp_online_qpn8_n) != 10240
    ):
        raise RuntimeError("fused Qwen4Exp HC received incompatible weights")

    down_workspace_ptr = int(down_layer._sm70_qwen4_exp_online_qpn8_workspace_ptr)
    up_workspace_ptr = int(up_layer._sm70_qwen4_exp_online_qpn8_workspace_ptr)
    if down_workspace_ptr != up_workspace_ptr:
        raise RuntimeError("fused Qwen4Exp HC requires one shared dense workspace")
    partials = getattr(down_layer, _HC_PARTIALS_ATTR, None)
    if not isinstance(partials, torch.Tensor):
        raise RuntimeError("fused Qwen4Exp HC requires prepared partials")

    m = xn.size(0)
    block_out = xn.new_empty((m, 2560))
    injection_out = xn.new_empty((m, 4))
    down_staging = xn.new_empty((m, _HC_DOWN_PADDED_N))
    lora_staging = xn.new_empty((m, 320))
    gate_staging = xn.new_empty((m, 10240))
    sm70_ops.fp8_qpn8_hc_dispatch_sm70_out(
        block_out,
        injection_out,
        down_staging,
        lora_staging,
        gate_staging,
        partials,
        down_workspace_ptr,
        xn,
        getattr(down_layer, _CODES_ATTR),
        getattr(down_layer, _SCALES_ATTR),
        getattr(up_layer, _CODES_ATTR),
        getattr(up_layer, _SCALES_ATTR),
    )
    return block_out, injection_out


__all__ = [
    "_shape_config",
    "maybe_apply_fused_hc",
    "maybe_apply_online_qpn8",
    "maybe_prepare_online_qpn8",
]
