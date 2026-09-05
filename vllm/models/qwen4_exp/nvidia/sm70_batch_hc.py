# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in FP16 batched HC with an isolated TP communication channel."""

import torch
from torch import nn

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.compilation.sm70_decode_graph import use_sm70_decode_graph_semantics
from vllm.forward_context import (
    get_forward_context,
    is_forward_context_available,
    is_uniform_decode_metadata,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
    UnquantizedLinearMethod,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

from .ops.hc import hc_gate_mix, hc_silu
from .sm70_fp16_gemv import Qwen38SM70FP16LinearMethod, _qwen38_sm70_fp16_gemv
from .sm70_fp16_hc import _qwen38_sm70_fp16_fused_hc

logger = init_logger(__name__)


def _channel():
    from vllm.distributed.parallel_state import get_tp_group

    try:
        device_comm = get_tp_group().device_communicator
        return getattr(device_comm, "sm70_hc_batch_comm", None)
    except (AssertionError, AttributeError, RuntimeError):
        return None


def _decode_context_ok() -> bool:
    if not is_forward_context_available():
        return False
    context = get_forward_context()
    key = "sm70_batch_hc_decode"
    if key not in context.additional_kwargs:
        context.additional_kwargs[key] = is_uniform_decode_metadata(
            context.attn_metadata
        )
    return bool(context.additional_kwargs[key])


def _supported_layers(child: nn.Module) -> bool:
    down = getattr(child, "input_mix_weight_down_block_inject", None)
    up = getattr(child, "input_mix_weight_up", None)
    # Unknown methods or LoRA wrappers must retain their own forward semantics.
    return (
        getattr(child, "use_combine", False)
        and type(down) is MergedColumnParallelLinear
        and type(up) is ReplicatedLinear
        and type(down.quant_method)
        in (UnquantizedLinearMethod, Qwen38SM70FP16LinearMethod)
        and type(up.quant_method) is UnquantizedLinearMethod
        and (
            getattr(child, "_sm70_qwen38_fp16_fused_hc", False)
            or not any(
                getattr(layer, "_sm70_f16_prepared", False) for layer in (down, up)
            )
        )
    )


def _supported_weights(down: torch.Tensor, up: torch.Tensor) -> bool:
    return (
        down.shape == (336, 10240)
        and up.shape == (10240, 320)
        and down.dtype == up.dtype == torch.float16
        and down.device == up.device
        and down.is_contiguous()
        and up.is_contiguous()
    )


def _copy_up_shard(child: nn.Module, up: torch.Tensor, rank: int) -> None:
    shard = up.view(4, 2560, 320)[:, rank * 640 : (rank + 1) * 640]
    packed = shard.reshape(2560, 320).contiguous()
    existing = getattr(child, "_sm70_batch_hc_up", None)
    if existing is not None:
        if (
            existing.shape != packed.shape
            or existing.dtype != packed.dtype
            or existing.device != packed.device
        ):
            raise RuntimeError(
                "HC reload changed a captured shard layout; rebuild graphs"
            )
        existing.copy_(packed)
    else:
        child.register_buffer("_sm70_batch_hc_up", packed, persistent=False)


@torch.no_grad()
def prepare_sm70_batch_hc(module: nn.Module) -> None:
    """Run after all quantization post-load hooks, never during a forward."""
    if not envs.VLLM_SM70_QWEN38_BATCH_HC_FP16:
        return
    if not current_platform.is_device_capability(70):
        return
    channel = _channel()
    if channel is None or not channel.supports_sm70_qwen38_hc_batch():
        logger.warning_once("SM70 batch HC unavailable: retaining original HC path.")
        return
    count = 0
    for child in module.modules():
        if not _supported_layers(child):
            continue
        down = child.input_mix_weight_down_block_inject.weight
        up = child.input_mix_weight_up.weight
        if not (down.is_cuda and _supported_weights(down, up)):
            continue
        if any(getattr(w, "_vllm_is_uva_offloaded", False) for w in (down, up)):
            continue
        _copy_up_shard(child, up, channel.rank)
        count += 1
    logger.info_once(
        "Prepared %d experimental SM70 FP16 batch HC up shards (%.1f MiB/rank); "
        "M1/prefill fallbacks preserved, dedicated communication channel.",
        count,
        count * 2560 * 320 * 2 / 1024**2,
    )


def _batch_hc(
    x: torch.Tensor,
    down: torch.Tensor,
    up: torch.Tensor,
    packed_up: torch.Tensor,
    legacy_fused: bool,
    down_gemv: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Keep all token-count decisions inside the opaque op. A prefill-first
    # dynamic compile must not bake its fallback into later decode graphs.
    channel = None
    if (
        not envs.VLLM_BATCH_INVARIANT
        and x.ndim == 2
        and 2 <= x.shape[0] <= 16
        and _supported_weights(down, up)
        and x.device == down.device == packed_up.device
        and packed_up.shape == (2560, 320)
        and packed_up.dtype == torch.float16
        and packed_up.is_contiguous()
        and _decode_context_ok()
    ):
        channel = _channel()
    if channel is not None and channel.can_sm70_qwen38_hc_batch(x):
        local_down = down.narrow(0, channel.rank * 80, 88)
        projected = torch.nn.functional.linear(x, local_down)
        lora = x.new_empty((x.shape[0], 320))
        injection = x.new_empty((x.shape[0], 4))
        block = x.new_empty((x.shape[0], 2560))
        ops.sm70_qwen38_hc_batch_down(channel._ptr, projected, injection, lora)
        gate = torch.nn.functional.linear(lora, packed_up)
        ops.sm70_qwen38_hc_batch_mix(channel._ptr, gate, x, block)
        logger.info_once("Using experimental SM70 batch HC (rows=%d).", x.shape[0])
        return block, injection

    if legacy_fused:
        return _qwen38_sm70_fp16_fused_hc(x, down, up)
    projected = (
        _qwen38_sm70_fp16_gemv(x, down)
        if down_gemv
        else torch.nn.functional.linear(x, down)
    )
    lora = hc_silu(projected[:, :320], 4)
    return hc_gate_mix(x, torch.nn.functional.linear(lora, up), 4), projected[
        :, 320:324
    ]


def _batch_hc_fake(
    x: torch.Tensor,
    down: torch.Tensor,
    up: torch.Tensor,
    packed_up: torch.Tensor,
    legacy_fused: bool,
    down_gemv: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    return x.new_empty((x.shape[0], 2560)), x.new_empty((x.shape[0], 4))


direct_register_custom_op(
    op_name="qwen38_sm70_batch_hc",
    op_func=_batch_hc,
    fake_impl=_batch_hc_fake,
)


def maybe_apply_sm70_batch_hc(child: nn.Module, x: torch.Tensor):
    packed = getattr(child, "_sm70_batch_hc_up", None)
    if (
        packed is None
        or envs.VLLM_BATCH_INVARIANT
        or not use_sm70_decode_graph_semantics()
    ):
        return None
    if not _supported_layers(child):
        return None
    down_layer = child.input_mix_weight_down_block_inject
    if not _supported_weights(down_layer.weight, child.input_mix_weight_up.weight):
        return None
    return torch.ops.vllm.qwen38_sm70_batch_hc(
        x,
        down_layer.weight,
        child.input_mix_weight_up.weight,
        packed,
        getattr(child, "_sm70_qwen38_fp16_fused_hc", False),
        type(down_layer.quant_method) is Qwen38SM70FP16LinearMethod,
    )
