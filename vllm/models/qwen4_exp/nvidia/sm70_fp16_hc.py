# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in fused checkpoint-FP16 HyperConnection decode route for SM70."""

from __future__ import annotations

import torch
from torch import nn

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

from .sm70_fp16_gemv import _exact_runtime_contract

logger = init_logger(__name__)

_HC_COUNT = 4
_HC_DIM = 2560
_HC_RANK = 320
_HC_HIDDEN = _HC_COUNT * _HC_DIM


@triton.jit
def _qwen38_hc_down_silu_inject_kernel(
    x_ptr,
    weight_ptr,
    lora_ptr,
    injection_ptr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    RANK_VALUE: tl.constexpr,
    HC_COUNT: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for block_start in tl.static_range(0, K, BLOCK_K):
        indices = block_start + offsets
        mask = indices < K
        x = tl.load(
            x_ptr + indices,
            mask=mask,
            other=0.0,
            eviction_policy="evict_last",
        )
        weight = tl.load(
            weight_ptr + row * K + indices,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        )
        acc += x.to(tl.float32) * weight.to(tl.float32)

    # Preserve the baseline GEMV -> FP16 -> SiLU boundary.
    value = tl.sum(acc, axis=0).to(tl.float16).to(tl.float32)
    is_lora = row < RANK_VALUE
    scaled = value / HC_COUNT
    tl.store(lora_ptr + row, scaled * tl.sigmoid(scaled), mask=is_lora)
    tl.store(injection_ptr + row - RANK_VALUE, value, mask=~is_lora)


@triton.jit
def _qwen38_hc_up_gate_mix_kernel(
    lora_ptr,
    weight_ptr,
    x_ptr,
    out_ptr,
    K: tl.constexpr,
    HC_DIMENSION: tl.constexpr,
    HC_COUNT: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    hidden = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    mask = offsets < K
    lora = tl.load(
        lora_ptr + offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)

    result = 0.0
    for stream in tl.static_range(HC_COUNT):
        row = stream * HC_DIMENSION + hidden
        weight = tl.load(
            weight_ptr + row * K + offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        )
        # Preserve the baseline GEMV -> FP16 gate -> sigmoid boundary.
        gate = tl.sum(lora * weight.to(tl.float32), axis=0)
        gate = gate.to(tl.float16).to(tl.float32)
        branch = tl.load(x_ptr + stream * HC_DIMENSION + hidden).to(tl.float32)
        result += tl.sigmoid(gate) * branch
    tl.store(out_ptr + hidden, result / HC_COUNT)


def _runtime_ok(
    x: torch.Tensor, down_weight: torch.Tensor, up_weight: torch.Tensor
) -> bool:
    return bool(
        x.ndim == 2
        and x.shape == (1, _HC_HIDDEN)
        and down_weight.shape == (_HC_RANK + _HC_COUNT + 12, _HC_HIDDEN)
        and up_weight.shape == (_HC_HIDDEN, _HC_RANK)
        and x.dtype == torch.float16
        and down_weight.dtype == torch.float16
        and up_weight.dtype == torch.float16
        and x.is_cuda
        and down_weight.is_cuda
        and up_weight.is_cuda
        and x.is_contiguous()
        and down_weight.is_contiguous()
        and up_weight.is_contiguous()
        and x.device == down_weight.device == up_weight.device
    )


def _qwen38_sm70_fp16_fused_hc(
    x: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not _runtime_ok(x, down_weight, up_weight):
        # Preserve the ordinary projection and FP16 materialization boundaries
        # for prefill and any unsupported runtime shape. This fallback lives
        # inside the opaque op so a prefill-first dynamic compile cannot bake
        # the M > 1 decision into subsequent decode graphs.
        down_and_injection = torch.nn.functional.linear(x, down_weight)
        lora = torch.ops.vllm.qwen4_exp_hc_silu(
            down_and_injection[..., :_HC_RANK], _HC_COUNT
        )
        injection = down_and_injection[..., _HC_RANK : _HC_RANK + _HC_COUNT]
        gate = torch.nn.functional.linear(lora, up_weight)
        block = torch.ops.vllm.qwen4_exp_hc_gate_mix(x, gate, _HC_COUNT)
        return block, injection
    lora = x.new_empty((1, _HC_RANK))
    injection = x.new_empty((1, _HC_COUNT))
    block = x.new_empty((1, _HC_DIM))
    _qwen38_hc_down_silu_inject_kernel[(_HC_RANK + _HC_COUNT,)](
        x,
        down_weight,
        lora,
        injection,
        K=_HC_HIDDEN,
        BLOCK_K=256,
        RANK_VALUE=_HC_RANK,
        HC_COUNT=_HC_COUNT,
        num_warps=4,
    )
    _qwen38_hc_up_gate_mix_kernel[(_HC_DIM,)](
        lora,
        up_weight,
        x,
        block,
        K=_HC_RANK,
        HC_DIMENSION=_HC_DIM,
        HC_COUNT=_HC_COUNT,
        BLOCK_K=512,
        num_warps=2,
    )
    logger.info_once("SM70 Qwen3.8 fused checkpoint-FP16 HC M=1 route enabled.")
    return block, injection


def _qwen38_sm70_fp16_fused_hc_fake(
    x: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    del down_weight, up_weight
    return (
        x.new_empty((*x.shape[:-1], _HC_DIM)),
        x.new_empty((*x.shape[:-1], _HC_COUNT)),
    )


direct_register_custom_op(
    op_name="qwen38_sm70_fp16_fused_hc",
    op_func=_qwen38_sm70_fp16_fused_hc,
    fake_impl=_qwen38_sm70_fp16_fused_hc_fake,
)


def maybe_apply_qwen38_sm70_fp16_fused_hc(
    down_layer: nn.Module,
    up_layer: nn.Module,
    x: torch.Tensor,
    enabled: bool,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not enabled:
        return None
    down_weight = getattr(down_layer, "weight", None)
    up_weight = getattr(up_layer, "weight", None)
    if down_weight is None or up_weight is None:
        return None
    if down_weight.shape != (
        _HC_RANK + _HC_COUNT + 12,
        _HC_HIDDEN,
    ) or up_weight.shape != (_HC_HIDDEN, _HC_RANK):
        return None
    return torch.ops.vllm.qwen38_sm70_fp16_fused_hc(x, down_weight, up_weight)


def enable_qwen38_sm70_fp16_fused_hc(
    module: nn.Module, dtype: torch.dtype, vllm_config=None
) -> None:
    """Mark exact base-model HC modules for the fused M=1 route."""
    if (
        not envs.VLLM_SM70_QWEN38_FUSED_HC_FP16
        or envs.VLLM_SM70_QWEN4_EXP_ONLINE_QPN8
        or dtype != torch.float16
        or not current_platform.is_device_capability((7, 0))
        or not _exact_runtime_contract(vllm_config)
    ):
        return

    enabled_count = 0
    for child in module.modules():
        if not (
            getattr(child, "use_combine", False)
            and getattr(child, "lora_rank", None) == _HC_RANK
            and getattr(child, "hc_count", None) == _HC_COUNT
            and getattr(child, "hidden_size", None) == _HC_DIM
            and hasattr(child, "input_mix_weight_down_block_inject")
            and hasattr(child, "input_mix_weight_up")
        ):
            continue
        child._sm70_qwen38_fp16_fused_hc = True
        enabled_count += 1

    if enabled_count:
        logger.info_once(
            "Prepared %d Qwen3.8 SM70 fused checkpoint-FP16 HC modules.",
            enabled_count,
        )


__all__ = [
    "enable_qwen38_sm70_fp16_fused_hc",
    "maybe_apply_qwen38_sm70_fp16_fused_hc",
]
