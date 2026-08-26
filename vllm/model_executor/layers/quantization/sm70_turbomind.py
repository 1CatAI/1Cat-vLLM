# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import Literal

import torch

from vllm import envs
from vllm.platforms import current_platform

U4_GROUP_SIZES = (32, 64, 128)
GPTQ_GROUP_SIZES = (128,)
COMPRESSED_UINT4_GROUP_SIZES = (32, 128)
MXFP4_GROUP_SIZE = 32
NVFP4_GROUP_SIZE = 16
STATE_ATTR = "_sm70_turbomind_linear"
SM70QuantBackend = Literal["auto", "marlin", "turbomind"]


@dataclass
class SM70TurboMindLinearState:
    weight: torch.Tensor
    scales: torch.Tensor
    group_size: int
    k_ld: int
    q_ld: int
    output_size: int
    op_kind: Literal["uint4", "mxfp4", "nvfp4"]


def quant_backend() -> SM70QuantBackend:
    return envs.get_sm70_quant_backend()


def use_turbomind(default_enabled: bool) -> bool:
    return envs.use_sm70_turbomind(default_enabled)


def forces_marlin() -> bool:
    return envs.force_sm70_marlin()


def is_exact_sm70_cuda(tensor: torch.Tensor, enabled: bool) -> bool:
    if not enabled or not tensor.is_cuda:
        return False
    return torch.cuda.get_device_capability(tensor.device) == (7, 0)


def is_exact_sm70_cuda_platform() -> bool:
    """Return true only for Volta SM70 CUDA workers.

    Quant-method selection runs before a layer owns a CUDA tensor, so it
    cannot use :func:`is_exact_sm70_cuda`. Keep this platform check separate
    from the tensor-based helpers used by linear weight preparation.
    """
    return current_platform.is_cuda() and current_platform.is_device_capability((7, 0))


def should_use_mxfp4_moe_turbomind() -> bool:
    """Select the native MXFP4 MoE path only on exact SM70."""
    return is_exact_sm70_cuda_platform() and use_turbomind(
        envs.VLLM_SM70_MXFP4_TURBOMIND
    )


def should_prepare_turbomind(
    tensor: torch.Tensor,
    default_enabled: bool,
) -> bool:
    return is_exact_sm70_cuda(tensor, use_turbomind(default_enabled))


def should_prepare_turbomind_or_marlin(
    tensor: torch.Tensor,
    default_enabled: bool,
) -> bool:
    return is_exact_sm70_cuda(tensor, use_turbomind(default_enabled) or forces_marlin())


def _get_u4_slices(x: torch.Tensor, dtype: torch.dtype) -> list[torch.Tensor]:
    if x.dtype == torch.int32:
        count = 8
    elif x.dtype == torch.uint8:
        count = 2
    else:
        raise TypeError(f"expected int32 or uint8 packed int4 tensor, got {x.dtype}")
    xs = []
    for _ in range(count):
        xs.append((x & 15).to(dtype))
        x = x >> 4
    return xs


def unpack_gptq_weight(qweight: torch.Tensor) -> torch.Tensor:
    xs = _get_u4_slices(qweight, torch.uint8)
    return torch.stack(xs, dim=1).reshape(-1, qweight.size(-1)).contiguous()


def unpack_gptq_zeros(qzeros: torch.Tensor) -> torch.Tensor:
    xs = _get_u4_slices(qzeros, torch.uint8)
    zeros = torch.stack(xs, dim=-1).reshape(qzeros.size(0), -1)
    return (zeros + 1).to(torch.float16).contiguous()


def unpack_compressed_weight(weight_packed: torch.Tensor) -> torch.Tensor:
    xs = _get_u4_slices(weight_packed, torch.uint8)
    weight = torch.stack(xs, dim=-1).reshape(*weight_packed.shape[:-1], -1)
    return weight.t().contiguous()


def unpack_compressed_zeros(weight_zero_point: torch.Tensor) -> torch.Tensor:
    xs = _get_u4_slices(weight_zero_point, torch.uint8)
    zeros = torch.stack(xs, dim=1).reshape(-1, weight_zero_point.size(-1))
    return zeros.t().to(torch.float16).contiguous()


def unpack_mxfp4_weight(weight_packed: torch.Tensor) -> torch.Tensor:
    if weight_packed.dim() > 2:
        weight_packed = torch.flatten(weight_packed, start_dim=-2)
    xs = _get_u4_slices(weight_packed, torch.uint8)
    weight = torch.flatten(
        torch.stack(xs, dim=-1),
        start_dim=-2,
    )
    return weight.t().contiguous()


def symmetric_int4_zeros_like(scales: torch.Tensor) -> torch.Tensor:
    return torch.full_like(scales, 8, dtype=torch.float16)


def _store_state(
    layer: torch.nn.Module,
    weight: torch.Tensor,
    scales: torch.Tensor,
    meta: torch.Tensor,
    group_size: int,
    output_size: int,
    op_kind: Literal["uint4", "mxfp4", "nvfp4"],
) -> None:
    state = SM70TurboMindLinearState(
        weight=weight,
        scales=scales,
        group_size=group_size,
        k_ld=int(meta[0]),
        q_ld=int(meta[1]),
        output_size=output_size,
        op_kind=op_kind,
    )
    setattr(layer, STATE_ATTR, state)


def has_prepared_linear(layer: torch.nn.Module) -> bool:
    return getattr(layer, STATE_ATTR, None) is not None


def prepare_gptq_linear(
    layer: torch.nn.Module,
    group_size: int,
    interleave_gated_silu: bool = False,
) -> None:
    if group_size not in GPTQ_GROUP_SIZES:
        raise RuntimeError(
            f"SM70 TurboMind GPTQ supports group_size 128, but got {group_size}."
        )
    if not hasattr(torch.ops._C, "uint4_sm70_prepare"):
        raise RuntimeError(
            "VLLM_SM70_GPTQ_TURBOMIND=1 requires a build with CUDA arch 7.0 "
            "and the SM70 TurboMind extension."
        )
    from vllm import _sm70_ops as sm70_ops

    qweight = unpack_gptq_weight(layer.qweight.data)
    scales = layer.scales.data.to(torch.float16).contiguous()
    zeros = unpack_gptq_zeros(layer.qzeros.data)
    tm_weight, tm_scales, meta = sm70_ops.uint4_sm70_prepare(
        qweight, scales, zeros, group_size, interleave_gated_silu
    )
    _store_state(
        layer,
        tm_weight,
        tm_scales,
        meta,
        group_size,
        qweight.size(1),
        "uint4",
    )


def prepare_compressed_uint4_linear(
    layer: torch.nn.Module,
    group_size: int,
    symmetric: bool,
    interleave_gated_silu: bool = False,
) -> None:
    if group_size not in COMPRESSED_UINT4_GROUP_SIZES:
        raise RuntimeError(
            "SM70 TurboMind compressed-tensors int4 supports "
            f"group_size 32/128, but got {group_size}."
        )
    if not hasattr(torch.ops._C, "uint4_sm70_prepare"):
        raise RuntimeError(
            "VLLM_SM70_COMPRESSED_TENSORS_TURBOMIND=1 requires a build with "
            "CUDA arch 7.0 and the SM70 TurboMind extension."
        )
    from vllm import _sm70_ops as sm70_ops

    qweight = unpack_compressed_weight(layer.weight_packed.data)
    scales = layer.weight_scale.data.t().to(torch.float16).contiguous()
    if symmetric:
        zeros = symmetric_int4_zeros_like(scales)
    else:
        zeros = unpack_compressed_zeros(layer.weight_zero_point.data)
    tm_weight, tm_scales, meta = sm70_ops.uint4_sm70_prepare(
        qweight, scales, zeros, group_size, interleave_gated_silu
    )
    _store_state(
        layer,
        tm_weight,
        tm_scales,
        meta,
        group_size,
        qweight.size(1),
        "uint4",
    )


def prepare_mxfp4_linear(
    layer: torch.nn.Module,
    interleave_gated_silu: bool = False,
) -> None:
    if not hasattr(torch.ops._C, "mxfp4_sm70_prepare"):
        raise RuntimeError(
            "VLLM_SM70_MXFP4_TURBOMIND=1 requires a build with CUDA arch 7.0 "
            "and the SM70 TurboMind extension."
        )
    from vllm import _sm70_ops as sm70_ops

    qweight = unpack_mxfp4_weight(layer.weight_packed.data)
    scales = layer.weight_scale.data.t().contiguous()
    tm_weight, tm_scales, meta = sm70_ops.mxfp4_sm70_prepare(
        qweight, scales, MXFP4_GROUP_SIZE, interleave_gated_silu
    )
    _store_state(
        layer,
        tm_weight,
        tm_scales,
        meta,
        MXFP4_GROUP_SIZE,
        qweight.size(1),
        "mxfp4",
    )


def prepare_nvfp4_linear(
    layer: torch.nn.Module,
    interleave_gated_silu: bool = False,
) -> None:
    if not hasattr(torch.ops._C, "nvfp4_sm70_prepare"):
        raise RuntimeError(
            "VLLM_SM70_NVFP4_TURBOMIND=1 requires a build with CUDA arch 7.0 "
            "and the SM70 TurboMind NVFP4 extension."
        )
    from vllm import _sm70_ops as sm70_ops

    qweight = unpack_mxfp4_weight(layer.weight.data)
    scales = (
        (
            layer.weight_scale.data.t().to(torch.float32)
            * layer.weight_global_scale.to(torch.float32)
        )
        .to(torch.float16)
        .contiguous()
    )
    tm_weight, tm_scales, meta = sm70_ops.nvfp4_sm70_prepare(
        qweight, scales, NVFP4_GROUP_SIZE, interleave_gated_silu
    )
    _store_state(
        layer,
        tm_weight,
        tm_scales,
        meta,
        NVFP4_GROUP_SIZE,
        qweight.size(1),
        "nvfp4",
    )


def apply_prepared_linear(
    layer: torch.nn.Module,
    x: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    state = getattr(layer, STATE_ATTR)
    reshaped_x = x.reshape(-1, x.shape[-1])
    out_shape = x.shape[:-1] + (state.output_size,)
    out = torch.empty(
        (reshaped_x.shape[0], state.output_size),
        dtype=x.dtype,
        device=x.device,
    )
    from vllm import _sm70_ops as sm70_ops

    if state.op_kind == "uint4":
        sm70_ops.awq_gemm_sm70_out(
            out,
            reshaped_x,
            state.weight,
            state.scales,
            state.group_size,
            state.k_ld,
            state.q_ld,
        )
    elif state.op_kind == "mxfp4":
        sm70_ops.mxfp4_gemm_sm70_out(
            out,
            reshaped_x,
            state.weight,
            state.scales,
            state.group_size,
            state.k_ld,
            state.q_ld,
        )
    elif state.op_kind == "nvfp4":
        sm70_ops.nvfp4_gemm_sm70_out(
            out,
            reshaped_x,
            state.weight,
            state.scales,
            state.group_size,
            state.k_ld,
            state.q_ld,
        )
    else:
        raise AssertionError(f"unknown SM70 TurboMind op kind: {state.op_kind}")
    if bias is not None:
        out.add_(bias)
    return out.reshape(out_shape)
