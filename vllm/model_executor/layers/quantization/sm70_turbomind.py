# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from vllm import envs

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


def should_prepare_turbomind(
    tensor: torch.Tensor,
    default_enabled: bool,
) -> bool:
    return is_exact_sm70_cuda(tensor, use_turbomind(default_enabled))


def should_prepare_turbomind_or_marlin(
    tensor: torch.Tensor,
    default_enabled: bool,
) -> bool:
    return is_exact_sm70_cuda(
        tensor, use_turbomind(default_enabled) or forces_marlin()
    )


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
            "SM70 TurboMind GPTQ supports group_size 128, "
            f"but got {group_size}."
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
        layer.weight_scale.data.t().to(torch.float32)
        * layer.weight_global_scale.to(torch.float32)
    ).to(torch.float16).contiguous()
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


# ---------------------------------------------------------------------------
# Hybrid linear-attention (GatedDeltaNet) policy for SM70 NVFP4
# ---------------------------------------------------------------------------
# Qwen3.6 hybrid layers expose NVFP4-able linears under ``*.linear_attn.*``.
# Aggressive-style checkpoints often run those on TurboMind safely; abliterated
# Medium/TC builds can emit garbage if forced through the fused path. Policy:
#   VLLM_SM70_NVFP4_LINEAR_ATTN=auto|tm|dequant  (default: auto)
#     auto    – TM when scale-health passes, else load-time dequant → half GEMM
#     tm      – always TurboMind for hybrid candidates
#     dequant – always dequant hybrid candidates (safe default for fragile bases)
#
# Non-candidate dense linears are unchanged (TM when enabled).

DEQUANT_ATTR = "_sm70_nvfp4_dequant_weight"
HYBRID_LINEAR_ATTN_LEAVES = (
    "in_proj_qkv",
    "in_proj_z",
    "out_proj",
)


def _layer_prefix(layer: torch.nn.Module) -> str:
    return str(getattr(layer, "prefix", "") or "")


def is_hybrid_linear_attn_nvfp4_candidate(layer: torch.nn.Module) -> bool:
    """True for GDN linears that are commonly NVFP4-packed (not a/b/norm/conv)."""
    prefix = _layer_prefix(layer)
    if "linear_attn" not in prefix:
        return False
    leaf = prefix.rsplit(".", 1)[-1]
    return leaf in HYBRID_LINEAR_ATTN_LEAVES


def hybrid_linear_attn_mode() -> Literal["auto", "tm", "dequant"]:
    return envs.get_sm70_nvfp4_linear_attn_mode()


def nvfp4_scale_health_ok(layer: torch.nn.Module) -> bool:
    """Cheap finite / dynamic-range check on NVFP4 scales before TM prepare."""
    try:
        if not hasattr(layer, "weight_scale") or not hasattr(layer, "weight_global_scale"):
            return False
        gs = layer.weight_global_scale.detach().float().reshape(-1)
        if gs.numel() == 0 or not torch.isfinite(gs).all():
            return False
        gmax = float(gs.abs().max().item())
        gmin = float(gs.abs().clamp_min(1e-30).min().item())
        if gmax > 1e6 or gmin < 1e-12:
            return False
        sc = layer.weight_scale.detach()
        if sc.dtype == torch.float8_e4m3fn:
            scf = sc.to(torch.float32)
        else:
            scf = sc.float()
        if not torch.isfinite(scf).all():
            return False
        pos = scf.abs()
        pos = pos[pos > 0]
        if pos.numel() > 0:
            ratio = float((pos.max() / pos.min().clamp_min(1e-30)).item())
            if ratio > 1e6:
                return False
        return True
    except Exception:
        return False


def has_dequant_linear(layer: torch.nn.Module) -> bool:
    return getattr(layer, DEQUANT_ATTR, None) is not None


def prepare_nvfp4_dequant_linear(
    layer: torch.nn.Module,
    dtype: torch.dtype = torch.float16,
) -> None:
    """Load-time dequant of packed NVFP4 weights to half/bf16 for F.linear."""
    from vllm.logger import init_logger
    from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
        dequantize_to_dtype,
    )

    logger = init_logger(__name__)
    weight = layer.weight.data
    scale = layer.weight_scale.data
    global_scale = layer.weight_global_scale.data
    # PerTensorScaleParameter may be multi-element for fused shards; take max.
    if global_scale.numel() > 1:
        global_scale = global_scale.reshape(-1).max().reshape(1)
    else:
        global_scale = global_scale.reshape(1)

    # CT/ModelOpt packed layout is linear (non-swizzled) after loader rename.
    w_hp = dequantize_to_dtype(
        weight.contiguous(),
        scale.contiguous(),
        global_scale.contiguous(),
        dtype=dtype,
        block_size=NVFP4_GROUP_SIZE,
        swizzle=False,
    )
    setattr(
        layer,
        DEQUANT_ATTR,
        Parameter(w_hp.contiguous(), requires_grad=False),
    )
    # Drop packed tensors to free VRAM (match TM prepare cleanup).
    device = w_hp.device
    layer.weight = Parameter(
        torch.empty(0, dtype=torch.uint8, device=device), requires_grad=False
    )
    layer.weight_scale = Parameter(
        torch.empty(0, dtype=torch.float8_e4m3fn, device=device), requires_grad=False
    )
    logger.info_once(
        "SM70 NVFP4 hybrid linear_attn dequant path enabled "
        "(VLLM_SM70_NVFP4_LINEAR_ATTN=%s).",
        hybrid_linear_attn_mode(),
    )


def apply_dequant_linear(
    layer: torch.nn.Module,
    x: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    w = getattr(layer, DEQUANT_ATTR)
    return F.linear(x, w.to(dtype=x.dtype), bias)


def should_dequant_hybrid_linear_attn(layer: torch.nn.Module) -> bool:
    if not is_hybrid_linear_attn_nvfp4_candidate(layer):
        return False
    mode = hybrid_linear_attn_mode()
    if mode == "dequant":
        return True
    if mode == "tm":
        return False
    # auto
    return not nvfp4_scale_health_ok(layer)


def try_prepare_sm70_nvfp4_linear(layer: torch.nn.Module) -> bool:
    """Prepare SM70 NVFP4 for a linear layer.

    Returns True if the layer was handled (TurboMind or hybrid dequant).
    Callers should fall back to Marlin / default kernels when False.
    """
    if not should_prepare_turbomind(layer.weight, envs.VLLM_SM70_NVFP4_TURBOMIND):
        return False

    if should_dequant_hybrid_linear_attn(layer):
        prepare_nvfp4_dequant_linear(layer, dtype=torch.float16)
        return True

    from vllm.logger import init_logger

    logger = init_logger(__name__)
    if is_hybrid_linear_attn_nvfp4_candidate(layer):
        logger.info_once(
            "SM70 NVFP4 hybrid linear_attn TurboMind path enabled "
            "(VLLM_SM70_NVFP4_LINEAR_ATTN=%s).",
            hybrid_linear_attn_mode(),
        )
    else:
        logger.info_once(
            "SM70 compressed-tensors/ModelOpt NVFP4 TurboMind W4A16 dense path "
            "enabled."
        )
    prepare_nvfp4_linear(layer)
    # Free packed storage after TM prepare (weights live in STATE_ATTR).
    device = layer.weight.device
    layer.weight = Parameter(
        torch.empty(0, dtype=torch.uint8, device=device), requires_grad=False
    )
    if hasattr(layer, "weight_scale"):
        layer.weight_scale = Parameter(
            torch.empty(0, dtype=torch.float8_e4m3fn, device=device),
            requires_grad=False,
        )
    return True


def try_apply_sm70_nvfp4_linear(
    layer: torch.nn.Module,
    x: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor | None:
    """Apply SM70 NVFP4 path if prepared; else return None."""
    if has_dequant_linear(layer):
        return apply_dequant_linear(layer, x, bias)
    if has_prepared_linear(layer):
        return apply_prepared_linear(layer, x, bias)
    return None
