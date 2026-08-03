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


# Block-scale magnitude threshold for NVFP4 scale-convention detection.
#
# Aggressive CT exports fold magnitude into FP8 block scales (often O(1)–448)
# with a large disk global; CT then stores ``global := 1/disk_global`` and
# TM/dequant do ``e2m1 * block * global`` (= block/disk_global).
#
# Medium/TC/cyber GPTQ exports leave *tiny* block scales (max ≪ 1) that
# already hold the true per-block scale, but still ship a large disk global.
# Applying 1/disk_global then multiplies under-scales weights by ~G → salad.
#
# Fix (do NOT do ``weight_scale *= global`` — that overflows fp8-e4m3 at 448):
# set ``weight_global_scale = 1.0`` (fp32) and leave block scales untouched.
# See /rivet-shared/plans/nvfp4-medium-scale-fix.md.
_NVFP4_TINY_BLOCK_SCALE_MAX = 1.0


def nvfp4_block_scales_are_tiny(layer: torch.nn.Module) -> bool:
    """True when per-group weight_scale looks like the "tiny-block" CT export."""
    if not hasattr(layer, "weight_scale"):
        return False
    sc = layer.weight_scale.detach()
    if sc.dtype == torch.float8_e4m3fn:
        scf = sc.to(torch.float32)
    else:
        scf = sc.float()
    if scf.numel() == 0 or not torch.isfinite(scf).all():
        return False
    return float(scf.abs().max().item()) < _NVFP4_TINY_BLOCK_SCALE_MAX


def normalize_nvfp4_global_scale_for_sm70(layer: torch.nn.Module) -> None:
    """Rewrite ``weight_global_scale`` for SM70-safe dequant/TM combine.

    Call **before** CT's usual ``global := 1/disk_global`` (or instead of it):

    - Tiny block scales (Medium/TC): set global to **1.0** (identity). Block
      scales already hold the true per-block value; do not touch weight_scale
      (fp8 overflow if you fold global into block scales without clamping).
    - Large block scales (Aggressive): ``global := 1/disk_global`` (existing CT
      convention expected by TurboMind / Marlin).
    """
    from vllm.logger import init_logger

    if not hasattr(layer, "weight_global_scale"):
        return
    raw = layer.weight_global_scale.detach().float()
    if raw.numel() == 0:
        return
    device = layer.weight_global_scale.device
    if nvfp4_block_scales_are_tiny(layer):
        layer.weight_global_scale = Parameter(
            torch.ones(1, dtype=torch.float32, device=device),
            requires_grad=False,
        )
        init_logger(__name__).info_once(
            "SM70 NVFP4: tiny block-scale convention (max |weight_scale| < 1); "
            "set weight_global_scale=1.0 (leave block scales untouched). "
            "Fixes Medium/TC GPTQ under-scale without fp8 overflow."
        )
        return
    # Aggressive / folded-block convention: CT stores the reciprocal.
    g = float(raw.reshape(-1).max().item())
    layer.weight_global_scale = Parameter(
        torch.tensor([1.0 / g], dtype=torch.float32, device=device),
        requires_grad=False,
    )


def combine_nvfp4_scales_for_sm70_tm(layer: torch.nn.Module) -> torch.Tensor:
    """Combine block + global scales for ``nvfp4_sm70_prepare``.

    Expects ``weight_global_scale`` already normalized via
    ``normalize_nvfp4_global_scale_for_sm70`` (or equivalent). Always
    ``block * global``; for Medium, global is 1.0 so this is a no-op multiply.
    """
    block = layer.weight_scale.data.t().to(torch.float32)
    g = layer.weight_global_scale.to(torch.float32)
    # Safety net if a caller skipped normalize_*: treat as identity global.
    if nvfp4_block_scales_are_tiny(layer):
        g_abs = float(g.detach().float().abs().max().item()) if g.numel() else 1.0
        # After proper normalize, g is 1.0. If still ~1/disk_global (≪ 1), skip.
        if g_abs < 0.5:
            return block.to(torch.float16).contiguous()
    return (block * g).to(torch.float16).contiguous()


def resolve_nvfp4_global_for_dequant(layer: torch.nn.Module) -> torch.Tensor:
    """Global scale for ``dequantize_to_dtype`` (``block * global``)."""
    global_scale = layer.weight_global_scale.data
    if global_scale.numel() > 1:
        global_scale = global_scale.reshape(-1).max().reshape(1)
    else:
        global_scale = global_scale.reshape(1)
    if nvfp4_block_scales_are_tiny(layer):
        g_abs = float(global_scale.detach().float().abs().max().item())
        if g_abs < 0.5:
            return torch.ones(1, dtype=torch.float32, device=global_scale.device)
    return global_scale.to(torch.float32)


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
    scales = combine_nvfp4_scales_for_sm70_tm(layer)
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
# Aggressive-style checkpoints run those on TurboMind safely. Medium/TC dense
# garbage was largely a block/global scale convention mismatch (see
# combine_nvfp4_scales_for_sm70_tm); hybrid dequant remains available as a
# stamp/env override for residual fragility.
#
# Policy resolution (see envs.get_sm70_nvfp4_linear_attn_mode):
#   1. env VLLM_SM70_NVFP4_LINEAR_ATTN=tm|dequant|auto  (override)
#   2. checkpoint quantization_config.sm70_linear_attn stamp
#   3. default tm  (speed; do not tax the good case)
#
# Stamp fragile builds at quant time, e.g.:
#   quantization_config["sm70_linear_attn"] = "dequant"
# so operators need no env. Non-candidate dense linears always use TM.

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


def _resolve_dequant_dtype(layer: torch.nn.Module) -> torch.dtype:
    """Prefer the layer/model compute dtype over a hardcoded fp16."""
    for attr in ("params_dtype", "dtype"):
        dt = getattr(layer, attr, None)
        if isinstance(dt, torch.dtype) and dt in (torch.float16, torch.bfloat16):
            return dt
    try:
        from vllm.config import get_current_vllm_config

        dt = get_current_vllm_config().model_config.dtype
        if dt in (torch.float16, torch.bfloat16):
            return dt
    except Exception:
        pass
    return torch.float16


def prepare_nvfp4_dequant_linear(
    layer: torch.nn.Module,
    dtype: torch.dtype | None = None,
) -> None:
    """Load-time dequant of packed NVFP4 weights to half/bf16 for F.linear."""
    from vllm.logger import init_logger
    from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
        dequantize_to_dtype,
    )

    logger = init_logger(__name__)
    if dtype is None:
        dtype = _resolve_dequant_dtype(layer)
    weight = layer.weight.data
    scale = layer.weight_scale.data
    # Same tiny-block convention handling as TurboMind combine.
    global_scale = resolve_nvfp4_global_for_dequant(layer)

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
    if hasattr(layer, "weight_global_scale"):
        layer.weight_global_scale = Parameter(
            torch.empty(0, dtype=torch.float32, device=device), requires_grad=False
        )
    logger.info_once(
        "SM70 NVFP4 hybrid linear_attn dequant path enabled "
        "(mode=%s, dtype=%s).",
        hybrid_linear_attn_mode(),
        dtype,
    )


def apply_dequant_linear(
    layer: torch.nn.Module,
    x: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    w = getattr(layer, DEQUANT_ATTR)
    # Weight already dequanted to compute dtype; cast only if mismatch.
    if w.dtype != x.dtype:
        w = w.to(dtype=x.dtype)
    return F.linear(x, w, bias)


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
        prepare_nvfp4_dequant_linear(layer)
        return True

    from vllm.logger import init_logger

    logger = init_logger(__name__)
    if is_hybrid_linear_attn_nvfp4_candidate(layer):
        logger.info_once(
            "SM70 NVFP4 hybrid linear_attn TurboMind path enabled "
            "(mode=%s).",
            hybrid_linear_attn_mode(),
        )
    else:
        logger.info_once(
            "SM70 compressed-tensors/ModelOpt NVFP4 TurboMind weight-only "
            "dense path enabled (W4A16 semantics; acts remain half/bf16)."
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
