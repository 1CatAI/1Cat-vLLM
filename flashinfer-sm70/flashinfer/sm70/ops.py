# SPDX-License-Identifier: Apache-2.0
"""FP16 Volta primitives with explicit outputs and preparation before capture."""

import math

import torch

from ._jit import prepare


def _fp16(*tensors):
    device = tensors[0].device
    if device.type != "cuda" or any(
        t.device != device or t.dtype != torch.float16 for t in tensors
    ):
        raise ValueError("SM70 primitives require same-device CUDA FP16 tensors")
    if torch.cuda.get_device_capability(device) != (7, 0):
        raise ValueError("SM70 primitives require a Volta device")


def _pdl(enable_pdl):
    if enable_pdl:
        raise ValueError("Programmatic dependent launch is unavailable on SM70")


@torch.library.custom_op("flashinfer_sm70::norm", mutates_args=("out", "input"))
def _norm(
    out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    gemma: bool,
    residual: bool,
) -> None:
    prepare("basics").norm(out, input, weight, eps, gemma, residual)


@_norm.register_fake
def _norm_fake(out, input, weight, eps, gemma, residual):
    return None


def _norm_args(input, weight, eps):
    _fp16(input, weight)
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be positive and finite")
    if input.ndim not in (2, 3) or not input.is_contiguous():
        raise ValueError("SM70 norm requires contiguous 2D/3D input")
    if (
        not 0 < input.shape[-1] <= 8192
        or weight.shape != (input.shape[-1],)
        or not weight.is_contiguous()
    ):
        raise ValueError(
            "SM70 norm requires a contiguous weight and hidden size 1..8192"
        )


def _rmsnorm(input, weight, eps, out, enable_pdl, gemma):
    _pdl(enable_pdl)
    _norm_args(input, weight, eps)
    if out is None:
        out = torch.empty_like(input)
    _fp16(out, input)
    if out.shape != input.shape or not out.is_contiguous():
        raise ValueError("out must match input shape and be contiguous")
    _norm(
        out.view(-1, input.shape[-1]),
        input.view(-1, input.shape[-1]),
        weight,
        eps,
        gemma,
        False,
    )
    return out


def rmsnorm(input, weight, eps=1e-6, out=None, enable_pdl=None):
    return _rmsnorm(input, weight, eps, out, enable_pdl, False)


def gemma_rmsnorm(input, weight, eps=1e-6, out=None, enable_pdl=None):
    return _rmsnorm(input, weight, eps, out, enable_pdl, True)


def _fused_norm(input, residual, weight, eps, enable_pdl, gemma):
    _pdl(enable_pdl)
    _norm_args(input, weight, eps)
    _fp16(input, residual)
    if residual.shape != input.shape or not residual.is_contiguous():
        raise ValueError("residual must match input shape and be contiguous")
    if torch._C._overlaps(input, residual):
        raise ValueError("input and residual must not alias")
    _norm(
        residual.view(-1, input.shape[-1]),
        input.view(-1, input.shape[-1]),
        weight,
        eps,
        gemma,
        True,
    )


def fused_add_rmsnorm(input, residual, weight, eps=1e-6, enable_pdl=None):
    _fused_norm(input, residual, weight, eps, enable_pdl, False)


def gemma_fused_add_rmsnorm(input, residual, weight, eps=1e-6, enable_pdl=None):
    _fused_norm(input, residual, weight, eps, enable_pdl, True)


@torch.library.custom_op("flashinfer_sm70::activation", mutates_args=("out",))
def _activation(out: torch.Tensor, input: torch.Tensor, kind: int) -> None:
    prepare("basics").activation(out, input, kind)


@_activation.register_fake
def _activation_fake(out, input, kind):
    return None


def _act(input, out, enable_pdl, kind):
    _pdl(enable_pdl)
    _fp16(input)
    if (
        input.ndim < 1
        or input.shape[-1] == 0
        or input.shape[-1] % 2
        or not input.is_contiguous()
    ):
        raise ValueError(
            "activation requires contiguous input with a positive even last dimension"
        )
    shape = (*input.shape[:-1], input.shape[-1] // 2)
    if out is None:
        out = torch.empty(shape, device=input.device, dtype=input.dtype)
    _fp16(input, out)
    if out.shape != shape or not out.is_contiguous():
        raise ValueError("invalid activation output shape or strides")
    _activation(out.view(-1, shape[-1]), input.view(-1, input.shape[-1]), kind)
    return out


def silu_and_mul(input, out=None, enable_pdl=None):
    return _act(input, out, enable_pdl, 0)


def gelu_and_mul(input, out=None, enable_pdl=None):
    return _act(input, out, enable_pdl, 1)


def gelu_tanh_and_mul(input, out=None, enable_pdl=None):
    return _act(input, out, enable_pdl, 2)


@torch.library.custom_op("flashinfer_sm70::rope", mutates_args=("out", "input"))
def _rope(
    out: torch.Tensor,
    input: torch.Tensor,
    positions: torch.Tensor,
    rotary_dim: int,
    interleave: bool,
    scale: float,
    theta: float,
) -> None:
    prepare("basics").rope(out, input, positions, rotary_dim, interleave, scale, theta)


@_rope.register_fake
def _rope_fake(out, input, positions, rotary_dim, interleave, scale, theta):
    return None


def _rope_args(q, k, pos_ids, rotary_dim, rope_scale, rope_theta):
    _fp16(q, k)
    if (
        q.ndim != 3
        or k.ndim != 3
        or q.shape[0] != k.shape[0]
        or q.shape[2] != k.shape[2]
    ):
        raise ValueError(
            "RoPE expects Q/K [tokens, heads, dimension] with matching tokens/dimension"
        )
    if q.stride(-1) != 1 or k.stride(-1) != 1:
        raise ValueError("RoPE requires contiguous head dimensions")
    if (
        pos_ids.device != q.device
        or pos_ids.dtype not in (torch.int32, torch.int64)
        or pos_ids.shape != (q.shape[0],)
        or not pos_ids.is_contiguous()
    ):
        raise ValueError(
            "positions must be contiguous same-device int32/int64 [tokens]"
        )
    rotary_dim = q.shape[-1] if rotary_dim is None else rotary_dim
    if not 0 < rotary_dim <= q.shape[-1] <= 8192 or rotary_dim % 2:
        raise ValueError(
            "rotary_dim must be positive, even and no larger than head dimension"
        )
    if not all(math.isfinite(v) and v > 0 for v in (rope_scale, rope_theta)):
        raise ValueError("RoPE scale and theta must be positive and finite")
    return rotary_dim


def apply_rope_pos_ids(
    q, k, pos_ids, rotary_dim=None, interleave=False, rope_scale=1, rope_theta=1e4
):
    rotary_dim = _rope_args(q, k, pos_ids, rotary_dim, rope_scale, rope_theta)
    qo, ko = torch.empty_like(q), torch.empty_like(k)
    _rope(qo, q, pos_ids, rotary_dim, interleave, float(rope_scale), float(rope_theta))
    _rope(ko, k, pos_ids, rotary_dim, interleave, float(rope_scale), float(rope_theta))
    return qo, ko


def apply_rope_pos_ids_inplace(
    q, k, pos_ids, rotary_dim=None, interleave=False, rope_scale=1, rope_theta=1e4
):
    rotary_dim = _rope_args(q, k, pos_ids, rotary_dim, rope_scale, rope_theta)
    # Native paired-element ownership makes input/output aliasing safe.
    _rope(q, q, pos_ids, rotary_dim, interleave, float(rope_scale), float(rope_theta))
    _rope(k, k, pos_ids, rotary_dim, interleave, float(rope_scale), float(rope_theta))
