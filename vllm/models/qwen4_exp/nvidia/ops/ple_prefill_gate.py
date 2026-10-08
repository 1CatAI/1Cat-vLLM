# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep PLE prefill values compact instead of expanding all HC streams."""

import math

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op


@triton.jit
def _prepare(
    Key,
    Query,
    Value,
    KeyNorm,
    QueryNorm,
    ConvNorm,
    Gates,
    KS: tl.constexpr,
    QS: tl.constexpr,
    VS: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    Eps: tl.constexpr,
    B: tl.constexpr,
):
    row, branch = tl.program_id(0), tl.program_id(1)
    d = tl.arange(0, B)
    mask = d < H
    offset = branch * H + d
    key = tl.load(Key + row * KS + offset, mask, other=0).to(tl.float32)
    query = tl.load(Query + row * QS + offset, mask, other=0).to(tl.float32)
    kw = 1 + tl.load(KeyNorm + offset, mask, other=0).to(tl.float32)
    qw = 1 + tl.load(QueryNorm + offset, mask, other=0).to(tl.float32)
    # Production prefill keeps these intermediates in FP32 after fusion.
    key = key * tl.rsqrt(tl.sum(key * key, 0) / H + Eps) * kw
    query = query * tl.rsqrt(tl.sum(query * query, 0) / H + Eps) * qw
    total = tl.sum(tl.where(mask, key * query, 0), 0)
    score = total / math.sqrt(H)
    magnitude = tl.where(tl.abs(score) < 1e-6, 1e-6, tl.abs(score))
    root = tl.sqrt_rn(magnitude)
    sign = tl.where(score > 0, 1, tl.where(score < 0, -1, 0))
    signed = sign * root
    gate = tl.sigmoid(signed)
    tl.store(Gates + row * G + branch, gate)
    value = tl.load(Value + row * VS + d, mask, other=0).to(tl.float32)
    full_gated = gate * value
    # Reduction sees the FP32 product; the following norm reads its FP16
    # materialization across the production kernel boundary.
    scale = tl.rsqrt(tl.sum(full_gated * full_gated, 0) / H + Eps)
    gated = full_gated.to(tl.float16).to(tl.float32)
    cw = 1 + tl.load(ConvNorm + offset, mask, other=0).to(tl.float32)
    # All input elements for this branch have been read before its key storage
    # becomes the convolution input. Other branches own disjoint rows.
    tl.store(Key + row * KS + offset, gated * scale * cw, mask)


@triton.jit
def _finish(
    Output,
    Value,
    Gates,
    Residual,
    N: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    OS: tl.constexpr,
    VS: tl.constexpr,
    RS: tl.constexpr,
    HasResidual: tl.constexpr,
    B: tl.constexpr,
):
    i = tl.program_id(0) * B + tl.arange(0, B)
    row = i // (G * H)
    branch = i // H % G
    d = i % H
    value = tl.load(Value + row * VS + d, i < N, other=0).to(tl.float32)
    gate = tl.load(Gates + row * G + branch, i < N, other=0).to(tl.float32)
    residual = (value * gate).to(tl.float16).to(tl.float32)
    offset = row * OS + branch * H + d
    conv = tl.load(Output + offset, i < N, other=0).to(tl.float32)
    result = residual + conv
    if HasResidual:
        hidden = tl.load(Residual + row * RS + branch * H + d, i < N, other=0)
        result = hidden.to(tl.float32) + result
    tl.store(Output + offset, result, i < N)


def _prepare_op(
    key: torch.Tensor,
    query: torch.Tensor,
    value: torch.Tensor,
    key_norm: torch.Tensor,
    query_norm: torch.Tensor,
    conv_norm: torch.Tensor,
    gates: torch.Tensor,
    eps: float,
) -> None:
    rows, groups = gates.shape
    hidden = value.shape[1]
    _prepare[(rows, groups)](
        key,
        query,
        value,
        key_norm,
        query_norm,
        conv_norm,
        gates,
        key.stride(0),
        query.stride(0),
        value.stride(0),
        hidden,
        groups,
        eps,
        triton.next_power_of_2(hidden),
        num_warps=4,
        enable_fp_fusion=False,
    )


def _prepare_fake(key, query, value, key_norm, query_norm, conv_norm, gates, eps):
    return None


def _finish_op(
    output: torch.Tensor,
    value: torch.Tensor,
    gates: torch.Tensor,
    residual: torch.Tensor | None = None,
) -> None:
    _finish[(triton.cdiv(output.numel(), 1024),)](
        output,
        value,
        gates,
        residual,
        output.numel(),
        value.shape[1],
        gates.shape[1],
        output.stride(0),
        value.stride(0),
        residual.stride(0) if residual is not None else 0,
        residual is not None,
        1024,
        num_warps=4,
        enable_fp_fusion=False,
    )


def _finish_fake(output, value, gates, residual=None):
    return None


direct_register_custom_op(
    op_name="qwen4_exp_ple_prefill_gate",
    op_func=_prepare_op,
    mutates_args=["key", "gates"],
    fake_impl=_prepare_fake,
)
direct_register_custom_op(
    op_name="qwen4_exp_ple_prefill_finish",
    op_func=_finish_op,
    mutates_args=["output"],
    fake_impl=_finish_fake,
)


@torch.compiler.assume_constant_result
def _is_sm70():
    return current_platform.is_device_capability(70)


def prefill_gate_reason(key, query, value, norms, groups):
    if not key.is_cuda or not _is_sm70():
        return "requires_sm70"
    if key.ndim != 2 or query.shape != key.shape or value.ndim != 2:
        return "invalid_tensor_shapes"
    if key.shape[0] == 0:
        return "empty_query_rows"
    if (
        key.dtype != torch.float16
        or query.dtype != key.dtype
        or value.dtype != key.dtype
    ):
        return "requires_fp16_operands"
    if groups != 4 or value.shape != (key.shape[0], 2560) or key.shape[1] != 10240:
        return "unqualified_hc_geometry"
    if not all(t.device == key.device and t.stride(-1) == 1 for t in (query, value)):
        return "unsupported_operand_layout"
    if (
        len(norms) != 3
        or key.stride(-1) != 1
        or key.stride(0) < key.shape[1]
        or any(
            w.device != key.device
            or w.numel() != 10240
            or not w.is_contiguous()
            or w.dtype not in (torch.float16, torch.float32)
            for w in norms
        )
    ):
        return "unsupported_norm_layout"
    return None


def prepare_prefill_gate(key, query, value, norms, groups, eps):
    """Consume the unique projected key and reuse it for convolution input."""
    gates = torch.empty((key.shape[0], groups), dtype=torch.float32, device=key.device)
    torch.ops.vllm.qwen4_exp_ple_prefill_gate(key, query, value, *norms, gates, eps)
    return key, gates


def finish_prefill_gate(output, value, gates, residual=None):
    """Reuse the convolution result for the FP16 PLE residual sum."""
    torch.ops.vllm.qwen4_exp_ple_prefill_finish(output, value, gates, residual)
    return output
