# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Range-preserving BF16 arithmetic for DFlash2 drafts on SM70.

The checkpoint was trained with BF16 activations, but Volta executes its dense
GEMMs in FP16. These kernels retain FP16 Tensor Core transport while preserving
BF16 residual range and rounding at the model's trained rounding points.

Adapted from haohervchb/sglang-V100@5526ef1c6a82.
"""

from __future__ import annotations

import torch
from torch import nn

from vllm.triton_utils import tl, triton

DFLASH_SM70_GATE_UP_INPUT_SCALE = 32.0
DFLASH_SM70_WIDE_OUTPUT_SCALE = 256.0


@triton.jit
def _round_fp32_to_bf16_rne(value):
    """Round FP32 to BF16 RNE and keep the result represented as FP32."""
    bits = value.to(tl.uint32, bitcast=True)
    rounded_bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return rounded_bits.to(tl.float32, bitcast=True)


@triton.jit
def _dflash_silu_and_mul_sm70_kernel(
    output_ptr,
    output_scales_ptr,
    gate_up_ptr,
    n_cols: tl.constexpr,
    gate_up_scale: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    input_offsets = row * 2 * n_cols + offsets
    output_offsets = row * n_cols + offsets

    gate = tl.load(gate_up_ptr + input_offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(gate_up_ptr + input_offsets + n_cols, mask=mask, other=0.0).to(
        tl.float32
    )
    gate = _round_fp32_to_bf16_rne(gate * gate_up_scale)
    up = _round_fp32_to_bf16_rne(up * gate_up_scale)
    activated_gate = _round_fp32_to_bf16_rne(gate * tl.sigmoid(gate))
    output = _round_fp32_to_bf16_rne(activated_gate * up)

    # A per-row power-of-two divisor retains exact transport scaling.
    max_abs = tl.max(tl.abs(output), axis=0)
    output_scale = tl.maximum(max_abs / 32752.0, 1.0)
    output_scale = tl.exp2(tl.ceil(tl.log2(output_scale)))
    tl.store(output_scales_ptr + row, output_scale)
    tl.store(output_ptr + output_offsets, output / output_scale, mask=mask)


@triton.jit
def _dflash_scale_output_sm70_kernel(
    output_ptr,
    row_scales_ptr,
    n_cols: tl.constexpr,
    residual_scale: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    row_offsets = row * n_cols + offsets
    value = tl.load(output_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
    row_scale = tl.load(row_scales_ptr + row).to(tl.float32)
    tl.store(
        output_ptr + row_offsets,
        value * (row_scale / residual_scale),
        mask=mask,
    )


@triton.jit
def _dflash_rmsnorm_sm70_kernel(
    output_ptr,
    residual_output_ptr,
    x_ptr,
    residual_ptr,
    weight_ptr,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    residual_scale: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    row_offsets = row * n_cols + offsets

    value = tl.load(x_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
    if HAS_RESIDUAL:
        residual = tl.load(residual_ptr + row_offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        value = _round_fp32_to_bf16_rne(value * residual_scale + residual)
        tl.store(residual_output_ptr + row_offsets, value, mask=mask)

    variance = tl.sum(value * value, axis=0) / n_cols
    normalized = value * tl.rsqrt(variance + eps)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    weight = _round_fp32_to_bf16_rne(weight)
    output = _round_fp32_to_bf16_rne(normalized * weight)
    tl.store(output_ptr + row_offsets, output, mask=mask)


def _num_warps(n_cols: int) -> int:
    return max(min(triton.next_power_of_2(triton.cdiv(n_cols, 256)), 16), 4)


def _silu_and_mul_reference(
    gate_up: torch.Tensor, gate_up_scale: float
) -> tuple[torch.Tensor, torch.Tensor]:
    gate, up = gate_up.float().chunk(2, dim=-1)
    gate = (gate * gate_up_scale).to(torch.bfloat16).float()
    up = (up * gate_up_scale).to(torch.bfloat16).float()
    activated = (gate * torch.sigmoid(gate)).to(torch.bfloat16).float()
    output = (activated * up).to(torch.bfloat16).float()
    max_abs = output.abs().amax(dim=-1)
    row_scales = torch.maximum(max_abs / 32752.0, torch.ones_like(max_abs))
    row_scales = torch.pow(2.0, torch.ceil(torch.log2(row_scales)))
    return (output / row_scales.unsqueeze(-1)).to(gate_up.dtype), row_scales


def dflash_silu_and_mul_sm70(
    gate_up: torch.Tensor,
    gate_up_scale: float = DFLASH_SM70_GATE_UP_INPUT_SCALE,
) -> tuple[torch.Tensor, torch.Tensor]:
    if gate_up.ndim < 2 or gate_up.shape[-1] % 2:
        raise ValueError(
            "DFlash SM70 SwiGLU expects [..., 2 * intermediate_size], "
            f"got {tuple(gate_up.shape)}."
        )
    if gate_up_scale <= 0:
        raise ValueError(f"gate_up_scale must be positive, got {gate_up_scale}.")
    if not gate_up.is_cuda:
        return _silu_and_mul_reference(gate_up, gate_up_scale)
    if gate_up.dtype != torch.float16:
        raise ValueError(
            f"DFlash SM70 SwiGLU requires FP16 CUDA transport, got {gate_up.dtype}."
        )
    if not gate_up.is_contiguous():
        gate_up = gate_up.contiguous()

    n_cols = gate_up.shape[-1] // 2
    n_rows = gate_up.numel() // (2 * n_cols)
    output = torch.empty(
        (*gate_up.shape[:-1], n_cols),
        dtype=gate_up.dtype,
        device=gate_up.device,
    )
    output_scales = torch.empty(n_rows, dtype=torch.float32, device=gate_up.device)
    block_size = triton.next_power_of_2(n_cols)
    _dflash_silu_and_mul_sm70_kernel[(n_rows,)](
        output,
        output_scales,
        gate_up,
        n_cols=n_cols,
        gate_up_scale=gate_up_scale,
        BLOCK_SIZE=block_size,
        num_warps=_num_warps(n_cols),
    )
    return output, output_scales


def dflash_scale_output_sm70(
    output: torch.Tensor,
    row_scales: torch.Tensor,
    residual_scale: float = DFLASH_SM70_WIDE_OUTPUT_SCALE,
) -> torch.Tensor:
    if output.ndim < 2:
        raise ValueError(f"DFlash SM70 output must have ndim >= 2, got {output.ndim}.")
    if residual_scale <= 0:
        raise ValueError(f"residual_scale must be positive, got {residual_scale}.")
    n_cols = output.shape[-1]
    n_rows = output.numel() // n_cols
    if row_scales.numel() != n_rows:
        raise ValueError(
            f"DFlash SM70 row-scale mismatch: rows={n_rows}, "
            f"scales={row_scales.numel()}."
        )
    if not output.is_cuda:
        return (output.float() * (row_scales.reshape(-1, 1) / residual_scale)).to(
            output.dtype
        )
    if output.dtype != torch.float16 or not output.is_contiguous():
        raise ValueError("DFlash SM70 W2 scaling requires contiguous CUDA FP16 output.")
    block_size = triton.next_power_of_2(n_cols)
    _dflash_scale_output_sm70_kernel[(n_rows,)](
        output,
        row_scales,
        n_cols=n_cols,
        residual_scale=residual_scale,
        BLOCK_SIZE=block_size,
        num_warps=_num_warps(n_cols),
    )
    return output


class DFlashSM70RMSNorm(nn.Module):
    """RMSNorm with FP32 residual transport and explicit BF16 RNE."""

    def __init__(
        self,
        hidden_size: int,
        eps: float,
        dtype: torch.dtype,
        residual_scale: float = DFLASH_SM70_WIDE_OUTPUT_SCALE,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.variance_epsilon = eps
        self.residual_scale = residual_scale
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))

    def _reference(
        self, x: torch.Tensor, residual: torch.Tensor | None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        value = x.float()
        residual_output = None
        if residual is not None:
            value = (
                (value * self.residual_scale + residual.float())
                .to(torch.bfloat16)
                .float()
            )
            residual_output = value
        variance = value.square().mean(dim=-1, keepdim=True)
        normalized = value * torch.rsqrt(variance + self.variance_epsilon)
        weight = self.weight.float().to(torch.bfloat16).float()
        output = (normalized * weight).to(torch.bfloat16).to(orig_dtype)
        if residual_output is None:
            return output
        return output, residual_output

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if not x.is_cuda:
            return self._reference(x, residual)
        if x.dtype != torch.float16:
            raise ValueError(
                f"DFlash SM70 RMSNorm requires FP16 CUDA transport, got {x.dtype}."
            )
        if not x.is_contiguous():
            x = x.contiguous()
        if residual is not None and not residual.is_contiguous():
            residual = residual.contiguous()

        n_rows = x.numel() // self.hidden_size
        output = torch.empty_like(x)
        has_residual = residual is not None
        residual_output = (
            torch.empty_like(x, dtype=torch.float32) if has_residual else output
        )
        residual_ptr = residual if residual is not None else x
        block_size = triton.next_power_of_2(self.hidden_size)
        _dflash_rmsnorm_sm70_kernel[(n_rows,)](
            output,
            residual_output,
            x,
            residual_ptr,
            self.weight,
            n_cols=self.hidden_size,
            eps=self.variance_epsilon,
            residual_scale=self.residual_scale,
            HAS_RESIDUAL=has_residual,
            BLOCK_SIZE=block_size,
            num_warps=_num_warps(self.hidden_size),
        )
        if has_residual:
            return output, residual_output
        return output


class DFlashSM70MLP(nn.Module):
    """Wrap the official Qwen MLP with overflow-safe BF16-equivalent math."""

    def __init__(self, mlp: nn.Module) -> None:
        super().__init__()
        self.gate_up_proj = mlp.gate_up_proj
        self.down_proj = mlp.down_proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x / DFLASH_SM70_GATE_UP_INPUT_SCALE)
        x, row_scales = dflash_silu_and_mul_sm70(gate_up)
        x, _ = self.down_proj(x)
        return dflash_scale_output_sm70(x, row_scales)
