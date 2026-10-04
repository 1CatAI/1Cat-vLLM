# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP16 GEMV with FP32 accumulation and a prefix SiLU epilogue on SM70."""

from __future__ import annotations

import math
from functools import lru_cache

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, tldevice, triton


@triton.jit
def _fp16_gemv_silu_ranges_kernel(
    x,
    weight,
    out,
    K: tl.constexpr,
    N: tl.constexpr,
    OUTPUT_N: tl.constexpr,
    PREFIX: tl.constexpr,
    PREFIX_START: tl.constexpr,
    SUFFIX_START: tl.constexpr,
    DIVISOR: tl.constexpr,
    M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ROWS: tl.constexpr,
    COMPENSATED: tl.constexpr,
):
    tiles: tl.constexpr = triton.cdiv(OUTPUT_N, ROWS)
    token, tile = tl.program_id(0) // tiles, tl.program_id(0) % tiles
    row = tile * ROWS + tl.arange(0, ROWS)
    active = row < N
    wr = tl.where(row < PREFIX, PREFIX_START + row, SUFFIX_START + row - PREFIX)
    offset = tl.arange(0, BLOCK_K)
    accum = tl.zeros((ROWS, BLOCK_K), tl.float32)
    if COMPENSATED:
        low = tl.zeros((ROWS, BLOCK_K), tl.float32)
    for start in tl.static_range(0, K, BLOCK_K):
        col = start + offset
        a = tl.load(x + token * K + col, col < K, 0, eviction_policy="evict_last")
        b = tl.load(
            weight + wr[:, None] * K + col[None, :],
            active[:, None] & (col[None, :] < K),
            0,
            eviction_policy="evict_first",
        )
        product = a[None, :].to(tl.float32) * b.to(tl.float32)
        if COMPENSATED:
            accum, low = _fp32_pair_sum(accum, low, product, 0.0)
        else:
            accum += product
    if COMPENSATED:
        high, low = tl.reduce((accum, low), 1, _fp32_pair_sum)
        value = _fp32_pair_to_fp16(high, low).to(tl.float32)
    else:
        value = tl.sum(accum, 1).to(tl.float16).to(tl.float32)
    scaled = value / DIVISOR
    value = tl.where(row < PREFIX, scaled * tl.sigmoid(scaled), value)
    tl.store(
        out + token * OUTPUT_N + row,
        tl.where(active, value, 0.0),
        (row < OUTPUT_N) & (token < M),
    )


@lru_cache
def _sm_count(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


class Sm70Fp16GemvSiluKernel:
    """Capability gate for two contiguous weight-row ranges and zero padding.

    Prefix outputs apply SiLU after the FP16 projection boundary and division;
    suffix outputs keep that boundary. The kernel has no model or TP predicate.
    """

    @staticmethod
    def can_implement(
        x: torch.Tensor,
        weight: torch.Tensor,
        out: torch.Tensor,
        active_columns: int,
        activated_columns: int,
        prefix_start: int,
        suffix_start: int,
        divisor: float,
    ) -> bool:
        return bool(
            current_platform.is_device_capability(70)
            and x.ndim == weight.ndim == out.ndim == 2
            and x.shape[0] > 0
            and x.shape[1] > 0
            and weight.shape[1] == x.shape[1]
            and out.shape[0] == x.shape[0]
            and 0 <= activated_columns <= active_columns <= out.shape[1]
            and active_columns > 0
            and prefix_start >= 0
            and suffix_start >= 0
            and prefix_start + activated_columns <= weight.shape[0]
            and suffix_start + active_columns - activated_columns <= weight.shape[0]
            and math.isfinite(divisor)
            and divisor > 0
            and all(
                t.is_cuda
                and t.dtype == torch.float16
                and t.is_contiguous()
                and t.device == x.device
                for t in (x, weight, out)
            )
            and all(
                out.data_ptr() + out.numel() * out.element_size() <= t.data_ptr()
                or t.data_ptr() + t.numel() * t.element_size() <= out.data_ptr()
                for t in (x, weight)
            )
        )

    @classmethod
    def apply_out(
        cls,
        x: torch.Tensor,
        weight: torch.Tensor,
        out: torch.Tensor,
        active_columns: int,
        activated_columns: int,
        prefix_start: int = 0,
        suffix_start: int = 0,
        divisor: float = 1.0,
        compensated: bool = False,
    ) -> None:
        if not cls.can_implement(
            x,
            weight,
            out,
            active_columns,
            activated_columns,
            prefix_start,
            suffix_start,
            divisor,
        ):
            raise ValueError("Unsupported SM70 FP16 GEMV/SiLU layout or row ranges")
        m, k = x.shape
        # More independent lanes hide the long row's load latency on a small
        # grid. Other geometries retain the established FP32 reduction tree.
        small_grid = active_columns * m <= 2 * _sm_count(x.device)
        if small_grid and 4096 <= k <= 16384 and m <= 2:
            block_k, warps = 2048, 8
        elif small_grid and 4096 <= k <= 16384 and m <= 4:
            block_k, warps = 512, 4
        else:
            block_k, warps = 256, 4
        # Short rows otherwise launch mostly empty CTAs. Four rows per CTA
        # retain the FP32 reduction for each row while amortizing dispatch.
        # This schedule is measured on rotating FP16 weights; batch width is
        # not an implementation boundary.
        rows = 4 if k <= 256 and active_columns * m >= 4 * _sm_count(x.device) else 1
        _fp16_gemv_silu_ranges_kernel[(m * triton.cdiv(out.shape[1], rows),)](
            x,
            weight,
            out,
            K=k,
            N=active_columns,
            OUTPUT_N=out.shape[1],
            PREFIX=activated_columns,
            PREFIX_START=prefix_start,
            SUFFIX_START=suffix_start,
            DIVISOR=divisor,
            M=m,
            BLOCK_K=block_k,
            ROWS=rows,
            COMPENSATED=compensated,
            num_warps=warps,
        )


@triton.jit
def _fp32_pair_sum(ah, al, bh, bl):
    # Error-free addition of the high components, then FP32 renormalization.
    # Both components stay FP32; there is no FP64 arithmetic or weight change.
    total = ah + bh
    bv = total - ah
    error = (ah - (total - bv)) + (bh - bv)
    low = (al + bl) + error
    high = total + low
    return high, low - (high - total)


@triton.jit
def _fp32_pair_to_fp16(high, low):
    """Keep the correction when FP32 rounding lands on an FP16 midpoint."""
    value = high + low
    bits = value.to(tl.uint32, bitcast=True)
    exponent = (bits >> 23) & 255
    shift = tl.minimum(24, tl.maximum(13, 126 - exponent.to(tl.int32)))
    significand = (bits & 0x7FFFFF) | 0x800000
    mask = (1 << shift) - 1
    midpoint = (
        (exponent >= 102)
        & (exponent <= 142)
        & ((significand & mask) == (1 << (shift - 1)))
    )
    correction = (high - value) + low
    step = tl.where((correction > 0) == ((bits >> 31) == 0), 1, -1)
    adjusted = bits + tl.where(midpoint & (correction != 0), step, 0).to(tl.uint32)
    return adjusted.to(tl.float32, bitcast=True).to(tl.float16)


@triton.jit
def _fp16_gate_up_kernel(
    x,
    weight,
    out,
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ROWS: tl.constexpr,
):
    tiles: tl.constexpr = triton.cdiv(N, ROWS)
    token, tile = tl.program_id(0) // tiles, tl.program_id(0) % tiles
    rows = tile * ROWS + tl.arange(0, ROWS)
    offsets = tl.arange(0, BLOCK_K)
    gate = tl.zeros((ROWS, BLOCK_K), tl.float32)
    up = tl.zeros((ROWS, BLOCK_K), tl.float32)
    gate_low = tl.zeros((ROWS, BLOCK_K), tl.float32)
    up_low = tl.zeros((ROWS, BLOCK_K), tl.float32)
    for start in tl.static_range(0, K, BLOCK_K):
        cols = start + offsets
        a = tl.load(x + token * K + cols, cols < K, 0).to(tl.float32)
        mask = (rows[:, None] < N) & (cols[None, :] < K)
        g = tl.load(
            weight + rows[:, None] * K + cols[None, :],
            mask,
            0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        u = tl.load(
            weight + (rows[:, None] + N) * K + cols[None, :],
            mask,
            0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        gate, gate_low = _fp32_pair_sum(gate, gate_low, a[None, :] * g, 0.0)
        up, up_low = _fp32_pair_sum(up, up_low, a[None, :] * u, 0.0)
    gh, gl = tl.reduce((gate, gate_low), 1, _fp32_pair_sum)
    uh, ul = tl.reduce((up, up_low), 1, _fp32_pair_sum)
    g = _fp32_pair_to_fp16(gh, gl).to(tl.float32)
    u = _fp32_pair_to_fp16(uh, ul).to(tl.float32)
    # Retain the activation's FP16 materialization before multiplication.
    # Use the native activation's FP32 exp/div arithmetic rather than a
    # sigmoid approximation before the FP16 materialization.
    silu = tl.div_rn(g, 1.0 + tldevice.exp(-g)).to(tl.float16).to(tl.float32)
    tl.store(out + token * N + rows, silu * u, (rows < N) & (token < M))


class Sm70Fp16GateUpKernel:
    """Split contiguous gate/up rows, FP32 reductions and FP16 boundaries."""

    @staticmethod
    def can_implement(x, weight, out) -> bool:
        return bool(
            weight.ndim == out.ndim == 2
            and weight.shape[0] == 2 * out.shape[1]
            and Sm70Fp16GemvSiluKernel.can_implement(
                x, weight, out, out.shape[1], 0, 0, 0, 1.0
            )
        )

    @classmethod
    def apply_out(cls, x, weight, out) -> None:
        if not cls.can_implement(x, weight, out):
            raise ValueError("Unsupported SM70 FP16 gate/up layout")
        m, k = x.shape
        n = out.shape[1]
        rows = 4 if k <= 256 and n * m >= 4 * _sm_count(x.device) else 1
        _fp16_gate_up_kernel[(m * triton.cdiv(n, rows),)](
            x,
            weight,
            out,
            M=m,
            K=k,
            N=n,
            BLOCK_K=256,
            ROWS=rows,
            num_warps=4,
        )


class Sm70Fp16CompensatedGemvKernel(Sm70Fp16GemvSiluKernel):
    """Measured linear candidate with FP32 error compensation.

    The existing range/SiLU provider retains its original accumulation mode.
    """

    @classmethod
    def apply_out(cls, *args, **kwargs):
        kwargs["compensated"] = True
        return super().apply_out(*args, **kwargs)
