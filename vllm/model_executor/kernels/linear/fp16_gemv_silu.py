# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP16 GEMV with FP32 accumulation and a prefix SiLU epilogue on SM70."""

from __future__ import annotations

import math
from functools import lru_cache

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton


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
    BLOCK_K: tl.constexpr,
):
    token, row = tl.program_id(0), tl.program_id(1)
    active = row < N
    wr = tl.where(row < PREFIX, PREFIX_START + row, SUFFIX_START + row - PREFIX)
    offset = tl.arange(0, BLOCK_K)
    accum = tl.zeros((BLOCK_K,), tl.float32)
    for start in tl.static_range(0, K, BLOCK_K):
        col = start + offset
        a = tl.load(
            x + token * K + col, active & (col < K), 0, eviction_policy="evict_last"
        )
        b = tl.load(
            weight + wr * K + col, active & (col < K), 0, eviction_policy="evict_first"
        )
        accum += a.to(tl.float32) * b.to(tl.float32)
    value = tl.sum(accum, 0).to(tl.float16).to(tl.float32)
    scaled = value / DIVISOR
    value = tl.where(row < PREFIX, scaled * tl.sigmoid(scaled), value)
    tl.store(out + token * OUTPUT_N + row, tl.where(active, value, 0.0))


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
            and 1 <= x.shape[0] <= 16
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
        _fp16_gemv_silu_ranges_kernel[(m, out.shape[1])](
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
            BLOCK_K=block_k,
            num_warps=warps,
        )
