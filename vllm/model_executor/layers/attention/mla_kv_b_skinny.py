# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Skinny per-head batched matmul for the two kv_b products of MLA decode.

``MLAAttention`` projects with the kv_b weight twice per layer: the query side
``q_nope[h] @ W_UK_T[h]`` and the value side ``attn_out[h] @ W_UV[h]``, both one
FP16 matrix per head (64 heads of 256 x 512). With at most a handful of rows
(decode and verify steps) the products are bound by reading those 16.8 MB each;
cuBLAS picks general-purpose tiles for them and streams 40-54 % of the card's
bandwidth. Here one program owns one head and ``BLOCK_N`` output columns, reads
its weight tile once for all rows and runs the dot on tensor cores.

Accumulation is FP32 over K, with a single rounding to the output dtype at the
end, like the cuBLAS kernels it replaces (``*_s884gemm_fp16`` / ``*_s1688gemm_
fp16`` accumulate in FP32; neither profile shows a split-K reduction). Only the
summation order differs, so results are not bit-identical to cuBLAS.

Needs the Triton of ``kernel_config.triton_sm7x_tensor_cores``; the layer
decides once whether to take this route (``KernelConfig.use_mla_kv_b_skinny``).
"""

from typing import NamedTuple

import torch

from vllm.triton_utils import tl, triton

# One MMA row tile: more rows than this and cuBLAS is no longer bandwidth-bound
# on the weight, so the caller keeps torch.bmm. A single row (decode without
# speculation) stays on cuBLAS too: there this kernel is 3-7 % slower on
# capability 70 and under 2 % faster on 75 (benchmarks/kernels/
# benchmark_mla_kv_b_skinny.py at one row); at five and seven rows, the verify
# rounds it was measured for, it is faster on both.
KV_B_SKINNY_MIN_ROWS = 2
KV_B_SKINNY_MAX_ROWS = 8
_BLOCK_M = 16  # smallest tl.dot row tile; rows past the batch are masked


class KvBTile(NamedTuple):
    block_n: int
    block_k: int
    num_warps: int


# Tiles per compute capability, as (N contiguous, K contiguous): "N contiguous"
# is W[h] = [K, N] row-major (W_UK_T, K = nope dim 256, N = latent 512); "K
# contiguous" is W[h] = [N, K] row-major (W_UV, K = latent 512, N = value 256).
# Chosen by benchmarks/kernels/benchmark_mla_kv_b_skinny.py --sweep --rows 1 5 7
# on GLM-5.3's shapes: the fastest sum over 1, 5 and 7 rows. Turing
# has 64 KiB of shared memory, which rules out the largest tiles there.
KV_B_SKINNY_TILES: dict[int, tuple[KvBTile, KvBTile]] = {
    70: (KvBTile(block_n=64, block_k=32, num_warps=4), KvBTile(64, 64, 2)),
    75: (KvBTile(block_n=128, block_k=32, num_warps=4), KvBTile(64, 64, 2)),
}


@triton.jit
def _kv_b_skinny_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    rows,
    K,
    x_stride_head,
    x_stride_row,
    w_stride_head,
    w_stride_k,
    w_stride_n,
    out_stride_head,
    out_stride_row,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    head = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    row_index = tl.arange(0, BLOCK_M)
    row_ok = row_index < rows
    offs_k = tl.arange(0, BLOCK_K)
    x_base = x_ptr + head * x_stride_head + row_index[:, None] * x_stride_row
    w_base = w_ptr + head * w_stride_head + cols[None, :] * w_stride_n
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k in range(0, K, BLOCK_K):
        # Weight tiles are never masked: K and N are whole multiples of the tile.
        w = tl.load(w_base + (k + offs_k)[:, None] * w_stride_k)
        x = tl.load(x_base + (k + offs_k)[None, :], mask=row_ok[:, None], other=0.0)
        acc += tl.dot(x, w)
    tl.store(
        out_ptr
        + head * out_stride_head
        + row_index[:, None] * out_stride_row
        + cols[None, :],
        acc.to(out_ptr.dtype.element_ty),
        mask=row_ok[:, None],
    )


def kv_b_skinny_tile(weight: torch.Tensor, capability: int) -> KvBTile:
    """The tile for ``weight`` ([heads, K, N]) on a card of ``capability`` (e.g.
    70); raises if there is none or it does not divide the shape."""
    if capability not in KV_B_SKINNY_TILES:
        raise ValueError(f"kv_b skinny kernel has no tile for capability {capability}")
    if weight.dtype != torch.float16:
        raise ValueError(f"kv_b skinny kernel needs an FP16 weight, got {weight.dtype}")
    if weight.dim() != 3:
        raise ValueError(f"kv_b skinny kernel needs [heads, K, N], got {weight.shape}")
    _, k, n = weight.shape
    n_contiguous, k_contiguous = KV_B_SKINNY_TILES[capability]
    tile = n_contiguous if weight.stride(2) == 1 else k_contiguous
    if weight.stride(2) != 1 and weight.stride(1) != 1:
        raise ValueError(f"kv_b weight has no contiguous axis: {weight.stride()}")
    if n % tile.block_n or k % tile.block_k:
        raise ValueError(
            f"kv_b skinny kernel tiles N by {tile.block_n} and K by {tile.block_k}, "
            f"got K={k}, N={n}"
        )
    return tile


def kv_b_bmm_skinny(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    tile: KvBTile,
) -> None:
    """``out[h] = x[h] @ weight[h]`` for at most ``KV_B_SKINNY_MAX_ROWS`` rows.

    ``x`` is [heads, rows, K], ``weight`` [heads, K, N] and ``out`` [heads, rows,
    N]; head and row strides are free (the layer passes transposed views), the
    last axis of ``x`` and ``out`` is contiguous, and ``weight`` has one
    contiguous axis (K or N). ``tile`` comes from ``kv_b_skinny_tile``, which
    also checks the weight's shape once.
    """
    heads, rows, k = x.shape
    n = weight.shape[2]
    if not 0 < rows <= KV_B_SKINNY_MAX_ROWS:
        raise ValueError(f"kv_b skinny kernel takes 1..{KV_B_SKINNY_MAX_ROWS} rows")
    if weight.shape != (heads, k, n) or out.shape != (heads, rows, n):
        raise ValueError(
            f"shape mismatch: x {tuple(x.shape)}, weight {tuple(weight.shape)}, "
            f"out {tuple(out.shape)}"
        )
    if x.dtype != torch.float16 or out.dtype != torch.float16:
        raise ValueError("kv_b skinny kernel takes FP16 activations")
    if x.stride(2) != 1 or out.stride(2) != 1:
        raise ValueError("kv_b skinny kernel needs a contiguous last axis")
    _kv_b_skinny_kernel[(heads, n // tile.block_n)](
        x,
        weight,
        out,
        rows,
        k,
        x.stride(0),
        x.stride(1),
        weight.stride(0),
        weight.stride(1),
        weight.stride(2),
        out.stride(0),
        out.stride(1),
        BLOCK_M=_BLOCK_M,
        BLOCK_N=tile.block_n,
        BLOCK_K=tile.block_k,
        num_warps=tile.num_warps,
        num_stages=1,
    )
