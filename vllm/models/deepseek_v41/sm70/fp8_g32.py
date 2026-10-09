# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 dense FP8 on SM70 without the FP16 dequant fallback (lane L-MOE, P5; PORT_DESIGN §1 first row, A2).

V4.1's dense weights are E4M3 with one UE8M0 (power-of-two) scale per 32 x 32 block. P2 dequantises them to FP16 at
load (exact; 2 B/param). Here they stay FP8 in TurboMind's SM70 layout (1 B/param + one FP16 scale per 32 K-values
of each output row, 1/16 extra) and one of three kernels runs per call, all reading that single copy:

* ``gemv``: M <= 8 (decode), a Triton GEMV that reads the TurboMind layout directly (32-row panels, 8-byte K chunks
  with the converter's [0, 2, 1, 3] byte order), decodes E4M3 with integer ops and accumulates in FP32. FP32 or FP16
  output, so it also serves the FP32-output projections (attention ``wo_b``, shared-expert down).
* ``turbomind``: FP16 output and ``GEMV_MAX_M_FP16 < M <= TM_MAX_M``, TurboMind's W8A16 HMMA GEMM with group 32
  (``sm70_884_8.cu`` group-32 tiles; FP32 accumulation, FP16 output).
* ``dequant``: larger M, or FP32 output above the GEMV range: ``fp8_sm70_dequantize_out`` rebuilds the FP16 weight
  (bitwise equal to the P2 fallback's) into a transient buffer and cuBLAS runs the product (FP16 or FP32 output).

Exactness: E4M3 x 2^e is a finite FP16 value for every code when -15 <= e <= 7 (smallest subnormal 2^-9 x 2^-15 =
2^-24; largest 448 x 2^7 = 57344 <= 65504, while 448 x 2^8 overflows) and the scale itself must be an FP16 normal
(e >= -14), so the window is [-14, 7]; the measured dense exponents are -13..-6 (MODEL_ARCHAEOLOGY §11), so every path multiplies the
same weights as the FP16 fallback and differs from it only in FP32 accumulation order. ``prepare_fp8_g32`` checks
every scale and fails loudly otherwise (Engram ``wkv``, with exponents down to -18, is not eligible).

Integration (L-CORE, fp8.py at P5): ``prepare_fp8_g32(weight, weight_scale_inv)`` once after loading, then
``fp8_g32_linear(x, w, out_dtype=...)`` in ``apply``. The bench is ``tests/models/deepseek_v41/test_moe_fp8g32.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm import _sm70_ops as sm70_ops
from vllm.triton_utils import tl, triton

GROUP = 32
PANEL = 32                # TurboMind FP8 B-operand panel: 32 output rows x K, stored [K/8][32][8]
_SCALE_EXP_MIN, _SCALE_EXP_MAX = -14, 7  # every E4M3 code x 2^e is a finite, exact FP16 value (module doc)
GEMV_MAX_M = 8            # the GEMV unrolls up to 8 rows
GEMV_MAX_M_FP16 = 2       # FP16 output: GEMV for M <= 2, TurboMind above (P5 bench, L-MOE progress)
TM_MAX_M = 64             # FP16 output: TurboMind up to here, dequant + cuBLAS above


@dataclass(frozen=True)
class Fp8G32Weight:
    tm_weight: torch.Tensor   # TurboMind-packed E4M3 bytes, [K, N] uint8 (panel order, see module doc)
    tm_scales: torch.Tensor   # FP16 group scales [K/32, N] (row-major; equal within a 32-row panel)
    k_ld: int                 # bytes per 32-row panel (= 32 * K)
    q_ld: int
    n: int
    k: int

    @property
    def nbytes(self) -> int:
        return self.tm_weight.numel() * self.tm_weight.element_size() + self.tm_scales.numel() * 2


def _scale_fp32(scale: torch.Tensor) -> torch.Tensor:
    if scale.dtype == torch.float8_e8m0fnu:
        return scale.to(torch.float32)
    if scale.dtype == torch.uint8:
        return torch.exp2(scale.to(torch.float32) - 127.0)
    if scale.dtype == torch.float32:
        return scale
    raise TypeError(f"FP8 g32 scale dtype {scale.dtype} (expected float8_e8m0fnu, uint8 E8M0 or float32)")


def prepare_fp8_g32(weight: torch.Tensor, scale: torch.Tensor) -> Fp8G32Weight:
    """weight [N, K] float8_e4m3fn (or its uint8/int8 bytes); scale [N/32, K/32] UE8M0 or FP32 powers of two.
    Returns the TurboMind-packed operands on weight's device."""
    if weight.ndim != 2 or not weight.is_cuda:
        raise TypeError("prepare_fp8_g32: weight must be a 2-D CUDA tensor [N, K]")
    if weight.dtype in (torch.uint8, torch.int8):
        weight = weight.view(torch.float8_e4m3fn)
    if weight.dtype != torch.float8_e4m3fn:
        raise TypeError(f"prepare_fp8_g32: weight dtype {weight.dtype} is not E4M3")
    n, k = weight.shape
    nan_codes = int(((weight.view(torch.uint8) & 0x7F) == 0x7F).sum().item())
    if nan_codes:
        # E4M3fn 0x7F / 0xFF are NaN; the FP16 fallback propagates them, while the g32 decoders would read +-480
        raise ValueError(f"prepare_fp8_g32: weight holds {nan_codes} E4M3 NaN codes (0x7F/0xFF); corrupt checkpoint?")
    if n % PANEL or k % GROUP:
        raise ValueError(f"prepare_fp8_g32: needs N % 32 == 0 and K % 32 == 0, got N={n} K={k}")
    s = _scale_fp32(scale.to(weight.device))
    if tuple(s.shape) != (n // GROUP, k // GROUP):
        raise ValueError(f"prepare_fp8_g32: scale shape {tuple(s.shape)} != ({n // GROUP}, {k // GROUP})")
    mant, exp = torch.frexp(s)
    if not bool((mant == 0.5).all()):
        raise ValueError("prepare_fp8_g32: scales must be exact powers of two (UE8M0)")
    e = exp - 1
    if int(e.min()) < _SCALE_EXP_MIN or int(e.max()) > _SCALE_EXP_MAX:
        raise ValueError(f"prepare_fp8_g32: scale exponents {int(e.min())}..{int(e.max())} outside the exact FP16 "
                         f"window [{_SCALE_EXP_MIN}, {_SCALE_EXP_MAX}]; keep this layer on the FP16 fallback")
    tm_weight, tm_scales, meta = sm70_ops.fp8_sm70_prepare(weight.contiguous(), s.contiguous(), GROUP)
    k_ld, q_ld = int(meta[0].item()), int(meta[1].item())
    if k_ld != PANEL * k or tuple(tm_scales.shape) != (k // GROUP, n) or tm_weight.numel() != n * k:
        raise RuntimeError(f"prepare_fp8_g32: unexpected TurboMind layout (k_ld={k_ld}, scales "
                           f"{tuple(tm_scales.shape)}); the GEMV assumes [N/32][K/8][32][8] panels")
    return Fp8G32Weight(tm_weight=tm_weight, tm_scales=tm_scales, k_ld=k_ld, q_ld=q_ld, n=n, k=k)


# ------------------------------------------------------------------------------------------------ decode GEMV


@triton.jit
def _x_scaled(x_ptr, i: tl.constexpr, stride_xm, kidx, scale, M):
    return tl.load(x_ptr + i * stride_xm + kidx, mask=(kidx >= 0) & (i < M), other=0.0).to(tl.float32) * scale


@triton.jit
def _row_out(out_ptr, i: tl.constexpr, stride_om, rows, acc, alpha, M, OUT_FP16: tl.constexpr):
    r = tl.sum(tl.sum(acc, axis=2), axis=0) * alpha
    mask = (rows >= 0) & (i < M)
    if OUT_FP16:
        tl.store(out_ptr + i * stride_om + rows, r.to(tl.float16), mask=mask)
    else:
        tl.store(out_ptr + i * stride_om + rows, r, mask=mask)


@triton.jit
def _fp8_tm_gemv_kernel(x_ptr, w_ptr, s_ptr, out_ptr, M, N, K, k_ld, alpha, stride_xm, stride_om,
                        M_PAD: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_KC: tl.constexpr, OUT_FP16: tl.constexpr):
    """out[:M, n0:n0+BLOCK_N] = alpha * x @ W^T. W bytes of a 32-row panel are [K/8][32][8] with the physical
    byte order [0, 2, 1, 3, 4, 6, 5, 7] inside each 8-byte chunk. E4M3 -> FP16 by bits: ((b & 0x7F) << 7 |
    (b & 0x80) << 8) is the value x 2^-8 (normals and subnormals alike), folded into the FP32 scale (x 256)."""
    n0 = tl.program_id(0) * BLOCK_N
    kc = tl.arange(0, BLOCK_KC)
    ni = (n0 % 32) + tl.arange(0, BLOCK_N)
    j = tl.arange(0, 8)
    logical_k = (j & 4) | ((j & 1) << 1) | ((j >> 1) & 1)
    w_base = w_ptr + (n0 // 32).to(tl.int64) * k_ld + kc[:, None, None] * 256 + ni[None, :, None] * 8 + j[None, None, :]
    a0 = tl.zeros((BLOCK_KC, BLOCK_N, 8), dtype=tl.float32)
    a1 = tl.zeros((BLOCK_KC, BLOCK_N, 8), dtype=tl.float32)
    a2 = tl.zeros((BLOCK_KC, BLOCK_N, 8), dtype=tl.float32)
    a3 = tl.zeros((BLOCK_KC, BLOCK_N, 8), dtype=tl.float32)
    a4 = tl.zeros((BLOCK_KC, BLOCK_N, 8), dtype=tl.float32)
    a5 = tl.zeros((BLOCK_KC, BLOCK_N, 8), dtype=tl.float32)
    a6 = tl.zeros((BLOCK_KC, BLOCK_N, 8), dtype=tl.float32)
    a7 = tl.zeros((BLOCK_KC, BLOCK_N, 8), dtype=tl.float32)
    for c0 in range(0, K // 8, BLOCK_KC):
        b = tl.load(w_base + c0 * 256).to(tl.uint16)
        w = (((b & 0x7F) << 7) | ((b & 0x80) << 8)).to(tl.float16, bitcast=True).to(tl.float32)
        kidx = (c0 + kc)[:, None] * 8 + logical_k[None, :]
        scale = tl.load(s_ptr + ((c0 + kc) // 4)[:, None] * N + n0).to(tl.float32) * 256.0
        a0 += w * _x_scaled(x_ptr, 0, stride_xm, kidx, scale, M)[:, None, :]
        if M_PAD > 1:
            a1 += w * _x_scaled(x_ptr, 1, stride_xm, kidx, scale, M)[:, None, :]
        if M_PAD > 2:
            a2 += w * _x_scaled(x_ptr, 2, stride_xm, kidx, scale, M)[:, None, :]
            a3 += w * _x_scaled(x_ptr, 3, stride_xm, kidx, scale, M)[:, None, :]
        if M_PAD > 4:
            a4 += w * _x_scaled(x_ptr, 4, stride_xm, kidx, scale, M)[:, None, :]
            a5 += w * _x_scaled(x_ptr, 5, stride_xm, kidx, scale, M)[:, None, :]
            a6 += w * _x_scaled(x_ptr, 6, stride_xm, kidx, scale, M)[:, None, :]
            a7 += w * _x_scaled(x_ptr, 7, stride_xm, kidx, scale, M)[:, None, :]
    rows = n0 + tl.arange(0, BLOCK_N)
    _row_out(out_ptr, 0, stride_om, rows, a0, alpha, M, OUT_FP16)
    if M_PAD > 1:
        _row_out(out_ptr, 1, stride_om, rows, a1, alpha, M, OUT_FP16)
    if M_PAD > 2:
        _row_out(out_ptr, 2, stride_om, rows, a2, alpha, M, OUT_FP16)
        _row_out(out_ptr, 3, stride_om, rows, a3, alpha, M, OUT_FP16)
    if M_PAD > 4:
        _row_out(out_ptr, 4, stride_om, rows, a4, alpha, M, OUT_FP16)
        _row_out(out_ptr, 5, stride_om, rows, a5, alpha, M, OUT_FP16)
        _row_out(out_ptr, 6, stride_om, rows, a6, alpha, M, OUT_FP16)
        _row_out(out_ptr, 7, stride_om, rows, a7, alpha, M, OUT_FP16)


# (N, K, M_PAD) -> (BLOCK_N, BLOCK_KC, num_warps): best of the P5 sweep for the V4.1 dense shapes at M = 3..8
# (bench/gemv_m8_sweep.log); register pressure of the M_PAD accumulators favours fewer warps there.
_GEMV_TABLE: dict[tuple[int, int, int], tuple[int, int, int]] = {
    (1792, 5120, 4): (4, 32, 1), (1792, 5120, 8): (4, 16, 1),
    (8192, 1280, 4): (32, 8, 4), (8192, 1280, 8): (32, 4, 2),
    (4096, 1280, 4): (8, 16, 1), (4096, 1280, 8): (32, 8, 4),
    (1024, 4096, 4): (4, 64, 2), (1024, 4096, 8): (2, 32, 1),
    (5120, 2048, 4): (8, 16, 1), (5120, 2048, 8): (32, 8, 4),
    (1152, 5120, 4): (4, 64, 2), (1152, 5120, 8): (4, 16, 1),
    (5120, 576, 4): (32, 8, 4), (5120, 576, 8): (32, 8, 4),
    (5120, 288, 4): (16, 4, 4), (5120, 288, 8): (16, 4, 4),
}


def _gemv_config(n: int, k: int, m_pad: int) -> tuple[int, int, int]:
    """(BLOCK_N, BLOCK_KC, num_warps) from the P5 sweeps (L2-busting rotation, V4.1 dense shapes): long K wants
    narrow row blocks with deep K chunks, short K wide row blocks; the accumulator tile shrinks as M grows."""
    hit = _GEMV_TABLE.get((n, k, m_pad))
    if hit is not None:
        return hit
    chunks = k // 8
    budget = {1: 4096, 2: 2048, 4: 2048}.get(m_pad, 1024)  # accumulator elements per row of x
    if k >= 4096:
        block_n = 8 if (n > 1280 and m_pad == 1) else 4
    else:
        block_n = 32 if (m_pad >= 4 and k <= 1280 and n >= 4096) else 16
    block_kc = 128
    while block_kc > 1 and (chunks % block_kc or block_n * block_kc * 8 > budget):
        block_kc //= 2
    return block_n, block_kc, 4 if m_pad <= 2 else 2


def fp8_g32_gemv(x: torch.Tensor, w: Fp8G32Weight, *, out_dtype: torch.dtype = torch.float32,
                 alpha: float = 1.0, out: torch.Tensor | None = None) -> torch.Tensor:
    """``alpha * x @ W^T`` for x [M <= 8, K] fp16 (row-major); FP32 accumulation; FP32 or FP16 output."""
    _check_x(x, w)
    m = x.shape[0]
    if m > GEMV_MAX_M:
        raise ValueError(f"fp8_g32_gemv serves at most {GEMV_MAX_M} rows, got {m}")
    out = _out(x, w, out_dtype, out)
    if m == 0:
        return out
    m_pad = max(1, triton.next_power_of_2(m))
    block_n, block_kc, warps = _gemv_config(w.n, w.k, m_pad)
    _fp8_tm_gemv_kernel[(w.n // block_n,)](
        x, w.tm_weight, w.tm_scales, out, m, w.n, w.k, w.k_ld, float(alpha), x.stride(0), out.stride(0),
        M_PAD=m_pad, BLOCK_N=block_n, BLOCK_KC=block_kc, OUT_FP16=out_dtype == torch.float16, num_warps=warps,
    )
    return out


# ------------------------------------------------------------------------------------- TurboMind / dequant


def fp8_g32_turbomind(x: torch.Tensor, w: Fp8G32Weight, out: torch.Tensor | None = None) -> torch.Tensor:
    """TurboMind W8A16 HMMA GEMM, group 32: x [M, K] fp16 -> [M, N] fp16."""
    _check_x(x, w)
    out = _out(x, w, torch.float16, out)
    if x.shape[0]:
        sm70_ops.fp8_gemm_sm70_out(out, x.contiguous(), w.tm_weight, w.tm_scales, GROUP, w.k_ld, w.q_ld, False)
    return out


def dequant_fp8_g32(w: Fp8G32Weight) -> torch.Tensor:
    """The FP16 weight [K, N] (bitwise the FP16 fallback's, transposed) rebuilt from the TurboMind layout."""
    dense = torch.empty((w.k, w.n), dtype=torch.float16, device=w.tm_weight.device)
    sm70_ops.fp8_sm70_dequantize_out(dense, w.tm_weight, w.tm_scales, GROUP)
    return dense


def fp8_g32_dequant_mm(x: torch.Tensor, w: Fp8G32Weight, *, out_dtype: torch.dtype = torch.float16,
                       out: torch.Tensor | None = None) -> torch.Tensor:
    """Large-M path: transient FP16 weight + cuBLAS (FP32 accumulation; FP16 or FP32 output)."""
    _check_x(x, w)
    out = _out(x, w, out_dtype, out)
    if x.shape[0]:
        torch.mm(x, dequant_fp8_g32(w), out_dtype=out_dtype, out=out)
    return out


def fp8_g32_path(m: int, out_dtype: torch.dtype = torch.float16) -> str:
    """Which kernel ``fp8_g32_linear`` runs for M rows."""
    if out_dtype == torch.float16:
        if m <= GEMV_MAX_M_FP16:
            return "gemv"
        return "turbomind" if m <= TM_MAX_M else "dequant"
    if out_dtype == torch.float32:
        return "gemv" if m <= GEMV_MAX_M else "dequant"
    raise TypeError(f"fp8_g32_linear: out_dtype {out_dtype} not supported (float16 or float32)")


def fp8_g32_linear(x: torch.Tensor, w: Fp8G32Weight, *, out_dtype: torch.dtype = torch.float16,
                   out: torch.Tensor | None = None) -> torch.Tensor:
    """x [M, K] fp16 -> x @ W^T [M, N] in ``out_dtype`` (FP32 accumulation on every path)."""
    path = fp8_g32_path(x.shape[0], out_dtype)
    if path == "gemv":
        return fp8_g32_gemv(x, w, out_dtype=out_dtype, out=out)
    if path == "turbomind":
        return fp8_g32_turbomind(x, w, out=out)
    return fp8_g32_dequant_mm(x, w, out_dtype=out_dtype, out=out)


def dequant_fp8_g32_reference(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Exact FP32 dequantisation of E4M3 [N, K] with block-32 scales (the FP16 fallback's values)."""
    if weight.dtype in (torch.uint8, torch.int8):
        weight = weight.view(torch.float8_e4m3fn)
    s = _scale_fp32(scale).repeat_interleave(GROUP, 0).repeat_interleave(GROUP, 1)
    return weight.float() * s[: weight.shape[0], : weight.shape[1]].to(weight.device)


def _check_x(x: torch.Tensor, w: Fp8G32Weight) -> None:
    if x.dtype != torch.float16 or x.ndim != 2 or x.shape[1] != w.k or (x.shape[0] and x.stride(1) != 1):
        raise TypeError(f"fp8_g32: x must be a row-major fp16 [M, {w.k}] matrix, got {x.dtype} {tuple(x.shape)}")


def _out(x: torch.Tensor, w: Fp8G32Weight, out_dtype: torch.dtype, out: torch.Tensor | None) -> torch.Tensor:
    if out is None:
        return torch.empty((x.shape[0], w.n), dtype=out_dtype, device=x.device)
    if out.dtype != out_dtype or tuple(out.shape) != (x.shape[0], w.n) or (x.shape[0] and out.stride(1) != 1):
        raise TypeError(f"fp8_g32: out must be row-major {out_dtype} [{x.shape[0]}, {w.n}]")
    return out
