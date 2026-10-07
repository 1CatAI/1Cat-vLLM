# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bit-exact quantise-dequantise (QAT) twins of DeepSeek-V4.1's in-model rounding steps.

PORT_DESIGN §4.2 (owner L-ATTN). The reference (``inference/kernel.py`` @ 2cba9e42) rounds three
activations onto low-precision grids *as model semantics*:

=====================  ======================  =====================================================
op                     reference call           algorithm (FP32 IEEE, RNE casts)
=====================  ======================  =====================================================
``fp8_block32_qdq``    ``act_quant(kv, 32,      per 32: ``amax = max(max|x|, fp32(1e-4))``;
(window KV)            "ue8m0", e8m0, True)``   ``s = 2^ceil_log2(amax * fp32(1/448))``;
                                                ``y = e4m3_rne(clamp(x / s, ±448)) * s``
``fp4_e4m3_qdq``       ``fp4_act_quant(latent,  per 16: ``amax = max(max|x|, 6 * 2^-9)``;
(compressed KV)        16, True, e4m3fn)``      ``s = e4m3_rne(amax / 6)`` (IEEE division, saturating);
                                                ``y = e2m1_rne(clamp(x / s, ±6)) * s``
``fp4_e8m0_qdq``       ``fp4_act_quant(k, 32,   per 32: ``amax = max(max|x|, 6 * 2^-126)``;
(indexer q and K)      True)``                  ``s = 2^ceil_log2(amax * fp32(1/6))``;
                                                ``y = e2m1_rne(clamp(x / s, ±6)) * s``
=====================  ======================  =====================================================

``ceil_log2(v) = exponent_bits(v) - 127 + (mantissa_bits(v) != 0)`` (ref ``fast_log2_ceil``).
``e4m3_rne``: round-to-nearest-even onto E4M3FN, saturating at ±448. ``e2m1_rne``: onto
{0, 0.5, 1, 1.5, 2, 3, 4, 6}, ties to the even mantissa (0.25->0, 0.75->1, 1.25->1, 1.75->2,
2.5->2, 3.5->4, 5->4). The sign of zero is preserved (the FP8/FP4 types carry a sign bit).

Every product ``grid_value * s`` is exact in FP32 (<= 7 significant bits), so the FP32 result
*is* the reference value; the reference stores it in BF16 (exact) and the port stores it in FP16,
which is exact whenever the value is an FP16 number (window KV blocks with ``e >= -15``, every
compressed-KV value, index q/k blocks with ``e >= -23``). ``out_dtype=torch.float32`` returns the
exact value.

Two implementations with identical bits:

* Triton (``impl="triton"``): integer round-to-nearest-even on the FP32 bit pattern, no
  transcendental functions (``tl.log2`` / ``tl.exp2`` are approximate on the SFU), IEEE division
  ``tl.math.div_rn`` for the non-power-of-two E4M3 scale. The ``*_tile`` device functions are
  importable by fused kernels (window-KV insert, compressed-KV store, indexer).
* torch (``impl="torch"``): different technique on purpose -- ``torch.float8_e4m3fn`` casts for
  E4M3 and a threshold table for E2M1. Divisions use tensor divisors: PyTorch turns a division by
  a CPU scalar into a multiplication by its reciprocal, which is not IEEE division.

Non-finite inputs: a block whose max|x| is not finite produces NaN for the whole block (fail loud;
the reference produces NaN/garbage there too).
"""

from __future__ import annotations

import torch

from vllm.models.deepseek_v41 import knobs
from vllm.triton_utils import tl, triton

FP8_E4M3_MAX = 448.0
FP4_E2M1_MAX = 6.0
FP8_BLOCK = 32          # window KV (ref fp8_block_size)
FP4_E8M0_BLOCK = 32     # indexer q / index-K (ref fp4_block_size)
FP4_E4M3_BLOCK = 16     # compressed KV (ref Attention._compress_kv)

# FP32 constants exactly as the reference kernel sees them (Python double -> FP32, RNE).
_FP8_AMAX_FLOOR = 1e-4                  # T.max(amax, 1e-4)
_FP8_MAX_INV = 1.0 / FP8_E4M3_MAX       # fp8_max_inv
_FP4_MAX_INV = 1.0 / FP4_E2M1_MAX       # fp4_max_inv
_FP4_E4M3_AMAX_FLOOR = 6.0 * 2.0**-9    # 6 * (2**-9): keeps an all-zero group's scale nonzero
_FP4_E8M0_AMAX_FLOOR = 6.0 * 2.0**-126  # 6 * (2**-126)

IMPL_ENV = "VLLM_DS41_ATTN_QAT_IMPL"
_IMPLS = ("triton", "torch")


def default_impl() -> str:
    return knobs.env_str(IMPL_ENV, "triton", choices=_IMPLS)


# =====================================================================================
# Triton device functions (bit-exact building blocks for every L-ATTN kernel)
# =====================================================================================


@triton.jit
def _pow2_i32(k):
    """2.0**k as FP32 for integer k in [-126, 127] (exact, built from bits)."""
    return ((k + 127) << 23).to(tl.float32, bitcast=True)


@triton.jit
def _rne_grid_abs(a, MANT_BITS: tl.constexpr, MIN_EXP: tl.constexpr):
    """Round a finite FP32 a >= 0 to the nearest value of a binary format with MANT_BITS explicit
    mantissa bits and minimum normal exponent MIN_EXP (subnormal step 2^(MIN_EXP-MANT_BITS)),
    ties to even. Pure integer arithmetic on the bit pattern; the caller saturates beforehand."""
    bits = a.to(tl.int32, bitcast=True)
    biased = (bits >> 23) & 0xFF
    e = biased - 127
    sig = (bits & 0x7FFFFF) | 0x800000
    e_eff = tl.maximum(e, MIN_EXP)
    shift = tl.minimum(23 - MANT_BITS + (e_eff - e), 30)
    q = sig >> shift
    rem = sig & ((1 << shift) - 1)
    half = 1 << (shift - 1)
    up = (rem > half) | ((rem == half) & ((q & 1) == 1))
    q = q + up.to(tl.int32)
    out = q.to(tl.float32) * _pow2_i32(e_eff - MANT_BITS)
    # biased == 0: zero or an FP32 subnormal (< 2^-126), far below both grids' half-step.
    return tl.where(biased == 0, 0.0, out)


@triton.jit
def _with_sign_of(mag, x):
    """|mag| with the sign bit of x (keeps -0.0 for negative inputs that round to zero)."""
    sign = x.to(tl.int32, bitcast=True) & (-2147483648)
    return (mag.to(tl.int32, bitcast=True) | sign).to(tl.float32, bitcast=True)


@triton.jit
def _block_amax_finite(x):
    """(max|x| over the last axis, block is finite). Triton's max/clamp drop NaN (fmax semantics), so a
    NaN element is detected explicitly: a block holding NaN or +-inf is reported non-finite."""
    a = tl.abs(x)
    amax = tl.max(a, axis=2)
    bad = tl.max(((a != a) | (a == float("inf"))).to(tl.int32), axis=2)
    return amax, bad == 0


@triton.jit
def e4m3_rne(x):
    """FP32 -> nearest E4M3FN value (RNE, saturating at +-448), returned as FP32. NaN stays NaN."""
    a = tl.minimum(tl.abs(x), 448.0)
    mag = _rne_grid_abs(a, 3, -6)
    return tl.where(x != x, x, _with_sign_of(mag, x))


@triton.jit
def e2m1_rne(x):
    """FP32 -> nearest E2M1 value (RNE, saturating at +-6), returned as FP32. NaN stays NaN."""
    a = tl.minimum(tl.abs(x), 6.0)
    mag = _rne_grid_abs(a, 1, 0)
    return tl.where(x != x, x, _with_sign_of(mag, x))


@triton.jit
def ceil_log2_i32(v):
    """ceil_log2(v) for finite normal v > 0 (ref fast_log2_ceil): exponent bits - 127 + (mantissa != 0)."""
    bits = v.to(tl.int32, bitcast=True)
    return ((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0).to(tl.int32)


@triton.jit
def fp8_block32_qdq_tile(x, BLOCK: tl.constexpr):
    """x: FP32 tile [R, NB, BLOCK] (blocks on the last axis) -> window-KV QAT values, FP32."""
    amax, finite = _block_amax_finite(x)
    amax = tl.maximum(amax, 1e-4)
    k = ceil_log2_i32(tl.where(finite, amax, 1.0) * (1.0 / 448.0))
    s3 = tl.expand_dims(_pow2_i32(k), 2)
    inv3 = tl.expand_dims(_pow2_i32(-k), 2)
    # x / 2^k == x * 2^-k exactly (Triton's "/" is div.full, not IEEE: never used here)
    y = e4m3_rne(tl.clamp(x * inv3, -448.0, 448.0)) * s3
    return tl.where(tl.expand_dims(finite, 2), y, float("nan"))


@triton.jit
def fp4_e8m0_qdq_tile(x, BLOCK: tl.constexpr):
    """x: FP32 tile [R, NB, BLOCK] -> indexer q / index-K QAT values (E2M1 x UE8M0), FP32."""
    amax, finite = _block_amax_finite(x)
    amax = tl.maximum(amax, 6.0 * 2.0**-126)
    k = ceil_log2_i32(tl.where(finite, amax, 1.0) * (1.0 / 6.0))
    s3 = tl.expand_dims(_pow2_i32(k), 2)
    inv3 = tl.expand_dims(_pow2_i32(-k), 2)
    y = e2m1_rne(tl.clamp(x * inv3, -6.0, 6.0)) * s3
    return tl.where(tl.expand_dims(finite, 2), y, float("nan"))


@triton.jit
def fp4_e4m3_qdq_tile(x, BLOCK: tl.constexpr):
    """x: FP32 tile [R, NB, BLOCK] -> compressed-KV QAT values (E2M1 x E4M3), FP32."""
    amax, finite = _block_amax_finite(x)
    amax = tl.maximum(tl.where(finite, amax, 1.0), 6.0 * 2.0**-9)
    s = e4m3_rne(tl.math.div_rn(amax, tl.full(amax.shape, 6.0, tl.float32)))
    s3 = tl.expand_dims(s, 2)
    y = e2m1_rne(tl.clamp(tl.math.div_rn(x, tl.broadcast_to(s3, x.shape)), -6.0, 6.0)) * s3
    return tl.where(tl.expand_dims(finite, 2), y, float("nan"))


@triton.jit
def _qdq_kernel(x_ptr, y_ptr, n_rows, x_row_stride, y_row_stride,
                N: tl.constexpr, BLOCK: tl.constexpr, KIND: tl.constexpr, ROWS: tl.constexpr):
    NB: tl.constexpr = N // BLOCK
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, N)
    mask = (rows < n_rows)[:, None]
    x = tl.load(x_ptr + rows[:, None].to(tl.int64) * x_row_stride + cols[None, :], mask=mask, other=0.0)
    x3 = tl.reshape(x.to(tl.float32), (ROWS, NB, BLOCK))
    if KIND == 0:
        y3 = fp8_block32_qdq_tile(x3, BLOCK)
    elif KIND == 1:
        y3 = fp4_e8m0_qdq_tile(x3, BLOCK)
    else:
        y3 = fp4_e4m3_qdq_tile(x3, BLOCK)
    y = tl.reshape(y3, (ROWS, N))
    tl.store(y_ptr + rows[:, None].to(tl.int64) * y_row_stride + cols[None, :],
             y.to(y_ptr.dtype.element_ty), mask=mask)


_KIND = {"fp8_block32": 0, "fp4_e8m0": 1, "fp4_e4m3": 2}


def _qdq_triton(x: torch.Tensor, block: int, kind: str, out_dtype: torch.dtype) -> torch.Tensor:
    n = x.shape[-1]
    if n & (n - 1):
        raise ValueError(f"QAT Triton path needs a power-of-two last dim, got {n}")
    x2 = x.reshape(-1, n)
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    y = torch.empty(x2.shape, dtype=out_dtype, device=x.device)
    rows = x2.shape[0]
    if rows:
        per = max(1, min(64, 4096 // n))
        _qdq_kernel[(triton.cdiv(rows, per),)](
            x2, y, rows, x2.stride(0), y.stride(0),
            N=n, BLOCK=block, KIND=_KIND[kind], ROWS=per, num_warps=4)
    return y.view(x.shape)


# =====================================================================================
# torch twins (independent technique; also the CPU / debugging path)
# =====================================================================================


def _fp32_bits(v: torch.Tensor) -> torch.Tensor:
    return v.contiguous().view(torch.int32)


def pow2_ceil_log2_torch(v: torch.Tensor) -> torch.Tensor:
    """2^ceil_log2(v) for FP32 v > 0 (finite, normal)."""
    bits = _fp32_bits(v)
    k = ((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    return ((k + 127) << 23).view(torch.float32)


def e4m3_rne_torch(y: torch.Tensor) -> torch.Tensor:
    """FP32 -> E4M3FN (RNE) -> FP32 with saturation at +-448 (torch's cast returns NaN above 480)."""
    return torch.clamp(y, -FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn).to(torch.float32)


def e2m1_rne_torch(y: torch.Tensor) -> torch.Tensor:
    """FP32 -> nearest E2M1 value (ties to even mantissa), saturating at +-6, as FP32."""
    a = y.abs()
    r = torch.zeros_like(a)
    r = torch.where(a > 0.25, 0.5, r)
    r = torch.where(a >= 0.75, 1.0, r)
    r = torch.where(a > 1.25, 1.5, r)
    r = torch.where(a >= 1.75, 2.0, r)
    r = torch.where(a > 2.5, 3.0, r)
    r = torch.where(a >= 3.5, 4.0, r)
    r = torch.where(a > 5.0, 6.0, r)
    r = torch.copysign(r, y)
    return torch.where(torch.isnan(y), y, r)


def _blocks(x: torch.Tensor, block: int) -> torch.Tensor:
    n = x.shape[-1]
    if n % block:
        raise ValueError(f"last dim {n} is not a multiple of the QAT block {block}")
    return x.float().reshape(*x.shape[:-1], n // block, block)


def _finish(y: torch.Tensor, amax: torch.Tensor, x: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    y = torch.where(torch.isfinite(amax).unsqueeze(-1), y, torch.full_like(y, float("nan")))
    return y.reshape(x.shape).to(out_dtype)


def fp8_block32_qdq_torch(x: torch.Tensor, out_dtype: torch.dtype = torch.float16) -> torch.Tensor:
    xb = _blocks(x, FP8_BLOCK)
    amax = xb.abs().amax(dim=-1)
    floor = torch.tensor(_FP8_AMAX_FLOOR, dtype=torch.float32, device=x.device)
    inv = torch.tensor(_FP8_MAX_INV, dtype=torch.float32, device=x.device)
    safe = torch.where(torch.isfinite(amax), torch.maximum(amax, floor), torch.ones_like(amax))
    s = pow2_ceil_log2_torch(safe * inv).unsqueeze(-1)
    y = e4m3_rne_torch(torch.clamp(xb / s, -FP8_E4M3_MAX, FP8_E4M3_MAX)) * s
    return _finish(y, amax, x, out_dtype)


def fp4_e8m0_qdq_torch(x: torch.Tensor, out_dtype: torch.dtype = torch.float16,
                       block: int = FP4_E8M0_BLOCK) -> torch.Tensor:
    xb = _blocks(x, block)
    amax = xb.abs().amax(dim=-1)
    floor = torch.tensor(_FP4_E8M0_AMAX_FLOOR, dtype=torch.float32, device=x.device)
    inv = torch.tensor(_FP4_MAX_INV, dtype=torch.float32, device=x.device)
    safe = torch.where(torch.isfinite(amax), torch.maximum(amax, floor), torch.ones_like(amax))
    s = pow2_ceil_log2_torch(safe * inv).unsqueeze(-1)
    y = e2m1_rne_torch(torch.clamp(xb / s, -FP4_E2M1_MAX, FP4_E2M1_MAX)) * s
    return _finish(y, amax, x, out_dtype)


def fp4_e4m3_qdq_torch(x: torch.Tensor, out_dtype: torch.dtype = torch.float16,
                       block: int = FP4_E4M3_BLOCK) -> torch.Tensor:
    xb = _blocks(x, block)
    amax = xb.abs().amax(dim=-1)
    floor = torch.tensor(_FP4_E4M3_AMAX_FLOOR, dtype=torch.float32, device=x.device)
    safe = torch.where(torch.isfinite(amax), torch.maximum(amax, floor), torch.ones_like(amax))
    s = e4m3_rne_torch(safe / torch.full_like(safe, FP4_E2M1_MAX)).unsqueeze(-1)  # tensor divisor: IEEE
    y = e2m1_rne_torch(torch.clamp(xb / s, -FP4_E2M1_MAX, FP4_E2M1_MAX)) * s
    return _finish(y, amax, x, out_dtype)


# =====================================================================================
# public entry points
# =====================================================================================


def _resolve(impl: str | None, x: torch.Tensor) -> str:
    impl = impl or default_impl()
    if impl not in _IMPLS:
        raise ValueError(f"unknown QAT impl {impl!r}; expected one of {_IMPLS}")
    if impl == "triton" and not x.is_cuda:
        raise ValueError("QAT impl 'triton' needs a CUDA tensor; pass impl='torch' for CPU tensors")
    return impl


def fp8_block32_qdq(x: torch.Tensor, out_dtype: torch.dtype = torch.float16,
                    impl: str | None = None) -> torch.Tensor:
    """Window-KV QAT (ref ``act_quant(x, 32, "ue8m0", e8m0, inplace=True)``) over the last dim."""
    if _resolve(impl, x) == "triton":
        return _qdq_triton(x, FP8_BLOCK, "fp8_block32", out_dtype)
    return fp8_block32_qdq_torch(x, out_dtype)


def fp4_e8m0_qdq(x: torch.Tensor, out_dtype: torch.dtype = torch.float16,
                 impl: str | None = None) -> torch.Tensor:
    """Indexer q / index-K QAT (ref ``fp4_act_quant(x, 32, inplace=True)``) over the last dim."""
    if _resolve(impl, x) == "triton":
        return _qdq_triton(x, FP4_E8M0_BLOCK, "fp4_e8m0", out_dtype)
    return fp4_e8m0_qdq_torch(x, out_dtype)


def fp4_e4m3_qdq(x: torch.Tensor, out_dtype: torch.dtype = torch.float16,
                 impl: str | None = None) -> torch.Tensor:
    """Compressed-KV QAT (ref ``fp4_act_quant(x, 16, True, scale_dtype=e4m3fn)``) over the last dim."""
    if _resolve(impl, x) == "triton":
        return _qdq_triton(x, FP4_E4M3_BLOCK, "fp4_e4m3", out_dtype)
    return fp4_e4m3_qdq_torch(x, out_dtype)
