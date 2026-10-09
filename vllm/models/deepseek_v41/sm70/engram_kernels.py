# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM70 Triton kernels of the DeepSeek-V4.1 Engram (lane L-ENGRAM, PORT_DESIGN §2.2 / §4.1).

* ``decode_rows``: 264-byte rows (256 E4M3 + 8 UE8M0) -> FP16 ``value * 2^(scale - 127 + row_bias)``. Exact: the
  value is assembled from integer fields (no FP8 hardware, no approximate exp2); 0x7F/0xFF stay NaN.
* ``post_wkv_gate_``: the gate of ref:m.py:350-365 fused after the wkv all-reduce, in place on the BF16 HC stream.
  FP32 math; BF16 bits are converted by hand (Volta has no BF16 instructions), rounding to nearest-even exactly
  like ``torch.Tensor.to(torch.bfloat16)``; the sign of the gate input follows ``copysign`` (sign bit, so -0.0
  stays negative).
"""

from __future__ import annotations

import torch

from vllm.models.deepseek_v41.common.contracts import ENGRAM_ROW_BYTES, HC, HIDDEN, NORM_EPS, STREAM_DTYPE
from vllm.triton_utils import tl, triton

GATE_CLAMP = 1e-6
_VALUE_BYTES = 256


@triton.jit
def _decode_rows_kernel(rows_ptr, out_ptr, stride_rt, stride_rs, stride_ot, row_bias,
                        VALUE_BYTES: tl.constexpr, SCALE_BLOCK: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    s = tl.program_id(1).to(tl.int64)
    offs = tl.arange(0, VALUE_BYTES)
    base = rows_ptr + t * stride_rt + s * stride_rs
    b = tl.load(base + offs).to(tl.int32)
    sc = tl.load(base + VALUE_BYTES + offs // SCALE_BLOCK).to(tl.int32)
    k = sc - 127 + row_bias
    ex = (b >> 3) & 0xF
    man = b & 7
    # E4M3 (fn, bias 7): normal = (8 + man) * 2^(ex - 10); subnormal = man * 2^-9
    e_eff = tl.where(ex == 0, k - 9, ex - 10 + k)
    m_eff = tl.where(ex == 0, man, man + 8).to(tl.float32)
    p2 = (tl.minimum(tl.maximum(e_eff + 127, 1), 254) << 23).to(tl.float32, bitcast=True)
    v = m_eff * p2
    # sign by bit (``-v`` may lower to ``0 - v`` and turn -0.0 into +0.0)
    v = (v.to(tl.int32, bitcast=True) | ((b & 0x80) << 24)).to(tl.float32, bitcast=True)
    v = tl.where((b & 0x7F) == 0x7F, float("nan"), v)
    tl.store(out_ptr + t * stride_ot + s * VALUE_BYTES + offs, v.to(tl.float16))


def decode_rows(rows: torch.Tensor, row_bias: int) -> torch.Tensor:
    """rows [n, S, 264] uint8 (inner dim contiguous) -> [n, S * 256] fp16 (exact when the service's row bias
    puts every scale in FP16's exact range; see common.engram.row_bias_for_exponents)."""
    if rows.dtype != torch.uint8 or rows.dim() != 3 or rows.shape[2] != ENGRAM_ROW_BYTES or rows.stride(2) != 1:
        raise ValueError(f"decode_rows: rows {tuple(rows.shape)} {rows.dtype} strides {rows.stride()}")
    n, s, _ = rows.shape
    out = torch.empty((n, s * _VALUE_BYTES), dtype=torch.float16, device=rows.device)
    if n == 0:
        return out
    _decode_rows_kernel[(n, s)](rows, out, rows.stride(0), rows.stride(1), out.stride(0), int(row_bias),
                                VALUE_BYTES=_VALUE_BYTES, SCALE_BLOCK=_VALUE_BYTES // (ENGRAM_ROW_BYTES - _VALUE_BYTES),
                                num_warps=2)
    return out


@triton.jit
def _bf16_bits_to_f32(x):
    return ((x.to(tl.int32) & 0xFFFF) << 16).to(tl.float32, bitcast=True)


@triton.jit
def _f32_to_bf16_bits(y):
    u = y.to(tl.uint32, bitcast=True)
    r = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
    r = tl.where(y != y, 0x7FC0, r)
    return r.to(tl.int16)


@triton.jit
def _post_wkv_gate_kernel(stream_ptr, kv_ptr, qk_ptr, alpha, eps, clamp, inv_sqrt_dim,
                          stride_st, stride_sh, stride_kvt,
                          DIM: tl.constexpr, HCM: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    t = (pid // HCM).to(tl.int64)
    c = pid % HCM
    sp = stream_ptr + t * stride_st + c * stride_sh
    kp = kv_ptr + t * stride_kvt + c * DIM
    vp = kv_ptr + t * stride_kvt + HCM * DIM
    qp = qk_ptr + c * DIM
    acc_h = tl.zeros([BLOCK], dtype=tl.float32)
    acc_k = tl.zeros([BLOCK], dtype=tl.float32)
    acc_d = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, DIM, BLOCK):
        o = off + tl.arange(0, BLOCK)
        m = o < DIM
        h = _bf16_bits_to_f32(tl.load(sp + o, mask=m, other=0))
        key = tl.load(kp + o, mask=m, other=0.0) * alpha
        q = tl.load(qp + o, mask=m, other=0.0)
        acc_h += h * h
        acc_k += key * key
        acc_d += (h * q) * key
    rstd = tl.rsqrt(tl.sum(acc_h, axis=0) / DIM + eps) * tl.rsqrt(tl.sum(acc_k, axis=0) / DIM + eps)
    dot = tl.sum(acc_d, axis=0) * rstd * inv_sqrt_dim
    g = tl.sqrt_rn(tl.maximum(tl.abs(dot), clamp, propagate_nan=tl.PropagateNan.ALL))   # clamp_min keeps NaN
    g = (g.to(tl.int32, bitcast=True) | (dot.to(tl.int32, bitcast=True) & -2147483648)).to(tl.float32, bitcast=True)
    gate = 1.0 / (1.0 + tl.exp(-g))
    for off in range(0, DIM, BLOCK):
        o = off + tl.arange(0, BLOCK)
        m = o < DIM
        h = _bf16_bits_to_f32(tl.load(sp + o, mask=m, other=0))
        val = tl.load(vp + o, mask=m, other=0.0) * alpha
        tl.store(sp + o, _f32_to_bf16_bits(h + gate * val), mask=m)


def post_wkv_gate_(stream: torch.Tensor, kv: torch.Tensor, qk: torch.Tensor, alpha: float) -> None:
    """In place on stream [n, 4, 5120] bf16: h_c += sigmoid(signed-sqrt(dot_c)) * value (ref:m.py:350-365).
    kv [n, 25600] float32 = all-reduced wkv output still carrying the power-of-two bias that ``alpha`` removes."""
    n = stream.shape[0]
    if stream.dtype != STREAM_DTYPE or tuple(stream.shape[1:]) != (HC, HIDDEN) or stream.stride(2) != 1:
        raise ValueError(f"post_wkv_gate_: stream {tuple(stream.shape)} {stream.dtype} strides {stream.stride()}")
    if kv.dtype != torch.float32 or tuple(kv.shape) != (n, (HC + 1) * HIDDEN) or kv.stride(1) != 1:
        raise ValueError(f"post_wkv_gate_: kv {tuple(kv.shape)} {kv.dtype}")
    if qk.dtype != torch.float32 or tuple(qk.shape) != (HC, HIDDEN) or not qk.is_contiguous():
        raise ValueError(f"post_wkv_gate_: qk {tuple(qk.shape)} {qk.dtype}")
    if n == 0:
        return
    bits = stream.view(torch.int16)
    _post_wkv_gate_kernel[(n * HC,)](bits, kv, qk, float(alpha), NORM_EPS, GATE_CLAMP, HIDDEN ** -0.5,
                                     bits.stride(0), bits.stride(1), kv.stride(0),
                                     DIM=HIDDEN, HCM=HC, BLOCK=1024, num_warps=4)
