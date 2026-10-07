# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 indexer kernels on FP16 index-K records (PORT_DESIGN §1 "Indexer" row; owner L-ATTN).

* ``index_q_rope_qat``: q [T, 32, 128] -> RoPE on the last 64 dims (interleaved, FP32) -> E2M1 x UE8M0/32 QAT.
* ``index_k_rope_qat_store``: K [N, 128] (= k_norm(wk(pre-RoPE latent)), FP32) -> RoPE at the group's first
  position -> same QAT -> FP16 rows of the index-K cache (+ optional export rows for the PP3 mirror).
* ``index_scores``: score[t, j] = sum_h relu(q[t,h] . k[j]) * w[t,h] in FP32 (cuBLAS FP16 GEMM with FP32
  accumulate/output per key tile, FP32 epilogue) -- no Hadamard (ref:m.py:550-557).
* ``topk_sorted``: top-min(512, n) by score, re-sorted ascending by position, -1 for unreachable picks
  (ref:m.py:577-580).
"""

from __future__ import annotations

import torch

from vllm.models.deepseek_v41.common import qat
from vllm.models.deepseek_v41.common.rope import apply_rope_torch
from vllm.triton_utils import tl, triton


@triton.jit
def _rope_fp4_tile(e, o, cs_ptr, pos, RD: tl.constexpr, D: tl.constexpr):
    """e, o: [R, D/2] even/odd FP32 halves of R rows at one position (pos scalar) -> [R, D] QAT FP32."""
    HALF: tl.constexpr = RD // 2
    i = tl.arange(0, D // 2)
    jj = i - (D - RD) // 2
    rot = jj >= 0
    c = tl.load(cs_ptr + pos * RD + jj, mask=rot, other=1.0)[None, :]
    s = tl.load(cs_ptr + pos * RD + HALF + jj, mask=rot, other=0.0)[None, :]
    re = tl.where(rot[None, :], e * c - o * s, e)
    ro = tl.where(rot[None, :], e * s + o * c, o)
    R: tl.constexpr = e.shape[0]
    x = tl.reshape(tl.join(re, ro), (R, D // 32, 32))
    return tl.reshape(qat.fp4_e8m0_qdq_tile(x, 32), (R, D))


@triton.jit
def _index_q_kernel(q_ptr, q_st, q_sh, out_ptr, pos_ptr, cs_ptr,
                    H: tl.constexpr, D: tl.constexpr, RD: tl.constexpr):
    t = tl.program_id(0)
    pos = tl.load(pos_ptr + t).to(tl.int64)
    h = tl.arange(0, H)[:, None]
    i = tl.arange(0, D // 2)[None, :]
    base = q_ptr + t.to(tl.int64) * q_st + h * q_sh
    e = tl.load(base + 2 * i).to(tl.float32)
    o = tl.load(base + 2 * i + 1).to(tl.float32)
    y = _rope_fp4_tile(e, o, cs_ptr, pos, RD, D)
    d = tl.arange(0, D)[None, :]
    tl.store(out_ptr + t.to(tl.int64) * H * D + h * D + d, y.to(out_ptr.dtype.element_ty))


@triton.jit
def _index_k_kernel(k_ptr, k_st, pos_ptr, cs_ptr, rows_ptr, slot_ptr, exp_ptr,
                    D: tl.constexpr, RD: tl.constexpr, EXPORT: tl.constexpr):
    n = tl.program_id(0)
    pos = tl.load(pos_ptr + n).to(tl.int64)
    i = tl.arange(0, D // 2)[None, :]
    e = tl.load(k_ptr + n.to(tl.int64) * k_st + 2 * i).to(tl.float32)
    o = tl.load(k_ptr + n.to(tl.int64) * k_st + 2 * i + 1).to(tl.float32)
    y = tl.reshape(_rope_fp4_tile(e, o, cs_ptr, pos, RD, D), (D,)).to(rows_ptr.dtype.element_ty)
    d = tl.arange(0, D)
    slot = tl.load(slot_ptr + n)
    if slot >= 0:
        tl.store(rows_ptr + slot * D + d, y)
    if EXPORT:
        tl.store(exp_ptr + n.to(tl.int64) * D + d, y)


def index_q_rope_qat(q: torch.Tensor, positions: torch.Tensor, cos_sin_cache: torch.Tensor,
                     impl: str = "sm70") -> torch.Tensor:
    """q [T, H, D] fp16 or fp32 -> QAT'd rotated q [T, H, D] fp16 (new tensor; FP16 holds the QAT values exactly
    for block exponents >= -23)."""
    T, H, D = q.shape
    if impl == "torch":
        return qat.fp4_e8m0_qdq(apply_rope_torch(q.float(), positions[:T], cos_sin_cache),
                                out_dtype=torch.float16, impl="torch")
    out = torch.empty((T, H, D), dtype=torch.float16, device=q.device)
    if T:
        if q.stride(-1) != 1:
            raise ValueError("index_q_rope_qat: q must be contiguous in the last dim")
        _index_q_kernel[(T,)](q, q.stride(0), q.stride(1), out, positions, cos_sin_cache,
                              H=H, D=D, RD=cos_sin_cache.shape[-1], num_warps=4)
    return out


def index_k_rope_qat_store(k: torch.Tensor, latent_pos: torch.Tensor, cos_sin_cache: torch.Tensor,
                           rows: torch.Tensor, slots: torch.Tensor, export: torch.Tensor | None,
                           impl: str = "sm70") -> None:
    """k [N, D] fp32 -> rotated + QAT'd FP16 rows at ``slots`` [N] (-1: skip); ``export`` [N, D] gets the same
    records (PP3 mirror of source 20)."""
    N, D = k.shape
    if N == 0:
        return
    if impl == "torch":
        rec = qat.fp4_e8m0_qdq(apply_rope_torch(k.float(), latent_pos, cos_sin_cache), out_dtype=rows.dtype,
                               impl="torch")
        keep = slots >= 0
        rows.index_copy_(0, slots[keep], rec[keep])
        if export is not None:
            export[:N] = rec
        return
    _index_k_kernel[(N,)](k, k.stride(0), latent_pos, cos_sin_cache, rows, slots,
                          export if export is not None else rows, D=D, RD=cos_sin_cache.shape[-1],
                          EXPORT=export is not None, num_warps=2)


def index_scores(q: torch.Tensor, weights: torch.Tensor, keys: torch.Tensor, out: torch.Tensor,
                 key_tile: int = 16384) -> torch.Tensor:
    """q [Q, H, D] fp16, weights [Q, H] fp32, keys [n, D] fp16 -> out[:, :n] = sum_h relu(q.k) * w (FP32)."""
    Q, H, D = q.shape
    n = keys.shape[0]
    q2 = q.reshape(Q * H, D)
    w3 = weights.float().reshape(Q, H, 1)
    for k0 in range(0, n, key_tile):
        k1 = min(n, k0 + key_tile)
        logits = torch.mm(q2, keys[k0:k1].t(), out_dtype=torch.float32).view(Q, H, k1 - k0)
        out[:, k0:k1] = (logits.relu_() * w3).sum(dim=1)
    return out[:, :n]


def topk_sorted(scores: torch.Tensor, ends: torch.Tensor, k: int) -> torch.Tensor:
    """scores [Q, n] FP32 (already -inf where masked), ends [Q] -> [Q, k] int32: the top-min(k, n) positions,
    ascending, -1 for picks >= end (and -1 padding when n < k)."""
    Q, n = scores.shape
    out = torch.full((Q, k), -1, dtype=torch.int32, device=scores.device)
    kk = min(k, n)
    if Q == 0 or kk == 0:
        return out
    idx = scores.topk(kk, dim=-1, sorted=False).indices.sort(dim=-1).values
    out[:, :kk] = torch.where(idx < ends[:, None].to(idx.dtype), idx, torch.full_like(idx, -1)).to(torch.int32)
    return out
