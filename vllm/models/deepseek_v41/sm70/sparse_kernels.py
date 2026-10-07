# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gather kernels over FP16 KV records (PORT_DESIGN §1 "Sparse attention kernel" row; owner L-ATTN).

Per query token t the reference (ref:k.py:310-403) attends over at most 128 window rows plus at most 512
compressed rows (indices -1 = invalid), with a per-head ``attn_sink`` logit in the denominator only:

    s_j  = scale * q . k_j                    (FP32)       m = max(-1e30, max_j s_j)
    l    = sum_j exp(s_j - m) + exp(sink - m)              o = sum_j exp(s_j - m) * k_j / l

K = V (MQA latent, all 512 dims). The finite running-max floor ``-1e30`` makes a row with no valid
index output exactly 0 (exp(sink + 1e30) = inf in the denominator), as in the reference.

Two implementations over the same gathered rows (query-tiled so the gather workspace stays bounded):
* ``torch``: FP32 everything (gather -> float -> einsum) -- the in-tree oracle;
* ``sm70``: cuBLAS batched FP16 GEMMs with FP32 accumulate and output (``bmm(out_dtype=float32)``), FP32
  softmax statistics, unnormalised probabilities rounded to FP16 for the PV GEMM (the reference rounds
  them to BF16), normalisation in FP32.
"""

from __future__ import annotations

import torch

from vllm.models.deepseek_v41.common import qat
from vllm.models.deepseek_v41.common.rope import apply_rope_torch
from vllm.triton_utils import tl, triton

RUNNING_MAX_FLOOR = -1e30


# ----------------------------------------------------------------------------- compressed-KV store
@triton.jit
def _ckv_store_kernel(lat_ptr, lat_st, pos_ptr, cs_ptr, rows_ptr, slot_ptr, exp_ptr,
                      D: tl.constexpr, RD: tl.constexpr, EXPORT: tl.constexpr):
    n = tl.program_id(0)
    HALF: tl.constexpr = RD // 2
    pos = tl.load(pos_ptr + n).to(tl.int64)
    i = tl.arange(0, D // 2)
    e = tl.load(lat_ptr + n.to(tl.int64) * lat_st + 2 * i).to(tl.float32)
    o = tl.load(lat_ptr + n.to(tl.int64) * lat_st + 2 * i + 1).to(tl.float32)
    jj = i - (D - RD) // 2
    rot = jj >= 0
    c = tl.load(cs_ptr + pos * RD + jj, mask=rot, other=1.0)
    s = tl.load(cs_ptr + pos * RD + HALF + jj, mask=rot, other=0.0)
    re = tl.where(rot, e * c - o * s, e)
    ro = tl.where(rot, e * s + o * c, o)
    x = tl.reshape(tl.join(re, ro), (1, D // 16, 16))
    y = tl.reshape(qat.fp4_e4m3_qdq_tile(x, 16), (D,)).to(rows_ptr.dtype.element_ty)
    d = tl.arange(0, D)
    slot = tl.load(slot_ptr + n)
    if slot >= 0:
        tl.store(rows_ptr + slot * D + d, y)
    if EXPORT:
        tl.store(exp_ptr + n.to(tl.int64) * D + d, y)


def ckv_rope_qat_store(latent: torch.Tensor, latent_pos: torch.Tensor, cos_sin_cache: torch.Tensor,
                       rows: torch.Tensor, slots: torch.Tensor, export: torch.Tensor | None,
                       impl: str = "sm70") -> None:
    """Compressed-KV write (ref:m.py:751-761): RoPE on the last 64 dims at the group's first position,
    E2M1 x E4M3/16 QAT, FP16 rows at ``slots`` [N] (-1: skip); ``export`` [N, 512] receives the same records."""
    N, D = latent.shape
    if N == 0:
        return
    if impl == "torch":
        rec = qat.fp4_e4m3_qdq(apply_rope_torch(latent.float(), latent_pos, cos_sin_cache), out_dtype=rows.dtype,
                               impl="torch")
        keep = slots >= 0
        rows.index_copy_(0, slots[keep], rec[keep])
        if export is not None:
            export[:N] = rec
        return
    if latent.stride(-1) != 1:
        raise ValueError("ckv_rope_qat_store: latent must be contiguous in the last dim")
    _ckv_store_kernel[(N,)](latent, latent.stride(0), latent_pos, cos_sin_cache, rows, slots,
                            export if export is not None else rows, D=D, RD=cos_sin_cache.shape[-1],
                            EXPORT=export is not None, num_warps=4)


# ----------------------------------------------------------------------------- index -> row translation
def logical_to_rows(idx: torch.Tensor, tok2req: torch.Tensor, block_table: torch.Tensor,
                    storage_block_size: int) -> torch.Tensor:
    """Compressed logical positions idx [T, K] (-1 pad) of each token's request -> cache rows [T, K] int64
    through the SOURCE layer's block table (row = block * storage + j % storage)."""
    valid = idx >= 0
    j = idx.clamp(min=0).to(torch.int64)
    blk = block_table[tok2req.to(torch.long)[:, None].expand_as(j), j // storage_block_size].to(torch.int64)
    return torch.where(valid, blk * storage_block_size + j % storage_block_size, torch.full_like(j, -1))


# ----------------------------------------------------------------------------- attention
def _gather(rows: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """rows [N, D], idx [Q, K] int64 (-1 allowed) -> [Q, K, D]; invalid entries are ZERO rows (the reference
    loads 0 for idx == -1): masking the score alone is not enough, since 0 * (inf or NaN garbage) in the PV
    product is NaN."""
    g = rows.index_select(0, idx.clamp(min=0).reshape(-1)).view(*idx.shape, rows.shape[-1])
    return g.masked_fill_((idx < 0)[..., None], 0)


def sparse_attention(q: torch.Tensor, sources: list[tuple[torch.Tensor, torch.Tensor]], sink: torch.Tensor,
                     scale: float, out: torch.Tensor, impl: str = "sm70", tile: int = 256) -> None:
    """q [T, H, D] fp16 (rotated); sources = [(rows [N_i, D] fp16, idx [T, K_i] int64 row ids, -1 invalid)];
    sink [H] fp32; writes out [T, H, D] fp16."""
    T, H, D = q.shape
    if T == 0:
        return
    if out.shape != q.shape:
        raise ValueError(f"out {tuple(out.shape)} != q {tuple(q.shape)}")
    for rows, idx in sources:
        if rows.dtype != torch.float16 or rows.shape[-1] != D or idx.shape[0] != T:
            raise ValueError(f"bad source rows {tuple(rows.shape)} {rows.dtype} idx {tuple(idx.shape)}")
    sinkf = sink.float()[:H]
    for t0 in range(0, T, tile):
        t1 = min(T, t0 + tile)
        kv = torch.cat([_gather(rows, idx[t0:t1]) for rows, idx in sources], dim=1)       # [Q, K, D] fp16
        valid = torch.cat([idx[t0:t1] >= 0 for _, idx in sources], dim=1)                # [Q, K]
        qt = q[t0:t1]
        if impl == "torch":
            s = torch.einsum("qhd,qkd->qhk", qt.float(), kv.float()) * scale
        elif impl == "sm70":
            s = torch.bmm(qt, kv.transpose(1, 2), out_dtype=torch.float32)
            s.mul_(scale)
        else:
            raise ValueError(f"unknown impl {impl!r}")
        s.masked_fill_(~valid[:, None, :], float("-inf"))
        m = s.amax(dim=-1).clamp(min=RUNNING_MAX_FLOOR)                                   # [Q, H]
        p = torch.exp(s - m[..., None])
        denom = p.sum(dim=-1) + torch.exp(sinkf[None, :] - m)
        if impl == "torch":
            o = torch.einsum("qhk,qkd->qhd", p, kv.float())
        else:
            o = torch.bmm(p.to(torch.float16), kv, out_dtype=torch.float32)
        out[t0:t1] = (o / denom[..., None]).to(out.dtype)
