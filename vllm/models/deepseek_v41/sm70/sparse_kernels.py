# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gather kernels over the KV records (PORT_DESIGN §1 "Sparse attention kernel" row; owner L-ATTN).

Window rows are FP16; compressed rows are packed FP4 records (``qat.unpack_ckv``) and are unpacked to the same
FP16 values right after the gather, so the attention math below is unchanged.

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

from vllm.models.deepseek_v41 import knobs
from vllm.models.deepseek_v41.common import qat
from vllm.models.deepseek_v41.common.rope import apply_rope_torch
from vllm.triton_utils import tl, triton

RUNNING_MAX_FLOOR = -1e30
FUSED_DECODE = knobs.env_bool("VLLM_DS41_ATTN_FUSED_DECODE", False)   # fused_decode_attention for small T (sm70 impl)
# Per-layer CUDA graphs of the decode attention break (gpubox lane): the metadata builders fill static buffers, every
# compressed group has exactly T latents (slot -1 = not stored), the compressor/indexer take no host decisions, and
# attention.py captures the per-layer segments as CUDA graphs keyed by (T, buffer addresses).
ATTN_GRAPH = knobs.env_bool("VLLM_DS41_ATTN_GRAPH", False)


# ----------------------------------------------------------------------------- compressor state rows
@triton.jit
def _state_store_kernel(src_ptr, src_st, slot_ptr, rows_ptr, D: tl.constexpr):
    n = tl.program_id(0)
    slot = tl.load(slot_ptr + n).to(tl.int64)
    if slot >= 0:
        d = tl.arange(0, D)
        v = tl.load(src_ptr + n.to(tl.int64) * src_st + d)
        tl.store(rows_ptr + slot * D + d, v)


def state_rows_store(src: torch.Tensor, slots: torch.Tensor, rows: torch.Tensor) -> None:
    """rows[slots[t]] = src[t] for slots[t] >= 0 (no host sync; the eager path's ``index_copy_(slots[keep])``)."""
    T, D = src.shape
    if T == 0:
        return
    if rows.shape[-1] != D or src.dtype != rows.dtype or src.stride(-1) != 1:
        raise ValueError(f"state_rows_store: src {tuple(src.shape)} {src.dtype} vs rows {tuple(rows.shape)} {rows.dtype}")
    _state_store_kernel[(T,)](src, src.stride(0), slots, rows, D=D, num_warps=4)


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
    codes, sbytes = qat.fp4_e4m3_quant_tile(x, 16)
    lo, hi = tl.split(tl.reshape(codes, (D // 2, 2)))                 # dims 2i (low nibble), 2i + 1 (high)
    nib = (lo | (hi << 4)).to(tl.uint8)
    sb = tl.reshape(sbytes, (D // 16,)).to(tl.uint8)
    j = tl.arange(0, D // 16)
    PACKED: tl.constexpr = D // 2 + D // 16
    slot = tl.load(slot_ptr + n).to(tl.int64)
    if slot >= 0:
        tl.store(rows_ptr + slot * PACKED + i, nib)
        tl.store(rows_ptr + slot * PACKED + D // 2 + j, sb)
    if EXPORT:
        tl.store(exp_ptr + n.to(tl.int64) * PACKED + i, nib)
        tl.store(exp_ptr + n.to(tl.int64) * PACKED + D // 2 + j, sb)


def ckv_rope_qat_store(latent: torch.Tensor, latent_pos: torch.Tensor, cos_sin_cache: torch.Tensor,
                       rows: torch.Tensor, slots: torch.Tensor, export: torch.Tensor | None,
                       impl: str = "sm70") -> None:
    """Compressed-KV write (ref:m.py:751-761): RoPE on the last 64 dims at the group's first position,
    E2M1 x E4M3/16 QAT, packed FP4 records (uint8 [*, 288]) at ``slots`` [N] (-1: skip); ``export`` [N, 288]
    receives the same records."""
    N, D = latent.shape
    if N == 0:
        return
    if rows.dtype != torch.uint8 or rows.shape[-1] != D // 2 + D // 16:
        raise TypeError(f"ckv_rope_qat_store: rows must be packed uint8 [*, {D // 2 + D // 16}], got {rows.dtype} "
                        f"{tuple(rows.shape)}")
    if impl == "torch":
        rec = qat.fp4_e4m3_pack_torch(apply_rope_torch(latent.float(), latent_pos, cos_sin_cache))
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
def _gather(rows: torch.Tensor, idx: torch.Tensor, impl: str = "sm70") -> torch.Tensor:
    """rows [N, D] fp16 or packed FP4 [N, 288] uint8, idx [Q, K] int64 (-1 allowed) -> [Q, K, D] fp16; invalid
    entries are ZERO rows (the reference loads 0 for idx == -1): masking the score alone is not enough, since
    0 * (inf or NaN garbage) in the PV product is NaN."""
    g = rows.index_select(0, idx.clamp(min=0).reshape(-1))
    if rows.dtype == torch.uint8:
        g = qat.unpack_ckv(g, impl="torch" if impl == "torch" else None)
    g = g.view(*idx.shape, g.shape[-1])
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
        fp16 = rows.dtype == torch.float16 and rows.shape[-1] == D
        packed = rows.dtype == torch.uint8 and rows.shape[-1] == qat.CKV_PACKED_DIM and D == 512
        if not (fp16 or packed) or idx.shape[0] != T:
            raise ValueError(f"bad source rows {tuple(rows.shape)} {rows.dtype} idx {tuple(idx.shape)}")
    sinkf = sink.float()[:H]
    if impl == "sm70" and FUSED_DECODE and T <= FUSED_DECODE_MAX_T and D == 512:
        fused_decode_attention(q, sources, sinkf, scale, out)
        return
    for t0 in range(0, T, tile):
        t1 = min(T, t0 + tile)
        kv = torch.cat([_gather(rows, idx[t0:t1], impl) for rows, idx in sources], dim=1)  # [Q, K, D] fp16
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


# ----------------------------------------------------------------------------- fused decode attention (gpubox lane)
# Split-K decode attention in two launches. Kernel 1: one program per (query token, 16-key block) gathers its rows
# once -- window rows from the FP16 cache, compressed rows from the packed FP4 cache (unpacked in registers) -- and
# computes, for every head, the block's softmax partials (max, sum, P.V accumulator). Kernel 2: one program per
# (token, head) merges the partials in fixed order, adds the attention sink and normalises. Replaces the ~18 eager
# launches of the tiled torch path per layer per decode step. Dims are handled as (even, odd) halves so a packed
# nibble byte maps onto one (even, odd) pair; q and the output are split / merged the same way.
# Numerics: FP32 throughout (the sm70 bmm path rounds p to FP16); within FP32 ulps of the torch oracle, not
# bit-identical. Gated by tolerance tests and the KL panel.

FUSED_DECODE_ENV = "VLLM_DS41_ATTN_FUSED_DECODE"
FUSED_DECODE_MAX_T = 16
_KB = 16            # keys per block


@triton.jit
def _fused_attn_partial_kernel(q_ptr, part_ml_ptr, part_acc_ptr, scale, swa_ptr, widx_ptr, ckv_ptr, cidx_ptr,
                               x_ptr, xidx_ptr,
                               H: tl.constexpr, D: tl.constexpr, NW: tl.constexpr, NC: tl.constexpr,
                               NX: tl.constexpr, KB: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    blk = tl.program_id(1)
    NSPLIT: tl.constexpr = (NW + NC + NX) // KB
    HALF: tl.constexpr = D // 2
    PACKED: tl.constexpr = D // 2 + D // 16
    i = tl.arange(0, HALF)
    kk = tl.arange(0, KB)
    if blk * KB < NW:                                        # window block: fp16 rows
        row = tl.load(widx_ptr + t * NW + blk * KB + kk)
        valid = row >= 0
        r = tl.where(valid, row, 0).to(tl.int64)
        k_e = tl.load(swa_ptr + r[:, None] * D + 2 * i[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
        k_o = tl.load(swa_ptr + r[:, None] * D + 2 * i[None, :] + 1, mask=valid[:, None], other=0.0).to(tl.float32)
    elif blk * KB < NW + NC:                                 # compressed block: packed FP4 rows
        row = tl.load(cidx_ptr + t * NC + (blk * KB - NW) + kk)
        valid = row >= 0
        r = tl.where(valid, row, 0).to(tl.int64)
        byte = tl.load(ckv_ptr + r[:, None] * PACKED + i[None, :], mask=valid[:, None], other=0).to(tl.int32)
        sb = tl.load(ckv_ptr + r[:, None] * PACKED + HALF + (i[None, :] // 8), mask=valid[:, None], other=0)
        sc = qat.e4m3_value(sb.to(tl.int32))
        k_e = qat.e2m1_value(byte & 0xF) * sc
        k_o = qat.e2m1_value((byte >> 4) & 0xF) * sc
    else:                                                    # extra fp16 rows (e.g. the DSpark block K/V)
        row = tl.load(xidx_ptr + t * NX + (blk * KB - NW - NC) + kk)
        valid = row >= 0
        r = tl.where(valid, row, 0).to(tl.int64)
        k_e = tl.load(x_ptr + r[:, None] * D + 2 * i[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
        k_o = tl.load(x_ptr + r[:, None] * D + 2 * i[None, :] + 1, mask=valid[:, None], other=0.0).to(tl.float32)
    for h in tl.static_range(H):
        qbase = q_ptr + (t * H + h) * D
        q_e = tl.load(qbase + 2 * i).to(tl.float32) * scale
        q_o = tl.load(qbase + 2 * i + 1).to(tl.float32) * scale
        s = tl.sum(k_e * q_e[None, :], axis=1) + tl.sum(k_o * q_o[None, :], axis=1)      # [KB]
        s = tl.where(valid, s, float("-inf"))
        m = tl.maximum(tl.max(s, axis=0), -1e30)             # RUNNING_MAX_FLOOR: an all-invalid block gives p = 0
        p = tl.exp(s - m)
        l = tl.sum(p, axis=0)
        pidx = (t * NSPLIT + blk) * H + h
        tl.store(part_ml_ptr + pidx * 2, m)
        tl.store(part_ml_ptr + pidx * 2 + 1, l)
        tl.store(part_acc_ptr + pidx * D + i, tl.sum(p[:, None] * k_e, axis=0))
        tl.store(part_acc_ptr + pidx * D + HALF + i, tl.sum(p[:, None] * k_o, axis=0))


@triton.jit
def _fused_attn_merge_kernel(part_ml_ptr, part_acc_ptr, sink_ptr, out_ptr, H: tl.constexpr, D: tl.constexpr,
                             NSPLIT: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1)
    HALF: tl.constexpr = D // 2
    i = tl.arange(0, HALF)
    m = tl.full((), -1e30, tl.float32)
    for s in tl.static_range(NSPLIT):
        m = tl.maximum(m, tl.load(part_ml_ptr + ((t * NSPLIT + s) * H + h) * 2))
    l = tl.zeros((), dtype=tl.float32)
    acc_e = tl.zeros((HALF,), dtype=tl.float32)
    acc_o = tl.zeros((HALF,), dtype=tl.float32)
    for s in tl.static_range(NSPLIT):
        pidx = (t * NSPLIT + s) * H + h
        alpha = tl.exp(tl.load(part_ml_ptr + pidx * 2) - m)
        l += tl.load(part_ml_ptr + pidx * 2 + 1) * alpha
        acc_e += tl.load(part_acc_ptr + pidx * D + i) * alpha
        acc_o += tl.load(part_acc_ptr + pidx * D + HALF + i) * alpha
    l = l + tl.exp(tl.load(sink_ptr + h) - m)
    obase = out_ptr + (t * H + h) * D
    tl.store(obase + 2 * i, (acc_e / l).to(tl.float16))
    tl.store(obase + 2 * i + 1, (acc_o / l).to(tl.float16))


_WS: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}


def _workspace(device: torch.device, T: int, nsplit: int, H: int, D: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-shape partial buffers, allocated once (attention runs outside the CUDA graphs: every allocation is a launch)."""
    key = (str(device), T, nsplit, H, D)
    ws = _WS.get(key)
    if ws is None:
        ws = (torch.empty((T, nsplit, H, 2), dtype=torch.float32, device=device),
              torch.empty((T, nsplit, H, D), dtype=torch.float32, device=device))
        _WS[key] = ws
    return ws


def fused_decode_attention(q: torch.Tensor, sources: list[tuple[torch.Tensor, torch.Tensor]], sink: torch.Tensor,
                           scale: float, out: torch.Tensor) -> None:
    """Decode-step sparse attention in two launches. ``sources`` = [(swa_rows fp16 [*, D], win_idx [T, 128]),
    optionally (ckv_rows packed uint8 [*, 288], ckv_idx [T, 512])]; idx -1 = invalid."""
    T, H, D = q.shape
    (swa_rows, widx), *rest = sources
    if swa_rows.dtype != torch.float16 or swa_rows.shape[-1] != D:
        raise ValueError("fused_decode_attention: window rows must be fp16 [*, D]")
    widx = widx.contiguous()
    NW = widx.shape[1]
    ckv_rows, cidx, NC = swa_rows, widx, 0           # unused unless given: no block of theirs is ever selected
    x_rows, xidx, NX = swa_rows, widx, 0
    for rows, idx in rest:                            # at most one packed (compressed) and one extra fp16 source
        if rows.dtype == torch.uint8:
            if rows.shape[-1] != qat.CKV_PACKED_DIM or D != 512 or NC:
                raise ValueError("fused_decode_attention: one packed uint8 [*, 288] source with D = 512")
            ckv_rows, cidx, NC = rows, idx.contiguous(), idx.shape[1]
        elif rows.dtype == torch.float16 and rows.shape[-1] == D and not NX:
            x_rows, xidx, NX = rows, idx.contiguous(), idx.shape[1]
        else:
            raise ValueError(f"fused_decode_attention: unsupported source {rows.dtype} {tuple(rows.shape)}")
    if NW % _KB or NC % _KB or NX % _KB:
        raise ValueError(f"fused_decode_attention: key counts must be multiples of {_KB}")
    if widx.dtype != torch.int64 or cidx.dtype != torch.int64 or xidx.dtype != torch.int64:
        raise TypeError("fused_decode_attention: index tensors must be int64 (as built by the metadata)")
    nsplit = (NW + NC + NX) // _KB
    part_ml, part_acc = _workspace(q.device, T, nsplit, H, D)
    _fused_attn_partial_kernel[(T, nsplit)](
        q.contiguous(), part_ml, part_acc, float(scale), swa_rows, widx, ckv_rows, cidx, x_rows, xidx,
        H=H, D=D, NW=NW, NC=NC, NX=NX, KB=_KB, num_warps=4)
    _fused_attn_merge_kernel[(T, H)](part_ml, part_acc, sink.float().contiguous(), out, H=H, D=D, NSPLIT=nsplit,
                                     num_warps=2)
