# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""q RoPE + window-KV RoPE + FP8 QAT + FP16 paged insert (PORT_DESIGN §1 "Attention q path" / "Window KV" rows).

Reference (ref:m.py:700-720, 770-772): ``q = wq_b(q_norm(wq_a x))`` with RoPE on the last 64 dims of every
head -- **no per-head q-norm** (the V4 fused kernel hard-wires one); ``kv = kv_norm(wkv x)``, RoPE on its
last 64 dims, then ``act_quant(kv, 32, "ue8m0", e8m0, inplace=True)`` over all 512 dims (RoPE tail
included). The port keeps the normed, rotated window KV in FP32 up to the QAT (``v100-semantic``) and
stores the exact QAT values as FP16 rows of the SWA cache at ``slot_mapping`` (-1 = no write).
RoPE: interleaved pairs, FP32 math, cos/sin from the layer's ``cos_sin_cache`` ([pos, cos(32)|sin(32)]).
"""

from __future__ import annotations

import torch

from vllm.models.deepseek_v41.common import qat
from vllm.models.deepseek_v41.common.rope import apply_rope_torch
from vllm.triton_utils import tl, triton


@triton.jit
def _q_rope_kv_insert_kernel(
    q_ptr, q_stride_t, q_stride_h,
    kv_ptr, kv_stride_t,
    pos_ptr, cs_ptr,
    rows_ptr, slot_ptr,
    H: tl.constexpr, D: tl.constexpr, RD: tl.constexpr,
):
    t = tl.program_id(0)
    HALF: tl.constexpr = RD // 2
    NOPE: tl.constexpr = D - RD
    pos = tl.load(pos_ptr + t).to(tl.int64)
    j = tl.arange(0, HALF)
    cos = tl.load(cs_ptr + pos * RD + j)
    sin = tl.load(cs_ptr + pos * RD + HALF + j)

    # q: rotate the last RD dims of every head in place (FP32 math, FP16 storage)
    h = tl.arange(0, H)
    qbase = q_ptr + t.to(tl.int64) * q_stride_t + h[:, None] * q_stride_h + NOPE + 2 * j[None, :]
    qe = tl.load(qbase).to(tl.float32)
    qo = tl.load(qbase + 1).to(tl.float32)
    tl.store(qbase, (qe * cos[None, :] - qo * sin[None, :]).to(q_ptr.dtype.element_ty))
    tl.store(qbase + 1, (qe * sin[None, :] + qo * cos[None, :]).to(q_ptr.dtype.element_ty))

    # window KV: rotate the tail, QAT over 32-blocks of all D dims, FP16 row store
    slot = tl.load(slot_ptr + t)
    if slot >= 0:
        i = tl.arange(0, D // 2)
        ke = tl.load(kv_ptr + t.to(tl.int64) * kv_stride_t + 2 * i).to(tl.float32)
        ko = tl.load(kv_ptr + t.to(tl.int64) * kv_stride_t + 2 * i + 1).to(tl.float32)
        jj = i - NOPE // 2
        rot = jj >= 0
        c = tl.load(cs_ptr + pos * RD + jj, mask=rot, other=1.0)
        s = tl.load(cs_ptr + pos * RD + HALF + jj, mask=rot, other=0.0)
        re = tl.where(rot, ke * c - ko * s, ke)
        ro = tl.where(rot, ke * s + ko * c, ko)
        x = tl.reshape(tl.join(re, ro), (1, D // 32, 32))
        y = tl.reshape(qat.fp8_block32_qdq_tile(x, 32), (D,))
        d = tl.arange(0, D)
        tl.store(rows_ptr + slot * D + d, y.to(rows_ptr.dtype.element_ty))


def q_rope_kv_insert(q: torch.Tensor, kv: torch.Tensor, positions: torch.Tensor, cos_sin_cache: torch.Tensor,
                     swa_rows: torch.Tensor, slot_mapping: torch.Tensor, impl: str = "sm70") -> None:
    """q [T, H, 512] fp16 (rotated in place); kv [T, 512] fp32/fp16 (normed, un-rotated);
    positions [T] int64; swa_rows [num_rows, 512] fp16 (the SWA cache viewed as rows);
    slot_mapping [T] int64 (-1: no write). Only the first ``slot_mapping.shape[0]`` tokens are processed."""
    T = slot_mapping.shape[0]
    if T == 0:
        return
    H, D = q.shape[1], q.shape[2]
    rd = cos_sin_cache.shape[-1]
    if kv.shape[-1] != D or swa_rows.shape[-1] != D or q.stride(-1) != 1 or kv.stride(-1) != 1:
        raise ValueError(f"q_rope_kv_insert: bad shapes q {tuple(q.shape)} kv {tuple(kv.shape)} "
                         f"rows {tuple(swa_rows.shape)}")
    if impl == "torch":
        pos = positions[:T]
        q[:T] = apply_rope_torch(q[:T], pos, cos_sin_cache)
        rec = qat.fp8_block32_qdq(apply_rope_torch(kv[:T].float(), pos, cos_sin_cache), out_dtype=swa_rows.dtype,
                                  impl="torch")
        keep = slot_mapping >= 0
        swa_rows.index_copy_(0, slot_mapping[keep], rec[keep])
        return
    if impl != "sm70":
        raise ValueError(f"unknown impl {impl!r}")
    if H & (H - 1):
        raise ValueError(f"q_rope_kv_insert sm70 path needs a power-of-two head count, got {H}")
    _q_rope_kv_insert_kernel[(T,)](
        q, q.stride(0), q.stride(1), kv, kv.stride(0), positions, cos_sin_cache, swa_rows, slot_mapping,
        H=H, D=D, RD=rd, num_warps=4)
