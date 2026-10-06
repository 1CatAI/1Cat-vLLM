# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Based on the supplied pack_iq2s.py; codebooks originate in gguf-py (MIT).
"""dense_mv / grouped-MoE planes for GGUF IQ2_S (type 22) expert rows.

Per lane and K128 group (four 32-weight steps):
  codes: 2 x uint4 = grid-index low bytes and sign words per step
  meta : uint4 = (index high bits, d16, scale nibbles, 0)
Weight = fp16(d * (1 + 2 * nibble) / 8) * grid_value, sign applied as an exact bit
flip. Grid table: 1024 entries x 8 bytes (two uint32 words per index).
"""

import gguf.quants as Q
import numpy as np

IQ2S = 7
ROWMAP = np.array(
    [((L >> 2) & 3) * 8 + (L & 3) + (4 if L & 16 else 0) for L in range(32)]
)


def iq2s_table():
    Q.IQ2_S.init_grid()
    g = np.asarray(Q.IQ2_S.grid).reshape(-1, 8).astype(np.uint8)
    return np.ascontiguousarray(g).view(np.uint32).reshape(-1).view(np.uint8)


def pack_iq2s(raw):
    N = raw.shape[0]
    b = raw.reshape(N, -1, 82)
    nb = b.shape[1]
    G, T = nb * 2, N // 32
    assert N % 32 == 0
    d = (
        b[..., 0:2].copy().view(np.uint16)[..., 0].astype(np.uint32)
    )  # (N, nb) fp16 bits
    lo = b[..., 2:34].reshape(N, nb, 8, 4)  # 8 steps x 4 index bytes
    sg = b[..., 34:66].reshape(
        N, nb, 8, 4
    )  # 8 steps x 4 sign bytes (one per 8 weights)
    qh = b[..., 66:74].reshape(N, nb, 8).astype(np.uint32)
    sc = b[..., 74:82].reshape(N, nb, 8).astype(np.uint32)
    lo_w = np.ascontiguousarray(lo).view(np.uint32).reshape(N, G, 4)  # word per step
    sg_w = np.ascontiguousarray(sg).view(np.uint32).reshape(N, G, 4)
    qh = qh.reshape(N, G, 4)
    hiw = np.zeros((N, G), np.uint32)
    for s in range(4):
        hiw |= (qh[..., s] & 0xFF) << (8 * s)  # 4 indices x 2 bits per step
    scw = np.zeros((N, G), np.uint32)
    scs = sc.reshape(N, G, 4)
    for s in range(4):
        scw |= (scs[..., s] & 0xFF) << (8 * s)
    d16 = np.repeat(d, 2, axis=1)  # (N, G)
    lane = lambda a: a.reshape((T, 32) + a.shape[1:])[:, ROWMAP]
    codes = np.empty((N, G, 2, 4), np.uint32)
    codes[:, :, 0] = lo_w
    codes[:, :, 1] = sg_w
    codes = lane(codes).transpose(0, 2, 3, 1, 4)  # (T, G, 2, L, 4)
    meta = np.stack([hiw, d16, scw, np.zeros_like(hiw)], -1)
    meta = lane(meta).transpose(0, 2, 1, 3)  # (T, G, L, 4)
    f = lambda a: np.ascontiguousarray(a).astype(np.uint32).view(np.uint8).reshape(-1)
    return IQ2S, f(codes), f(meta)


BLOCK_BYTES = {18: 98, 21: 110, 22: 82}


def expert_planes(raw, gtype):
    """Pack [N, payload] GGUF rows of an IQ3_XXS/IQ3_S/IQ2_S expert projection."""
    if gtype == 22:
        return pack_iq2s(raw)
    from vllm.model_executor.layers.quantization.gguf_dmv_formats import pack

    return pack(raw, gtype)


def expert_table(gtype):
    if gtype == 22:
        return iq2s_table()
    from vllm.model_executor.layers.quantization.gguf_dmv_formats import tables

    return tables()
