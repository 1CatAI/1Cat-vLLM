# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Based on the supplied pack.py for dense_mv2/shexp2.
"""Lossless GGUF code extraction and packing for dense_mv.cu.

Codes stay exact integers. Per-group coefficients are rounded to FP16 the same
way as the canonical TurboMind path (scale*code+min evaluated in FP16).
"""

import numpy as np

Q4K, Q5K, Q6K, LUT4, Q8 = 0, 1, 2, 3, 4
# GGUF type ids
T_Q8_0, T_Q4_K, T_Q5_K, T_Q6_K, T_IQ4_NL, T_IQ4_XS = 8, 12, 13, 14, 20, 23
KV = np.array(
    [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113],
    dtype=np.float32,
)


def _f16(b):
    return np.ascontiguousarray(b).view(np.float16).astype(np.float32)


def _k_scales(sc12):
    # sc12: (..., 12) uint8 -> sc, mn (..., 8)
    sc = np.empty(sc12.shape[:-1] + (8,), np.uint8)
    mn = np.empty_like(sc)
    for j in range(4):
        sc[..., j] = sc12[..., j] & 63
        mn[..., j] = sc12[..., j + 4] & 63
    for j in range(4, 8):
        sc[..., j] = (sc12[..., j + 4] & 0xF) | ((sc12[..., j - 4] >> 6) << 4)
        mn[..., j] = (sc12[..., j + 4] >> 4) | ((sc12[..., j] >> 6) << 4)
    return sc.astype(np.float32), mn.astype(np.float32)


def decode(raw, gtype):
    """Extract integer codes and coefficients from GGUF rows."""
    N = raw.shape[0]
    if gtype in (T_Q4_K, T_Q5_K):
        B = 144 if gtype == T_Q4_K else 176
        b = raw.reshape(N, -1, B)
        nb = b.shape[1]
        d = _f16(b[..., 0:2])[..., 0]
        dmin = _f16(b[..., 2:4])[..., 0]
        sc, mn = _k_scales(b[..., 4:16])
        if gtype == T_Q4_K:
            qs = b[..., 16:144]
            qh = None
        else:
            qh = b[..., 16:48]
            qs = b[..., 48:176]
        q = np.empty((N, nb, 256), np.uint8)
        for i in range(4):
            c = qs[..., 32 * i : 32 * i + 32]
            lo = c & 0xF
            hi = c >> 4
            if qh is not None:
                lo = lo | (((qh >> (2 * i)) & 1) << 4)
                hi = hi | (((qh >> (2 * i + 1)) & 1) << 4)
            q[..., 64 * i : 64 * i + 32] = lo
            q[..., 64 * i + 32 : 64 * i + 64] = hi
        s = d[..., None] * sc
        m = -dmin[..., None] * mn
        return (
            Q4K if gtype == T_Q4_K else Q5K,
            q.reshape(N, -1),
            s.reshape(N, -1),
            m.reshape(N, -1),
            32,
        )
    if gtype == T_Q6_K:
        b = raw.reshape(N, -1, 210)
        nb = b.shape[1]
        ql, qh = b[..., 0:128], b[..., 128:192]
        sc = b[..., 192:208].view(np.int8).astype(np.float32)
        d = _f16(b[..., 208:210])[..., 0]
        q = np.empty((N, nb, 256), np.uint8)
        for n in range(2):
            low = ql[..., 64 * n : 64 * n + 64]
            h = qh[..., 32 * n : 32 * n + 32]
            o = 128 * n
            q[..., o : o + 32] = (low[..., :32] & 0xF) | (((h >> 0) & 3) << 4)
            q[..., o + 32 : o + 64] = (low[..., 32:] & 0xF) | (((h >> 2) & 3) << 4)
            q[..., o + 64 : o + 96] = (low[..., :32] >> 4) | (((h >> 4) & 3) << 4)
            q[..., o + 96 : o + 128] = (low[..., 32:] >> 4) | (((h >> 6) & 3) << 4)
        s = d[..., None] * sc
        return Q6K, q.reshape(N, -1), s.reshape(N, -1), None, 16
    if gtype == T_IQ4_XS:
        b = raw.reshape(N, -1, 136)
        nb = b.shape[1]
        d = _f16(b[..., 0:2])[..., 0]
        sh = np.ascontiguousarray(b[..., 2:4]).view(np.uint16)[..., 0].astype(np.int32)
        sl = b[..., 4:8].astype(np.int32)
        qs = b[..., 8:136]
        q = np.empty((N, nb, 256), np.uint8)
        s = np.empty((N, nb, 8), np.float32)
        for ib in range(8):
            ls = ((sl[..., ib // 2] >> (4 * (ib % 2))) & 0xF) | (
                ((sh >> (2 * ib)) & 3) << 4
            )
            s[..., ib] = d * (ls - 32)
            c = qs[..., 16 * ib : 16 * ib + 16]
            q[..., 32 * ib : 32 * ib + 16] = c & 0xF
            q[..., 32 * ib + 16 : 32 * ib + 32] = c >> 4
        return LUT4, q.reshape(N, -1), s.reshape(N, -1), None, 32
    if gtype == T_IQ4_NL:
        b = raw.reshape(N, -1, 18)
        d = _f16(b[..., 0:2])[..., 0]
        qs = b[..., 2:18]
        q = np.concatenate([qs & 0xF, qs >> 4], axis=-1)
        return LUT4, q.reshape(N, -1), d.reshape(N, -1), None, 32
    if gtype == T_Q8_0:
        b = raw.reshape(N, -1, 34)
        d = _f16(b[..., 0:2])[..., 0]
        q = b[..., 2:34].copy()  # int8 bit patterns
        return Q8, q.reshape(N, -1), d.reshape(N, -1), None, 32
    raise ValueError(gtype)


def reconstruct(fmt, q, s, m, gs):
    """FP16 weights exactly as the kernel builds them."""
    s16 = s.astype(np.float16)
    if fmt in (Q4K, Q5K):
        # fp16 fma: compute in fp32 then round once
        w = (
            q.astype(np.float32).reshape(q.shape[0], -1, gs)
            * s16.astype(np.float32)[..., None]
            + m.astype(np.float16).astype(np.float32)[..., None]
        ).astype(np.float16)
    elif fmt == Q6K:
        w = (
            (q.astype(np.float32) - 32).reshape(q.shape[0], -1, gs)
            * s16.astype(np.float32)[..., None]
        ).astype(np.float16)
    elif fmt == LUT4:
        w = (
            KV[q].reshape(q.shape[0], -1, gs) * s16.astype(np.float32)[..., None]
        ).astype(np.float16)
    else:
        w = (
            q.view(np.int8).astype(np.float32).reshape(q.shape[0], -1, gs)
            * s16.astype(np.float32)[..., None]
        ).astype(np.float16)
    return w.reshape(q.shape[0], -1)


ROWMAP = np.array(
    [((L >> 2) & 3) * 8 + (L & 3) + (4 if L & 16 else 0) for L in range(32)]
)


def pack(fmt, q, s, m, gs):
    """Return (codes, high, scale) uint8 arrays laid out for dense_mv."""
    N, K = q.shape
    T = (N + 31) // 32
    S = (K + 31) // 32
    G = (S + 3) // 4
    Sp = G * 4
    qp = np.zeros((T * 32, Sp * 32), np.uint8)
    qp[:N, :K] = q
    # (T, 32 lanes, Sp, 32 k)
    Q = qp.reshape(T, 32, Sp, 32)[:, ROWMAP].astype(np.uint32)
    lo = Q & 0xF
    sp = np.zeros((T * 32, Sp * 32 // gs), np.float32)
    sp[:N, : K // gs] = s
    S16 = sp.astype(np.float16).view(np.uint16).astype(np.uint32)
    S16 = S16.reshape(T, 32, Sp, 32 // gs)[:, ROWMAP]
    if m is not None:
        mp = np.zeros_like(sp)
        mp[:N, : K // gs] = m
        M16 = mp.astype(np.float16).view(np.uint16).astype(np.uint32)
        M16 = M16.reshape(T, 32, Sp, 1)[:, ROWMAP]
    high = np.zeros(0, np.uint8)
    if fmt in (Q4K, Q5K, Q6K):
        regs = np.zeros((T, 32, Sp, 4), np.uint32)
        for c in range(4):
            for j in range(4):
                regs[..., c] |= lo[..., 8 * c + 2 * j] << (4 * j)
                regs[..., c] |= lo[..., 8 * c + 2 * j + 1] << (4 * j + 16)
        codes = regs.transpose(0, 2, 1, 3)  # (T, Sp, L, 4)
    elif fmt == LUT4:
        regs = np.zeros((T, 32, Sp, 4), np.uint32)
        for c in range(4):
            for b in range(4):
                byte = Q[..., 8 * c + 2 * b] | (Q[..., 8 * c + 2 * b + 1] << 4)
                regs[..., c] |= byte << (8 * b)
        codes = regs.transpose(0, 2, 1, 3)
    else:  # Q8: 8 regs per step -> (T, Sp, 2, L, 4)
        regs = np.zeros((T, 32, Sp, 8), np.uint32)
        for c in range(8):
            for b in range(4):
                regs[..., c] |= Q[..., 4 * c + b] << (8 * b)
        codes = regs.reshape(T, 32, Sp, 2, 4).transpose(0, 2, 3, 1, 4)
    if fmt == Q5K:
        hb = (Q >> 4) & 1
        H = np.zeros((T, 32, Sp), np.uint32)
        for c in range(4):
            for j in range(4):
                H |= hb[..., 8 * c + 2 * j] << (4 * c + j)
                H |= hb[..., 8 * c + 2 * j + 1] << (16 + 4 * c + j)
        high = H.reshape(T, 32, G, 4).transpose(0, 2, 1, 3)  # (T, G, L, 4 steps)
    elif fmt == Q6K:
        hb = (Q >> 4) & 3
        H = np.zeros((T, 32, Sp, 2), np.uint32)
        for c in range(4):
            for j in range(4):
                p = 2 * (4 * (c & 1) + j)
                H[..., c >> 1] |= hb[..., 8 * c + 2 * j] << p
                H[..., c >> 1] |= hb[..., 8 * c + 2 * j + 1] << (16 + p)
        # (T, L, G, 2 halves(steps 0-1 / 2-3), 2 steps, 2 words) -> (T, G, j, L, 4)
        high = H.reshape(T, 32, G, 2, 4).transpose(0, 2, 3, 1, 4)
    if fmt in (Q4K, Q5K):
        sc = S16[..., 0] | (M16[..., 0] << 16)
    elif fmt == Q6K:
        sc = S16[..., 0] | (S16[..., 1] << 16)
    else:
        sc = S16[..., 0] | (S16[..., 0] << 16)
    scale = sc.reshape(T, 32, G, 4).transpose(0, 2, 1, 3)  # (T, G, L, 4)
    f = lambda a: np.ascontiguousarray(a).astype(np.uint32).view(np.uint8).reshape(-1)
    return f(codes), (f(high) if high.size else high), f(scale)


def pack_device(fmt, q, s, m, gs, device):
    """Device twin of ``pack``: identical bytes, returned as device uint8 tensors.

    ``q``/``s``/``m`` are the same NumPy inputs ``pack`` takes; the integer
    register assembly runs on ``device`` in int64 and is emitted as the same
    little-endian uint32 words. FP16 conversion uses the same round-to-nearest.
    """
    import torch

    q = torch.from_numpy(np.ascontiguousarray(q)).to(device)
    N, K = q.shape
    T = (N + 31) // 32
    S = (K + 31) // 32
    G = (S + 3) // 4
    Sp = G * 4
    rowmap = torch.from_numpy(ROWMAP).to(device)
    qp = torch.zeros((T * 32, Sp * 32), dtype=torch.uint8, device=device)
    qp[:N, :K] = q
    Q = qp.view(T, 32, Sp, 32)[:, rowmap].long()
    lo = Q & 0xF

    def half_bits(values, cols):
        padded = torch.zeros((T * 32, cols), dtype=torch.float32, device=device)
        padded[:N, : values.shape[1]] = torch.from_numpy(
            np.ascontiguousarray(values, dtype=np.float32)
        ).to(device)
        return padded.half().view(torch.int16).long() & 0xFFFF

    S16 = half_bits(s, Sp * 32 // gs).view(T, 32, Sp, 32 // gs)[:, rowmap]
    if m is not None:
        M16 = half_bits(m, Sp * 32 // gs).view(T, 32, Sp, 1)[:, rowmap]
    high = None
    if fmt in (Q4K, Q5K, Q6K):
        regs = torch.zeros((T, 32, Sp, 4), dtype=torch.long, device=device)
        for c in range(4):
            for j in range(4):
                regs[..., c] |= lo[..., 8 * c + 2 * j] << (4 * j)
                regs[..., c] |= lo[..., 8 * c + 2 * j + 1] << (4 * j + 16)
        codes = regs.permute(0, 2, 1, 3)
    elif fmt == LUT4:
        regs = torch.zeros((T, 32, Sp, 4), dtype=torch.long, device=device)
        for c in range(4):
            for b in range(4):
                byte = Q[..., 8 * c + 2 * b] | (Q[..., 8 * c + 2 * b + 1] << 4)
                regs[..., c] |= byte << (8 * b)
        codes = regs.permute(0, 2, 1, 3)
    else:
        regs = torch.zeros((T, 32, Sp, 8), dtype=torch.long, device=device)
        for c in range(8):
            for b in range(4):
                regs[..., c] |= Q[..., 4 * c + b] << (8 * b)
        codes = regs.view(T, 32, Sp, 2, 4).permute(0, 2, 3, 1, 4)
    if fmt == Q5K:
        hb = (Q >> 4) & 1
        H = torch.zeros((T, 32, Sp), dtype=torch.long, device=device)
        for c in range(4):
            for j in range(4):
                H |= hb[..., 8 * c + 2 * j] << (4 * c + j)
                H |= hb[..., 8 * c + 2 * j + 1] << (16 + 4 * c + j)
        high = H.view(T, 32, G, 4).permute(0, 2, 1, 3)
    elif fmt == Q6K:
        hb = (Q >> 4) & 3
        H = torch.zeros((T, 32, Sp, 2), dtype=torch.long, device=device)
        for c in range(4):
            for j in range(4):
                p = 2 * (4 * (c & 1) + j)
                H[..., c >> 1] |= hb[..., 8 * c + 2 * j] << p
                H[..., c >> 1] |= hb[..., 8 * c + 2 * j + 1] << (16 + p)
        high = H.view(T, 32, G, 2, 4).permute(0, 2, 3, 1, 4)
    if fmt in (Q4K, Q5K):
        sc = S16[..., 0] | (M16[..., 0] << 16)
    elif fmt == Q6K:
        sc = S16[..., 0] | (S16[..., 1] << 16)
    else:
        sc = S16[..., 0] | (S16[..., 0] << 16)
    scale = sc.reshape(T, 32, G, 4).permute(0, 2, 1, 3)

    def words(a):
        a = a.contiguous()
        return (
            (a - ((a >> 31) & 1) * (1 << 32))
            .to(torch.int32)
            .view(torch.uint8)
            .reshape(-1)
        )

    empty = torch.zeros(0, dtype=torch.uint8, device=device)
    return words(codes), (words(high) if high is not None else empty), words(scale)
