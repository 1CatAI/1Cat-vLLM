# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Based on the supplied pack3.py; codebooks originate in gguf-py (MIT).
"""dense_mv planes for grid-codebook GGUF formats (IQ3_S, IQ3_XXS).

Per lane and K128 group (four 32-weight steps):
  codes: 3 x uint4 = grid-index bytes (steps 0-1, steps 2-3) and explicit sign words
  meta : IQ3_S uint2 = (four high-index bytes, d16 | scale nibbles << 16)
         IQ3_XXS uint32 = original d16 | sub-scale nibbles << 16
Weight = fp16(d * (1 + 2 * nibble)) * grid_value, sign applied as an exact bit flip.
"""

import gguf.quants as Q
import numpy as np

IQ3S, IQ3X = 5, 6
ROWMAP = np.array(
    [((L >> 2) & 3) * 8 + (L & 3) + (4 if L & 16 else 0) for L in range(32)]
)


def _grid(cls):
    cls.init_grid()
    g = np.asarray(cls.grid).reshape(-1, 4).astype(np.uint8)
    return np.ascontiguousarray(g).view(np.uint32).reshape(-1)


def tables(fmt=None):
    """Shared IQ3_S (512 words) and IQ3_XXS (256 words) grids."""
    return (
        np.concatenate([_grid(Q.IQ3_S), _grid(Q.IQ3_XXS)])
        .astype(np.uint32)
        .view(np.uint8)
    )


def _split_signs(w):
    """32 sign bits in weight order -> even weights in bits 0-15, odd in 16-31."""
    out = np.zeros_like(w)
    for p in range(16):
        out |= ((w >> (2 * p)) & 1) << p
        out |= ((w >> (2 * p + 1)) & 1) << (16 + p)
    return out


def pack(raw, gtype):
    N = raw.shape[0]
    if gtype == 21:
        b = raw.reshape(N, -1, 110)
        nb = b.shape[1]
        d = b[..., 0:2].copy().view(np.float16)[..., 0].astype(np.float32)
        idx = b[..., 2:66].reshape(N, nb, 8, 8)
        hi = b[..., 66:74].reshape(N, nb, 8)
        signs = b[..., 74:106].reshape(N, nb, 8, 4)
        sc = b[..., 106:110]
        nib = np.stack([sc & 15, sc >> 4], -1).reshape(N, nb, 8)
        fmt = IQ3S
    elif gtype == 18:
        b = raw.reshape(N, -1, 98)
        nb = b.shape[1]
        d = b[..., 0:2].copy().view(np.float16)[..., 0].astype(np.float32)
        idx = b[..., 2:66].reshape(N, nb, 8, 8)
        aux = b[..., 66:98].copy().view(np.uint32).reshape(N, nb, 8)
        ks = np.frombuffer(Q.IQ2_XXS.ksigns, dtype=np.uint8)
        s7 = (aux[..., None] >> np.array([0, 7, 14, 21], np.uint32)) & 127
        signs = ks[s7]
        nib = (aux >> 28).astype(np.uint8)
        hi = np.zeros((N, nb, 8), np.uint8)
        fmt = IQ3X
    else:
        raise ValueError(gtype)
    S, G, T = nb * 8, nb * 2, N // 32
    assert N % 32 == 0
    idxw = (
        np.ascontiguousarray(idx).reshape(N, S, 8).view(np.uint32).reshape(N, G, 4, 2)
    )
    sgw = np.ascontiguousarray(signs).reshape(N, S, 4).view(np.uint32).reshape(N, G, 4)
    d16 = np.repeat(
        d.astype(np.float16).view(np.uint16).astype(np.uint32), 2, axis=1
    )  # (N, G)
    nibg = nib.reshape(N, G, 4).astype(np.uint32)
    dsw = d16 | (
        (
            nibg[..., 0]
            | (nibg[..., 1] << 4)
            | (nibg[..., 2] << 8)
            | (nibg[..., 3] << 12)
        )
        << 16
    )
    lane = lambda a: a.reshape((T, 32) + a.shape[1:])[:, ROWMAP]
    codes = np.empty((N, G, 3, 4), np.uint32)
    codes[:, :, 0] = idxw[:, :, 0:2].reshape(N, G, 4)
    codes[:, :, 1] = idxw[:, :, 2:4].reshape(N, G, 4)
    codes[:, :, 2] = _split_signs(sgw)
    codes = lane(codes).transpose(0, 2, 3, 1, 4)  # (T, G, 3, L, 4)
    if fmt == IQ3S:
        hig = hi.reshape(N, G, 4).astype(np.uint32)
        hiw = (
            hig[..., 0] | (hig[..., 1] << 8) | (hig[..., 2] << 16) | (hig[..., 3] << 24)
        )
        meta = lane(np.stack([hiw, dsw], -1)).transpose(0, 2, 1, 3)  # (T, G, L, 2)
    else:
        meta = lane(dsw).transpose(0, 2, 1)  # (T, G, L)
    f = lambda a: np.ascontiguousarray(a).astype(np.uint32).view(np.uint8).reshape(-1)
    return fmt, f(codes), f(meta)


def compact_lut4_scale(scale):
    """Pack four duplicated half scales into two consecutive half2 words."""
    w = scale.view(np.uint32).reshape(-1, 4)
    lo = w & 0xFFFF
    out = np.empty((w.shape[0], 2), np.uint32)
    out[:, 0] = lo[:, 0] | (lo[:, 1] << 16)
    out[:, 1] = lo[:, 2] | (lo[:, 3] << 16)
    return out.view(np.uint8).reshape(-1)
