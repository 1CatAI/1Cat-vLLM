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


IQ2_FORMATS = {16: 7, 17: 8, 22: 9}
IQ2_VALUES = np.array([-43, -25, -8, 8, 25, 43], np.int8)


def iq2_reverse_table(kind):
    """Invert the eight-value magnitude grid using a 3^8-entry table.

    This table is only read by canonical restoration, never by the M8 kernel.
    It keeps the original grid indices and sign masks exactly recoverable.
    """
    from vllm.model_executor.layers.quantization.gguf_lattice_transcode import (
        lattice_grid,
    )

    grid = lattice_grid(kind).astype(np.int8)
    digits = np.searchsorted(np.array([8, 25, 43]), grid).astype(np.int32)
    keys = (digits * (3 ** np.arange(8, dtype=np.int32))).sum(axis=1)
    if len(np.unique(keys)) != len(keys):
        raise ValueError("IQ2 grid contains duplicate magnitude tuples")
    result = np.full(3**8, 65535, np.uint16)
    result[keys] = np.arange(len(keys), dtype=np.uint16)
    return result.view(np.uint8)


def pack_iq2(raw, kind):
    """Lossless signed nibbles with original d16 and eight scale nibbles/K128.

    grid*small is exactly representable in FP16. The M8 decoder multiplies
    that product by original d and rounds only the final reconstructed weight.
    """
    from vllm.model_executor.layers.quantization.gguf_dense_hmma_formats import (
        pack as pack_planes,
    )
    from vllm.model_executor.layers.quantization.gguf_lattice_transcode import (
        lattice_grid,
        transcode_lattice,
    )
    from vllm.transformers_utils.gguf_tensor_reader import quant_size

    if kind not in IQ2_FORMATS:
        raise ValueError(kind)
    canonical = transcode_lattice(raw, kind)
    n, k = canonical.shape
    values = lattice_grid(kind)[canonical.indices].reshape(n, k)
    signs = (canonical.signs[..., None] >> np.arange(8, dtype=np.uint8)) & 1
    values = (values * (1 - 2 * signs.astype(np.int8)).reshape(n, k)).astype(np.int8)
    q = np.searchsorted(IQ2_VALUES, values).astype(np.uint8)
    if not np.array_equal(IQ2_VALUES[q], values):
        raise ValueError("IQ2 grid does not fit the signed six-value codebook")
    codes, _, _ = pack_planes(3, q, np.zeros((n, k // 32)), None, 32)
    _, size = quant_size(kind)
    blocks = np.ascontiguousarray(raw).reshape(n, -1, size)
    groups = k // 128
    ds = np.repeat(blocks[..., :2].copy().view(np.uint16).reshape(n, -1), 2, axis=1)
    if kind == 16:
        words = blocks[..., 2:].copy().view(np.uint32).reshape(n, -1, 8, 2)
        nib = np.repeat(words[..., 1] >> 28, 2, axis=2).reshape(n, groups, 8)
    else:
        tail = blocks[..., 66:] if kind == 17 else blocks[..., 74:]
        nib = (tail[..., None] >> np.array([0, 4], np.uint8)) & 15
        nib = nib.reshape(n, groups, 8).astype(np.uint32)
    packed = np.bitwise_or.reduce(nib << (4 * np.arange(8, dtype=np.uint32)), axis=-1)
    meta = np.stack([ds.astype(np.uint32), packed], axis=-1)
    meta = meta.reshape(n // 32, 32, groups, 2)[:, ROWMAP].transpose(0, 2, 1, 3)
    return (
        IQ2_FORMATS[kind],
        codes,
        np.ascontiguousarray(meta).view(np.uint8).reshape(-1),
    )
