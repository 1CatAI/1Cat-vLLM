# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lossless IQ4_XS records retaining original nested scales and nibble codes."""

import gguf
import numpy as np


def _shape(n: int, k: int):
    if n <= 0 or n % 32 or k <= 0 or k % 256:
        raise ValueError("IQ4_XS records require complete N32 and K256 tiles")
    return n // 32, k // 256


def pack_iq4_xs_records(data: np.ndarray) -> np.ndarray:
    """Permute original bits into aligned N32/K32 packets without expansion."""
    if data.dtype != np.uint8 or data.ndim != 2 or data.shape[1] % 136:
        raise ValueError("IQ4_XS needs uint8 packed rows with complete 136B blocks")
    n, row_bytes = data.shape
    nt, nb = _shape(n, row_bytes // 136 * 256)
    blocks = data.reshape(nt, 32, nb, 136).transpose(0, 2, 1, 3)
    qs = blocks[..., 8:].reshape(nt, nb, 32, 8, 16)
    codes = np.concatenate((qs & np.uint8(15), qs >> np.uint8(4)), axis=-1)
    packets = (codes[..., ::2] | (codes[..., 1::2] << np.uint8(4))).transpose(
        0, 1, 3, 2, 4
    )
    parts = [packets.reshape(nt, nb * 4096)]
    for start, stop in ((0, 2), (2, 4), (4, 8)):
        parts.append(blocks[..., start:stop].reshape(nt, nb * 32 * (stop - start)))
    return np.ascontiguousarray(np.concatenate(parts, axis=1).reshape(-1))


def _planes(records: np.ndarray, n: int, k: int):
    nt, nb = _shape(n, k)
    if records.dtype != np.uint8 or records.ndim != 1:
        raise ValueError("IQ4_XS records require a flat uint8 array")
    if records.size != n * nb * 136:
        raise ValueError("IQ4_XS record bytes differ from the original payload")
    tiles = records.reshape(nt, nb * 4352)
    payload = tiles[:, : nb * 4096].reshape(nt, nb, 8, 32, 16)
    d = tiles[:, nb * 4096 : nb * 4160].reshape(nt, nb, 32, 2)
    hi = tiles[:, nb * 4160 : nb * 4224].reshape(nt, nb, 32, 2)
    lo = tiles[:, nb * 4224 :].reshape(nt, nb, 32, 4)
    return payload, d, hi, lo


def unpack_iq4_xs_records(records: np.ndarray, n: int, k: int) -> np.ndarray:
    """Independent inverse recovering every original byte, including metadata."""
    payload, d, hi, lo = _planes(records, n, k)
    nt, nb = n // 32, k // 256
    codes = np.empty((*payload.shape[:-1], 32), dtype=np.uint8)
    codes[..., ::2] = payload & np.uint8(15)
    codes[..., 1::2] = payload >> np.uint8(4)
    original_qs = codes[..., :16] | (codes[..., 16:] << np.uint8(4))
    blocks = np.empty((nt, nb, 32, 136), dtype=np.uint8)
    blocks[..., :2] = d
    blocks[..., 2:4] = hi
    blocks[..., 4:8] = lo
    blocks[..., 8:] = original_qs.transpose(0, 1, 3, 2, 4).reshape(nt, nb, 32, 128)
    return np.ascontiguousarray(blocks.transpose(0, 2, 1, 3).reshape(n, nb * 136))


def dequantize_iq4_xs_records(
    records: np.ndarray, n: int, k: int, *, dtype=np.float32
) -> np.ndarray:
    """Decode both scale levels in FP32 before the optional final FP16 rounding."""
    if np.dtype(dtype) not in (np.dtype(np.float32), np.dtype(np.float16)):
        raise ValueError("IQ4_XS reader supports Float and Half operands")
    payload, original_d, original_hi, original_lo = _planes(records, n, k)
    d = original_d.copy().view("<f2").astype(np.float32).squeeze(-1)
    hi = original_hi.copy().view("<u2").squeeze(-1)
    lo = original_lo.copy().view("<u4").squeeze(-1)
    groups = np.arange(8, dtype=np.uint32)
    scale_codes = ((lo[..., None] >> (4 * groups)) & 15) | (
        ((hi[..., None].astype(np.uint32) >> (2 * groups)) & 3) << 4
    )
    small = scale_codes.astype(np.int32) - 32
    scales = (d[..., None] * small.astype(np.float32)).transpose(0, 1, 3, 2)
    codes = np.empty((*payload.shape[:-1], 32), dtype=np.uint8)
    codes[..., ::2] = payload & np.uint8(15)
    codes[..., 1::2] = payload >> np.uint8(4)
    # Reuse the existing official IQ4 LUT rather than maintaining another table.
    lut = np.asarray(gguf.quants.IQ4_NL.kvalues, dtype=np.float32)
    weights = scales[..., None] * lut[codes]
    return np.ascontiguousarray(
        weights.transpose(0, 3, 1, 2, 4).reshape(n, k).astype(dtype)
    )
