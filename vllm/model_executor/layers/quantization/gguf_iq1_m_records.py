# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Source-sized IQ1_M records retaining its distributed original scale."""

import numpy as np


def pack_iq1_m_records(raw: np.ndarray) -> np.ndarray:
    """Interleave K128 indices/high bits; keep all original scale words."""
    if raw.dtype != np.uint8:
        raise TypeError("IQ1_M records require original uint8 blocks")
    if (
        raw.ndim != 2
        or raw.shape[0] == 0
        or raw.shape[0] % 32
        or raw.shape[1] == 0
        or raw.shape[1] % 56
    ):
        raise ValueError("IQ1_M records require N32 and complete 56-byte blocks")
    n, row_bytes = raw.shape
    tiles, blocks = n // 32, row_bytes // 56
    source = raw.reshape(tiles, 32, blocks, 56).transpose(0, 2, 1, 3)
    indices = (
        source[..., :32].reshape(tiles, blocks, 32, 2, 16).transpose(0, 1, 3, 2, 4)
    )
    high = source[..., 32:48].reshape(tiles, blocks, 32, 2, 8).transpose(0, 1, 3, 2, 4)
    payload = np.concatenate(
        (indices.reshape(tiles, blocks, 2, 512), high.reshape(tiles, blocks, 2, 256)),
        axis=3,
    )
    scales = source[..., 48:56].reshape(tiles, blocks * 256)
    records = np.concatenate(
        (payload.reshape(tiles, blocks * 1536), scales), axis=1
    ).reshape(-1)
    assert records.nbytes == raw.nbytes
    return records
