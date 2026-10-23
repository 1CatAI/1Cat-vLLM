# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Original-byte Q4_K records for aligned shared-activation readers."""

import numpy as np


def pack_q4_k_records(raw: np.ndarray) -> np.ndarray:
    """Interleave K128 packet planes across N32 without expanding metadata.

    Each column supplies four aligned 16-byte packets per K128. Original
    d, dmin and the twelve packed scale/min bytes remain together in one
    16-byte record per K256. No scale is multiplied or converted here.
    """
    if raw.dtype != np.uint8:
        raise TypeError("Q4_K records require original uint8 blocks")
    if (
        raw.ndim != 2
        or raw.shape[0] == 0
        or raw.shape[0] % 32
        or raw.shape[1] == 0
        or raw.shape[1] % 144
    ):
        raise ValueError("Q4_K records require N32 and complete 144-byte blocks")
    n, row_bytes = raw.shape
    tiles, blocks = n // 32, row_bytes // 144
    source = raw.reshape(tiles, 32, blocks, 144).transpose(0, 2, 1, 3)
    payload = (
        source[..., 16:]
        .reshape(tiles, blocks, 32, 2, 4, 16)
        .transpose(0, 1, 3, 4, 2, 5)
        .reshape(tiles, blocks * 4096)
    )
    metadata = source[..., :16].reshape(tiles, blocks * 512)
    records = np.concatenate((payload, metadata), axis=1).reshape(-1)
    assert records.nbytes == raw.nbytes
    return records
