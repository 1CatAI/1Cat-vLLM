# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Source-sized Q2_K records for aligned shared-activation readers."""

import numpy as np


def pack_q2_k_records(raw: np.ndarray) -> np.ndarray:
    """Keep original quant bits, scale/min nibbles and d/dmin unchanged."""
    if raw.dtype != np.uint8:
        raise TypeError("Q2_K records require original uint8 blocks")
    if (
        raw.ndim != 2
        or raw.shape[0] == 0
        or raw.shape[0] % 32
        or raw.shape[1] == 0
        or raw.shape[1] % 84
    ):
        raise ValueError("Q2_K records require N32 and complete 84-byte blocks")
    n, row_bytes = raw.shape
    tiles, blocks = n // 32, row_bytes // 84
    source = raw.reshape(tiles, 32, blocks, 84).transpose(0, 2, 1, 3)
    bits = (
        source[..., 16:80]
        .reshape(tiles, blocks, 32, 2, 2, 16)
        .transpose(0, 1, 3, 4, 2, 5)
        .reshape(tiles, blocks, 2, 1024)
    )
    scales = (
        source[..., :16]
        .reshape(tiles, blocks, 32, 2, 8)
        .transpose(0, 1, 3, 2, 4)
        .reshape(tiles, blocks, 2, 256)
    )
    payload = np.concatenate((bits, scales), axis=3).reshape(tiles, blocks * 2560)
    original_d = source[..., 80:84].reshape(tiles, blocks * 128)
    records = np.concatenate((payload, original_d), axis=1).reshape(-1)
    assert records.nbytes == raw.nbytes
    return records
