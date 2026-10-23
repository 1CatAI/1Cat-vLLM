# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Source-sized IQ2_S records for shared-activation gated-pair readers."""

import numpy as np


def pack_iq2_s_records(raw: np.ndarray) -> np.ndarray:
    """Interleave original K128 index, sign and high-bit/scale packets.

    Each column retains two 16-byte planes and one 8-byte metadata packet
    per K128. Original d follows the payload, once per K256. Every source
    bit is preserved; scales and codebook indices are never expanded.
    """
    if raw.dtype != np.uint8:
        raise TypeError("IQ2_S records require original uint8 blocks")
    if (
        raw.ndim != 2
        or raw.shape[0] == 0
        or raw.shape[0] % 32
        or raw.shape[1] == 0
        or raw.shape[1] % 82
    ):
        raise ValueError("IQ2_S records require N32 and complete 82-byte blocks")
    n, row_bytes = raw.shape
    tiles, blocks = n // 32, row_bytes // 82
    source = raw.reshape(tiles, 32, blocks, 82).transpose(0, 2, 1, 3)
    indices = (
        source[..., 2:34].reshape(tiles, blocks, 32, 2, 16).transpose(0, 1, 3, 2, 4)
    )
    signs = (
        source[..., 34:66].reshape(tiles, blocks, 32, 2, 16).transpose(0, 1, 3, 2, 4)
    )
    high = source[..., 66:74].reshape(tiles, blocks, 32, 2, 4).transpose(0, 1, 3, 2, 4)
    scales = (
        source[..., 74:82].reshape(tiles, blocks, 32, 2, 4).transpose(0, 1, 3, 2, 4)
    )
    payload = np.concatenate(
        (
            indices.reshape(tiles, blocks, 2, 512),
            signs.reshape(tiles, blocks, 2, 512),
            np.concatenate((high, scales), axis=4).reshape(tiles, blocks, 2, 256),
        ),
        axis=3,
    ).reshape(tiles, blocks * 2560)
    original_d = source[..., :2].reshape(tiles, blocks * 64)
    records = np.concatenate((payload, original_d), axis=1).reshape(-1)
    assert records.nbytes == raw.nbytes
    return records
