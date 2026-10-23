# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Source-sized IQ2_XS records for shared-activation gated-pair readers."""

import numpy as np


def pack_iq2_xs_records(raw: np.ndarray) -> np.ndarray:
    """Keep packed indices/signs, scales and original d without expansion."""
    if raw.dtype != np.uint8:
        raise TypeError("IQ2_XS records require original uint8 blocks")
    if (
        raw.ndim != 2
        or raw.shape[0] == 0
        or raw.shape[0] % 32
        or raw.shape[1] == 0
        or raw.shape[1] % 74
    ):
        raise ValueError("IQ2_XS records require N32 and complete 74-byte blocks")
    n, row_bytes = raw.shape
    tiles, blocks = n // 32, row_bytes // 74
    source = raw.reshape(tiles, 32, blocks, 74).transpose(0, 2, 1, 3)
    words = (
        source[..., 2:66]
        .reshape(tiles, blocks, 32, 2, 2, 16)
        .transpose(0, 1, 3, 4, 2, 5)
        .reshape(tiles, blocks, 2, 1024)
    )
    scales = (
        source[..., 66:74]
        .reshape(tiles, blocks, 32, 2, 4)
        .transpose(0, 1, 3, 2, 4)
        .reshape(tiles, blocks, 2, 128)
    )
    payload = np.concatenate((words, scales), axis=3).reshape(tiles, blocks * 2304)
    original_d = source[..., :2].reshape(tiles, blocks * 64)
    records = np.concatenate((payload, original_d), axis=1).reshape(-1)
    assert records.nbytes == raw.nbytes
    return records
