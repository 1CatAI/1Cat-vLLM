# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lossless storage conversions needed to admit GGUF tensor parallelism."""

import numpy as np


def q2_0_to_q4_1(data: np.ndarray) -> np.ndarray:
    """Split each Q2_0 block into two Q4_1 blocks without requantizing values.

    Q2_0 represents d * (code - 1) with four codes and one FP16 scale per
    64 values. Q4_1 represents d * code + min per 32 values. Reusing d and
    setting min=-d preserves every value, while allowing a K=160 TP shard.
    Both block formats follow llama.cpp 002a12ad (MIT); no kernel is copied.
    """
    if data.dtype != np.uint8 or data.shape[-1] % 18:
        raise ValueError("Q2_0 repack needs whole 18-byte blocks")
    blocks = data.reshape(-1, 18)
    codes = (blocks[:, 2:, None] >> np.arange(0, 8, 2, dtype=np.uint8)) & 3
    codes = codes.reshape(-1, 2, 32)
    result = np.empty((len(blocks), 2, 20), dtype=np.uint8)
    result[:, :, :2] = blocks[:, None, :2]
    result[:, :, 2:4] = blocks[:, None, :2]
    # IEEE FP16 negation changes only the sign bit, with no rounding.
    result[:, :, 3] ^= 0x80
    result[:, :, 4:] = codes[:, :, :16] | (codes[:, :, 16:] << 4)
    return result.reshape(*data.shape[:-1], data.shape[-1] // 18 * 40)
