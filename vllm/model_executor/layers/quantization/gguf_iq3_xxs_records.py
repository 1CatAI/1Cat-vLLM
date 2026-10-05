# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Source-sized IQ3_XXS records for aligned, shared-activation pair readers."""

import numpy as np


def pack_iq3_xxs_records(raw: np.ndarray) -> np.ndarray:
    """Permute original blocks; preserve every index, sign, scale and d bit.

    An N32 tile stores K128 records with two 16-byte index packets and one
    original 16-byte sign/scale packet per column. Each packet plane
    interleaves columns at 16-byte boundaries. The original d plane follows
    the payload, retaining one d per K256 block, including odd K128 starts.
    This is storage preparation only; it does not admit a model kernel.
    """
    if raw.dtype != np.uint8:
        raise TypeError("IQ3_XXS records require original uint8 blocks")
    if (
        raw.ndim != 2
        or raw.shape[0] == 0
        or raw.shape[0] % 32
        or raw.shape[1] == 0
        or raw.shape[1] % 98
    ):
        raise ValueError("IQ3_XXS records require N32 and complete 98-byte blocks")
    n, row_bytes = raw.shape
    tiles, blocks = n // 32, row_bytes // 98
    source = raw.reshape(tiles, 32, blocks, 98).transpose(0, 2, 1, 3)
    indices = (
        source[..., 2:66]
        .reshape(tiles, blocks, 32, 2, 2, 16)
        .transpose(0, 1, 3, 4, 2, 5)
    )
    sign_scale = (
        source[..., 66:98]
        .reshape(tiles, blocks, 32, 2, 16)
        .transpose(0, 1, 3, 2, 4)[:, :, :, None, :, :]
    )
    payload = np.concatenate((indices, sign_scale), axis=3).reshape(
        tiles, blocks * 3072
    )
    original_d = source[..., :2].reshape(tiles, blocks * 64)
    records = np.concatenate((payload, original_d), axis=1).reshape(-1)
    assert records.nbytes == raw.nbytes
    return records
