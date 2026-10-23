# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lossless signed-nibble records for integer-dot IQ expert experiments.

IQ codebooks correlate adjacent weights to compress them. Their decoded scalar
values still fit in a 16-entry integer lookup table. Expand only the indices
and signs at load time, keeping the original FP16 base scale and integer subscales.
"""

import numpy as np

from vllm.model_executor.layers.quantization.gguf_lattice_transcode import (
    lattice_grid,
    transcode_lattice,
)
from vllm.transformers_utils.gguf_tensor_reader import quant_size

SIGNED_LEVELS = {
    18: np.array(
        [-62, -52, -44, -36, -28, -20, -12, -4, 4, 12, 20, 28, 36, 44, 52, 62],
        dtype=np.int8,
    ),
    21: np.arange(-15, 16, 2, dtype=np.int8),
    22: np.array([-43, -25, -8, 8, 25, 43], dtype=np.int8),
}


def pack_lattice_lut(data: np.ndarray, source_type: int) -> np.ndarray:
    """Return [N,K/32,20] with 16 code bytes and four metadata bytes.

    Metadata preserves the base FP16 scale and one or two integer odd
    subscales. No floating coefficient rounding or reconstructed weight
    storage is introduced.
    """
    if source_type not in SIGNED_LEVELS:
        raise ValueError("Signed-nibble codec supports IQ3_XXS, IQ3_S and IQ2_S")
    canonical = transcode_lattice(data, source_type)
    n, k = canonical.shape
    values = lattice_grid(source_type)[canonical.indices].reshape(n, k)
    signs = (canonical.signs[..., None] >> np.arange(8, dtype=np.uint8)) & 1
    values = (values * (1 - 2 * signs.astype(np.int8)).reshape(n, k)).astype(np.int16)
    levels = SIGNED_LEVELS[source_type]
    indices = np.searchsorted(levels, values)
    if np.any(indices >= len(levels)) or not np.array_equal(levels[indices], values):
        raise ValueError("Source codebook contains a value outside the scalar LUT")
    codes = indices.astype(np.uint8).reshape(n, k // 32, 32)
    _, size = quant_size(source_type)
    blocks = np.ascontiguousarray(data).reshape(n, k // 256, size)
    base = blocks[..., :2]
    if not np.isfinite(base.copy().view("<f2")).all():
        raise ValueError("Source coefficients must be finite")
    if source_type == 18:
        word = blocks[..., 66:98].copy().view("<u4")
        odd = (1 + 2 * (word >> 28)).astype(np.uint8)[..., None]
    else:
        local = blocks[..., 74:82] if source_type == 22 else blocks[..., 106:110]
        nibble = (local[..., None] >> np.array([0, 4], np.uint8)) & 15
        odd = (1 + 2 * nibble.reshape(n, k // 256, 8, -1)).astype(np.uint8)
    records = np.zeros((n, k // 256, 8, 20), dtype=np.uint8)
    records[..., :16] = (codes[..., ::2] | (codes[..., 1::2] << 4)).reshape(
        n, k // 256, 8, 16
    )
    records[..., 16:18] = base[..., None, :]
    records[..., 18 : 18 + odd.shape[-1]] = odd
    return records.reshape(n, k // 32, 20)
