# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest

from vllm.model_executor.layers.quantization.gguf_transcode import (
    pack_affine_high_plane,
    reconstruction_error,
    transcode_affine,
)
from vllm.transformers_utils.gguf_tensor_reader import dequantize, quant_size


def source(weight_type, scale):
    block, size = quant_size(weight_type)
    n, k = 32, 768
    raw = np.random.default_rng(20261003 + weight_type).integers(
        0, 256, (n * k // block, size), dtype=np.uint8
    )
    d = np.frombuffer(np.float16(scale).tobytes(), np.uint8)
    if weight_type in (11, 14):
        raw[:, -2:] = d
    else:
        raw[:, :2] = d
        if weight_type in (7, 13):
            raw[:, 2:4] = np.frombuffer(np.float16(-scale * 2).tobytes(), np.uint8)
    return raw.reshape(n, -1)


@pytest.mark.parametrize("weight_type", [6, 7, 11, 13, 14])
@pytest.mark.parametrize("scale", [0.0009765625, 0.001337])
def test_bitplane_codes_and_coefficients_match_reference(weight_type, scale):
    data = source(weight_type, scale)
    canonical = transcode_affine(data, weight_type)
    expected = dequantize(data, weight_type)
    error = reconstruction_error(canonical, expected)
    if scale == 0.0009765625 or weight_type in (6, 7):
        np.testing.assert_array_equal(canonical.dequantize(), expected)
    else:
        assert error["relative_l2"] < 0.001
    low_bits = 2 if canonical.bits == 3 else 4
    high_bits = canonical.bits - low_bits
    words = pack_affine_high_plane(canonical)
    high = (
        words[..., None]
        >> (np.arange(canonical.group_size, dtype=np.uint32) * high_bits)
    ) & ((1 << high_bits) - 1)
    low = canonical.codes & ((1 << low_bits) - 1)
    reconstructed = low | (
        high.reshape(canonical.codes.shape).astype(np.uint8) << low_bits
    )
    np.testing.assert_array_equal(reconstructed, canonical.codes)
    for rank in range(4):
        shard = canonical.tp_slice(rank, 4, axis=1)
        np.testing.assert_array_equal(
            shard.dequantize(), canonical.dequantize()[:, rank * 192 : (rank + 1) * 192]
        )
