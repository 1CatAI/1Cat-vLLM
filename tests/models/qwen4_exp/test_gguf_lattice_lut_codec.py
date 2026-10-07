# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gguf
import numpy as np
import pytest

from vllm.model_executor.layers.quantization.gguf_lattice_lut import (
    SIGNED_LEVELS,
    pack_lattice_lut,
)
from vllm.transformers_utils.gguf_tensor_reader import quant_size


@pytest.mark.parametrize("kind", [18, 21, 22])
def test_signed_nibbles_match_official_dequantization(kind):
    _, size = quant_size(kind)
    rng = np.random.default_rng(2048 + kind)
    raw = rng.integers(0, 256, (9, 5, size), dtype=np.uint8)
    d = rng.uniform(0.001, 0.04, (9, 5, 1)).astype("<f2")
    raw[..., :2] = d.view(np.uint8)
    source = raw.reshape(9, -1)
    untouched = source.copy()
    records = pack_lattice_lut(source, kind)
    packed = records[..., :16]
    indices = np.stack((packed & 15, packed >> 4), axis=-1).reshape(9, 40, 32)
    values = SIGNED_LEVELS[kind][indices].astype(np.float32)
    base = records[..., 16:18].copy().view("<f2").astype(np.float32)
    odd = records[..., 18:20].astype(np.float32)
    if kind == 22:
        coefficients = base * np.repeat(odd, 16, axis=-1) * 0.125
    else:
        coefficients = base * odd[..., :1] * (0.25 if kind == 18 else 1)
    restored = (values * coefficients).reshape(9, 1280)
    reference = gguf.quants.dequantize(source, gguf.GGMLQuantizationType(kind))
    np.testing.assert_array_equal(restored, reference)
    np.testing.assert_array_equal(source, untouched)
