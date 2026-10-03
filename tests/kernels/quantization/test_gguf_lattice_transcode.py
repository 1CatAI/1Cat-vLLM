# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gguf
import numpy as np
import pytest

from vllm.model_executor.layers.quantization.gguf_lattice_transcode import (
    lattice_grid,
    transcode_lattice,
)
from vllm.transformers_utils.gguf_tensor_reader import quant_size


def source(weight_type, scale, n=3, k=768):
    block, size = quant_size(weight_type)
    data = np.random.default_rng(20261003 + weight_type).integers(
        0, 256, (n * k // block, size), dtype=np.uint8
    )
    half = np.array([scale], dtype="<f2")
    if weight_type == 29:
        bits = half.view("<u2")[0]
        words = data[:, 48:].copy().view("<u2")
        words &= 0x0FFF
        words |= ((bits >> (4 * np.arange(4, dtype=np.uint16))) & 15) << 12
        data[:, 48:] = words.view(np.uint8)
    else:
        data[:, :2] = half.view(np.uint8)
    return data.reshape(n, -1)


@pytest.mark.parametrize("weight_type", [16, 17, 18, 19, 21, 22, 29])
@pytest.mark.parametrize("scale", [0.0009765625, 0.00137])
def test_lattice_official_reconstruction_and_tp(weight_type, scale):
    data = source(weight_type, scale)
    p = transcode_lattice(data, weight_type)
    expected = gguf.quants.dequantize(data, gguf.GGMLQuantizationType(weight_type))
    actual = p.dequantize()
    if scale == 0.0009765625:
        np.testing.assert_array_equal(actual, expected)
    else:
        assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < 0.001
    assert p.indices.dtype == np.uint16
    assert p.indices.max() < lattice_grid(weight_type).shape[0]
    assert p.signs.dtype == np.uint8 and p.deltas.dtype == np.uint8
    parts = [p.tp_slice(rank, 4, axis=1).dequantize() for rank in range(4)]
    np.testing.assert_array_equal(np.concatenate(parts, axis=1), actual)


def test_lattice_group_cut_is_rejected():
    p = transcode_lattice(source(18, 0.0009765625, k=256), 18)
    with pytest.raises(ValueError, match="cuts a lattice group"):
        p.tp_slice(0, 16, axis=1)
