# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest

from vllm.model_executor.layers.quantization.gguf_transcode import (
    reconstruction_error,
    transcode_affine,
)
from vllm.transformers_utils.gguf_tensor_reader import dequantize, quant_size


def source_blocks(weight_type, rows=32, k=768, scale=0.00390625):
    if weight_type in (41, 42):
        k = 640
    block, size = quant_size(weight_type)
    data = np.random.default_rng(20261003 + weight_type).integers(
        0,
        256,
        (rows * k // block, size),
        dtype=np.uint8,
    )
    offset = 80 if weight_type == 10 else (0 if weight_type in (41, 42) else size - 2)
    data[:, offset : offset + 2] = np.frombuffer(np.float16(scale).tobytes(), np.uint8)
    if weight_type == 10:
        data[:, 82:84] = np.frombuffer(np.float16(-0.0078125).tobytes(), np.uint8)
    return data.reshape(rows, -1)


def oracle(data, weight_type):
    if weight_type != 41:
        return dequantize(data, weight_type)
    # The installed gguf-py has the Q1 enum but no dequantizer. This follows
    # llama.cpp dequantize_row_q1_0: least-significant bit first in each byte.
    blocks = data.reshape(-1, 18)
    d = blocks[:, :2].copy().view("<f2").astype(np.float32)
    values = (
        np.unpackbits(blocks[:, 2:], axis=1, bitorder="little").astype(np.float32) * 2
        - 1
    )
    return (values * d).reshape(data.shape[0], -1)


@pytest.mark.parametrize("weight_type", [10, 34, 35, 41, 42])
def test_u2_affine_and_ternary_match_official_reconstruction(weight_type):
    data = source_blocks(weight_type)
    canonical = transcode_affine(data, weight_type)
    assert canonical.bits == 2
    np.testing.assert_array_equal(canonical.dequantize(), oracle(data, weight_type))
    assert reconstruction_error(canonical, oracle(data, weight_type))["max_abs"] == 0
    for rank in range(4):
        shard = canonical.tp_slice(rank, 4, axis=1)
        width = canonical.codes.shape[1] // 4
        np.testing.assert_array_equal(
            shard.dequantize(),
            canonical.dequantize()[:, rank * width : (rank + 1) * width],
        )


def test_q1_scale_at_fp16_limit_does_not_overflow_canonical_coefficients():
    data = source_blocks(41, scale=65504)
    canonical = transcode_affine(data, 41)
    assert np.all(np.isfinite(canonical.scales))
    np.testing.assert_array_equal(canonical.dequantize(), oracle(data, 41))


def test_q2_k_expansion_reports_fp16_rounding():
    data = source_blocks(10, scale=0.001337)
    result = reconstruction_error(transcode_affine(data, 10), oracle(data, 10))
    assert 0 < result["relative_l2"] < 0.001
