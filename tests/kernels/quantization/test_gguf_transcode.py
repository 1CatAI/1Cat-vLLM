# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gguf
import numpy as np
import pytest

from vllm.model_executor.layers.quantization.gguf_transcode import (
    reconstruction_error,
    transcode_affine_group32,
)


def packed(weight_type, rows=16, k=640):
    block, size = gguf.GGML_QUANT_SIZES[weight_type]
    k = 768 if weight_type == 12 else k
    random = np.random.default_rng(20261003 + weight_type)
    data = random.integers(0, 256, (rows * k // block, size), dtype=np.uint8)
    data[:, :2] = np.frombuffer(np.float16(0.00390625).tobytes(), dtype=np.uint8)
    if weight_type in (3, 12):
        data[:, 2:4] = np.frombuffer(np.float16(-0.0078125).tobytes(), dtype=np.uint8)
    return data.reshape(rows, -1)


@pytest.mark.parametrize("weight_type", [2, 3, 8, 12])
def test_group32_integer_codes_match_official_dequantization(weight_type):
    source = packed(weight_type)
    canonical = transcode_affine_group32(source, weight_type)
    reference = gguf.quants.dequantize(source, gguf.GGMLQuantizationType(weight_type))
    np.testing.assert_array_equal(canonical.dequantize(), reference)
    assert reconstruction_error(canonical, reference)["max_abs"] == 0


def test_expanded_q4_k_coefficients_report_rounding_error():
    source = packed(12)
    blocks = source.reshape(-1, 144)
    blocks[:, :2] = np.frombuffer(np.float16(0.001337).tobytes(), dtype=np.uint8)
    blocks[:, 2:4] = np.frombuffer(np.float16(0.000739).tobytes(), dtype=np.uint8)
    canonical = transcode_affine_group32(source, 12)
    reference = gguf.quants.dequantize(source, gguf.GGMLQuantizationType.Q4_K)
    result = reconstruction_error(canonical, reference)
    assert 0 < result["relative_l2"] < 0.001
    assert 0 < result["max_abs"] < 0.001


def test_tp4_reblocks_q4_k_to_complete_group32_shards():
    source = packed(12)
    canonical = transcode_affine_group32(source, 12)
    full = canonical.dequantize()
    span = full.shape[1] // 4
    for rank in range(4):
        shard = canonical.tp_slice(rank, 4, axis=1)
        np.testing.assert_array_equal(
            shard.dequantize(), full[:, rank * span : (rank + 1) * span]
        )
    with pytest.raises(ValueError, match="cuts an affine group"):
        canonical.tp_slice(0, 48, axis=1)
