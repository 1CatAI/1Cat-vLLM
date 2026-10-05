# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import pytest

from vllm.model_executor.layers.quantization.gguf_dp4a_formats import (
    transcode_integer_dot,
)
from vllm.transformers_utils.gguf_tensor_reader import dequantize, quant_size


@pytest.mark.parametrize("kind", [12, 14, 23])
@pytest.mark.parametrize("k", [1536, 2560])
def test_integer_dot_preserves_official_fp32_weights_and_packed_codes(kind, k):
    n = 32
    _, size = quant_size(kind)
    rng = np.random.default_rng(kind + k)
    blocks = rng.integers(0, 256, (n, k // 256, size), dtype=np.uint8)
    d = rng.uniform(0.001, 0.02, blocks.shape[:2]).astype("<f2")
    start = 208 if kind == 14 else 0
    blocks[:, :, start : start + 2] = d[..., None].view(np.uint8)
    if kind == 12:
        blocks[:, :, 2:4] = (d * np.float16(0.5))[..., None].view(np.uint8)
    raw = blocks.reshape(n, -1)
    codec = transcode_integer_dot(raw, kind)
    np.testing.assert_array_equal(codec.dequantize(), dequantize(raw, kind))
    packed, original_d, small, dmin, mins = codec.packed()
    packets = packed.transpose(0, 2, 1).copy().reshape(n, -1)
    if kind == 14:
        decoded = packets.view(np.int8).reshape(n, k)
    else:
        decoded = (
            (
                packets[..., None].astype(np.uint32)
                >> (4 * np.arange(8, dtype=np.uint32))
            )
            & 15
        ).reshape(n, k)
    np.testing.assert_array_equal(decoded, codec.codes)
    np.testing.assert_array_equal(
        original_d.T, codec.d.repeat(256 // codec.group_size, axis=1)
    )
    np.testing.assert_array_equal(small.T, codec.small_scales)
    assert original_d.dtype == np.float16 and small.dtype == np.int8
    if kind == 12:
        np.testing.assert_array_equal(dmin.T, codec.dmin.repeat(8, axis=1))
        np.testing.assert_array_equal(mins.T, codec.small_mins)
    else:
        assert not dmin.size and not mins.size
