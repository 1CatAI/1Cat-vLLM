# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gguf
import numpy as np
import pytest

from vllm.model_executor.layers.quantization.gguf_lut_transcode import transcode_lut4
from vllm.transformers_utils.gguf_tensor_reader import quant_size


def source(weight_type, *, n=3, k=768, scale=0.015625, seed=20261003, nv_max=0x7E):
    block, size = quant_size(weight_type)
    rng = np.random.default_rng(seed)
    data = rng.integers(0, 256, (n, k // block, size), dtype=np.uint8)
    if weight_type in (20, 23):
        data[..., :2] = np.array([scale], dtype="<f2").view(np.uint8)
    elif weight_type == 39:
        data[..., :1] = rng.integers(115, 133, (*data.shape[:2], 1), dtype=np.uint8)
    else:
        data[..., :4] = rng.choice(
            np.array([0, 1, 0x38, nv_max, 0x7F], np.uint8), size=(*data.shape[:2], 4)
        )
    return data.reshape(n, -1)


@pytest.mark.parametrize("weight_type", [20, 23, 39, 40])
@pytest.mark.parametrize("scale", [0.015625, 0.01337])
def test_lut4_official_reconstruction(weight_type, scale):
    data = source(weight_type, scale=scale)
    canonical = transcode_lut4(data, weight_type)
    expected = gguf.quants.dequantize(data, gguf.GGMLQuantizationType(weight_type))
    actual = canonical.dequantize()
    if weight_type == 23 and scale != 0.015625:
        assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < 0.001
    else:
        np.testing.assert_array_equal(actual, expected)
    assert canonical.codes.dtype == np.uint8
    assert canonical.codes.max() < 16
    # K=768 -> K=192 cuts IQ4_XS superblocks but retains LUT4 groups.
    parts = [canonical.tp_slice(rank, 4, axis=1).dequantize() for rank in range(4)]
    np.testing.assert_array_equal(np.concatenate(parts, axis=1), actual)


def test_lut4_tp_rejects_group_cut():
    canonical = transcode_lut4(source(20, k=96), 20)
    with pytest.raises(ValueError, match="cuts a LUT4 group"):
        canonical.tp_slice(0, 4, axis=1)


@pytest.mark.parametrize("exponent,reason", [(255, "overflows"), (90, "underflows")])
def test_mxfp4_unrepresentable_scale_is_rejected(exponent, reason):
    data = source(39, n=1, k=32)
    data[0, 0] = exponent
    with pytest.raises(ValueError, match=reason):
        transcode_lut4(data, 39)
