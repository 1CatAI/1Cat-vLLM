# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only official IQ4_XS oracle; no platform or CUDA imports are needed."""

import runpy
from pathlib import Path

import gguf
import numpy as np
import pytest

_API = runpy.run_path(
    str(
        Path(__file__).resolve().parents[3]
        / "vllm/model_executor/layers/quantization/gguf_iq4_native.py"
    )
)
pack = _API["pack_iq4_xs_records"]
unpack = _API["unpack_iq4_xs_records"]
dequantize = _API["dequantize_iq4_xs_records"]


def source():
    rng = np.random.default_rng(20261005)
    blocks = rng.integers(0, 256, (64, 2, 136), dtype=np.uint8)
    # Cover signed zeros, subnormals, non-dyadic coefficients, and large d.
    ds = np.resize(
        np.array(
            [0.0, -0.0, 2**-24, -(2**-24), 0.01337, -0.01337, 65504, -65504], "<f2"
        ),
        (64, 2),
    )
    blocks[..., :2] = ds[..., None].view(np.uint8)
    codes = np.arange(64 * 2 * 8, dtype=np.uint16).reshape(64, 2, 8) % 64
    hi = np.sum(((codes >> 4) & 3) << (2 * np.arange(8)), axis=-1, dtype=np.uint16)
    blocks[..., 2:4] = hi[..., None].astype("<u2").view(np.uint8)
    lo = ((codes[..., ::2] & 15) | ((codes[..., 1::2] & 15) << 4)).astype(np.uint8)
    blocks[..., 4:8] = lo
    return blocks.reshape(64, 272)


def test_iq4_native_official_float_and_half_bits():
    original = source()
    records = pack(original)
    assert records.nbytes == original.nbytes
    np.testing.assert_array_equal(unpack(records, 64, 512), original)
    expected = gguf.quants.dequantize(original, gguf.GGMLQuantizationType.IQ4_XS)
    for dtype, bits in ((np.float32, np.uint32), (np.float16, np.uint16)):
        with np.errstate(over="ignore"):
            actual = dequantize(records, 64, 512, dtype=dtype)
            reference = expected.astype(dtype)
        np.testing.assert_array_equal(actual.view(bits), reference.view(bits))
    # Independent output-row shards must concatenate to the original record
    # stream: each N32 tile is complete and has no cross-tile metadata.
    np.testing.assert_array_equal(
        records, np.concatenate([pack(original[:32]), pack(original[32:])])
    )


@pytest.mark.parametrize("n,width", [(31, 136), (32, 137), (32, 0), (0, 136)])
def test_iq4_native_rejects_partial_blocks(n, width):
    with pytest.raises(ValueError):
        pack(np.zeros((n, width), dtype=np.uint8))


def test_iq4_native_rejects_truncated_stream():
    records = pack(source())
    with pytest.raises(ValueError, match="bytes"):
        unpack(records[:-1], 64, 512)
    with pytest.raises(ValueError, match="Float and Half"):
        dequantize(records, 64, 512, dtype=np.float64)
