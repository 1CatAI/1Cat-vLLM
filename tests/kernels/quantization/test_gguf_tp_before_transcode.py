# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import fields

import numpy as np
import pytest

from vllm.model_executor.layers.quantization.gguf_lattice_transcode import (
    LATTICE_TYPES,
    transcode_lattice,
)
from vllm.model_executor.layers.quantization.gguf_lut_transcode import (
    LUT4_TYPES,
    transcode_lut4,
)
from vllm.model_executor.layers.quantization.gguf_transcode import (
    tp_slice_packed,
    transcode_affine,
)
from vllm.transformers_utils.gguf_tensor_reader import quant_size

TYPES = (
    2,
    3,
    6,
    7,
    8,
    10,
    11,
    12,
    13,
    14,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    29,
    34,
    35,
    39,
    40,
    41,
    42,
)


def packed_source(kind, n=128, k=1024):
    block, size = quant_size(kind)
    raw = np.random.default_rng(kind).integers(
        0, 256, (n, k // block, size), dtype=np.uint8
    )
    d = np.array([0.00390625], dtype="<f2")
    offset = 80 if kind == 10 else size - 2 if kind in (11, 14, 34, 35) else 0
    if kind == 39:
        raw[..., :1] = 127
    elif kind == 40:
        raw[..., :4] = 0x38
    elif kind == 29:
        words = raw[..., 48:].copy().view("<u2")
        words &= 0x0FFF
        words |= ((d.view("<u2")[0] >> (4 * np.arange(4, dtype=np.uint16))) & 15) << 12
        raw[..., 48:] = words.view(np.uint8)
    else:
        raw[..., offset : offset + 2] = d.view(np.uint8)
        if kind in (3, 7, 12, 13):
            raw[..., 2:4] = d.view(np.uint8)
        elif kind == 10:
            raw[..., 82:84] = d.view(np.uint8)
    return raw.reshape(n, -1)


def convert(source, kind):
    if kind in LATTICE_TYPES:
        return transcode_lattice(source, kind)
    if kind in LUT4_TYPES:
        return transcode_lut4(source, kind)
    return transcode_affine(source, kind)


@pytest.mark.parametrize("kind", TYPES)
@pytest.mark.parametrize("axis", [0, 1])
def test_tp_before_transcode_preserves_every_canonical_field(kind, axis):
    source = packed_source(kind)
    canonical = convert(source, kind)
    for rank in range(4):
        reference = canonical.tp_slice(rank, 4, axis=axis)
        raw = tp_slice_packed(source, kind, rank, 4, axis=axis)
        assert raw is not None
        candidate = convert(raw, kind)
        for field in fields(reference):
            expected, actual = (
                getattr(reference, field.name),
                getattr(candidate, field.name),
            )
            if isinstance(expected, np.ndarray):
                assert actual.dtype == expected.dtype
                np.testing.assert_array_equal(actual, expected)
            else:
                assert actual == expected
        if kind in LATTICE_TYPES:
            for actual, expected in zip(
                candidate.mma884_storage(), reference.mma884_storage()
            ):
                np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("kind", [18, 22, 23, 42])
def test_unaligned_original_blocks_retain_canonical_tp(kind):
    # IQ K=768 -> 192 and Q2_0 K=640 -> 160 cut source blocks, while
    # the expanded scale groups remain legal TP4 boundaries.
    source = packed_source(kind, k=640 if kind == 42 else 768)
    assert tp_slice_packed(source, kind, 0, 4, axis=1) is None
    candidate = convert(source, kind).tp_slice(0, 4, axis=1)
    assert candidate.dequantize().shape[1] == (160 if kind == 42 else 192)


@pytest.mark.parametrize("rank,size,axis", [(4, 4, 0), (0, 0, 0), (0, 4, 2)])
def test_invalid_tp_boundary_is_rejected(rank, size, axis):
    with pytest.raises(ValueError, match="Invalid GGUF TP"):
        tp_slice_packed(packed_source(18), 18, rank, size, axis=axis)
