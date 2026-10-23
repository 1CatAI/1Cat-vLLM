# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import runpy
from pathlib import Path

import numpy as np
import pytest

pack_q4_k_records = runpy.run_path(
    str(
        Path(__file__).parents[3]
        / "vllm/model_executor/layers/quantization/gguf_q4_k_records.py"
    )
)["pack_q4_k_records"]


def unpack_independently(records, n, blocks):
    original = np.empty((n, blocks * 144), dtype=np.uint8)
    for row in range(n):
        macro = records[(row // 32) * blocks * 4608 : (row // 32 + 1) * blocks * 4608]
        column = row % 32
        for block in range(blocks):
            start = blocks * 4096 + block * 512 + column * 16
            original[row, block * 144 : block * 144 + 16] = macro[start : start + 16]
            for half in range(2):
                for packet in range(4):
                    start = (block * 2 + half) * 2048 + packet * 512 + column * 16
                    target = block * 144 + 16 + half * 64 + packet * 16
                    original[row, target : target + 16] = macro[start : start + 16]
    return original


@pytest.mark.parametrize("n,blocks", [(32, 1), (64, 3), (96, 20)])
def test_original_fields_and_interleaved_packets(n, blocks):
    raw = np.random.default_rng(161).integers(0, 256, (n, blocks * 144), np.uint8)
    before = raw.copy()
    records = pack_q4_k_records(raw)
    assert records.flags.c_contiguous and records.nbytes == raw.nbytes
    np.testing.assert_array_equal(unpack_independently(records, n, blocks), raw)
    np.testing.assert_array_equal(raw, before)


def test_strided_source_rows_preserve_every_byte():
    raw = np.random.default_rng(162).integers(0, 256, (64, 288), np.uint8)[::2]
    np.testing.assert_array_equal(
        unpack_independently(pack_q4_k_records(raw), 32, 2), raw
    )


@pytest.mark.parametrize("shape", [(0, 144), (31, 144), (32, 0), (32, 143), (32,)])
def test_reject_partial_blocks_or_output_tiles(shape):
    with pytest.raises(ValueError):
        pack_q4_k_records(np.empty(shape, dtype=np.uint8))


def test_reject_already_converted_storage():
    with pytest.raises(TypeError):
        pack_q4_k_records(np.empty((32, 144), dtype=np.float16))
