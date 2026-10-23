# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

import gguf
import numpy as np
import regex as re


def test_shared_float_book_matches_canonical_lookup_and_official_integers():
    source = (
        Path(__file__).parents[3]
        / "csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/transform.h"
    ).read_text()
    body = source.split("static uint32_t iq_values(uint32_t nibbles)", 1)[1]
    body = body.split("\n  }", 1)[0]
    constants = [int(value, 16) for value in re.findall(r"0x([0-9A-F]+)U", body)]
    # The production helper selects four bytes from each pair of eight-byte
    # halves. Read its table constants rather than maintaining another table.
    low = constants[1:3]
    high = constants[3:5]
    table = np.array(gguf.quants.IQ4_NL.kvalues, dtype=np.float32)
    codes = np.arange(65536, dtype=np.uint32)
    for lane in range(4):
        index = (codes >> (4 * lane)) & 15
        byte = index & 7
        lo = np.where(byte < 4, low[0], low[1]).astype(np.uint32)
        hi = np.where(byte < 4, high[0], high[1]).astype(np.uint32)
        biased = np.where(index < 8, lo, hi) >> ((byte & 3) * 8)
        canonical = ((biased & 255).astype(np.int32) - 128).astype(np.float16)
        direct = table[index]
        np.testing.assert_array_equal(canonical.astype(np.float32), direct)
