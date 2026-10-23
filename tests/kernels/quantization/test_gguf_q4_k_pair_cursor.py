# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import runpy
from pathlib import Path

import numpy as np

pack = runpy.run_path(
    str(
        Path(__file__).parents[3]
        / "vllm/model_executor/layers/quantization/gguf_q4_k_records.py"
    )
)["pack_q4_k_records"]


def test_split_k_cursor_and_register_fragments_match_original_fields():
    raw = np.random.default_rng(163).integers(0, 256, (64, 20 * 144), np.uint8)
    original = raw.reshape(64, 20, 144)
    records = pack(raw)
    for tile in range(2):
        macro = records[tile * 20 * 4608 : (tile + 1) * 20 * 4608]
        for split in range(8):
            first_part = split * 5
            for column in range(32):
                payload = first_part * 2048 + column * 16
                metadata = 20 * 4096 + (first_part // 2) * 512 + column * 16
                half_block = first_part & 1
                cached = None
                for step in range(5):
                    if cached is None or half_block == 0:
                        cached = macro[metadata : metadata + 16].copy()
                    block = original[tile * 32 + column, (first_part + step) // 2]
                    np.testing.assert_array_equal(cached, block[:16])
                    for segment in range(8):
                        group = half_block * 4 + segment // 2
                        low = cached[4 + group % 4]
                        minimum = cached[8 + group % 4]
                        upper = cached[12 + group % 4]
                        scale = (
                            (upper & 15) | ((low >> 6) << 4) if half_block else low & 63
                        )
                        offset = (
                            (upper >> 4) | ((minimum >> 6) << 4)
                            if half_block
                            else minimum & 63
                        )
                        if group < 4:
                            assert scale == block[4 + group] & 63
                            assert offset == block[8 + group] & 63
                        else:
                            assert scale == (block[8 + group] & 15) | (
                                (block[group] >> 6) << 4
                            )
                            assert offset == (block[8 + group] >> 4) | (
                                (block[4 + group] >> 6) << 4
                            )
                        for fragment in range(2):
                            packet = (segment // 4) * 2 + (segment & 1)
                            start = payload + packet * 512 + fragment * 8
                            q = macro[start : start + 8] >> ((segment // 2 & 1) * 4)
                            original_start = (
                                16
                                + half_block * 64
                                + (segment // 4) * 32
                                + (segment & 1) * 16
                                + fragment * 8
                            )
                            expected = block[original_start : original_start + 8]
                            expected = expected >> ((segment // 2 & 1) * 4)
                            np.testing.assert_array_equal(q & 15, expected & 15)
                    payload += 2048
                    metadata += half_block * 512
                    half_block ^= 1
