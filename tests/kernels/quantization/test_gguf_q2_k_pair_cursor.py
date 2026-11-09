# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import runpy
from pathlib import Path

import numpy as np

pack_q2_k_records = runpy.run_path(
    str(
        Path(__file__).parents[3]
        / "vllm/model_executor/layers/quantization/gguf_q2_k_records.py"
    )
)["pack_q2_k_records"]


def test_all_split_k_starts_reconstruct_original_affine_fields():
    rng = np.random.default_rng(1290)
    source = rng.integers(0, 256, (64, 20, 84), dtype=np.uint8)
    d = rng.uniform(-1, 1, (64, 20, 2)).astype("<f2")
    source[..., 80:84] = d.view(np.uint8).reshape(64, 20, 4)
    records = pack_q2_k_records(source.reshape(64, 1680))
    for tile in range(2):
        for col in range(32):
            for first_part in range(0, 40, 5):
                macro = tile * 20 * 32 * 84
                cursor = macro + first_part * 1280 + col * 16
                scale_cursor = macro + first_part * 1280 + 1024 + col * 8
                d_cursor = macro + 20 * 2560 + (first_part // 2) * 128 + col * 4
                half_block = first_part & 1
                cached = None
                for part in range(first_part, first_part + 5):
                    if part == first_part or half_block == 0:
                        cached = records[d_cursor : d_cursor + 4].view("<f2").copy()
                    bits = np.concatenate(
                        (
                            records[cursor : cursor + 16],
                            records[cursor + 512 : cursor + 528],
                        )
                    )
                    scales = records[scale_cursor : scale_cursor + 8]
                    block = source[tile * 32 + col, part // 2]
                    np.testing.assert_array_equal(cached, block[80:84].view("<f2"))
                    for segment in range(8):
                        scale = int(scales[segment])
                        expected_scale = int(block[half_block * 8 + segment])
                        assert scale & 15 == expected_scale & 15
                        assert scale >> 4 == expected_scale >> 4
                        for fragment in range(2):
                            for i in range(8):
                                offset = (segment & 1) * 16 + fragment * 8 + i
                                q = (int(bits[offset]) >> (2 * (segment // 2))) & 3
                                base = (
                                    half_block * 128 + segment * 16 + fragment * 8 + i
                                )
                                expected = (
                                    int(block[16 + (base // 128) * 32 + base % 32])
                                    >> (2 * ((base % 128) // 32))
                                ) & 3
                                assert q == expected
                    cursor += 1280
                    scale_cursor += 1280
                    d_cursor += half_block * 128
                    half_block ^= 1
