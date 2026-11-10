# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import runpy
from pathlib import Path

import numpy as np

pack_iq2_xxs_records = runpy.run_path(
    str(
        Path(__file__).parents[3]
        / "vllm/model_executor/layers/quantization/gguf_iq2_xxs_records.py"
    )
)["pack_iq2_xxs_records"]


def test_all_split_k_starts_reconstruct_original_decoder_fields():
    rng = np.random.default_rng(1290)
    source = rng.integers(0, 256, (64, 20, 66), dtype=np.uint8)
    d = rng.uniform(-1, 1, (64, 20)).astype("<f2")
    source[..., :2] = d.view(np.uint8).reshape(64, 20, 2)
    records = pack_iq2_xxs_records(source.reshape(64, 1320))
    for tile in range(2):
        for col in range(32):
            for first_part in range(0, 40, 5):
                macro = tile * 20 * 32 * 66
                cursor = macro + first_part * 1024 + col * 16
                d_cursor = macro + 20 * 2048 + (first_part // 2) * 64 + col * 2
                half_block = first_part & 1
                cached_d = None
                for part in range(first_part, first_part + 5):
                    if part == first_part or half_block == 0:
                        cached_d = records[d_cursor : d_cursor + 2].view("<f2")[0]
                    words = np.concatenate(
                        (
                            records[cursor : cursor + 16].view("<u4"),
                            records[cursor + 512 : cursor + 528].view("<u4"),
                        )
                    )
                    block = source[tile * 32 + col, part // 2]
                    assert cached_d == block[:2].view("<f2")[0]
                    for segment in range(8):
                        for fragment in range(2):
                            octet = 2 * segment + fragment
                            group = octet // 4
                            indices, aux = (
                                int(words[2 * group]),
                                int(words[2 * group + 1]),
                            )
                            offset = 2 + half_block * 32 + group * 8
                            expected_indices = int.from_bytes(
                                block[offset : offset + 4], "little"
                            )
                            expected_aux = int.from_bytes(
                                block[offset + 4 : offset + 8], "little"
                            )
                            assert (indices >> (8 * (octet & 3))) & 255 == (
                                expected_indices >> (8 * (octet & 3))
                            ) & 255
                            assert (aux >> (7 * (octet & 3))) & 127 == (
                                expected_aux >> (7 * (octet & 3))
                            ) & 127
                            assert aux >> 28 == expected_aux >> 28
                    cursor += 1024
                    d_cursor += half_block * 64
                    half_block ^= 1
