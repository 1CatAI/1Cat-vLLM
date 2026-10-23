# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import runpy
from pathlib import Path

import numpy as np

pack_iq3_xxs_records = runpy.run_path(
    str(
        Path(__file__).parents[3]
        / "vllm/model_executor/layers/quantization/gguf_iq3_xxs_records.py"
    )
)["pack_iq3_xxs_records"]


def test_all_split_k_starts_reconstruct_original_decoder_fields():
    rng = np.random.default_rng(1290)
    source = rng.integers(0, 256, (64, 20, 98), dtype=np.uint8)
    # Include exact finite original d values without interpreting other fields.
    d = rng.uniform(-1, 1, (64, 20)).astype("<f2")
    source[..., :2] = d.view(np.uint8).reshape(64, 20, 2)
    records = pack_iq3_xxs_records(source.reshape(64, 1960))
    for tile in range(2):
        for col in range(32):
            for first_part in range(0, 40, 5):
                macro = tile * 20 * 32 * 98
                cursor = macro + first_part * 1536 + col * 16
                d_cursor = macro + 20 * 3072 + (first_part // 2) * 64 + col * 2
                half_block = first_part & 1
                cached_d = None
                for part in range(first_part, first_part + 5):
                    if part == first_part or half_block == 0:
                        cached_d = records[d_cursor : d_cursor + 2].view("<f2")[0]
                    packets = [
                        records[cursor + offset : cursor + offset + 16].view("<u4")
                        for offset in (0, 512, 1024)
                    ]
                    indices = np.concatenate(packets[:2])
                    aux_words = packets[2]
                    block = source[tile * 32 + col, part // 2]
                    assert cached_d == block[:2].view("<f2")[0]
                    for segment in range(8):
                        for fragment in range(2):
                            octet = 2 * segment + fragment
                            packed = int(indices[octet // 2]) >> ((octet & 1) * 16)
                            aux = int(aux_words[octet // 4])
                            original_octet = half_block * 16 + octet
                            assert packed & 255 == block[2 + 2 * original_octet]
                            assert (packed >> 8) & 255 == block[3 + 2 * original_octet]
                            offset = 66 + 4 * (original_octet // 4)
                            expected_aux = int.from_bytes(
                                block[offset : offset + 4], "little"
                            )
                            assert aux == expected_aux
                            sign_index = (aux >> (7 * (octet & 3))) & 127
                            expected_signs = sign_index | (
                                (sign_index.bit_count() & 1) << 7
                            )
                            assert expected_signs.bit_count() % 2 == 0
                    cursor += 1536
                    d_cursor += half_block * 64
                    half_block ^= 1
