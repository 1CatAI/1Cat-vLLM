# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import runpy
from pathlib import Path

import numpy as np

pack_iq2_s_records = runpy.run_path(
    str(
        Path(__file__).parents[3]
        / "vllm/model_executor/layers/quantization/gguf_iq2_s_records.py"
    )
)["pack_iq2_s_records"]


def test_all_split_k_starts_reconstruct_original_decoder_fields():
    rng = np.random.default_rng(1290)
    source = rng.integers(0, 256, (64, 20, 82), dtype=np.uint8)
    d = rng.uniform(-1, 1, (64, 20)).astype("<f2")
    source[..., :2] = d.view(np.uint8).reshape(64, 20, 2)
    records = pack_iq2_s_records(source.reshape(64, 1640))
    for tile in range(2):
        for col in range(32):
            for first_part in range(0, 40, 5):
                macro = tile * 20 * 32 * 82
                cursor = macro + first_part * 1280
                d_cursor = macro + 20 * 2560 + (first_part // 2) * 64 + col * 2
                half_block = first_part & 1
                cached_d = None
                for part in range(first_part, first_part + 5):
                    if part == first_part or half_block == 0:
                        cached_d = records[d_cursor : d_cursor + 2].view("<f2")[0]
                    indices = records[cursor + col * 16 : cursor + col * 16 + 16]
                    signs = records[cursor + 512 + col * 16 : cursor + 528 + col * 16]
                    metadata = records[
                        cursor + 1024 + col * 8 : cursor + 1032 + col * 8
                    ]
                    high = int.from_bytes(metadata[:4], "little")
                    scales = int.from_bytes(metadata[4:], "little")
                    block = source[tile * 32 + col, part // 2]
                    assert cached_d == block[:2].view("<f2")[0]
                    for segment in range(8):
                        for fragment in range(2):
                            octet = 2 * segment + fragment
                            base = half_block * 128 + segment * 16 + fragment * 8
                            index = int(indices[octet]) | (
                                ((high >> (2 * octet)) & 3) << 8
                            )
                            original = int(block[2 + base // 8]) | (
                                (
                                    (
                                        int(block[66 + base // 32])
                                        >> (2 * ((base // 8) % 4))
                                    )
                                    & 3
                                )
                                << 8
                            )
                            assert index == original
                            assert int(signs[octet]) == int(block[34 + base // 8])
                            assert (scales >> (4 * segment)) & 15 == (
                                int(block[74 + base // 32]) >> (4 * ((base // 16) % 2))
                            ) & 15
                    cursor += 1280
                    d_cursor += half_block * 64
                    half_block ^= 1
