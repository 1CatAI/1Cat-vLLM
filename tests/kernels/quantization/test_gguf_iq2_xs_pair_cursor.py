# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import runpy
from pathlib import Path

import numpy as np

pack_iq2_xs_records = runpy.run_path(
    str(
        Path(__file__).parents[3]
        / "vllm/model_executor/layers/quantization/gguf_iq2_xs_records.py"
    )
)["pack_iq2_xs_records"]


def test_all_split_k_starts_reconstruct_original_decoder_fields():
    rng = np.random.default_rng(1290)
    source = rng.integers(0, 256, (64, 20, 74), dtype=np.uint8)
    d = rng.uniform(-1, 1, (64, 20)).astype("<f2")
    source[..., :2] = d.view(np.uint8).reshape(64, 20, 2)
    records = pack_iq2_xs_records(source.reshape(64, 1480))
    for tile in range(2):
        for col in range(32):
            for first_part in range(0, 40, 5):
                macro = tile * 20 * 32 * 74
                cursor = macro + first_part * 1152 + col * 16
                scale_cursor = macro + first_part * 1152 + 1024 + col * 4
                d_cursor = macro + 20 * 2304 + (first_part // 2) * 64 + col * 2
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
                    scales = int.from_bytes(
                        records[scale_cursor : scale_cursor + 4], "little"
                    )
                    block = source[tile * 32 + col, part // 2]
                    assert cached_d == block[:2].view("<f2")[0]
                    for segment in range(8):
                        for fragment in range(2):
                            octet = 2 * segment + fragment
                            packed = (
                                int(words[octet // 2]) >> (16 * (octet & 1))
                            ) & 65535
                            original_octet = half_block * 16 + octet
                            offset = 2 + 2 * original_octet
                            expected = int.from_bytes(
                                block[offset : offset + 2], "little"
                            )
                            assert packed & 511 == expected & 511
                            assert packed >> 9 == expected >> 9
                            sign_index = packed >> 9
                            assert (
                                sign_index | ((sign_index.bit_count() & 1) << 7)
                            ).bit_count() % 2 == 0
                            base = half_block * 128 + segment * 16
                            assert (scales >> (4 * segment)) & 15 == (
                                int(block[66 + base // 32]) >> (4 * ((base // 16) % 2))
                            ) & 15
                    cursor += 1152
                    scale_cursor += 1152
                    d_cursor += half_block * 64
                    half_block ^= 1
