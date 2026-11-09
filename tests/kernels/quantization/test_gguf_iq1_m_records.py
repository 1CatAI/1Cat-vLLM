# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import runpy
from pathlib import Path

import numpy as np
import pytest

pack = runpy.run_path(
    str(
        Path(__file__).parents[3]
        / "vllm/model_executor/layers/quantization/gguf_iq1_m_records.py"
    )
)["pack_iq1_m_records"]


@pytest.mark.parametrize("n,blocks", [(32, 1), (64, 3), (96, 20)])
def test_original_bits_and_distributed_scales(n, blocks):
    raw = np.random.default_rng(1290).integers(0, 256, (n, blocks * 56), dtype=np.uint8)
    before = raw.copy()
    records = pack(raw)
    assert records.nbytes == raw.nbytes
    np.testing.assert_array_equal(raw, before)
    recovered = np.empty_like(raw)
    for tile in range(n // 32):
        macro = tile * blocks * 32 * 56
        for block in range(blocks):
            for col in range(32):
                row = recovered[tile * 32 + col, block * 56 : (block + 1) * 56]
                scale = macro + blocks * 1536 + block * 256 + col * 8
                row[48:56] = records[scale : scale + 8]
                for half in range(2):
                    cursor = macro + (2 * block + half) * 768
                    index = cursor + col * 16
                    high = cursor + 512 + col * 8
                    assert index % 16 == 0 and high % 8 == 0
                    row[half * 16 : (half + 1) * 16] = records[index : index + 16]
                    row[32 + half * 8 : 40 + half * 8] = records[high : high + 8]
    np.testing.assert_array_equal(recovered, raw)


def test_all_split_k_starts_decode_original_indices_and_scales():
    source = np.random.default_rng(1290).integers(0, 256, (64, 20, 56), dtype=np.uint8)
    records = pack(source.reshape(64, 1120))
    for tile in range(2):
        for col in range(32):
            for first in range(0, 40, 5):
                macro = tile * 20 * 32 * 56
                payload = macro + first * 768 + col * 16
                high = macro + first * 768 + 512 + col * 8
                scales = macro + 20 * 1536 + first // 2 * 256 + col * 8
                half = first & 1
                cached = None
                for part in range(first, first + 5):
                    if part == first or half == 0:
                        cached = records[scales : scales + 8].view("<u2").copy()
                    assert cached is not None
                    block = source[tile * 32 + col, part // 2]
                    words = block[48:56].view("<u2")
                    d = (
                        (int(cached[0]) >> 12)
                        | ((int(cached[1]) >> 12) << 4)
                        | ((int(cached[2]) >> 12) << 8)
                        | ((int(cached[3]) >> 12) << 12)
                    )
                    assert d == sum((int(words[j]) >> 12) << (4 * j) for j in range(4))
                    qh = records[high : high + 8]
                    indices = records[payload : payload + 16]
                    for segment in range(8):
                        small = (
                            int(cached[half * 2 + segment // 4]) >> (3 * (segment & 3))
                        ) & 7
                        expected = (
                            int(words[half * 2 + segment // 4]) >> (3 * (segment & 3))
                        ) & 7
                        assert small == expected
                        for fragment in range(2):
                            octet = 2 * segment + fragment
                            aux = (int(qh[octet // 2]) >> (4 * (octet & 1))) & 15
                            expected_aux = (
                                int(block[32 + half * 8 + octet // 2])
                                >> (4 * (octet & 1))
                            ) & 15
                            assert aux == expected_aux
                            assert int(indices[octet]) | ((aux & 7) << 8) == int(
                                block[half * 16 + octet]
                            ) | ((expected_aux & 7) << 8)
                    payload += 768
                    high += 768
                    scales += half * 256
                    half ^= 1


@pytest.mark.parametrize("shape", [(0, 56), (33, 56), (32, 0), (32, 55)])
def test_invalid_shape(shape):
    with pytest.raises(ValueError):
        pack(np.zeros(shape, dtype=np.uint8))


def test_invalid_dtype():
    with pytest.raises(TypeError):
        pack(np.zeros((32, 56), dtype=np.int8))


def test_strided_source():
    source = np.random.default_rng(1290).integers(0, 256, (64, 224), dtype=np.uint8)
    np.testing.assert_array_equal(pack(source[:, ::2]), pack(source[:, ::2].copy()))
