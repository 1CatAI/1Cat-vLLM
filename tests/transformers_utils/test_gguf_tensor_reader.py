# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import struct

import numpy as np
import pytest

from vllm.transformers_utils.gguf_tensor_reader import (
    Q2_0,
    GGUFReader,
    dequantize,
    quant_size,
    quant_type_name,
)


def write_q2(path, k=64, *, truncate=False):
    name = b"blk.0.ffn_down_exps.weight"
    header = struct.pack("<4sIQQ", b"GGUF", 3, 1, 0)
    directory = (
        struct.pack("<Q", len(name))
        + name
        + struct.pack("<IQQQIQ", 3, k, 2, 3, Q2_0, 0)
    )
    prefix = header + directory
    prefix += b"\0" * (-len(prefix) % 32)
    block = np.zeros((6, 18), dtype=np.uint8)
    block[:, :2] = np.frombuffer(np.float16(0.5).tobytes(), dtype=np.uint8)
    block[:, 2:] = 0xE4  # sequential [-1, 0, 1, 2] codes in every byte
    payload = block.tobytes()
    path.write_bytes(prefix + (payload[:-1] if truncate else payload))


def test_q2_stacked_reader_and_cpu_dequantization(tmp_path):
    path = tmp_path / "q2.gguf"
    write_q2(path)
    reader = GGUFReader(path)
    tensor = reader.tensors[0]
    assert tensor.name == "blk.0.ffn_down_exps.weight"
    assert quant_type_name(tensor.tensor_type) == "Q2_0"
    assert quant_size(tensor.tensor_type) == (64, 18)
    assert list(tensor.shape) == [64, 2, 3]
    assert tensor.data.shape == (3, 2, 18)
    expected = np.tile(np.array([-0.5, 0, 0.5, 1], np.float32), 16)
    actual = dequantize(tensor.data, Q2_0)
    assert actual.shape == (3, 2, 64)
    np.testing.assert_array_equal(actual, np.broadcast_to(expected, actual.shape))
    assert np.shares_memory(tensor.data, reader.data)


@pytest.mark.parametrize(
    "k,truncate,error",
    [
        (96, False, "positive multiple of 64"),
        (64, True, "Truncated"),
    ],
)
def test_q2_rejects_invalid_block_layout_or_truncated_payload(
    tmp_path, k, truncate, error
):
    path = tmp_path / "bad.gguf"
    write_q2(path, k, truncate=truncate)
    with pytest.raises(ValueError, match=error):
        GGUFReader(path)


def test_q2_scale_sign_and_multi_block_rows():
    blocks = np.zeros((2, 18), dtype=np.uint8)
    scales = np.array([-2, 0.125], dtype="<f2")
    blocks[:, :2] = scales.view(np.uint8).reshape(2, 2)
    blocks[:, 2:] = 0xE4
    actual = dequantize(blocks.reshape(1, 36), Q2_0)
    codes = np.tile([-1, 0, 1, 2], 16)
    expected = np.concatenate([codes * -2, codes * 0.125]).astype(np.float32)
    np.testing.assert_array_equal(actual[0], expected)
