# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gguf
import numpy as np
import pytest

from vllm.transformers_utils.gguf_rows import PackedGGUFRowReader


def iq4_rows(rows, hidden=160):
    rng = np.random.default_rng(940)
    data = rng.integers(0, 256, (rows, hidden // 32, 18), dtype=np.uint8)
    scales = np.full((rows, hidden // 32), 0.03125, dtype=np.float16)
    data[:, :, :2] = scales.view(np.uint8).reshape(rows, hidden // 32, 2)
    return data.reshape(rows, -1)


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_selected_rows_match_official_dequantization(tmp_path, monkeypatch, dtype):
    from vllm.transformers_utils import gguf_rows

    path = tmp_path / "packed.bin"
    source = iq4_rows(23)
    source.tofile(path)
    mapped = np.memmap(path, dtype=np.uint8, mode="r", shape=source.shape)
    reader = PackedGGUFRowReader(mapped, 20, 160, logical_rows=21)
    assert reader.data is mapped
    ids = np.array([[20, 2, 20], [0, 2, 7]], dtype=np.int64)
    official = gguf.quants.dequantize(source, gguf.GGMLQuantizationType.IQ4_NL)
    original = gguf_rows.dequantize
    calls = []

    def requested_only(data, source_type):
        calls.append(data.shape)
        return original(data, source_type)

    monkeypatch.setattr(gguf_rows, "dequantize", requested_only)
    actual = reader.lookup(ids, dtype)
    np.testing.assert_array_equal(actual, official[ids].astype(dtype))
    assert calls == [(4, source.shape[1])]
    np.testing.assert_array_equal(mapped, source)


def test_empty_and_invalid_ids_preserve_storage():
    source = iq4_rows(3)
    reader = PackedGGUFRowReader(source, 20, 160, logical_rows=2)
    assert reader.lookup(np.empty((0, 4), dtype=np.int64)).shape == (0, 4, 160)
    for ids in (np.array([-1]), np.array([2]), np.array([2**63], dtype=np.uint64)):
        with pytest.raises(IndexError):
            reader.lookup(ids)
    with pytest.raises(TypeError, match="integers"):
        reader.lookup(np.array([0.5]))
    with pytest.raises(ValueError, match="FP16 or FP32"):
        reader.lookup(np.array([0]), np.float64)
    assert reader.data is source


def test_rejects_malformed_packed_rows():
    source = iq4_rows(3)
    for data, width, rows in (
        (source.astype(np.int8), 160, 3),
        (source.reshape(-1), 160, 3),
        (source, 159, 3),
        (source[:, :-1], 160, 3),
        (source, 160, 4),
        (source, 160, 0),
    ):
        with pytest.raises(ValueError):
            PackedGGUFRowReader(data, 20, width, rows)


def test_reports_fp16_overflow_without_clipping():
    data = iq4_rows(1)
    blocks = data.reshape(1, 5, 18)
    blocks[:, :, :2] = (
        np.full((1, 5), 1024, dtype=np.float16).view(np.uint8).reshape(1, 5, 2)
    )
    blocks[:, :, 2:] = 0xFF
    reader = PackedGGUFRowReader(data, 20, 160)
    with pytest.raises(ValueError, match="overflow FP16"):
        reader.lookup(np.array([0]))
    actual = reader.lookup(np.array([0]), np.float32)
    expected = gguf.quants.dequantize(data, gguf.GGMLQuantizationType.IQ4_NL)
    np.testing.assert_array_equal(actual, expected)


def test_packed_row_reader_uses_gather_before_official_dequantization():
    import torch

    source = iq4_rows(7)
    calls = []

    class Gather:
        def apply(self, ids, output):
            calls.append(ids.tolist())
            output.copy_(torch.from_numpy(source)[ids])

    reader = PackedGGUFRowReader(source, 20, 160, packed_row_gather=Gather())
    ids = np.array([[6, 0, 6], [2, 6, 0]], dtype=np.int64)
    reference = PackedGGUFRowReader(source, 20, 160).lookup(ids)
    np.testing.assert_array_equal(reader.lookup(ids), reference)
    assert calls == [[0, 2, 6]]
