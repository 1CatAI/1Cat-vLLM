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
    original = gguf_rows._decode_rows
    calls = []

    def requested_only(data, source_type):
        calls.append(data.shape)
        return original(data, source_type)

    monkeypatch.setattr(gguf_rows, "_decode_rows", requested_only)
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


def test_iq4_nl_decoder_is_bit_identical_to_gguf():
    from vllm.transformers_utils import gguf_rows

    rng = np.random.default_rng(941)
    data = rng.integers(0, 256, (257, 5, 18), dtype=np.uint8)
    # Cover signs, subnormals, zero and large scales; only skip non-finite ones.
    scales = rng.integers(0, 2**16, (257, 5), dtype=np.uint16).view(np.float16)
    scales[~np.isfinite(scales)] = np.float16(1.5)
    scales[0] = [0, -0.0, 6e-8, -65504, 2**-14]
    data[:, :, :2] = scales.view(np.uint8).reshape(257, 5, 2)
    data = data.reshape(257, -1)
    expected = gguf.quants.dequantize(data, gguf.GGMLQuantizationType.IQ4_NL)
    actual = gguf_rows._decode_rows(data, 20)
    assert actual.dtype == expected.dtype == np.float32
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


def _recording_reader(source, cache_rows, logical_rows=None):
    requested = []

    def gather(rows):
        requested.append(rows.tolist())
        return source[rows]

    reader = PackedGGUFRowReader(
        source,
        20,
        160,
        logical_rows,
        cache_bytes=cache_rows * source.shape[1],
        gather=gather,
    )
    return reader, requested


def test_row_cache_reads_only_misses_in_lru_order():
    source = iq4_rows(16)
    official = gguf.quants.dequantize(source, gguf.GGMLQuantizationType.IQ4_NL)
    reader, requested = _recording_reader(source, 4)
    assert reader.cache is not None and reader.cache.capacity == 4

    def check(ids):
        ids = np.asarray(ids, dtype=np.int64)
        actual = reader.lookup(ids)
        expected = official[ids].astype(np.float16)
        np.testing.assert_array_equal(actual.view(np.uint16), expected.view(np.uint16))
        packed = reader.lookup_packed_iq4nl(ids)
        np.testing.assert_array_equal(packed, source[ids])

    check([[1, 2], [1, 3]])
    check([3, 2, 1])
    assert requested == [[1, 2, 3]]
    # Row 5 fills the last slot; row 6 then evicts the least recent row, 2.
    check([5, 1])
    check([6])
    check([1, 5, 6, 4])
    assert requested == [[1, 2, 3], [5], [6], [4]]
    check([2, 3])
    assert requested == [[1, 2, 3], [5], [6], [4], [2, 3]]


def test_row_cache_bypasses_wide_batches():
    source = iq4_rows(16)
    official = gguf.quants.dequantize(source, gguf.GGMLQuantizationType.IQ4_NL)
    reader, requested = _recording_reader(source, 2)
    ids = np.array([0, 1, 2, 1], dtype=np.int64)
    np.testing.assert_array_equal(reader.lookup(ids), official[ids].astype(np.float16))
    np.testing.assert_array_equal(reader.lookup(ids[:2], np.float32), official[:2])
    assert requested == [[0, 1, 2], [0, 1]]
    assert reader.cache is not None and list(reader.cache.slots) == [0, 1]
    assert PackedGGUFRowReader(source, 20, 160, cache_bytes=89).cache is None
    with pytest.raises(ValueError, match="non-negative"):
        PackedGGUFRowReader(source, 20, 160, cache_bytes=-1)


def test_row_cache_keeps_value_checks_and_stores_only_fp16_rows():
    data = iq4_rows(3)
    blocks = data.reshape(3, 5, 18)
    blocks[0, :, :2] = np.full(5, 1024, dtype=np.float16).view(np.uint8).reshape(5, 2)
    blocks[0, :, 2:] = 0xFF
    blocks[2, 0, :2] = np.array([np.inf], dtype=np.float16).view(np.uint8)
    reader, _ = _recording_reader(data, 8)
    with pytest.raises(ValueError, match="overflow FP16"):
        reader.lookup(np.array([0, 1]))
    with pytest.raises(ValueError, match="overflow FP16"):
        reader.lookup_packed_iq4nl(np.array([0]))
    with pytest.raises(ValueError, match="nonfinite"):
        reader.lookup(np.array([2]), np.float32)
    assert reader.cache is not None and not reader.cache.slots
    # FP32 output admits the finite row, which must not later leak into FP16.
    expected = gguf.quants.dequantize(data[:2], gguf.GGMLQuantizationType.IQ4_NL)
    np.testing.assert_array_equal(reader.lookup(np.array([0, 1]), np.float32), expected)
    assert list(reader.cache.slots) == [1]
    with pytest.raises(ValueError, match="overflow FP16"):
        reader.lookup(np.array([0]))


def test_packed_row_reader_uses_gather_before_official_dequantization():
    from vllm.models.qwen4_exp.nvidia.gguf_embedding import _row_gather_callable

    torch = pytest.importorskip("torch")
    source = iq4_rows(7)
    calls = []

    class Gather:
        def apply(self, ids, output):
            calls.append(ids.tolist())
            output.copy_(torch.from_numpy(source)[ids])

    gather = _row_gather_callable(Gather(), source.shape[1])
    reader = PackedGGUFRowReader(source, 20, 160, gather=gather)
    ids = np.array([[6, 0, 6], [2, 6, 0]], dtype=np.int64)
    reference = PackedGGUFRowReader(source, 20, 160).lookup(ids)
    np.testing.assert_array_equal(reader.lookup(ids), reference)
    assert calls == [[0, 2, 6]]
