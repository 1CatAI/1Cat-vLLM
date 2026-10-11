# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read selected packed GGUF rows without expanding an embedding table."""

from collections import OrderedDict
from collections.abc import Callable

import gguf
import numpy as np

from vllm.transformers_utils.gguf_tensor_reader import dequantize, quant_size

_IQ4_NL = int(gguf.GGMLQuantizationType.IQ4_NL)
_IQ4_NL_VALUES = np.asarray(gguf.quants.IQ4_NL.kvalues, dtype=np.float32)
_FP16_MAX = float(np.finfo(np.float16).max)
# |scale| * max|code| (127) stays within FP16 for every finite scale up to this.
_IQ4_NL_SAFE_SCALE = 512


def _decode_rows(data: np.ndarray, source_type: int) -> np.ndarray:
    """Dequantize packed rows to FP32, bit-identical to gguf-py."""
    if source_type != _IQ4_NL:
        return dequantize(data, source_type)
    # gguf-py IQ4_NL: FP32(scale) * FP32(codebook[nibble]); the low nibbles
    # hold elements 0-15 of each 32-element block and the high nibbles 16-31.
    # One table lookup over the 32 codes avoids its per-call reshape overhead.
    rows = data.shape[0]
    blocks = data.reshape(rows, -1, 18)
    scales = blocks[..., :2].copy().view(np.float16).astype(np.float32)
    packed = blocks[..., 2:]
    codes = np.concatenate((packed & 15, packed >> 4), axis=-1)
    return (scales * _IQ4_NL_VALUES[codes]).reshape(rows, -1)


class _PackedRowCache:
    """LRU of validated packed rows keyed by logical row ID.

    Hot n-gram rows repeat across decode rounds. Serving them from anonymous
    memory skips the mapped-file page fault each cold repeat would take.
    """

    def __init__(self, capacity: int, row_bytes: int):
        # np.empty reserves address space; pages are committed as slots fill.
        self.values = np.empty((capacity, row_bytes), dtype=np.uint8)
        self.slots: OrderedDict[int, int] = OrderedDict()
        self.capacity = capacity
        self.hits = self.misses = 0

    def split(self, unique: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return cache slots for ``unique`` (-1 when absent) and the misses."""
        slots = np.full(unique.shape[0], -1, dtype=np.int64)
        get = self.slots.get
        touch = self.slots.move_to_end
        for position, row in enumerate(unique.tolist()):
            slot = get(row)
            if slot is not None:
                touch(row)
                slots[position] = slot
        missing = np.flatnonzero(slots < 0)
        self.hits += unique.shape[0] - missing.shape[0]
        self.misses += missing.shape[0]
        return slots, missing

    def insert(self, rows: np.ndarray, values: np.ndarray) -> None:
        """Store ``values`` for ``rows`` (all absent), evicting LRU entries."""
        slots = np.empty(rows.shape[0], dtype=np.int64)
        table = self.slots
        for position, row in enumerate(rows.tolist()):
            if len(table) < self.capacity:
                slot = len(table)
            else:
                _, slot = table.popitem(last=False)
            table[row] = slot
            slots[position] = slot
        self.values[slots] = values


class PackedGGUFRowReader:
    def __init__(
        self,
        data: np.ndarray,
        source_type: int,
        hidden_size: int,
        logical_rows: int | None = None,
        cache_bytes: int = 0,
        gather: Callable[[np.ndarray], np.ndarray] | None = None,
    ):
        block, size = quant_size(source_type)
        if data.dtype != np.uint8 or data.ndim != 2:
            raise ValueError("GGUF row storage must be packed uint8 [rows,bytes]")
        if hidden_size <= 0 or hidden_size % block:
            raise ValueError("GGUF row width cuts a source quantization block")
        if data.shape[1] != hidden_size // block * size:
            raise ValueError("GGUF packed row width disagrees with logical width")
        rows = data.shape[0] if logical_rows is None else logical_rows
        if not 0 < rows <= data.shape[0]:
            raise ValueError("Invalid GGUF logical row count")
        if cache_bytes < 0:
            raise ValueError("GGUF row cache size must be non-negative")
        # Keep the reader/mmap owner's view. Only requested rows are copied.
        self.data = data
        self.source_type = int(source_type)
        self.hidden_size = hidden_size
        self.logical_rows = rows
        # Optional byte-identical replacement for ``data[rows]`` (sorted int64).
        self._gather = gather
        capacity = min(rows, cache_bytes // data.shape[1])
        self.cache = _PackedRowCache(capacity, data.shape[1]) if capacity else None

    def _unique_ids(self, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not np.issubdtype(ids.dtype, np.integer):
            raise TypeError("GGUF row IDs must be integers")
        if ids.min() < 0 or ids.max() >= self.logical_rows:
            raise IndexError("GGUF row ID outside the logical vocabulary")
        return np.unique(ids.reshape(-1), return_inverse=True)

    def _read(self, rows: np.ndarray) -> np.ndarray:
        return self.data[rows] if self._gather is None else self._gather(rows)

    def _validate(self, packed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return per-row (finite, fits-FP16) flags of the decoded values."""
        if self.source_type == _IQ4_NL:
            scales = packed.reshape(packed.shape[0], -1, 18)[..., :2]
            scales = scales.copy().view(np.float16).astype(np.float32)
            scales = scales.reshape(packed.shape[0], -1)
            finite = np.isfinite(scales).all(axis=-1)
            fits = finite & (np.abs(scales) <= _IQ4_NL_SAFE_SCALE).all(axis=-1)
            unusual = np.flatnonzero(finite & ~fits)
            if unusual.size:
                values = _decode_rows(packed[unusual], self.source_type)
                fits[unusual] = (np.abs(values) <= _FP16_MAX).all(axis=-1)
            return finite, fits
        values = _decode_rows(packed, self.source_type)
        finite = np.isfinite(values).all(axis=-1)
        return finite, finite & (np.abs(values) <= _FP16_MAX).all(axis=-1)

    @staticmethod
    def _check(finite: np.ndarray, fits: np.ndarray, fp16: bool) -> None:
        if not finite.all():
            raise ValueError("GGUF selected rows contain nonfinite values")
        if fp16 and not fits.all():
            raise ValueError("GGUF selected rows overflow FP16 output")

    def _fetch(self, unique: np.ndarray, fp16: bool) -> np.ndarray:
        """Packed rows for sorted ``unique`` IDs, raising on invalid values."""
        cache = self.cache
        # A batch wider than the cache would evict slots it is still reading.
        if cache is None or unique.size > cache.capacity:
            packed = self._read(unique)
            self._check(*self._validate(packed), fp16)
            return packed
        slots, missing = cache.split(unique)
        if not missing.size:
            return cache.values[slots]
        rows = unique[missing]
        fetched = self._read(rows)
        finite, fits = self._validate(fetched)
        self._check(finite, fits, fp16)
        packed = np.empty((unique.size, self.data.shape[1]), dtype=np.uint8)
        hit = slots >= 0
        packed[hit] = cache.values[slots[hit]]
        packed[missing] = fetched
        # Cached rows are valid for every output dtype.
        cache.insert(rows[fits], fetched[fits])
        return packed

    def lookup(self, ids: np.ndarray, dtype=np.float16) -> np.ndarray:
        ids = np.asarray(ids)
        if not np.issubdtype(ids.dtype, np.integer):
            raise TypeError("GGUF row IDs must be integers")
        dtype = np.dtype(dtype)
        if dtype not in (np.dtype(np.float16), np.dtype(np.float32)):
            raise ValueError("GGUF row output requires FP16 or FP32")
        output_shape = (*ids.shape, self.hidden_size)
        if not ids.size:
            return np.empty(output_shape, dtype=dtype)
        unique, inverse = self._unique_ids(ids)
        packed = self._fetch(unique, dtype == np.float16)
        values = _decode_rows(packed, self.source_type).astype(dtype)
        return values[inverse].reshape(output_shape)

    def lookup_packed_iq4nl(self, ids: np.ndarray) -> np.ndarray:
        """Copy selected IQ4_NL rows, retaining the FP16 decoder checks."""
        if self.source_type != _IQ4_NL:
            raise ValueError("Packed result transport requires IQ4_NL rows")
        ids = np.asarray(ids)
        if not np.issubdtype(ids.dtype, np.integer):
            raise TypeError("GGUF row IDs must be integers")
        output_shape = (*ids.shape, self.data.shape[1])
        if not ids.size:
            return np.empty(output_shape, dtype=np.uint8)
        unique, inverse = self._unique_ids(ids)
        return self._fetch(unique, fp16=True)[inverse].reshape(output_shape)
