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


class _DecodedRowCache:
    """LRU of validated FP16 rows keyed by logical row ID.

    Hot n-gram rows repeat across decode rounds; serving them from here skips
    the mapped-file page touch and the dequantization of each repeat.
    """

    def __init__(self, capacity: int, width: int):
        # np.empty reserves address space; pages are committed as slots fill.
        self.values = np.empty((capacity, width), dtype=np.float16)
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

    def insert(self, rows: np.ndarray, values: np.ndarray) -> np.ndarray:
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
        return slots


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
        capacity = min(rows, cache_bytes // (hidden_size * 2))
        self.cache = _DecodedRowCache(capacity, hidden_size) if capacity else None

    def _decode_checked(self, rows: np.ndarray, dtype: np.dtype) -> np.ndarray:
        packed = self.data[rows] if self._gather is None else self._gather(rows)
        values = _decode_rows(packed, self.source_type)
        if not np.isfinite(values).all():
            raise ValueError("GGUF selected rows contain nonfinite values")
        if dtype == np.float16 and np.any(np.abs(values) > np.finfo(dtype).max):
            raise ValueError("GGUF selected rows overflow FP16 output")
        return values.astype(dtype)

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
        if ids.min() < 0 or ids.max() >= self.logical_rows:
            raise IndexError("GGUF row ID outside the logical vocabulary")
        unique, inverse = np.unique(ids.reshape(-1), return_inverse=True)
        cache = self.cache
        # A batch wider than the cache would evict slots it is still reading.
        if cache is None or dtype != np.float16 or unique.size > cache.capacity:
            values = self._decode_checked(unique, dtype)
            return values[inverse].reshape(output_shape)
        slots, missing = cache.split(unique)
        if missing.size:
            rows = unique[missing]
            slots[missing] = cache.insert(rows, self._decode_checked(rows, dtype))
        return cache.values[slots[inverse]].reshape(output_shape)
