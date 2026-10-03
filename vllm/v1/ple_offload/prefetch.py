# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded exact-row cache and coherent sampled-key mailbox reader.

The mailbox is advisory: skipped publications and misses use the ordinary
lookup. Keys include every raw ngram token, so request slot reuse and batch
reordering cannot select a different row. Each PLE layer owns its own cache.
"""

import ctypes
from collections import OrderedDict

import torch

from vllm.model_executor.kernels.ple.host_result import _atomics


class SampledKeyReader:
    """Read a single-producer seqlock without acknowledging or blocking GPU work.

    The producer publishes an odd generation before changing any payload and
    an even generation after all system-visible writes complete. Each payload
    row is [valid, oldest token, ..., newest token]. A skipped generation is
    harmless; a changing generation causes a miss, never a partial snapshot.
    """

    def __init__(self, payload: torch.Tensor, flag: torch.Tensor):
        if (
            payload.device.type != "cpu"
            or payload.dtype != torch.int32
            or payload.ndim != 2
            or payload.shape[1] < 3
            or not payload.is_contiguous()
            or not payload.is_shared()
            or flag.device.type != "cpu"
            or flag.dtype != torch.int32
            or not flag.numel()
            or not flag.is_contiguous()
            or not flag.is_shared()
        ):
            raise ValueError("Sampled keys require shared CPU int32 payload/flag")
        functions = _atomics()
        if functions is None:
            raise RuntimeError("System acquire/release atomics are unavailable")
        self._load = functions[1]
        self._payload_load = functions[1]
        self._payload = payload
        self._flag_pointer = flag.data_ptr()
        self._seen = 0

    def read(self) -> list[tuple[int, ...]] | None:
        before = self._load(self._flag_pointer, 2)
        if before == self._seen or before & 1:
            return None
        # Payload words are atomic on both sides. Ordinary tensor copying
        # during an overwrite would be a data race even if the later version
        # check rejected that snapshot.
        pointer = self._payload.data_ptr()
        columns = self._payload.shape[1]
        rows = [
            [
                ctypes.c_int32(
                    self._payload_load(pointer + 4 * (row * columns + col), 0)
                ).value
                for col in range(columns)
            ]
            for row in range(self._payload.shape[0])
        ]
        after = self._load(self._flag_pointer, 2)
        if before != after or after & 1:
            return None
        self._seen = after
        if any(r[0] not in (0, 1) for r in rows):
            raise RuntimeError("Invalid sampled-key validity marker")
        return [tuple(r[1:]) for r in rows if r[0]]


class ExactRowPrefetchCache:
    """Own immutable copies of a bounded number of exact lookup results."""

    def __init__(self, capacity_rows: int):
        if capacity_rows <= 0:
            raise ValueError("Prefetch capacity must be positive")
        self.capacity_rows = capacity_rows
        self._rows: OrderedDict[tuple[int, ...], torch.Tensor] = OrderedDict()

    @property
    def resident_bytes(self) -> int:
        return sum(r.numel() * r.element_size() for r in self._rows.values())

    def put(self, keys: list[tuple[int, ...]], result: torch.Tensor) -> None:
        if result.device.type != "cpu" or result.ndim != 2 or len(keys) != len(result):
            raise ValueError("Prefetch rows must match a CPU result matrix")
        for key, row in zip(keys, result, strict=True):
            if key in self._rows and not torch.equal(self._rows[key], row):
                raise RuntimeError("The same ngram key produced different row bytes")
            self._rows[key] = row.clone()
            self._rows.move_to_end(key)
            while len(self._rows) > self.capacity_rows:
                self._rows.popitem(last=False)

    def copy(self, keys: list[tuple[int, ...]], destination: torch.Tensor) -> bool:
        if (
            destination.device.type != "cpu"
            or destination.ndim != 2
            or len(keys) != len(destination)
        ):
            raise ValueError("Prefetch destination must match requested CPU rows")
        if any(key not in self._rows for key in keys):
            return False
        rows = [self._rows[key] for key in keys]
        if any(
            row.shape != destination.shape[1:] or row.dtype != destination.dtype
            for row in rows
        ):
            raise ValueError("Cached row layout differs from the consumer")
        for index, (key, row) in enumerate(zip(keys, rows, strict=True)):
            destination[index].copy_(row)
            self._rows.move_to_end(key)
        return True
