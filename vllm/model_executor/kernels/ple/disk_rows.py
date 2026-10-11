# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Byte-preserving CPU gathers from retained file-backed embedding shards."""

import ctypes
import statistics
import sys
import threading
import time
import weakref
from collections.abc import Sequence

import numpy as np
import torch

_cache_lock = threading.Lock()
_cache_allocated_bytes = 0


def _release_cache(bytes_: int) -> None:
    global _cache_allocated_bytes
    with _cache_lock:
        _cache_allocated_bytes -= bytes_


class MappedRowGatherKernel:
    """Prefetch selected cold pages and gather into the caller's output."""

    def __init__(
        self,
        pointers: Sequence[int],
        shard_size: int,
        num_rows: int,
        row_bytes: int,
        cache_bytes: int = 0,
    ):
        self.pointers = torch.tensor(pointers, dtype=torch.int64, device="cpu")
        self.shard_size = shard_size
        self.num_rows = num_rows
        self.row_bytes = row_bytes
        self.operator = torch.ops._C.ple_disk_gather_u8
        self._apply_lock = threading.Lock()
        self.cache_ids = self.cache_rows = None
        self.cache_bytes = 0
        global _cache_allocated_bytes
        if cache_bytes and hasattr(torch.ops._C, "ple_disk_cached_gather_u8"):
            with _cache_lock:
                capacity = min(
                    num_rows,
                    max(0, cache_bytes - _cache_allocated_bytes) // (row_bytes + 8),
                )
                if capacity:
                    self.cache_ids = torch.full((capacity,), -1, dtype=torch.int64)
                    self.cache_rows = torch.empty(
                        (capacity, row_bytes), dtype=torch.uint8
                    )
                    self.cache_bytes = capacity * (row_bytes + 8)
                    _cache_allocated_bytes += self.cache_bytes
                    weakref.finalize(self, _release_cache, self.cache_bytes)

    def apply(self, ids: torch.Tensor, output: torch.Tensor) -> None:
        if self.cache_ids is not None:
            # The offload runner is normally serial. Keep key/payload publication
            # safe when qualification or external callers use another thread.
            with self._apply_lock:
                torch.ops._C.ple_disk_cached_gather_u8(
                    ids,
                    self.pointers,
                    self.shard_size,
                    self.num_rows,
                    self.row_bytes,
                    output,
                    self.cache_ids,
                    self.cache_rows,
                )
            return
        self.operator(
            ids, self.pointers, self.shard_size, self.num_rows, self.row_bytes, output
        )

    def qualify(self) -> dict:
        count = min(64, self.num_rows)
        values = [i * (self.num_rows - 1) // max(1, count - 1) for i in range(count)]
        ids = torch.tensor(values, dtype=torch.int64, device="cpu")
        output = torch.empty((count, self.row_bytes), dtype=torch.uint8, device="cpu")
        bases = self.pointers.tolist()

        def reference():
            rows = np.empty((count, self.row_bytes), dtype=np.uint8)
            for row, index in enumerate(values):
                shard, local = divmod(index, self.shard_size)
                ctypes.memmove(
                    rows.ctypes.data + row * self.row_bytes,
                    bases[shard] + local * self.row_bytes,
                    self.row_bytes,
                )
            output.numpy()[:] = rows

        reference()
        expected = output.clone()
        self.apply(ids, output)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        timings: list[list[float]] = [[], []]
        for repeat in range(9):
            for arm in (0, 1) if repeat % 2 == 0 else (1, 0):
                start = time.perf_counter_ns()
                for _ in range(8):
                    if arm:
                        self.apply(ids, output)
                    else:
                        reference()
                timings[arm].append((time.perf_counter_ns() - start) / 8000)
        # These are explicitly warm startup timings. Cold I/O acceptance must
        # additionally be measured in the model or with physical cold pages.
        return {
            "byte_check": True,
            "cache_bytes": self.cache_bytes,
            "warm_startup_us": [statistics.median(samples) for samples in timings],
            "warm_samples_us": timings,
        }


def prepare_mapped_row_gather(
    *,
    pointers: Sequence[int],
    shard_size: int,
    num_rows: int,
    row_bytes: int,
    file_backed: bool,
    enabled: bool,
    cache_bytes: int = 0,
) -> tuple[MappedRowGatherKernel | None, dict]:
    if not enabled:
        return None, {"selected": None, "reason": "disabled_by_policy"}
    if sys.platform != "linux" or not file_backed:
        return None, {"selected": None, "reason": "linux_file_mapping_required"}
    if not pointers or min(shard_size, num_rows, row_bytes, *pointers) <= 0:
        return None, {"selected": None, "reason": "invalid_geometry"}
    import vllm._custom_ops  # noqa: F401 - load the shipped registration

    if not hasattr(torch.ops._C, "ple_disk_gather_u8"):
        return None, {"selected": None, "reason": "native_operator_unavailable"}
    kernel = MappedRowGatherKernel(
        pointers, shard_size, num_rows, row_bytes, cache_bytes
    )
    try:
        measurement = kernel.qualify()
    except (RuntimeError, AssertionError) as error:
        return None, {
            "selected": None,
            "reason": "startup_check_failed",
            "detail": str(error),
        }
    return kernel, {"selected": "MappedRowGatherKernel", **measurement}
