# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native PLE request lane: mapped request in, packed rows out, no Python.

The default request path costs about a millisecond per forward before the
first CPU row is touched: a D2H copy and event wait in the GPU process, a
ZMQ hop, then Python hashing and lookup in the owner. Here TP0's model stream
writes the request into host-mapped memory with one kernel, and a native
thread in the owner (``csrc/ple_fast_lane.cpp``) produces the same packed
rows into every TP rank's mapped result buffer and releases its flag.
"""

import threading
import time

import torch
from cuda.bindings import driver as cuda_driver

from vllm.logger import init_logger
from vllm.model_executor.kernels.ple.host_result import _shared_zeros
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

# header: sequence, tokens, requests, owner-active, reserved...
_HEADER = 8
_SEQ, _TOKENS, _REQS, _ACTIVE = 0, 1, 2, 3
_STAT_NAMES = (
    "requests",
    "rows",
    "hits",
    "misses",
    "total_ns",
    "max_ns",
    "error",
    "last_seq",
    "hash_ns",
    "lookup_ns",
    "reset_wait_ns",
    "prefetch_positions",
    "prefetch_pages",
    "cold_pages",
    "window_checked",
    "window_matched",
)
_ERRORS = {1: "malformed request", 2: "nonfinite rows", 3: "FP16 overflow"}


def request_shape(max_reqs: int, max_tokens: int) -> tuple[int]:
    # header | query_start_loc [max_reqs + 1] | context [max_reqs, 2] | ids
    return (_HEADER + max_reqs + 1 + max_reqs * 2 + max_tokens,)


@triton.jit(do_not_specialize=["num_tokens", "num_reqs", "seq"])
def _publish_request_kernel(
    input_ids,
    query_start_loc,
    context,
    request,
    num_tokens,
    num_reqs,
    seq,
    MAX_REQS: tl.constexpr,
    CONTEXT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    lane = tl.arange(0, BLOCK)
    starts = request + 8
    context_out = starts + MAX_REQS + 1
    ids_out = context_out + MAX_REQS * 2
    start_mask = lane < num_reqs + 1
    value = tl.load(query_start_loc + lane, mask=start_mask, other=0)
    tl.store(starts + lane, value.to(tl.int64), mask=start_mask)
    context_mask = lane < num_reqs * 2
    row = lane // 2
    column = lane % 2
    value = tl.load(context + row * CONTEXT + column, mask=context_mask, other=0)
    tl.store(context_out + lane, value.to(tl.int64), mask=context_mask)
    for begin in range(0, num_tokens, BLOCK):
        index = begin + lane
        mask = index < num_tokens
        value = tl.load(input_ids + index, mask=mask, other=0)
        tl.store(ids_out + index, value.to(tl.int64), mask=mask)
    tl.debug_barrier()
    tl.store(request + 1, num_tokens.to(tl.int64))
    tl.store(request + 2, num_reqs.to(tl.int64))
    tl.debug_barrier()
    # Release the request body before the owner can observe its sequence.
    tl.atomic_xchg(request, seq.to(tl.int64), sem="release", scope="sys")


class FastLaneRequest:
    """TP0's host-mapped request slot, written by one kernel per forward."""

    def __init__(self, device: torch.device, max_reqs: int, max_tokens: int):
        self.device = device
        self.max_reqs = max_reqs
        self.max_tokens = max_tokens
        self.host = _shared_zeros(request_shape(max_reqs, max_tokens), torch.int64)
        self._registered = 0
        self._seq = 0
        self._block = min(1024, triton.next_power_of_2(max(max_reqs * 2 + 1, 64)))
        with torch.accelerator.device_index(device.index):
            torch.cuda.init()
            torch.cuda.current_stream(device)
            size = self.host.numel() * self.host.element_size()
            status = cuda_driver.cuMemHostRegister(self.host.data_ptr(), size, 3)
            if status[0].value != 0:
                raise RuntimeError(f"PLE fast lane registration failed: {status[0]}")
            self._registered = self.host.data_ptr()
            self._mapped = torch.ops._C.get_cuda_view_from_cpu_tensor(self.host)
            if self._mapped.data_ptr() != self.host.data_ptr():
                raise RuntimeError("PLE fast lane has no stable mapped CUDA view")

    def validate_registration(self) -> None:
        if self.host.data_ptr() != self._registered or not self.host.is_pinned():
            raise RuntimeError("PLE fast lane storage changed during registration")

    @property
    def active(self) -> bool:
        """Whether the owner's native thread is serving this slot."""
        return bool(self.host[_ACTIVE].item())

    def publish(
        self,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
        num_reqs: int,
        num_tokens: int,
    ) -> None:
        if not (0 < num_reqs <= self.max_reqs and 0 < num_tokens <= self.max_tokens):
            raise ValueError("PLE fast lane request exceeds its capacity")
        self._seq += 1
        _publish_request_kernel[(1,)](
            input_ids,
            query_start_loc,
            ngram_context,
            self._mapped,
            num_tokens,
            num_reqs,
            self._seq,
            MAX_REQS=self.max_reqs,
            CONTEXT=ngram_context.stride(0),
            BLOCK=self._block,
            num_warps=4,
        )

    def close(self) -> None:
        if self._registered:
            with torch.accelerator.device_index(self.device.index):
                torch.accelerator.synchronize()
                cuda_driver.cuMemHostUnregister(self._registered)
            self._registered = 0


class FastLaneOwner:
    """Runs the native lane in the PLE owner and surfaces its failures."""

    def __init__(self, request: torch.Tensor, handle: int, report_s: float = 60):
        self._request = request
        self._handle = handle
        self._report_s = report_s
        self._stop = threading.Event()
        self._last: dict[str, int] = {}
        request[_ACTIVE] = 1
        self._thread = threading.Thread(
            target=self._monitor, name="ple-fast-lane-monitor", daemon=True
        )
        self._thread.start()

    def stats(self) -> dict[str, int]:
        values = torch.ops._C.ple_fast_lane_stats(self._handle).tolist()
        return dict(zip(_STAT_NAMES, values))

    def _monitor(self) -> None:
        last_report = time.monotonic()
        while not self._stop.wait(0.5):
            stats = self.stats()
            if stats["error"]:
                self._request[_ACTIVE] = 0
                logger.critical(
                    "PLE fast lane stopped: %s; terminating the PLE owner.",
                    _ERRORS.get(stats["error"], stats["error"]),
                )
                # Waiting consumers can never be served now; fail loudly.
                import os

                os._exit(1)
            if time.monotonic() - last_report >= self._report_s:
                last_report = time.monotonic()
                self._report(stats)

    def _report(self, stats: dict[str, int]) -> None:
        delta = {k: stats[k] - self._last.get(k, 0) for k in stats}
        self._last = stats
        requests = max(delta["requests"], 1)
        rows = max(delta["hits"] + delta["misses"], 1)
        logger.info(
            "PLE fast lane: %d requests, mean %.1f us (hash %.1f, reset wait "
            "%.1f, rows %.1f), max %.1f us, row hit %.1f%%, prefetched %d "
            "positions / %d pages, cold pages/request %.1f, window match %d/%d",
            delta["requests"],
            delta["total_ns"] / requests / 1e3,
            delta["hash_ns"] / requests / 1e3,
            delta["reset_wait_ns"] / requests / 1e3,
            delta["lookup_ns"] / requests / 1e3,
            stats["max_ns"] / 1e3,
            100 * delta["hits"] / rows,
            delta["prefetch_positions"],
            delta["prefetch_pages"],
            delta["cold_pages"] / requests,
            delta["window_matched"],
            delta["window_checked"],
        )

    def close(self) -> None:
        self._stop.set()
        self._request[_ACTIVE] = 0
        torch.ops._C.ple_fast_lane_stop(self._handle)
