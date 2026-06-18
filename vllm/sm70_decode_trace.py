# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
import time
from collections import defaultdict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TypeVar

import torch

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

T = TypeVar("T")

_TRACE_COUNTS: defaultdict[str, int] = defaultdict(int)


def sm70_decode_event_trace_enabled() -> bool:
    return bool(envs.VLLM_SM70_DECODE_EVENT_TRACE)


def _rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return -1


def _should_log(label: str, elapsed_ms: float) -> bool:
    threshold_ms = envs.VLLM_SM70_DECODE_EVENT_TRACE_THRESHOLD_MS
    if elapsed_ms < threshold_ms:
        return False
    _TRACE_COUNTS[label] += 1
    every = max(1, envs.VLLM_SM70_DECODE_EVENT_TRACE_EVERY)
    count = _TRACE_COUNTS[label]
    return count <= 4 or count % every == 0


@contextmanager
def sm70_decode_trace_range(label: str) -> Iterator[None]:
    if not sm70_decode_event_trace_enabled():
        yield
        return

    pushed = False
    if torch.cuda.is_available():
        try:
            torch.cuda.nvtx.range_push(label)
            pushed = True
        except RuntimeError:
            pushed = False

    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if pushed:
            torch.cuda.nvtx.range_pop()
        if _should_log(label, elapsed_ms):
            logger.warning(
                "SM70 decode event trace: label=%s elapsed_ms=%.3f pid=%s rank=%s",
                label,
                elapsed_ms,
                os.getpid(),
                _rank(),
            )


def sm70_trace_call(label: str, fn: Callable[[], T]) -> T:
    with sm70_decode_trace_range(label):
        return fn()


def sm70_trace_event_sync(event: torch.Event, label: str) -> None:
    with sm70_decode_trace_range(label):
        event.synchronize()
