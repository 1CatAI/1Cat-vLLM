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


def _should_log_values(label, elapsed_ms, threshold_ms, every, counts) -> bool:
    if elapsed_ms < threshold_ms:
        return False
    counts[label] += 1
    count = counts[label]
    return count <= 4 or count % max(1, every) == 0


def _should_log(label: str, elapsed_ms: float) -> bool:
    threshold = envs.VLLM_SM70_DECODE_EVENT_TRACE_THRESHOLD_MS
    if elapsed_ms < threshold:
        return False
    return _should_log_values(
        label,
        elapsed_ms,
        threshold,
        envs.VLLM_SM70_DECODE_EVENT_TRACE_EVERY,
        _TRACE_COUNTS,
    )


@contextmanager
def _trace_range(
    label: str, should_log: Callable[[str, float], bool]
) -> Iterator[None]:
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
        if should_log(label, elapsed_ms):
            logger.warning(
                "SM70 decode event trace: label=%s elapsed_ms=%.3f pid=%s rank=%s",
                label,
                elapsed_ms,
                os.getpid(),
                _rank(),
            )


@contextmanager
def sm70_decode_trace_range(label: str) -> Iterator[None]:
    if not sm70_decode_event_trace_enabled():
        yield
        return
    with _trace_range(label, _should_log):
        yield


class DecodeEventTracer:
    """Engine-local trace counts and policy for migrated runtime consumers."""

    def __init__(self, policy):
        self.policy = policy
        self.counts: defaultdict[str, int] = defaultdict(int)

    def _should_log(self, label: str, elapsed_ms: float) -> bool:
        return _should_log_values(
            label,
            elapsed_ms,
            self.policy.event_threshold_ms,
            self.policy.event_every,
            self.counts,
        )

    def synchronize(self, event: torch.Event, label: str) -> None:
        if not self.policy.events:
            event.synchronize()
            return
        with _trace_range(label, self._should_log):
            event.synchronize()


def sm70_trace_call(label: str, fn: Callable[[], T]) -> T:
    with sm70_decode_trace_range(label):
        return fn()


def sm70_trace_event_sync(event: torch.Event, label: str) -> None:
    with sm70_decode_trace_range(label):
        event.synchronize()
