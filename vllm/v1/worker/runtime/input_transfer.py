# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Per-runner input transfer lifecycle, independent of model/platform policy."""

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from logging import Logger
from typing import Any

import torch

from vllm.v1.utils import RING_PREP, RING_PREP_SLOTS, CpuGpuBuffer


class InputTransferSession:
    def __init__(
        self,
        *,
        eligible: bool = False,
        trace_enabled: bool = False,
        trace_every: int = 16,
        logger: Logger | None = None,
        trace_prefix: str = "Input preparation",
        staged_message: str = "",
        synchronize: Callable[[Any, str], None] | None = None,
    ):
        self.eligible = eligible
        self.trace_enabled = trace_enabled
        self.trace_every = trace_every
        self.logger = logger
        self.trace_prefix = trace_prefix
        self.staged_message = staged_message
        self.synchronize = synchronize or (lambda event, label: event.synchronize())
        self.event: torch.Event | None = None
        self.active = False
        self.trace_step = 0
        self.logged = False
        # VLLM_V41_RING_PREP: the event of each of the last RING_PREP_SLOTS
        # staged steps (their copies read that ring slot).
        self.ring_events: list[torch.cuda.Event | None] = [None] * RING_PREP_SLOTS

    def can_stage(
        self,
        *,
        num_tokens: int,
        num_reqs: int,
        has_previous_sample: bool,
        has_encoder_inputs: bool,
        has_draft_tokens: bool,
        has_accepted_event: bool,
    ) -> bool:
        return (
            self.eligible
            and num_tokens == 1
            and num_reqs == 1
            and has_previous_sample
            and not has_encoder_inputs
            and not has_draft_tokens
            and not has_accepted_event
        )

    def copy_buffer(self, buffer: CpuGpuBuffer, n: int | None = None) -> torch.Tensor:
        return buffer.copy_to_gpu_staged(n) if self.active else buffer.copy_to_gpu(n)

    def copy_positions(self, buffer: CpuGpuBuffer, n: int) -> torch.Tensor:
        src, dst = buffer.cpu[:, :n], buffer.gpu[:, :n]
        return (
            buffer.copy_view_to_gpu_staged(src, dst)
            if self.active
            else dst.copy_(src, non_blocking=True)
        )

    def commit_block_table(self, block_table: Any, num_reqs: int) -> None:
        if self.active:
            block_table.commit_block_table_staged(num_reqs)
        else:
            block_table.commit_block_table(num_reqs)

    @contextmanager
    def prepare(
        self, *, skip_sync: bool = False, trace_override: bool = False
    ) -> Iterator[None]:
        trace_enabled = self.trace_enabled or trace_override
        step = self.trace_step
        trace_log = trace_enabled and step % self.trace_every == 0
        if trace_enabled:
            self.trace_step += 1
        previous = self.active
        self.active = skip_sync
        sync_ms = 0.0
        started = False
        mode = (
            "no_event"
            if self.event is None
            else "staged_event"
            if skip_sync
            else "event"
        )
        try:
            ring = skip_sync and RING_PREP["on"]
            if ring:
                RING_PREP["step"] += 1
                slot = RING_PREP["step"] % RING_PREP_SLOTS
                previous_use = self.ring_events[slot]
                if previous_use is not None:
                    previous_use.synchronize()
            if self.event is not None:
                if skip_sync:
                    if not self.logged:
                        if self.logger is not None:
                            self.logger.info(self.staged_message)
                        self.logged = True
                else:
                    sync_start = time.perf_counter() if trace_log else 0.0
                    from vllm.v1 import v41_trace

                    if v41_trace.ENABLED:
                        v41_trace.mark("psync", done=int(self.event.query()))
                    self.synchronize(
                        self.event, "GPUModelRunner.prepare_inputs_event.synchronize"
                    )
                    if v41_trace.ENABLED:
                        v41_trace.mark("psync_end")
                    if trace_log:
                        sync_ms = (time.perf_counter() - sync_start) * 1000.0
            body_start = time.perf_counter() if trace_log else 0.0
            started = True
            try:
                yield
            finally:
                if self.event is not None:
                    self.event.record()
                if ring:
                    ring_event = torch.cuda.Event()
                    ring_event.record()
                    self.ring_events[slot] = ring_event
        finally:
            self.active = previous
            if started and trace_log and self.logger is not None:
                body_ms = (time.perf_counter() - body_start) * 1000.0
                self.logger.info(
                    "%s kind=input_prep step=%d mode=%s sync_ms=%.3f body_ms=%.3f",
                    self.trace_prefix,
                    step,
                    mode,
                    sync_ms,
                    body_ms,
                )
