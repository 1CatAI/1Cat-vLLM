# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Nonblocking GPU timing feedback for mixed-prefill scheduling."""

from collections import deque

import torch

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import MixedPrefillTiming


class MixedPrefillTimer:
    def __init__(self) -> None:
        self.active: tuple[torch.cuda.Event, MixedPrefillTiming] | None = None
        self.pending: deque[
            tuple[torch.cuda.Event, torch.cuda.Event, MixedPrefillTiming]
        ] = deque()

    def begin(self, output: SchedulerOutput) -> None:
        # Abandon an incomplete measurement (e.g. an execute without sampling).
        self.active = None
        if output.mixed_prefill_tokens <= 0 or output.mixed_decode_tokens <= 0:
            return
        # Bound bookkeeping even if a device/connector holds the GPU for long.
        if len(self.pending) >= 8:
            return
        start = torch.cuda.Event(enable_timing=True)
        start.record()
        self.active = (
            start,
            MixedPrefillTiming(
                prefill_tokens=output.mixed_prefill_tokens,
                decode_tokens=output.mixed_decode_tokens,
                budget_tokens=output.mixed_prefill_budget,
                elapsed_ms=0.0,
            ),
        )

    def finish(self) -> MixedPrefillTiming | None:
        if self.active is not None:
            start, sample = self.active
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            self.pending.append((start, end, sample))
            self.active = None
        ready = None
        # A delayed sample carries its own token counts. Never synchronize the
        # current step or mistake a previous step for the current batch.
        while self.pending and self.pending[0][1].query():
            start, end, ready = self.pending.popleft()
            ready.elapsed_ms = start.elapsed_time(end)
        return ready
