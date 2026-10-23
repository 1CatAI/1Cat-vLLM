# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Per-owner event collection and reporting, shared by runners and proposers."""

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

Events = list[tuple[str, torch.cuda.Event, torch.cuda.Event]]
ProfileInput = Events | dict[str, Any]


@dataclass(frozen=True)
class ProfileReport:
    label: str
    preferred: tuple[str, ...]
    metadata: tuple[tuple[str, str], ...]
    interval: bool = True
    append_other: bool = True
    spec_steps: bool = False
    gpu_sums: tuple[tuple[str, tuple[str, ...]], ...] = ()
    cpu_fields: tuple[str, ...] = ()
    wall_start: str | None = None


class StepProfiler:
    """Own totals only; in-flight contexts belong to the executing step."""

    def __init__(
        self,
        *,
        enabled: bool,
        interval: int,
        report: ProfileReport,
        logger: Any,
        report_rank: Callable[[], bool],
    ):
        self.enabled = enabled
        self.interval = interval
        self.layout = report
        self.logger = logger
        self.report_rank = report_rank
        self.totals: dict[str, float] = {}
        self.calls = self.spec_steps = self.last_calls = self.last_spec_steps = 0
        self.last_totals: dict[str, float] = {}

    @staticmethod
    def start(events: ProfileInput | None) -> torch.cuda.Event | None:
        if events is None:
            return None
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    @staticmethod
    def finish(
        events: ProfileInput | None, name: str, start: torch.cuda.Event | None
    ) -> None:
        if events is None or start is None:
            return
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        event_list = events["events"] if isinstance(events, dict) else events
        event_list.append((name, start, end))

    # Both context shapes use the same collector, including the disabled fast path.
    start_context = start
    finish_context = finish

    @staticmethod
    def add_cpu_ms(cpu_ms: dict[str, float] | None, name: str, start: float) -> None:
        if cpu_ms is not None:
            cpu_ms[name] = (
                cpu_ms.get(name, 0.0) + (time.perf_counter() - start) * 1000.0
            )

    @classmethod
    def add_cpu_context(
        cls, ctx: dict[str, Any] | None, name: str, start: float
    ) -> None:
        cls.add_cpu_ms(ctx["cpu_ms"] if ctx is not None else None, name, start)

    def report(self, ctx: dict[str, Any] | None) -> None:
        if ctx is None:
            return
        events = ctx["events"]
        if events is None:
            return
        if events:
            events[-1][2].synchronize()
        timings: dict[str, float] = {}
        for name, start, end in events:
            timings[name] = timings.get(name, 0.0) + start.elapsed_time(end)
        for name, stages in self.layout.gpu_sums:
            timings[name] = sum(timings.get(stage, 0.0) for stage in stages)
        timings.update(ctx.get("cpu_ms", {}))
        timings.update((key, ctx[key]) for key in self.layout.cpu_fields)
        if self.layout.wall_start is not None:
            timings["total_wall_cpu"] = (
                time.perf_counter() - ctx[self.layout.wall_start]
            ) * 1000.0
        self.calls += 1
        for name, value in timings.items():
            self.totals[name] = self.totals.get(name, 0.0) + value
        if self.layout.spec_steps:
            self.spec_steps += bool(ctx["has_spec_decode_metadata"])
        if self.calls != 1 and self.calls % self.interval != 0:
            return
        if not self.report_rank():
            return
        keys = [key for key in self.layout.preferred if key in self.totals]
        if self.layout.append_other:
            keys.extend(sorted(key for key in self.totals if key not in keys))
        metadata = "".join(
            f" {label}={ctx[key]}" for label, key in self.layout.metadata
        )
        spec = f" spec_steps={self.spec_steps}" if self.layout.spec_steps else ""
        summary = " ".join(f"{key}={self.totals[key] / self.calls:.3f}" for key in keys)
        message = self.layout.label + " avg_ms calls=%d%s%s %s"
        self.logger.info(
            message,
            self.calls,
            spec,
            metadata,
            summary,
        )
        if not self.layout.interval:
            return
        interval_calls = self.calls - self.last_calls
        spec = (
            f" interval_spec_steps={self.spec_steps - self.last_spec_steps}"
            if self.layout.spec_steps
            else ""
        )
        interval_values = {
            key: (self.totals[key] - self.last_totals.get(key, 0.0)) / interval_calls
            for key in keys
        }
        summary = " ".join(
            f"{key}={value:.3f}" for key, value in interval_values.items()
        )
        message = (
            self.layout.label + " interval_avg_ms calls=%d interval_calls=%d%s%s %s"
        )
        self.logger.info(
            message,
            self.calls,
            interval_calls,
            spec,
            metadata,
            summary,
        )
        self.last_totals = dict(self.totals)
        self.last_calls = self.calls
        self.last_spec_steps = self.spec_steps

    def report_proposal(
        self,
        events: Events | None,
        cpu_ms: dict[str, float],
        batch_size: int,
        num_tokens: int,
    ) -> None:
        self.report(
            dict(
                events=events,
                cpu_ms=cpu_ms,
                batch_size=batch_size,
                num_tokens=num_tokens,
            )
        )
