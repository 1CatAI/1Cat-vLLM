# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded, opt-in host observations of the existing prefill controller."""

import json
import logging
import math
import os
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MixedPrefillTraceConfig:
    output_dir: str
    max_events: int = 16384
    max_bytes: int = 33554432


def mixed_prefill_trace_config(value: object) -> MixedPrefillTraceConfig | None:
    if not isinstance(value, Mapping):
        return None
    raw = value.get("mixed_prefill_budget_trace")
    if raw is None:
        return None
    if not isinstance(raw, Mapping) or set(raw) - {
        "output_dir",
        "max_events",
        "max_bytes",
    }:
        raise ValueError("Invalid mixed_prefill_budget_trace configuration")
    directory = raw.get("output_dir")
    if not isinstance(directory, str) or not Path(directory).is_absolute():
        raise ValueError("mixed_prefill_budget_trace requires an absolute output_dir")
    events = raw.get("max_events", 16384)
    size = raw.get("max_bytes", 33554432)
    if type(events) is not int or not 8 <= events <= 200000:
        raise ValueError("mixed_prefill_budget_trace max_events must be 8..200000")
    if type(size) is not int or not 4096 <= size <= 67108864:
        raise ValueError("mixed_prefill_budget_trace max_bytes must be 4096..67108864")
    return MixedPrefillTraceConfig(directory, events, size)


def _number(value: float | None) -> float | str | None:
    if value is None or math.isfinite(value):
        return value
    return str(value)


def budget_snapshot(budget: Any) -> dict[str, Any]:
    return {
        "tokens": budget.tokens,
        "ms_per_token": _number(budget.ms_per_token),
        "target_ms": budget.target_ms,
        "min_tokens": budget.min_tokens,
        "max_tokens": budget.max_tokens,
    }


def _request_snapshot(request: Any) -> dict[str, Any]:
    return {
        "request_id": request.request_id,
        "status": str(request.status),
        "num_prompt_tokens": request.num_prompt_tokens,
        "num_tokens": request.num_tokens,
        "num_computed_tokens": request.num_computed_tokens,
        "num_output_tokens": request.num_output_tokens,
        "num_output_placeholders": request.num_output_placeholders,
        "num_tokens_with_spec": request.num_tokens_with_spec,
        "num_in_flight_tokens": request.num_in_flight_tokens,
        "next_decode_eligible_step": request.next_decode_eligible_step,
        "is_prefill_chunk": request.is_prefill_chunk,
        "max_tokens": request.max_tokens,
    }


def _scheduler_snapshot(scheduler: Any) -> dict[str, Any]:
    return {
        "current_step": scheduler.current_step,
        "running": [_request_snapshot(request) for request in scheduler.running],
        "waiting_count": len(scheduler.waiting),
        "skipped_waiting_count": len(scheduler.skipped_waiting),
        "budget": budget_snapshot(scheduler.mixed_prefill_budget),
        "queue_depth": None,
        "queue_scope": "not_observed",
    }


def _policy_snapshot(scheduler: Any) -> dict[str, Any]:
    config = getattr(scheduler, "scheduler_config", None)
    return {
        "max_num_scheduled_tokens": getattr(
            scheduler, "max_num_scheduled_tokens", None
        ),
        "long_prefill_token_threshold": getattr(
            config, "long_prefill_token_threshold", None
        ),
        "adaptive_long_prefill_threshold": getattr(
            scheduler, "adaptive_long_prefill_threshold", None
        ),
        "balanced_prefill_chunks": getattr(scheduler, "balanced_prefill_chunks", None),
        "need_mamba_block_aligned_split": getattr(
            scheduler, "need_mamba_block_aligned_split", None
        ),
        "mamba_state_block_size": getattr(scheduler, "mamba_state_block_size", None),
        "mamba_state_retention_interval": getattr(
            scheduler, "mamba_state_retention_interval", None
        ),
        "mamba_dense_boundaries_on_contention": getattr(
            scheduler, "mamba_dense_boundaries_on_contention", None
        ),
    }


def _sample_snapshot(sample: Any) -> dict[str, Any] | None:
    if sample is None:
        return None
    return {
        "prefill_tokens": sample.prefill_tokens,
        "decode_tokens": sample.decode_tokens,
        "budget_tokens": sample.budget_tokens,
        "elapsed_ms": _number(sample.elapsed_ms),
        "is_pipeline": sample.is_pipeline,
    }


def _sample_reason(before: Mapping[str, Any], sample: Any) -> str:
    """Describe the unchanged controller guards, without driving its decision."""
    if sample is None:
        return "missing_sample"
    if before["target_ms"] <= 0:
        return "disabled_target"
    if sample.prefill_tokens <= 0:
        return "no_prefill"
    if sample.budget_tokens <= 0:
        return "standalone_no_budget"
    if sample.decode_tokens <= 0 and not sample.is_pipeline:
        return "non_pipeline_without_decode"
    if not math.isfinite(sample.elapsed_ms) or sample.elapsed_ms <= 0:
        return "invalid_elapsed"
    if sample.prefill_tokens < sample.budget_tokens * 0.75 and not (
        sample.is_pipeline and sample.elapsed_ms > before["target_ms"]
    ):
        return "small_fast_tail"
    return "accepted_by_existing_guards"


class MixedPrefillTrace:
    """Write scalar observations; never alter requests, samples, or budgets."""

    def __init__(self, config: MixedPrefillTraceConfig, scheduler: Any) -> None:
        self.config = config
        self._seq = 0
        self._bytes = 0
        self._disabled = False
        directory = Path(config.output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"engine-{os.getpid()}-mixed-prefill-budget.jsonl"
        descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        self._stream = os.fdopen(descriptor, "w", encoding="utf-8", buffering=1)
        pp_size = scheduler.parallel_config.pipeline_parallel_size
        self._record(
            "init",
            None,
            {
                "config": asdict(config),
                "budget": budget_snapshot(scheduler.mixed_prefill_budget),
                "scheduling_policy": _policy_snapshot(scheduler),
                "pipeline_parallel_size": pp_size,
                "max_concurrent_batches": scheduler.vllm_config.max_concurrent_batches,
                "source_file": __file__,
                "reset_policy": "none; observer never resets controller state",
                "queue_depth": None,
                "queue_scope": "not_observed",
            },
        )

    def _line(self, kind: str, step: int | None, payload: dict[str, Any]) -> str:
        return (
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": kind,
                    "seq": self._seq,
                    "engine_pid": os.getpid(),
                    "wall_time_ns": time.time_ns(),
                    "monotonic_ns": time.monotonic_ns(),
                    "scheduled_step": step,
                    **payload,
                },
                ensure_ascii=True,
                allow_nan=False,
                separators=(",", ":"),
            )
            + "\n"
        )

    def _record(self, kind: str, step: int | None, payload: dict[str, Any]) -> None:
        if self._disabled:
            return
        try:
            line = self._line(kind, step, payload)
            size = len(line.encode("utf-8"))
            if self._seq >= self.config.max_events - 1 or (
                self._bytes + size + 1024 > self.config.max_bytes
            ):
                line = self._line(
                    "exhausted",
                    step,
                    {
                        "events_written": self._seq,
                        "bytes_written": self._bytes,
                        "coverage_complete": False,
                    },
                )
                assert self._bytes + len(line.encode("utf-8")) <= self.config.max_bytes
                self._disabled = True
            self._stream.write(line)
            self._bytes += len(line.encode("utf-8"))
            self._seq += 1
            if self._disabled:
                self._stream.close()
        except (OSError, TypeError, ValueError, AssertionError):
            self._disabled = True
            logger.warning(
                "Mixed prefill trace failed; diagnostic coverage is incomplete",
                exc_info=True,
            )

    def begin_schedule(
        self,
        scheduler: Any,
        prefill_remaining: Mapping[str, int],
        resident_tokens: Mapping[str, int],
        has_inflight_decode: bool,
        control_mixed_prefill: bool,
    ) -> None:
        if self._disabled:
            return
        self._record(
            "schedule_before",
            scheduler.current_step,
            {
                **_scheduler_snapshot(scheduler),
                "prefill_remaining": dict(prefill_remaining),
                "resident_tokens": dict(resident_tokens),
                "has_inflight_decode": has_inflight_decode,
                "control_mixed_prefill": control_mixed_prefill,
            },
        )

    def end_schedule(self, scheduler: Any, output: Any) -> None:
        if self._disabled:
            return
        scheduled = dict(output.num_scheduled_tokens)
        positions = {}
        for req_id, count in scheduled.items():
            request = scheduler.requests.get(req_id)
            if request is not None:
                positions[req_id] = {
                    "start": request.num_computed_tokens - count,
                    "end": request.num_computed_tokens,
                }
        self._record(
            "schedule_after",
            output.scheduled_step,
            {
                **_scheduler_snapshot(scheduler),
                "scheduled_tokens": scheduled,
                "scheduled_positions": positions,
                "total_num_scheduled_tokens": output.total_num_scheduled_tokens,
                "mixed_prefill_tokens": output.mixed_prefill_tokens,
                "mixed_decode_tokens": output.mixed_decode_tokens,
                "mixed_prefill_budget": output.mixed_prefill_budget,
                "mixed_prefill_is_pp": output.mixed_prefill_is_pp,
                "preempted_req_ids": sorted(output.preempted_req_ids),
                "finished_req_ids": sorted(output.finished_req_ids),
            },
        )

    def feedback(
        self,
        scheduler: Any,
        output: Any,
        sample: Any,
        before: dict[str, Any],
    ) -> None:
        if self._disabled:
            return
        cost = None
        if sample is not None and sample.prefill_tokens > 0:
            cost = _number(sample.elapsed_ms / sample.prefill_tokens)
        self._record(
            "feedback",
            output.scheduled_step,
            {
                "current_step": scheduler.current_step,
                "sample": _sample_snapshot(sample),
                "before": before,
                "after": budget_snapshot(scheduler.mixed_prefill_budget),
                "sample_cost_ms_per_token": cost,
                "sample_reason": _sample_reason(before, sample),
                "scheduled_tokens": dict(output.num_scheduled_tokens),
            },
        )


def create_mixed_prefill_trace(
    value: object, scheduler: Any
) -> MixedPrefillTrace | None:
    config = mixed_prefill_trace_config(value)
    return None if config is None else MixedPrefillTrace(config, scheduler)
