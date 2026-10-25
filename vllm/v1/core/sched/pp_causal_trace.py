# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in CPU scheduler evidence for synthetic PP phase experiments.

No tensor reads, synchronization or serialization is done inside a model step.
Only request IDs containing an explicit diagnostic tag activate collection.
"""

import atexit
import json
import os
import time
from pathlib import Path
from typing import Any


class PPCausalTrace:
    @classmethod
    def from_config(cls, config: object) -> "PPCausalTrace | None":
        if config is None:
            return None
        if not isinstance(config, dict):
            raise ValueError("pp_causal_trace must be a configuration dictionary")
        directory = config.get("directory")
        if not isinstance(directory, str) or not directory:
            raise ValueError("PP causal trace requires a directory string")
        tags = config.get("tags", ["causal-", "two_phase-"])
        if (
            not isinstance(tags, (list, tuple))
            or not tags
            or not all(isinstance(tag, str) and bool(tag.strip()) for tag in tags)
        ):
            raise ValueError("PP causal trace requires a nonempty request tag")
        limit = config.get("max_events", 200000)
        if type(limit) is not int or limit < 4:
            raise ValueError("PP causal trace event limit must be at least four")
        return cls(Path(directory), tuple(tag.strip() for tag in tags), limit)

    def __init__(self, directory: Path, tags: tuple[str, ...], limit: int):
        self.directory = directory
        self.tags = tags
        self.limit = limit
        self.events: list[dict[str, Any]] = []
        self.pending: set[int] = set()
        self.control_steps: set[int] = set()
        self.active_step: int | None = None
        self.dropped = 0
        self.segment = 0
        self.session = f"{os.getpid()}-{time.monotonic_ns()}"
        self.clock = {
            "wall_time_ns": time.time_ns(),
            "monotonic_ns": time.monotonic_ns(),
            "implementation": time.get_clock_info("monotonic").implementation,
        }
        atexit.register(self.flush)

    def selected(self, request_id: str) -> bool:
        return any(tag in request_id for tag in self.tags)

    def requests(self, scheduler) -> dict[str, Any]:
        return {
            key: request
            for key, request in scheduler.requests.items()
            if self.selected(key)
        }

    @staticmethod
    def snapshot(requests) -> dict[str, dict[str, Any]]:
        fields = (
            "num_prompt_tokens",
            "num_computed_tokens",
            "num_output_tokens",
            "num_tokens_with_spec",
            "num_output_placeholders",
            "num_in_flight_tokens",
            "next_decode_eligible_step",
        )
        return {
            key: {field: int(getattr(request, field)) for field in fields}
            | {"status": str(request.status)}
            for key, request in requests.items()
        }

    def append(self, event: str, scheduler, **values):
        if len(self.events) >= self.limit:
            self.dropped += 1
            return
        self.events.append(
            {
                "event": event,
                "monotonic_ns": time.monotonic_ns(),
                "scheduler_current_step": scheduler.current_step,
                "feedback_depth": getattr(scheduler, "pp_feedback_depth", 1),
                **values,
            }
        )

    def schedule_begin(self, scheduler):
        requests = self.requests(scheduler)
        if not requests:
            self.active_step = None
            return
        self.active_step = scheduler.current_step
        self.append(
            "schedule_begin",
            scheduler,
            requests=self.snapshot(requests),
            running_ids=[r.request_id for r in scheduler.running],
            waiting_count=len(scheduler.waiting),
            traced_pending_steps=sorted(self.pending),
        )

    def schedule_end(self, scheduler, output):
        if self.active_step != scheduler.current_step:
            return
        if output.total_num_scheduled_tokens > 0:
            self.pending.add(output.scheduled_step)
        elif len(self.control_steps) < self.limit:
            self.control_steps.add(output.scheduled_step)
        else:
            self.dropped += 1
        self.append(
            "schedule_end",
            scheduler,
            scheduled_step=output.scheduled_step,
            query_tokens=dict(output.num_scheduled_tokens),
            draft_lengths={
                key: len(value)
                for key, value in output.scheduled_spec_decode_tokens.items()
            },
            total_query_tokens=output.total_num_scheduled_tokens,
            is_model_forward=output.total_num_scheduled_tokens > 0,
            requests=self.snapshot(self.requests(scheduler)),
        )

    def feedback_begin(self, scheduler, output, result):
        if output.scheduled_step not in self.pending | self.control_steps:
            return
        samples = result.sampled_token_ids
        rows = {}
        for request_id, query in output.num_scheduled_tokens.items():
            if not self.selected(request_id):
                continue
            index = result.req_id_to_index.get(request_id)
            tokens = list(samples[index]) if samples and index is not None else []
            draft_count = len(output.scheduled_spec_decode_tokens.get(request_id, ()))
            accepted = len(tokens) - 1 if draft_count and tokens else None
            rows[request_id] = {
                "query_tokens": query,
                "draft_tokens": draft_count,
                "sampled_token_ids_before_stop_trim": tokens,
                "accepted_draft_tokens_before_stop_trim": accepted,
                "rejected_draft_tokens_before_stop_trim": draft_count - accepted
                if accepted is not None
                else None,
            }
        self.append(
            "feedback_begin",
            scheduler,
            scheduled_step=output.scheduled_step,
            results=rows,
            requests=self.snapshot(self.requests(scheduler)),
        )

    def feedback_end(self, scheduler, output):
        if output.scheduled_step not in self.pending | self.control_steps:
            return
        self.append(
            "feedback_end",
            scheduler,
            scheduled_step=output.scheduled_step,
            requests=self.snapshot(self.requests(scheduler)),
        )
        self.pending.discard(output.scheduled_step)
        self.control_steps.discard(output.scheduled_step)
        if not self.pending and not self.requests(scheduler):
            self.flush()

    def flush(self):
        if not self.events and not self.dropped:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"scheduler-{self.session}-{self.segment:04d}.jsonl"
        temporary = path.with_suffix(".jsonl.tmp")
        header = {
            "record": "header",
            "schema_version": 1,
            "session": self.session,
            "clock": self.clock,
            "tags": self.tags,
            "scope": "CPU schedule/feedback; not PP GPU transfer duration",
            "overhead": (
                "Enabled unmatched calls still scan CPU request IDs; no tensor reads"
            ),
        }
        footer = {
            "record": "footer",
            "events": len(self.events),
            "dropped_events": self.dropped,
            "pending_steps": sorted(self.pending),
            "unreturned_control_steps": sorted(self.control_steps),
        }
        with temporary.open("x", encoding="utf-8") as stream:
            for row in (header, *self.events, footer):
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")
        temporary.replace(path)
        self.segment += 1
        self.events.clear()
        self.dropped = 0
