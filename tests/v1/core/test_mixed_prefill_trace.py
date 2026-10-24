# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import json
from types import SimpleNamespace as NS

import pytest

from vllm.v1.core.sched.mixed_prefill import MixedPrefillBudget
from vllm.v1.core.sched.mixed_prefill_trace import (
    MixedPrefillTrace,
    MixedPrefillTraceConfig,
    budget_snapshot,
    create_mixed_prefill_trace,
    mixed_prefill_trace_config,
)
from vllm.v1.outputs import MixedPrefillTiming


def scheduler():
    request = NS(
        request_id="request-a",
        status="RUNNING",
        num_prompt_tokens=8192,
        num_tokens=8192,
        num_computed_tokens=8192,
        num_output_tokens=0,
        num_output_placeholders=1,
        num_tokens_with_spec=8196,
        num_in_flight_tokens=768,
        next_decode_eligible_step=23,
        is_prefill_chunk=False,
        max_tokens=512,
    )
    return NS(
        mixed_prefill_budget=MixedPrefillBudget(768, 250, initial_tokens=32),
        current_step=21,
        running=[request],
        waiting=[],
        skipped_waiting=[],
        requests={request.request_id: request},
        parallel_config=NS(pipeline_parallel_size=7),
        vllm_config=NS(max_concurrent_batches=2),
    )


def output(step=21):
    return NS(
        scheduled_step=step,
        num_scheduled_tokens={"request-a": 768},
        total_num_scheduled_tokens=768,
        mixed_prefill_tokens=768,
        mixed_decode_tokens=0,
        mixed_prefill_budget=0,
        mixed_prefill_is_pp=True,
        preempted_req_ids=set(),
        finished_req_ids=set(),
    )


def records(trace):
    return [json.loads(line) for line in trace.path.read_text().splitlines()]


@pytest.mark.parametrize(
    "value", [None, {}, object(), {"mixed_prefill_budget_trace": None}]
)
def test_default_off_has_no_constructor_side_effects(value):
    assert create_mixed_prefill_trace(value, object()) is None


@pytest.mark.parametrize(
    "changes",
    [
        {"extra": 1},
        {"max_events": True},
        {"max_events": 7},
        {"max_bytes": 4095},
        {"max_bytes": 67108865},
        {"output_dir": "relative"},
    ],
)
def test_invalid_configuration_is_rejected(tmp_path, changes):
    raw = {"output_dir": str(tmp_path), **changes}
    with pytest.raises(ValueError):
        mixed_prefill_trace_config({"mixed_prefill_budget_trace": raw})


def test_observations_do_not_mutate_scheduler_requests_or_output(tmp_path):
    state = scheduler()
    scheduled = output()
    previous = json.dumps(state.__dict__, default=vars, sort_keys=True)
    before_output = copy.deepcopy(scheduled.__dict__)
    trace = MixedPrefillTrace(MixedPrefillTraceConfig(str(tmp_path)), state)
    trace.begin_schedule(state, {"request-a": 0}, {}, True, True)
    trace.end_schedule(state, scheduled)
    assert json.dumps(state.__dict__, default=vars, sort_keys=True) == previous
    assert scheduled.__dict__ == before_output
    data = records(trace)
    assert [row["kind"] for row in data] == [
        "init",
        "schedule_before",
        "schedule_after",
    ]
    assert data[1]["has_inflight_decode"] and not data[1]["resident_tokens"]
    assert data[2]["scheduled_positions"] == {"request-a": {"start": 7424, "end": 8192}}
    assert data[2]["running"][0]["num_output_placeholders"] == 1
    assert data[2]["running"][0]["next_decode_eligible_step"] == 23
    assert data[2]["queue_depth"] is None


@pytest.mark.parametrize(
    "sample,reason",
    [
        (None, "missing_sample"),
        (MixedPrefillTiming(768, 0, 0, 500, is_pipeline=True), "standalone_no_budget"),
        (MixedPrefillTiming(8, 0, 32, 100, is_pipeline=True), "small_fast_tail"),
        (
            MixedPrefillTiming(16, 0, 32, 309.4, is_pipeline=True),
            "accepted_by_existing_guards",
        ),
        (MixedPrefillTiming(0, 1, 32, 1, is_pipeline=True), "no_prefill"),
        (
            MixedPrefillTiming(32, 0, 32, 100, is_pipeline=False),
            "non_pipeline_without_decode",
        ),
        (MixedPrefillTiming(32, 1, 32, float("nan")), "invalid_elapsed"),
    ],
)
def test_feedback_preserves_original_budget_update_and_step(tmp_path, sample, reason):
    state = scheduler()
    scheduled = output(step=19)
    trace = MixedPrefillTrace(MixedPrefillTraceConfig(str(tmp_path)), state)
    before = budget_snapshot(state.mixed_prefill_budget)
    reference = copy.deepcopy(state.mixed_prefill_budget)
    state.mixed_prefill_budget.update(sample)
    reference.update(sample)
    trace.feedback(state, scheduled, sample, before)
    assert vars(state.mixed_prefill_budget) == vars(reference)
    row = records(trace)[-1]
    assert row["scheduled_step"] == 19 and row["current_step"] == 21
    assert row["sample_reason"] == reason
    assert row["before"] == before
    assert row["after"] == budget_snapshot(reference)


def test_budget_16_quantization_dead_zone_is_observed_not_repaired(tmp_path):
    state = scheduler()
    trace = MixedPrefillTrace(MixedPrefillTraceConfig(str(tmp_path)), state)
    for index in range(31):
        sample = MixedPrefillTiming(
            16,
            0,
            32 if index == 0 else 16,
            309.4 if index == 0 else 200,
            is_pipeline=True,
        )
        before = budget_snapshot(state.mixed_prefill_budget)
        state.mixed_prefill_budget.update(sample)
        trace.feedback(state, output(step=index + 1), sample, before)
    assert state.mixed_prefill_budget.tokens == 16
    assert records(trace)[-1]["after"]["tokens"] == 16


def test_event_limit_has_one_exhausted_marker_and_preserves_policy(tmp_path):
    state = scheduler()
    trace = MixedPrefillTrace(
        MixedPrefillTraceConfig(str(tmp_path), max_events=8), state
    )
    for _ in range(20):
        trace.end_schedule(state, output())
    data = records(trace)
    assert len(data) == 8 and data[-1]["kind"] == "exhausted"
    assert not data[-1]["coverage_complete"]
    assert state.mixed_prefill_budget.tokens == 32


def test_byte_limit_and_existing_file_are_fail_closed(tmp_path):
    state = scheduler()
    config = MixedPrefillTraceConfig(str(tmp_path), max_bytes=4096)
    trace = MixedPrefillTrace(config, state)
    for _ in range(30):
        trace.end_schedule(state, output())
    assert trace.path.stat().st_size <= 4096
    assert records(trace)[-1]["kind"] == "exhausted"
    original = trace.path.read_bytes()
    with pytest.raises(FileExistsError):
        MixedPrefillTrace(config, state)
    assert trace.path.read_bytes() == original


def test_async_final_prefill_captures_updated_placeholders_and_eligibility(
    tmp_path,
    monkeypatch,
):
    from tests.v1.core.test_mixed_prefill_budget import _pp_scheduler
    from tests.v1.core.test_scheduler import create_requests

    state = _pp_scheduler(monkeypatch, max_num_batched_tokens=768)
    trace = MixedPrefillTrace(MixedPrefillTraceConfig(str(tmp_path)), state)
    state._mixed_prefill_trace = trace
    request = create_requests(1, num_tokens=64, req_ids=["observed-final"])[0]
    state.add_request(request)
    step = state.schedule()
    row = records(trace)[-1]
    actual = next(
        item for item in row["running"] if item["request_id"] == request.request_id
    )
    assert row["kind"] == "schedule_after"
    assert actual["num_output_tokens"] == 0
    assert actual["num_output_placeholders"] == 1
    assert actual["next_decode_eligible_step"] == step.scheduled_step + 2
    assert row["scheduled_positions"][request.request_id] == {"start": 0, "end": 64}
