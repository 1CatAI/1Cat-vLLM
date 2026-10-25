# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Combined real-scheduler contracts; no model execution or GPU timing claims."""

import json

import pytest

from tests.v1.core import utils
from tests.v1.core.test_pp_decode_stagger import (
    FeedbackQueue,
    local_model,
    make_scheduler,
    prime_pair,
)
from vllm.config import SchedulerConfig, VllmConfig
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.pp_causal_trace import PPCausalTrace
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import RequestStatus

pytestmark = pytest.mark.cpu_test
assert local_model is not None  # Metadata-only fixture, not a downloaded model.


def _state(request, scheduler):
    return (
        request.num_computed_tokens,
        request.num_in_flight_tokens,
        request.num_output_placeholders,
        request.next_decode_eligible_step,
        tuple(request.spec_token_ids),
        tuple(
            tuple(group)
            for group in scheduler.kv_cache_manager.get_block_ids(request.request_id)
        ),
    )


def _quota_prefill(scheduler):
    requests = utils.create_requests(
        num_requests=3,
        num_tokens=1792,
        max_tokens=128,
        req_ids=["ppphase-a", "ppphase-b", "ppphase-c"],
    )
    queue = FeedbackQueue(scheduler, accepted=(1, 2, 5))
    scheduler.add_request(requests[0])
    assert queue.step().num_scheduled_tokens == {"ppphase-a": 768}
    for request in requests[1:]:
        scheduler.add_request(request)
    assert queue.step().num_scheduled_tokens == dict.fromkeys(
        [r.request_id for r in requests], 256
    )
    assert scheduler._early_prefill_quota.members == tuple(
        r.request_id for r in requests
    )
    for _ in range(4):
        output = queue.step()
        assert output.num_scheduled_tokens == {
            "ppphase-a": 128,
            "ppphase-b": 320,
            "ppphase-c": 320,
        }
    assert [r.num_computed_tokens for r in requests] == [1536] * 3
    output = queue.step()
    assert output.num_scheduled_tokens == dict.fromkeys(
        [r.request_id for r in requests], 256
    )
    assert [r.num_computed_tokens for r in requests] == [1792] * 3
    assert [r.num_output_tokens for r in requests] == [0] * 3
    assert [r.num_output_placeholders for r in requests] == [1] * 3
    assert all(
        r.next_decode_eligible_step == output.scheduled_step + 2 for r in requests
    )
    assert not scheduler._early_prefill_quota.members
    assert scheduler._early_prefill_quota.retired == frozenset(
        r.request_id for r in requests
    )
    return queue, requests


def test_both_features_are_default_off_and_legacy_observer_is_none():
    config = SchedulerConfig.default_factory()
    assert not config.pp_prefill_quota_compensation and not config.pp_decode_stagger
    legacy = object.__new__(Scheduler)
    assert legacy._pp_causal_trace is None
    assert legacy._early_prefill_quota is None


@pytest.mark.parametrize("stagger", [False, True])
def test_quota_final_prefill_then_three_to_two_decode_preserves_feedback(
    monkeypatch, local_model, stagger
):
    scheduler = make_scheduler(
        monkeypatch,
        local_model,
        enabled=stagger,
        quota=True,
        adaptive=True,
        budget=768,
    )
    queue, requests = _quota_prefill(scheduler)
    # Final-prefill placeholders retain the original two-step feedback delay.
    assert not queue.step().num_scheduled_tokens
    assert [r.num_output_tokens for r in requests] == [1] * 3
    assert [r.num_output_placeholders for r in requests] == [0] * 3
    scheduler.finish_requests(requests[2].request_id, RequestStatus.FINISHED_ABORTED)
    deferred = _state(requests[1], scheduler)
    first = queue.step()
    assert len(first.num_scheduled_tokens) == (1 if stagger else 2)
    if stagger:
        assert _state(requests[1], scheduler) == deferred
    for index in range(5):
        output = queue.step()
        if stagger:
            assert list(output.num_scheduled_tokens) == [
                requests[(index + 1) % 2].request_id
            ]
        else:
            assert len(output.num_scheduled_tokens) == (0 if index % 2 == 0 else 2)
    assert not scheduler._early_prefill_quota.members


@pytest.mark.parametrize("skipped", [False, True])
@pytest.mark.parametrize("stagger", [False, True])
def test_quota_on_waiting_prefill_keeps_both_resident_decoders(
    monkeypatch, local_model, skipped, stagger
):
    scheduler = make_scheduler(
        monkeypatch,
        local_model,
        enabled=stagger,
        quota=True,
        adaptive=True,
        budget=768,
    )
    scheduler.mixed_prefill_enabled = True
    queue, residents = prime_pair(scheduler)
    scheduler.mixed_prefill_budget.tokens = 16
    incoming = utils.create_requests(1, num_tokens=1536, req_ids=["incoming"])[0]
    scheduler.add_request(incoming)
    if skipped:
        assert scheduler.waiting.pop_request() is incoming
        scheduler.skipped_waiting.add_request(incoming)
    output = queue.step()
    assert all(r.request_id in output.num_scheduled_tokens for r in residents)
    assert output.num_scheduled_tokens[incoming.request_id] == 16
    assert output.mixed_prefill_budget == 16
    assert output.mixed_decode_tokens == 10
    assert not scheduler._early_prefill_quota.members


def test_new_prefill_after_phase_split_preserves_inflight_decoder_deadline(
    monkeypatch, local_model
):
    scheduler = make_scheduler(
        monkeypatch,
        local_model,
        quota=True,
        adaptive=True,
        budget=768,
    )
    scheduler.mixed_prefill_enabled = True
    scheduler.mixed_prefill_budget.tokens = 16
    queue, residents = prime_pair(scheduler)
    assert list(queue.step().num_scheduled_tokens) == [residents[0].request_id]
    blocked_until = residents[0].next_decode_eligible_step
    incoming = utils.create_requests(1, num_tokens=1536, req_ids=["incoming"])[0]
    scheduler.add_request(incoming)
    output = queue.step()
    assert output.scheduled_step < blocked_until
    assert output.num_scheduled_tokens == {residents[1].request_id: 5, "incoming": 16}
    assert residents[0].next_decode_eligible_step == blocked_until
    assert output.mixed_prefill_budget == output.mixed_prefill_tokens == 16
    assert output.mixed_decode_tokens == 5
    assert not scheduler._early_prefill_quota.members


@pytest.mark.parametrize("failure", ["preempt", "zero"])
def test_quota_on_does_not_turn_failed_allocation_into_stagger_credit(
    monkeypatch, local_model, failure
):
    scheduler = make_scheduler(
        monkeypatch, local_model, quota=True, adaptive=True, budget=768
    )
    queue, requests = prime_pair(scheduler)
    if failure == "preempt":
        original = scheduler.kv_cache_manager.allocate_slots
        attempts = 0

        def allocate(*args, **kwargs):
            nonlocal attempts
            attempts += 1
            return None if attempts == 1 else original(*args, **kwargs)

        monkeypatch.setattr(scheduler.kv_cache_manager, "allocate_slots", allocate)
        assert list(queue.step().num_scheduled_tokens) == [requests[0].request_id]
        assert requests[1].status == RequestStatus.PREEMPTED
        assert requests[1].request_id in queue.step().num_scheduled_tokens
    else:
        original_balance = scheduler._balanced_prefill_chunk
        monkeypatch.setattr(
            scheduler,
            "_balanced_prefill_chunk",
            lambda request, *args: (
                0 if request is requests[0] else original_balance(request, *args)
            ),
        )
        assert list(queue.step().num_scheduled_tokens) == [requests[1].request_id]
    assert not scheduler._early_prefill_quota.members


def test_post_final_pause_respects_decode_deadline_with_quota_enabled(
    monkeypatch, local_model
):
    scheduler = make_scheduler(
        monkeypatch,
        local_model,
        quota=True,
        adaptive=True,
        budget=768,
    )
    queue, requests = _quota_prefill(scheduler)
    scheduler.set_pause_state(PauseState.PAUSED_ALL)
    assert not queue.step().num_scheduled_tokens
    scheduler.set_pause_state(PauseState.UNPAUSED)
    scheduler.finish_requests(requests[2].request_id, RequestStatus.FINISHED_ABORTED)
    assert list(queue.step().num_scheduled_tokens) == [requests[0].request_id]
    assert list(queue.step().num_scheduled_tokens) == [requests[1].request_id]


@pytest.mark.parametrize("stagger", [False, True])
def test_both_observers_bind_complete_async_and_quota_state(
    monkeypatch, local_model, tmp_path, stagger
):
    def config(**kwargs):
        kwargs["additional_config"] = {
            "pp_causal_trace": {
                "directory": str(tmp_path / "pp"),
                "tags": ["ppphase-"],
                "max_events": 5000,
            },
            "mixed_prefill_budget_trace": {
                "output_dir": str(tmp_path / "mixed"),
                "max_events": 5000,
                "max_bytes": 1048576,
            },
        }
        return VllmConfig(**kwargs)

    monkeypatch.setattr(utils, "VllmConfig", config)
    scheduler = make_scheduler(
        monkeypatch,
        local_model,
        enabled=stagger,
        quota=True,
        adaptive=True,
        budget=768,
    )
    order = []
    quota = scheduler._early_prefill_quota
    original_observe = quota.observe
    original_pp_end = scheduler._pp_causal_trace.schedule_end
    original_mixed_end = scheduler._mixed_prefill_trace.end_schedule

    def observed(*args, **kwargs):
        original_observe(*args, **kwargs)
        order.append((kwargs["step"], "quota", bool(quota.members)))

    def mixed_end(engine, output):
        original_mixed_end(engine, output)
        order.append((output.scheduled_step, "mixed", bool(quota.members)))

    def pp_end(engine, output):
        original_pp_end(engine, output)
        order.append((output.scheduled_step, "pp", bool(quota.members)))

    monkeypatch.setattr(quota, "observe", observed)
    monkeypatch.setattr(scheduler._mixed_prefill_trace, "end_schedule", mixed_end)
    monkeypatch.setattr(scheduler._pp_causal_trace, "schedule_end", pp_end)
    queue, requests = _quota_prefill(scheduler)
    assert order[-3:] == [(7, "quota", False), (7, "mixed", False), (7, "pp", False)]
    for _ in range(3):
        queue.step()
        trace = scheduler._pp_causal_trace
        last = [event for event in trace.events if event["event"] == "feedback_end"][-1]
        assert last["requests"] == PPCausalTrace.snapshot(trace.requests(scheduler))
        mixed = [
            json.loads(line)
            for line in scheduler._mixed_prefill_trace.path.read_text().splitlines()
        ]
        feedback = [row for row in mixed if row["kind"] == "feedback"][-1]
        assert feedback["scheduled_step"] == last["scheduled_step"]
    scheduler.finish_requests(
        [r.request_id for r in requests], RequestStatus.FINISHED_ABORTED
    )
    queue.step()
    queue.step()
    scheduler._pp_causal_trace.flush()
