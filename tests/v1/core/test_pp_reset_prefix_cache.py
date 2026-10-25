# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Force-reset invalidates scheduled frames, not speculative token counts."""

import time
from pathlib import Path

import pytest
from transformers import OPTConfig

from tests.v1.core import utils
from vllm.config import SchedulerConfig
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def local_model(tmp_path: Path) -> str:
    model = tmp_path / "metadata-only"
    OPTConfig(
        hidden_size=32,
        num_hidden_layers=12,
        num_attention_heads=4,
        ffn_dim=64,
        max_position_embeddings=256,
        vocab_size=512,
        architectures=["OPTForCausalLM"],
    ).save_pretrained(model)
    return str(model)


def make_scheduler(
    monkeypatch, local_model, depth=4, *, spec_steps=4, budget=64, v2=True
):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1" if v2 else "0")
    monkeypatch.setattr("vllm.config.parallel.current_platform.device_count", lambda: 7)

    def scheduler_config(**kwargs):
        return SchedulerConfig(pp_async_queue_depth=depth if v2 else None, **kwargs)

    monkeypatch.setattr(utils, "SchedulerConfig", scheduler_config)
    scheduler = utils.create_scheduler(
        model=local_model,
        skip_tokenizer_init=True,
        async_scheduling=True,
        pipeline_parallel_size=7 if v2 else 1,
        max_num_seqs=8,
        max_num_batched_tokens=budget,
        max_model_len=128,
        enable_prefix_caching=True,
    )
    assert isinstance(scheduler, AsyncScheduler)
    # As in test_pp_async_queue_depth, exercise the method-independent CPU
    # scheduling contract without constructing/loading a speculative model.
    scheduler.num_spec_tokens = spec_steps
    scheduler._spec_token_placeholders = (-1,) * spec_steps
    return scheduler


def result(output: SchedulerOutput, tokens: list[int]) -> ModelRunnerOutput:
    req_ids = list(output.num_scheduled_tokens)
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={rid: index for index, rid in enumerate(req_ids)},
        sampled_token_ids=[tokens.copy() for _ in req_ids],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def add_prefilled_requests(scheduler):
    requests = utils.create_requests(
        num_requests=3, num_tokens=8, max_tokens=64, ignore_eos=True
    )
    for request in requests:
        scheduler.add_request(request)
    output = scheduler.schedule()
    assert output.scheduled_step == scheduler.current_step == 1
    scheduler.update_from_output(output, result(output, [101]))
    assert all(request.num_in_flight_tokens == 0 for request in requests)
    return requests


def next_frame(scheduler):
    for _ in range(8):
        output = scheduler.schedule()
        assert output.scheduled_step == scheduler.current_step
        if output.total_num_scheduled_tokens:
            return output
        scheduler.update_from_output(output, result(output, []))
    raise AssertionError("decode never became eligible")


def state(requests):
    return [
        (
            request.num_computed_tokens,
            request.num_output_placeholders,
            tuple(request.output_token_ids),
            tuple(request.spec_token_ids),
        )
        for request in requests
    ]


@pytest.mark.parametrize("depth", [2, 4, 7])
@pytest.mark.parametrize("accepted_length", [1, 2, 5])
def test_reset_reschedule_drops_old_frame_before_rejection_accounting(
    monkeypatch, local_model, depth, accepted_length
):
    scheduler = make_scheduler(monkeypatch, local_model, depth)
    requests = add_prefilled_requests(scheduler)
    old = next_frame(scheduler)
    assert set(old.num_scheduled_tokens.values()) == {5}
    assert all(request.num_output_placeholders == 5 for request in requests)
    assert scheduler.reset_prefix_cache(reset_running_requests=True)
    assert all(
        request.last_prefix_cache_reset_step == old.scheduled_step
        for request in requests
    )
    fresh = scheduler.schedule()
    assert fresh.scheduled_step > old.scheduled_step
    assert len(fresh.num_scheduled_tokens) == 3
    before = state(requests)
    in_flight = [request.num_in_flight_tokens for request in requests]

    scheduler.update_from_output(
        old, result(old, list(range(201, 201 + accepted_length)))
    )
    assert state(requests) == before
    assert [request.num_in_flight_tokens for request in requests] == [
        n - 5 for n in in_flight
    ]
    scheduler.update_from_output(fresh, result(fresh, [301]))
    assert all(list(request.output_token_ids) == [101, 301] for request in requests)
    assert all(request.num_output_placeholders == 0 for request in requests)
    assert all(request.num_in_flight_tokens == 0 for request in requests)

    # Post-reset speculative acceptance still performs ordinary bookkeeping.
    normal = next_frame(scheduler)
    scheduler.update_from_output(normal, result(normal, [401, 402]))
    assert all(
        list(request.output_token_ids) == [101, 301, 401, 402] for request in requests
    )
    assert all(request.num_computed_tokens == 11 for request in requests)
    assert all(request.num_output_placeholders == 0 for request in requests)


@pytest.mark.parametrize("depth", [2, 4, 7])
def test_reset_discards_empty_partial_prefill_frame(monkeypatch, local_model, depth):
    scheduler = make_scheduler(monkeypatch, local_model, depth, budget=8)
    (request,) = utils.create_requests(num_requests=1, num_tokens=20, max_tokens=64)
    scheduler.add_request(request)
    old = scheduler.schedule()
    assert old.total_num_scheduled_tokens == 8
    assert request.is_prefill_chunk and request.num_output_placeholders == 0
    assert scheduler.reset_prefix_cache(reset_running_requests=True)
    fresh = scheduler.schedule()
    before = state([request])
    scheduler.update_from_output(old, result(old, []))
    assert state([request]) == before
    assert request.num_in_flight_tokens == 8
    scheduler.update_from_output(fresh, result(fresh, []))
    second = scheduler.schedule()
    scheduler.update_from_output(second, result(second, []))
    final = scheduler.schedule()
    assert final.total_num_scheduled_tokens == 4
    scheduler.update_from_output(final, result(final, [301]))
    assert list(request.output_token_ids) == [301]
    assert request.num_in_flight_tokens == 0


@pytest.mark.parametrize("depth", [2, 4, 7])
def test_repeated_reset_keeps_both_older_generations_stale(
    monkeypatch, local_model, depth
):
    scheduler = make_scheduler(monkeypatch, local_model, depth)
    requests = add_prefilled_requests(scheduler)
    first = next_frame(scheduler)
    assert scheduler.reset_prefix_cache(reset_running_requests=True)
    second = scheduler.schedule()
    assert scheduler.reset_prefix_cache(reset_running_requests=True)
    current = scheduler.schedule()
    before = state(requests)
    scheduler.update_from_output(first, result(first, [201, 202]))
    assert state(requests) == before
    scheduler.update_from_output(second, result(second, [301]))
    assert state(requests) == before
    scheduler.update_from_output(current, result(current, [401]))
    assert all(list(request.output_token_ids) == [101, 401] for request in requests)
    assert all(request.num_in_flight_tokens == 0 for request in requests)


@pytest.mark.parametrize("spec_steps", [0, 4])
def test_reset_discards_multiple_in_flight_frames(monkeypatch, local_model, spec_steps):
    scheduler = make_scheduler(
        monkeypatch, local_model, v2=False, spec_steps=spec_steps
    )
    requests = add_prefilled_requests(scheduler)
    first = scheduler.schedule()
    second = scheduler.schedule()
    assert len(first.num_scheduled_tokens) == len(second.num_scheduled_tokens) == 3
    assert scheduler.reset_prefix_cache(reset_running_requests=True)
    current = scheduler.schedule()
    before = state(requests)
    scheduler.update_from_output(first, result(first, [201]))
    scheduler.update_from_output(second, result(second, [202]))
    assert state(requests) == before
    scheduler.update_from_output(current, result(current, [301]))
    assert all(list(request.output_token_ids) == [101, 301] for request in requests)
    assert all(request.num_in_flight_tokens == 0 for request in requests)


def test_ordinary_preemption_still_accepts_in_flight_output(monkeypatch, local_model):
    scheduler = make_scheduler(monkeypatch, local_model)
    requests = add_prefilled_requests(scheduler)
    output = next_frame(scheduler)
    request = scheduler.running.pop()
    scheduler._preempt_request(request, time.monotonic())
    assert request.last_prefix_cache_reset_step == 0
    scheduler.update_from_output(output, result(output, [201, 202]))
    assert request.status == RequestStatus.PREEMPTED
    assert all(list(req.output_token_ids) == [101, 201, 202] for req in requests)
    assert all(req.num_output_placeholders == 0 for req in requests)
    assert all(req.num_in_flight_tokens == 0 for req in requests)


def test_cancellation_and_reset_do_not_restore_finished_request(
    monkeypatch, local_model
):
    scheduler = make_scheduler(monkeypatch, local_model)
    requests = add_prefilled_requests(scheduler)
    old = next_frame(scheduler)
    cancelled = requests[0]
    scheduler.finish_requests([cancelled.request_id], RequestStatus.FINISHED_ABORTED)
    assert scheduler.reset_prefix_cache(reset_running_requests=True)
    current = scheduler.schedule()
    scheduler.update_from_output(old, result(old, [201]))
    scheduler.update_from_output(current, result(current, [301]))
    assert cancelled.request_id not in scheduler.requests
    assert cancelled.status == RequestStatus.FINISHED_ABORTED
    assert list(cancelled.output_token_ids) == [101]
    assert all(list(req.output_token_ids) == [101, 301] for req in requests[1:])


@pytest.mark.parametrize("depth", [2, 4, 7])
def test_reset_invalidates_all_workers_before_partial_resumption(
    monkeypatch, local_model, depth
):
    scheduler = make_scheduler(monkeypatch, local_model, depth)
    requests = add_prefilled_requests(scheduler)
    old = next_frame(scheduler)
    assert scheduler.reset_prefix_cache(reset_running_requests=True)
    # Only one request fits. The other two still own worker-side request slots
    # and old feedback, despite their physical cache blocks already being free.
    scheduler.max_num_scheduled_tokens = 8
    fresh = scheduler.schedule()
    expected = {request.request_id for request in requests}
    assert len(fresh.scheduled_new_reqs) == 1
    assert fresh.prefix_cache_reset_req_ids == expected
    assert not fresh.preempted_req_ids
    scheduler.update_from_output(old, result(old, [201, 202]))
    assert all(list(request.output_token_ids) == [101] for request in requests)

    following = scheduler.schedule()
    assert following.prefix_cache_reset_req_ids is None
    # Dispatching a later batch must not mutate the first batch's snapshot.
    assert fresh.prefix_cache_reset_req_ids == expected


def test_reset_invalidation_survives_empty_steps_repeated_reset_and_cancellation(
    monkeypatch, local_model
):
    scheduler = make_scheduler(monkeypatch, local_model)
    requests = add_prefilled_requests(scheduler)
    assert scheduler.reset_prefix_cache(reset_running_requests=True)
    # A repeated reset before schedule() sees no running requests, but must
    # retain the first reset's pending notifications.
    assert scheduler.reset_prefix_cache(reset_running_requests=True)
    scheduler.finish_requests([requests[0].request_id], RequestStatus.FINISHED_ABORTED)
    scheduler._pause_state = PauseState.PAUSED_ALL
    empty = scheduler.schedule()
    expected = {request.request_id for request in requests}
    assert empty.total_num_scheduled_tokens == 0
    assert empty.prefix_cache_reset_req_ids == expected
    assert requests[0].request_id in empty.finished_req_ids
    assert scheduler.schedule().prefix_cache_reset_req_ids is None
    assert empty.prefix_cache_reset_req_ids == expected

    scheduler._pause_state = PauseState.UNPAUSED
    resumed = scheduler.schedule()
    assert resumed.prefix_cache_reset_req_ids is None
    assert len(resumed.scheduled_new_reqs) == 2
    assert scheduler.reset_prefix_cache(reset_running_requests=True)
    new_reset = scheduler.schedule()
    assert new_reset.prefix_cache_reset_req_ids == {
        request.request_id for request in requests[1:]
    }
    assert empty.prefix_cache_reset_req_ids == expected
