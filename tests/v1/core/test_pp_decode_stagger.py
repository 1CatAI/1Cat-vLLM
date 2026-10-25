# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU state-machine tests using the actual scheduler and feedback updates."""

from collections import deque
from pathlib import Path

import pytest
from transformers import OPTConfig

from tests.v1.core import utils
from vllm.config import SchedulerConfig
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import SchedulingPolicy
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
from vllm.v1.request import RequestStatus

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def local_model(tmp_path: Path) -> str:
    model = tmp_path / "metadata-only"
    OPTConfig(
        hidden_size=32,
        num_hidden_layers=14,
        num_attention_heads=4,
        ffn_dim=64,
        max_position_embeddings=2048,
        vocab_size=65536,
        architectures=["OPTForCausalLM"],
    ).save_pretrained(model)
    return str(model)


def make_scheduler(
    monkeypatch,
    local_model,
    *,
    enabled=True,
    spec=4,
    budget=128,
    quota=False,
    adaptive=False,
):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setattr("vllm.config.parallel.current_platform.device_count", lambda: 7)

    def config(**kwargs):
        return SchedulerConfig(
            pp_async_queue_depth=2,
            pp_decode_stagger=enabled,
            pp_prefill_quota_compensation=quota,
            **kwargs,
        )

    monkeypatch.setattr(utils, "SchedulerConfig", config)
    scheduler = utils.create_scheduler(
        model=local_model,
        skip_tokenizer_init=True,
        async_scheduling=True,
        pipeline_parallel_size=7,
        max_num_seqs=8,
        max_num_batched_tokens=budget,
        long_prefill_token_threshold=128 if adaptive else 0,
        long_prefill_token_threshold_adaptive=adaptive,
        max_model_len=2048,
    )
    assert isinstance(scheduler, AsyncScheduler)
    assert scheduler.pp_feedback_depth == 2
    scheduler.num_spec_tokens = spec
    scheduler._spec_token_placeholders = (-1,) * spec
    return scheduler


class FeedbackQueue:
    """Simulate enqueue-before-oldest-feedback with two outstanding frames.

    Model output is deterministic per request, independent of batch grouping.
    Empty control frames count toward queue depth, like the real engine loop.
    The model itself is not mocked into the scheduler's scheduling decisions.
    """

    def __init__(self, scheduler, *, accepted=(1,)):
        self.scheduler = scheduler
        self.frames: deque[tuple[SchedulerOutput, dict[str, bool]]] = deque()
        self.accepted = accepted
        self.eos_next: set[str] = set()
        self.records: list[dict[str, int]] = []
        self.feedback_counts: dict[str, int] = {}
        self.accepted_sizes: set[int] = set()

    def step(self):
        scheduler = self.scheduler
        before = {r.request_id: r.next_decode_eligible_step for r in scheduler.running}
        inflight = {
            rid
            for output, _partial in self.frames
            for rid in output.num_scheduled_tokens
        }
        output = scheduler.schedule()
        for rid in output.num_scheduled_tokens:
            # A partial prefill may continue before its prior chunk's feedback;
            # resident decoding must retain the existing feedback dependency.
            if rid in before:
                assert scheduler.current_step >= before[rid]
                if scheduler.requests[rid].num_output_tokens:
                    assert rid not in inflight
        partial = {
            rid: scheduler.requests[rid].is_prefill_chunk
            for rid in output.num_scheduled_tokens
        }
        self.frames.append((output, partial))
        self.records.append(dict(output.num_scheduled_tokens))
        if len(self.frames) >= scheduler.pp_feedback_depth:
            old, partial = self.frames.popleft()
            ids = list(old.num_scheduled_tokens)
            samples: list[list[int]] = []
            for rid in ids:
                request = scheduler.requests.get(rid)
                if partial[rid] or request is None or request.is_finished():
                    samples.append([])
                    continue
                if rid in self.eos_next:
                    samples.append([utils.EOS_TOKEN_ID])
                    self.eos_next.remove(rid)
                    continue
                count = min(
                    1 + len(old.scheduled_spec_decode_tokens.get(rid, ())),
                    self.accepted[
                        self.feedback_counts.get(rid, 0) % len(self.accepted)
                    ],
                )
                self.feedback_counts[rid] = self.feedback_counts.get(rid, 0) + 1
                self.accepted_sizes.add(count)
                samples.append(
                    [1000 + request.num_output_tokens + i for i in range(count)]
                )
            scheduler.update_from_output(
                old,
                ModelRunnerOutput(
                    req_ids=ids,
                    req_id_to_index={rid: i for i, rid in enumerate(ids)},
                    sampled_token_ids=samples,
                    logprobs=None,
                    prompt_logprobs_dict={},
                    pooler_output=[],
                ),
            )
        return output


def prime_pair(scheduler, *, count=2, accepted=(1,)):
    requests = utils.create_requests(num_requests=count, num_tokens=8, max_tokens=128)
    for request in requests:
        scheduler.add_request(request)
    queue = FeedbackQueue(scheduler, accepted=accepted)
    assert len(queue.step().num_scheduled_tokens) == count
    assert not queue.step().num_scheduled_tokens
    assert all(request.num_output_tokens == 1 for request in requests)
    return queue, requests


@pytest.mark.parametrize("spec", [0, 1, 4])
@pytest.mark.parametrize("accepted", [(1,), (1, 2, 5)])
def test_same_phase_staggers_without_advancing_feedback(
    monkeypatch, local_model, spec, accepted
):
    scheduler = make_scheduler(monkeypatch, local_model, spec=spec)
    queue, requests = prime_pair(scheduler, accepted=accepted)
    ids = [r.request_id for r in requests]
    deferred = requests[1]

    def deferred_state():
        return (
            deferred.num_computed_tokens,
            deferred.num_output_placeholders,
            deferred.next_decode_eligible_step,
            tuple(deferred.spec_token_ids),
            tuple(tuple(g) for g in scheduler.kv_cache_manager.get_block_ids(ids[1])),
        )

    before = deferred_state()
    assert list(queue.step().num_scheduled_tokens) == [ids[0]]
    assert deferred_state() == before
    for i in range(9):
        output = queue.step()
        assert list(output.num_scheduled_tokens) == [ids[(i + 1) % 2]]
        assert all(r.num_output_placeholders >= 0 for r in requests)
    if spec == 4 and accepted == (1, 2, 5):
        assert queue.accepted_sizes == {1, 2, 5}


def test_flag_off_retains_two_request_batch_and_empty_control_step(
    monkeypatch, local_model
):
    scheduler = make_scheduler(monkeypatch, local_model, enabled=False)
    queue, requests = prime_pair(scheduler)
    ids = {r.request_id for r in requests}
    for _ in range(4):
        assert set(queue.step().num_scheduled_tokens) == ids
        assert not queue.step().num_scheduled_tokens


def test_variable_draft_rows_preserve_phase_and_rejection_accounting(
    monkeypatch, local_model
):
    scheduler = make_scheduler(monkeypatch, local_model)
    queue, requests = prime_pair(scheduler, accepted=(1, 2, 5))
    observed_sizes = set()
    for i in range(12):
        request = requests[i % 2]
        draft_count = (0, 1, 4)[i % 3]
        scheduler.update_draft_token_ids(
            DraftTokenIds(
                req_ids=[request.request_id],
                draft_token_ids=[list(range(draft_count))],
            )
        )
        output = queue.step()
        assert output.num_scheduled_tokens == {request.request_id: draft_count + 1}
        observed_sizes.add(draft_count + 1)
        assert all(r.num_output_placeholders >= 0 for r in requests)
    assert observed_sizes == {1, 2, 5}


def test_pause_then_resume_can_reestablish_two_phases(monkeypatch, local_model):
    scheduler = make_scheduler(monkeypatch, local_model)
    queue, requests = prime_pair(scheduler)
    queue.step()
    scheduler._pause_state = PauseState.PAUSED_ALL
    for _ in range(3):
        assert not queue.step().num_scheduled_tokens
    scheduler._pause_state = PauseState.UNPAUSED
    for i in range(6):
        assert list(queue.step().num_scheduled_tokens) == [requests[i % 2].request_id]


def test_already_staggered_requests_are_not_delayed(monkeypatch, local_model):
    scheduler = make_scheduler(monkeypatch, local_model)
    a, b = utils.create_requests(num_requests=2, num_tokens=8, max_tokens=128)
    queue = FeedbackQueue(scheduler)
    scheduler.add_request(a)
    assert list(queue.step().num_scheduled_tokens) == [a.request_id]
    scheduler.add_request(b)
    assert list(queue.step().num_scheduled_tokens) == [b.request_id]
    for i in range(8):
        assert list(queue.step().num_scheduled_tokens) == [(a, b)[i % 2].request_id]


@pytest.mark.parametrize("finish", ["abort", "eos"])
def test_three_to_two_transition_is_reconsidered(monkeypatch, local_model, finish):
    scheduler = make_scheduler(monkeypatch, local_model)
    queue, requests = prime_pair(scheduler, count=3)
    a, b, c = requests
    if finish == "abort":
        scheduler.finish_requests(c.request_id, RequestStatus.FINISHED_ABORTED)
    else:
        queue.eos_next.add(c.request_id)
        assert len(queue.step().num_scheduled_tokens) == 3
        assert not queue.step().num_scheduled_tokens
        assert c.is_finished()
    for i in range(6):
        assert list(queue.step().num_scheduled_tokens) == [(a, b)[i % 2].request_id]


@pytest.mark.parametrize("abort_deferred", [False, True])
def test_abort_during_phase_split_leaves_no_persistent_lane(
    monkeypatch, local_model, abort_deferred
):
    scheduler = make_scheduler(monkeypatch, local_model)
    queue, requests = prime_pair(scheduler)
    assert list(queue.step().num_scheduled_tokens) == [requests[0].request_id]
    aborted = requests[int(abort_deferred)]
    survivor = requests[int(not abort_deferred)]
    scheduler.finish_requests(aborted.request_id, RequestStatus.FINISHED_ABORTED)
    for _ in range(6):
        output = queue.step()
        assert aborted.request_id not in output.num_scheduled_tokens
    assert survivor.num_output_tokens >= 3


@pytest.mark.parametrize("skipped", [False, True])
def test_waiting_work_preserves_existing_admission(monkeypatch, local_model, skipped):
    scheduler = make_scheduler(monkeypatch, local_model)
    queue, requests = prime_pair(scheduler)
    (new_request,) = utils.create_requests(
        num_requests=1, num_tokens=8, req_ids=["new"]
    )
    scheduler.add_request(new_request)
    if skipped:
        assert scheduler.waiting.pop_request() is new_request
        scheduler.skipped_waiting.add_request(new_request)
    output = queue.step()
    assert set(output.num_scheduled_tokens) == {
        requests[0].request_id,
        requests[1].request_id,
        "new",
    }


def test_running_prefill_does_not_split_resident_decoders(monkeypatch, local_model):
    scheduler = make_scheduler(monkeypatch, local_model)
    queue, requests = prime_pair(scheduler)
    (prefill,) = utils.create_requests(
        num_requests=1, num_tokens=512, req_ids=["prefill"]
    )
    scheduler.add_request(prefill)
    assert "prefill" in queue.step().num_scheduled_tokens
    assert prefill.is_prefill_chunk
    queue.step()
    output = queue.step()
    assert all(r.request_id in output.num_scheduled_tokens for r in requests)
    assert "prefill" in output.num_scheduled_tokens


@pytest.mark.parametrize("preempt_first", [False, True])
def test_allocation_failure_never_counts_as_successful_decode(
    monkeypatch, local_model, preempt_first
):
    scheduler = make_scheduler(monkeypatch, local_model)
    queue, requests = prime_pair(scheduler)
    a, b = requests
    if preempt_first:
        scheduler.policy = SchedulingPolicy.PRIORITY
        a.priority = 1
        b.priority = 0
    real_allocate = scheduler.kv_cache_manager.allocate_slots
    call_count = 0

    def fail_once(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return None
        return real_allocate(*args, **kwargs)

    monkeypatch.setattr(scheduler.kv_cache_manager, "allocate_slots", fail_once)
    output = queue.step()
    if preempt_first:
        assert not output.num_scheduled_tokens
        assert a.status == RequestStatus.PREEMPTED
        assert b.request_id in queue.step().num_scheduled_tokens
    else:
        assert list(output.num_scheduled_tokens) == [a.request_id]
        assert b.status == RequestStatus.PREEMPTED
        assert call_count == 2
        # Pending work disables staggering while the preempted request resumes.
        assert b.request_id in queue.step().num_scheduled_tokens


def test_no_tokens_for_first_request_does_not_defer_second(monkeypatch, local_model):
    scheduler = make_scheduler(monkeypatch, local_model)
    queue, requests = prime_pair(scheduler)
    real_balance = scheduler._balanced_prefill_chunk
    # Exercise a genuine zero-token branch before slot allocation. The second
    # request must still proceed; no successful-first marker exists yet.
    monkeypatch.setattr(
        scheduler,
        "_balanced_prefill_chunk",
        lambda request, *args, **kwargs: (
            0 if request is requests[0] else real_balance(request, *args, **kwargs)
        ),
    )
    output = queue.step()
    assert list(output.num_scheduled_tokens) == [requests[1].request_id]


def test_completed_outputs_match_with_and_without_stagger(monkeypatch, local_model):
    completed = []
    for enabled in (False, True):
        scheduler = make_scheduler(monkeypatch, local_model, enabled=enabled)
        queue, requests = prime_pair(scheduler, accepted=(1, 2, 5))
        for _ in range(400):
            queue.step()
            if scheduler.get_num_unfinished_requests() == 0:
                break
        else:
            pytest.fail("finite requests failed to complete")
        completed.append([list(r.output_token_ids) for r in requests])
    assert completed[0] == completed[1]
