# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import deque
from unittest.mock import patch

import pytest

from tests.v1.core import utils as scheduler_test_utils
from tests.v1.core.test_mixed_prefill_budget import offline_scheduler_model
from tests.v1.core.utils import create_requests, create_scheduler
from vllm.config import SchedulerConfig
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.request_queue import SchedulingPolicy
from vllm.v1.outputs import ModelRunnerOutput

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]

# Reuse the metadata-only local OPT fixture; no tokenizer/model is downloaded.
assert offline_scheduler_model is not None


def make_scheduler(monkeypatch, *, enabled=True, pp=7, asynchronous=True):
    original_config = scheduler_test_utils.SchedulerConfig

    def config(**kwargs):
        return original_config(pp_prefill_quota_compensation=enabled, **kwargs)

    monkeypatch.setattr(scheduler_test_utils, "SchedulerConfig", config)
    monkeypatch.setattr("vllm.config.parallel.current_platform.device_count", lambda: 7)
    scheduler = create_scheduler(
        async_scheduling=asynchronous,
        pipeline_parallel_size=pp,
        max_num_batched_tokens=768,
        max_model_len=16384,
        long_prefill_token_threshold=128,
        long_prefill_token_threshold_adaptive=True,
    )
    scheduler.max_model_len = 16384
    scheduler.use_v2_model_runner = True
    scheduler.pp_feedback_depth = 2
    scheduler.mixed_prefill_enabled = True
    return scheduler


def feedback(scheduler, step):
    ids = list(step.num_scheduled_tokens)
    scheduler.update_from_output(
        step,
        ModelRunnerOutput(
            req_ids=ids,
            req_id_to_index={rid: i for i, rid in enumerate(ids)},
            sampled_token_ids=[[] for _ in ids],
        ),
    )


def start_skew(scheduler, first_count):
    requests = create_requests(3, num_tokens=8192, req_ids=["a", "b", "c"])
    for request in requests[:first_count]:
        scheduler.add_request(request)
    first = scheduler.schedule()
    for request in requests[first_count:]:
        scheduler.add_request(request)
    second = scheduler.schedule()
    return requests, first, second


def positions(requests):
    return [request.num_computed_tokens for request in requests]


def test_flag_defaults_off_and_cli_round_trips(tmp_path):
    from vllm.engine.arg_utils import EngineArgs
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    assert not SchedulerConfig.default_factory().pp_prefill_quota_compensation
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    args = EngineArgs.from_cli_args(
        parser.parse_args(
            [
                "--model",
                str(tmp_path),
                "--skip-tokenizer-init",
                "--pp-prefill-quota-compensation",
            ]
        )
    )
    assert args.pp_prefill_quota_compensation


@pytest.mark.parametrize("first_count,expected_steps", [(2, 1), (1, 4)])
def test_early_group_repays_service_skew_across_real_feedback(
    monkeypatch, first_count, expected_steps
):
    scheduler = make_scheduler(monkeypatch)
    requests, first, second = start_skew(scheduler, first_count)
    assert positions(requests) == (
        [640, 640, 256] if first_count == 2 else [1024, 256, 256]
    )
    controller = scheduler._early_prefill_quota
    assert controller is not None and controller.members == ("a", "b", "c")
    pending = deque([first, second])
    budget_before = vars(scheduler.mixed_prefill_budget).copy()
    for _ in range(expected_steps):
        feedback(scheduler, pending.popleft())
        assert requests[0].num_computed_tokens > requests[0].num_in_flight_tokens
        step = scheduler.schedule()
        pending.append(step)
        assert list(step.num_scheduled_tokens) == ["a", "b", "c"]
        assert min(step.num_scheduled_tokens.values()) >= 128
        assert max(step.num_scheduled_tokens.values()) <= 512
        assert sum(step.num_scheduled_tokens.values()) == 768
    assert len(set(positions(requests))) == 1
    assert vars(scheduler.mixed_prefill_budget) == budget_before


@pytest.mark.parametrize("enabled", [False, True])
def test_synchronized_group_keeps_original_shares(monkeypatch, enabled):
    scheduler = make_scheduler(monkeypatch, enabled=enabled)
    requests = create_requests(3, num_tokens=8192, req_ids=["a", "b", "c"])
    for request in requests:
        scheduler.add_request(request)
    first = scheduler.schedule()
    assert first.num_scheduled_tokens == {"a": 256, "b": 256, "c": 256}
    assert scheduler.schedule().num_scheduled_tokens == first.num_scheduled_tokens
    if enabled:
        assert not scheduler._early_prefill_quota.members
    else:
        assert scheduler._early_prefill_quota is None


@pytest.mark.parametrize("pp,asynchronous", [(1, True), (7, False)])
def test_other_scheduler_modes_do_not_enable_compensation(
    monkeypatch, pp, asynchronous
):
    scheduler = make_scheduler(monkeypatch, pp=pp, asynchronous=asynchronous)
    requests, first, _ = start_skew(scheduler, 2)
    feedback(scheduler, first)
    assert scheduler.schedule().num_scheduled_tokens == {"a": 256, "b": 256, "c": 256}
    assert positions(requests) == [896, 896, 512]


def test_late_arrival_closes_group_and_does_not_steal_existing_share(monkeypatch):
    scheduler = make_scheduler(monkeypatch)
    _, first, _ = start_skew(scheduler, 1)
    feedback(scheduler, first)
    incoming = create_requests(1, num_tokens=8192, req_ids=["later"])[0]
    scheduler.add_request(incoming)
    step = scheduler.schedule()
    assert step.num_scheduled_tokens == {"a": 192, "b": 192, "c": 192, "later": 192}
    assert not scheduler._early_prefill_quota.members
    assert scheduler._early_prefill_quota.retired == frozenset(["a", "b", "c"])
    assert scheduler.schedule().num_scheduled_tokens == step.num_scheduled_tokens


@pytest.mark.parametrize(
    "field,value", [("num_preemptions", 1), ("last_prefix_cache_reset_step", 1)]
)
def test_replayed_requests_cannot_freeze_a_new_group(monkeypatch, field, value):
    scheduler = make_scheduler(monkeypatch)
    requests = create_requests(3, num_tokens=8192, req_ids=["a", "b", "c"])
    scheduler.add_request(requests[0])
    scheduler.schedule()
    setattr(requests[0], field, value)
    for request in requests[1:]:
        scheduler.add_request(request)
    scheduler.schedule()
    assert not scheduler._early_prefill_quota.members


@pytest.mark.parametrize("reason", ["priority", "pause"])
def test_priority_change_and_pause_cancel_active_group(monkeypatch, reason):
    scheduler = make_scheduler(monkeypatch)
    requests, _, _ = start_skew(scheduler, 2)
    if reason == "priority":
        scheduler.policy = SchedulingPolicy.PRIORITY
        requests[1].priority = -1
        step = scheduler.schedule()
        assert step.num_scheduled_tokens == {"a": 256, "b": 256, "c": 256}
    else:
        scheduler.set_pause_state(PauseState.PAUSED_ALL)
    assert not scheduler._early_prefill_quota.members


def test_zero_work_does_not_receive_credit_or_extend_deadline(monkeypatch):
    scheduler = make_scheduler(monkeypatch)
    requests, first, _ = start_skew(scheduler, 2)
    feedback(scheduler, first)
    controller = scheduler._early_prefill_quota
    deadline = controller.expires_step
    scheduler.need_mamba_block_aligned_split = True
    with patch.object(scheduler, "_mamba_block_aligned_split", return_value=0):
        while scheduler.current_step <= deadline:
            assert not scheduler.schedule().num_scheduled_tokens
    assert positions(requests) == [640, 640, 256]
    scheduler.need_mamba_block_aligned_split = False
    assert controller.expires_step == deadline
    assert not controller.members
    assert scheduler.schedule().num_scheduled_tokens == {"a": 256, "b": 256, "c": 256}


def test_boundary_clipping_is_repaid_using_actual_scheduled_work(monkeypatch):
    scheduler = make_scheduler(monkeypatch)
    requests, first, _ = start_skew(scheduler, 2)
    feedback(scheduler, first)
    scheduler.need_mamba_block_aligned_split = True

    # Exercise the real downstream clipping call, retaining only 128 of the
    # laggard's planned 512 tokens, as a recurrent boundary/encoder may require.
    def clip(request, count):
        return min(count, 128) if request.request_id == "c" else count

    with patch.object(scheduler, "_mamba_block_aligned_split", side_effect=clip):
        step = scheduler.schedule()
    assert step.num_scheduled_tokens == {"a": 128, "b": 128, "c": 128}
    assert positions(requests) == [768, 768, 384]
    scheduler.need_mamba_block_aligned_split = False
    step = scheduler.schedule()
    assert step.num_scheduled_tokens == {"a": 128, "b": 128, "c": 512}
    assert positions(requests) == [896, 896, 896]


def test_allocation_failure_retires_group_and_keeps_actual_ledger(monkeypatch):
    scheduler = make_scheduler(monkeypatch)
    requests, first, _ = start_skew(scheduler, 2)
    feedback(scheduler, first)
    allocate = scheduler.kv_cache_manager.allocate_slots
    attempts = 0

    def fail_once(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        return None if attempts == 1 else allocate(*args, **kwargs)

    with patch.object(
        scheduler.kv_cache_manager, "allocate_slots", side_effect=fail_once
    ):
        step = scheduler.schedule()
    # The lowest-priority last request is preempted by the original algorithm.
    # The current aligned chunk remains valid; later requests use normal caps.
    assert step.num_scheduled_tokens == {"a": 128, "b": 256}
    assert requests[0].num_computed_tokens == 768
    assert requests[1].num_computed_tokens == 896
    assert requests[2].num_preemptions == 1
    assert "c" not in step.num_scheduled_tokens
    assert not scheduler._early_prefill_quota.members


def test_resident_decode_budget_remains_authoritative(monkeypatch):
    scheduler = make_scheduler(monkeypatch)
    requests, first, _ = start_skew(scheduler, 2)
    feedback(scheduler, first)
    # Final-prefill placeholder is a real async in-flight decoder, even before
    # a visible output token exists. It ends the cohort before mixed limiting.
    requests[0].num_computed_tokens = requests[0].num_prompt_tokens
    requests[0].is_prefill_chunk = False
    requests[0].num_output_placeholders = 1
    requests[0].next_decode_eligible_step = scheduler.current_step + 2
    scheduler.mixed_prefill_budget.tokens = 16
    step = scheduler.schedule()
    assert step.mixed_prefill_budget == 16
    assert step.mixed_prefill_tokens == 16
    assert step.mixed_decode_tokens == 0
    assert not scheduler._early_prefill_quota.members


@pytest.mark.parametrize("arrival_step,eligible", [(997, True), (998, False)])
def test_actual_995_997_admission_with_prior_feedback_and_late_counterexample(
    monkeypatch, arrival_step, eligible
):
    # Frozen trace seq2983..2989: A is scheduled at995/996, feedback995
    # arrives before997, then B/C enter. This is *not* an all-in-flight group.
    scheduler = make_scheduler(monkeypatch)
    scheduler.current_step = 994
    requests = create_requests(3, num_tokens=8192, req_ids=["a", "b", "c"])
    scheduler.add_request(requests[0])
    first = scheduler.schedule()
    second = scheduler.schedule()
    assert first.scheduled_step == 995 and second.scheduled_step == 996
    feedback(scheduler, first)
    assert positions(requests) == [1536, 0, 0]
    assert requests[0].num_in_flight_tokens == 768
    if arrival_step == 998:
        third = scheduler.schedule()
        feedback(scheduler, second)
        pending = deque([third])
    else:
        pending = deque([second])
    for request in requests[1:]:
        scheduler.add_request(request)
    admission = scheduler.schedule()
    pending.append(admission)
    if not eligible:
        assert not scheduler._early_prefill_quota.members
        assert scheduler.schedule().num_scheduled_tokens == {
            "a": 256,
            "b": 256,
            "c": 256,
        }
        return
    assert positions(requests) == [1792, 256, 256]
    assert [request.num_in_flight_tokens for request in requests] == [1024, 256, 256]
    assert scheduler._early_prefill_quota.members == ("a", "b", "c")
    for _ in range(8):
        feedback(scheduler, pending.popleft())
        step = scheduler.schedule()
        pending.append(step)
        assert min(step.num_scheduled_tokens.values()) >= 128
    assert positions(requests) == [2816, 2816, 2816]


def test_real_4480_splitter_uses_actual_positions_and_preserves_recovery(monkeypatch):
    scheduler = make_scheduler(monkeypatch)
    requests, first, _ = start_skew(scheduler, 2)
    feedback(scheduler, first)
    # A synthetic boundary fixture (not a naturally reachable depth-2
    # admission trajectory) near the real retained boundary. Bind
    # the planner's successful-service ledger to the same already served prefix.
    controller = scheduler._early_prefill_quota
    for request, position in zip(requests, [4736, 4736, 4352]):
        first_step, _ = controller._service[request.request_id]
        request.num_computed_tokens = position
        controller._service[request.request_id] = (first_step, position)
    scheduler.need_mamba_block_aligned_split = True
    scheduler.mamba_state_block_size = 4480
    scheduler.mamba_state_retention_interval = 0
    scheduler.mamba_dense_boundaries_on_contention = True
    first_clipped = scheduler.schedule()
    assert first_clipped.num_scheduled_tokens == {"a": 128, "b": 128, "c": 128}
    assert positions(requests) == [4864, 4864, 4480]
    recovered = scheduler.schedule()
    assert recovered.num_scheduled_tokens == {"a": 128, "b": 128, "c": 512}
    assert positions(requests) == [4992, 4992, 4992]


def test_actual_1088_1090_two_then_one_admission_after_feedback(monkeypatch):
    scheduler = make_scheduler(monkeypatch)
    scheduler.current_step = 1087
    requests = create_requests(3, num_tokens=8192, req_ids=["a", "b", "c"])
    for request in requests[:2]:
        scheduler.add_request(request)
    first = scheduler.schedule()
    second = scheduler.schedule()
    assert positions(requests) == [768, 768, 0]
    feedback(scheduler, first)
    scheduler.add_request(requests[2])
    third = scheduler.schedule()
    assert third.scheduled_step == 1090
    assert positions(requests) == [1024, 1024, 256]
    assert [request.num_in_flight_tokens for request in requests] == [640, 640, 256]
    for pending in [second, third]:
        feedback(scheduler, pending)
        assert scheduler.schedule().num_scheduled_tokens == {
            "a": 128,
            "b": 128,
            "c": 512,
        }
    assert positions(requests) == [1280, 1280, 1280]


def test_preemption_removes_already_scheduled_member_without_service_credit(
    monkeypatch,
):
    scheduler = make_scheduler(monkeypatch)
    requests, first, _ = start_skew(scheduler, 2)
    feedback(scheduler, first)
    scheduler.policy = SchedulingPolicy.PRIORITY
    # Equal priority remains eligible, but original priority preemption selects
    # a (latest arrival timestamp) even after it was scheduled this step.
    for request, arrival in zip(requests, [100.0, 2.0, 1.0]):
        request.arrival_time = arrival
    allocate = scheduler.kv_cache_manager.allocate_slots
    failed = False

    def fail_b_once(request, *args, **kwargs):
        nonlocal failed
        if request.request_id == "b" and not failed:
            failed = True
            return None
        return allocate(request, *args, **kwargs)

    with patch.object(
        scheduler.kv_cache_manager, "allocate_slots", side_effect=fail_b_once
    ):
        step = scheduler.schedule()
    assert step.num_scheduled_tokens == {"b": 128, "c": 256}
    assert requests[0].num_preemptions == 1
    assert "a" not in step.num_scheduled_tokens
    assert positions(requests)[1:] == [768, 512]
    assert not scheduler._early_prefill_quota.members
    assert not scheduler._early_prefill_quota._service


def test_nonzero_cached_start_cannot_be_counted_as_actual_service(monkeypatch):
    scheduler = make_scheduler(monkeypatch)
    requests = create_requests(3, num_tokens=8192, req_ids=["a", "b", "c"])
    # The existing WAITING path accepts a nonzero position after async KV load.
    requests[0].num_computed_tokens = 256
    scheduler.add_request(requests[0])
    scheduler.schedule()
    assert requests[0].num_computed_tokens == 1024
    assert "a" not in scheduler._early_prefill_quota._service
    for request in requests[1:]:
        scheduler.add_request(request)
    scheduler.schedule()
    assert not scheduler._early_prefill_quota.members


def test_zero_work_before_admission_does_not_extend_freshness(monkeypatch):
    scheduler = make_scheduler(monkeypatch)
    requests = create_requests(3, num_tokens=8192, req_ids=["a", "b", "c"])
    scheduler.add_request(requests[0])
    scheduler.schedule()
    scheduler.need_mamba_block_aligned_split = True
    with patch.object(scheduler, "_mamba_block_aligned_split", return_value=0):
        assert not scheduler.schedule().num_scheduled_tokens
        assert not scheduler.schedule().num_scheduled_tokens
    scheduler.need_mamba_block_aligned_split = False
    for request in requests[1:]:
        scheduler.add_request(request)
    scheduler.schedule()
    # All work is still in flight and the skew is only 768, but the first
    # request's successful admission is too old. Idle work cannot renew it.
    assert positions(requests) == [1024, 256, 256]
    assert not scheduler._early_prefill_quota.members


def test_actual_sequential_899_900_901_admission_exits_frozen_pair(monkeypatch):
    scheduler = make_scheduler(monkeypatch)
    scheduler.current_step = 898
    requests = create_requests(3, num_tokens=8192, req_ids=["a", "b", "c"])
    scheduler.add_request(requests[0])
    first = scheduler.schedule()
    scheduler.add_request(requests[1])
    second = scheduler.schedule()
    assert first.scheduled_step == 899 and second.scheduled_step == 900
    assert positions(requests) == [1152, 384, 0]
    assert scheduler._early_prefill_quota.members == ("a", "b")
    feedback(scheduler, first)
    scheduler.add_request(requests[2])
    third = scheduler.schedule()
    # New members end an already frozen group even inside the initial window.
    # This deliberately leaves the observed sequential-admission skew unpaid.
    assert third.scheduled_step == 901
    assert third.num_scheduled_tokens == {"a": 256, "b": 256, "c": 256}
    assert positions(requests) == [1408, 640, 256]
    assert [request.num_in_flight_tokens for request in requests] == [640, 640, 256]
    assert not scheduler._early_prefill_quota.members
    assert scheduler._early_prefill_quota.retired == frozenset(["a", "b"])
    assert not scheduler._early_prefill_quota._service
