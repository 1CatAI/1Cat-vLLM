# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import patch

import pytest

from vllm.v1.core.sched.mixed_prefill import MixedPrefillBudget
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import MixedPrefillTiming, ModelRunnerOutput
from vllm.v1.worker.mixed_prefill import MixedPrefillTimer

from .utils import create_requests, create_scheduler

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def test_budget_adapts_to_gpu_cost_and_ignores_short_boundaries():
    budget = MixedPrefillBudget(8192, 250)
    assert budget.tokens == 512
    budget.update(MixedPrefillTiming(512, 16, 512, 400))
    assert budget.tokens == 320
    budget.update(MixedPrefillTiming(8, 16, 320, 150))
    assert budget.tokens == 320
    for _ in range(30):
        n = budget.tokens
        budget.update(MixedPrefillTiming(n, 16, n, n * 0.1))
    assert 2400 <= budget.tokens <= 2512


@pytest.mark.parametrize("elapsed", [0, -1, float("nan"), float("inf")])
def test_invalid_timing_does_not_change_budget(elapsed):
    budget = MixedPrefillBudget(8192, 250)
    budget.update(MixedPrefillTiming(512, 16, 512, elapsed))
    assert budget.tokens == 512


def test_timer_never_waits_and_keeps_delayed_sample_token_counts():
    events = []

    class Event:
        def __init__(self, **kwargs):
            self.ready = False
            events.append(self)

        def record(self):
            pass

        def query(self):
            return self.ready

        def elapsed_time(self, end):
            assert end.ready
            return 300.0

        def synchronize(self):
            pytest.fail("GPU feedback must never synchronize")

    timer = MixedPrefillTimer()
    output = SchedulerOutput.make_empty()
    output.mixed_prefill_tokens = output.mixed_prefill_budget = 512
    output.mixed_decode_tokens = 16
    with patch("torch.cuda.Event", Event):
        timer.begin(output)
        assert timer.finish() is None
        events[-1].ready = True
        timer.begin(SchedulerOutput.make_empty())
        sample = timer.finish()
        assert sample == MixedPrefillTiming(512, 16, 512, 300)
        assert len(events) == 2  # No events for pure decode.


def _resident(scheduler):
    req = create_requests(num_requests=1, num_tokens=32)[0]
    scheduler.add_request(req)
    step = scheduler.schedule()
    scheduler.update_from_output(
        step,
        ModelRunnerOutput(
            req_ids=[req.request_id],
            req_id_to_index={req.request_id: 0},
            sampled_token_ids=[[42]],
        ),
    )
    return req


def test_running_prefill_cannot_starve_later_resident_decode():
    scheduler = create_scheduler(max_num_batched_tokens=1024)
    scheduler.mixed_prefill_enabled = True
    resident = _resident(scheduler)
    prefill = create_requests(num_requests=1, num_tokens=1536)[0]
    prefill.request_id = "long"
    scheduler.add_request(prefill)
    first = scheduler.schedule()
    assert first.num_scheduled_tokens[resident.request_id] == 1
    assert first.num_scheduled_tokens["long"] == 512
    scheduler.update_from_output(
        first,
        ModelRunnerOutput(
            req_ids=[resident.request_id, "long"],
            req_id_to_index={resident.request_id: 0, "long": 1},
            sampled_token_ids=[[43], []],
        ),
    )
    # The next step simulates a queue order in which prefill is first.
    scheduler.running.reverse()
    step = scheduler.schedule()
    assert step.num_scheduled_tokens[resident.request_id] > 0
    assert step.num_scheduled_tokens["long"] <= 512
    assert step.mixed_prefill_tokens == 512
    assert step.mixed_decode_tokens > 0


def test_pure_prefill_and_disabled_control_keep_full_budget():
    scheduler = create_scheduler(max_num_batched_tokens=1024)
    long = create_requests(num_requests=1, num_tokens=1536)[0]
    scheduler.add_request(long)
    step = scheduler.schedule()
    assert step.num_scheduled_tokens[long.request_id] == 1024
    assert step.mixed_prefill_tokens == 0

    scheduler = create_scheduler(max_num_batched_tokens=1024)
    resident = _resident(scheduler)
    scheduler.mixed_prefill_enabled = True
    scheduler.scheduler_config.mixed_prefill_step_latency_ms = 0
    long = create_requests(num_requests=1, num_tokens=1536, req_ids=["long"])[0]
    scheduler.add_request(long)
    step = scheduler.schedule()
    assert step.num_scheduled_tokens["long"] == 1023
    assert step.num_scheduled_tokens[resident.request_id] == 1
    assert step.mixed_prefill_tokens == 0


def test_adaptive_threshold_floor_does_not_override_mixed_latency_budget():
    scheduler = create_scheduler(
        max_num_batched_tokens=8192,
        long_prefill_token_threshold=128,
        long_prefill_token_threshold_adaptive=True,
    )
    resident = _resident(scheduler)
    scheduler.mixed_prefill_enabled = True
    incoming = create_requests(num_requests=1, num_tokens=16384)[0]
    incoming.request_id = "incoming"
    scheduler.add_request(incoming)
    output = scheduler.schedule()
    assert output.num_scheduled_tokens[resident.request_id] > 0
    assert output.num_scheduled_tokens[incoming.request_id] == 512
    assert output.mixed_prefill_tokens == 512
