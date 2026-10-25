# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real CPU Scheduler regression for the PP feedback-depth contract."""

import pytest

from tests.v1.core import utils
from vllm.v1.core.sched.async_scheduler import AsyncScheduler

pytestmark = pytest.mark.cpu_test


def make_pp_scheduler(monkeypatch, depth, *, budget=32):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setattr("vllm.config.parallel.current_platform.device_count", lambda: 7)
    real_scheduler_config = utils.SchedulerConfig

    def scheduler_config_with_depth(**kwargs):
        return real_scheduler_config(pp_async_queue_depth=depth, **kwargs)

    monkeypatch.setattr(utils, "SchedulerConfig", scheduler_config_with_depth)
    scheduler = utils.create_scheduler(
        async_scheduling=True,
        pipeline_parallel_size=7,
        max_num_seqs=8,
        max_num_batched_tokens=budget,
        max_model_len=128,
    )
    assert isinstance(scheduler, AsyncScheduler)
    assert scheduler.use_v2_model_runner
    assert scheduler.pp_feedback_depth == (7 if depth is None else depth)
    assert scheduler.vllm_config.max_concurrent_batches == scheduler.pp_feedback_depth
    scheduler.num_spec_tokens = 4
    scheduler._spec_token_placeholders = (-1,) * scheduler.num_spec_tokens
    return scheduler


@pytest.mark.parametrize("depth", [None, 2, 4, 7])
def test_real_scheduler_blocks_decode_until_matching_feedback(monkeypatch, depth):
    scheduler = make_pp_scheduler(monkeypatch, depth)
    (request,) = utils.create_requests(num_requests=1, num_tokens=8)
    scheduler.add_request(request)
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {request.request_id: 8}
    eligible = scheduler.current_step + scheduler.pp_feedback_depth
    assert request.next_decode_eligible_step == eligible

    for _ in range(scheduler.pp_feedback_depth - 1):
        output = scheduler.schedule()
        assert request.request_id not in output.num_scheduled_tokens

    output = scheduler.schedule()
    assert scheduler.current_step == eligible
    assert output.num_scheduled_tokens == {request.request_id: 5}
    assert len(output.scheduled_spec_decode_tokens[request.request_id]) == 4
    assert request.next_decode_eligible_step == eligible + scheduler.pp_feedback_depth


@pytest.mark.parametrize("depth", [2, 4, 7])
def test_real_scheduler_defers_only_after_final_prefill_chunk(monkeypatch, depth):
    scheduler = make_pp_scheduler(monkeypatch, depth, budget=8)
    (request,) = utils.create_requests(num_requests=1, num_tokens=20)
    scheduler.add_request(request)

    for expected_tokens in (8, 8):
        output = scheduler.schedule()
        assert output.num_scheduled_tokens == {request.request_id: expected_tokens}
        assert request.is_prefill_chunk
        assert request.next_decode_eligible_step <= scheduler.current_step

    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {request.request_id: 4}
    assert not request.is_prefill_chunk
    assert request.next_decode_eligible_step == scheduler.current_step + depth

    for _ in range(depth - 1):
        assert request.request_id not in scheduler.schedule().num_scheduled_tokens
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {request.request_id: 5}
