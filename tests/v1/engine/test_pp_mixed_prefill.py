# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import deque
from concurrent.futures import Future
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.engine.core import EngineCore
from vllm.v1.engine.mixed_prefill import (
    finish_pp_mixed_prefill_timing,
    start_pp_mixed_prefill_timing,
)
from vllm.v1.outputs import MixedPrefillTiming, ModelRunnerOutput

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def _batch(step, prefill=32, decode=0):
    output = SchedulerOutput.make_empty()
    output.scheduled_step = step
    output.num_scheduled_tokens = {f"r{step}": prefill + decode}
    output.total_num_scheduled_tokens = prefill + decode
    output.mixed_prefill_tokens = output.mixed_prefill_budget = prefill
    output.mixed_decode_tokens = decode
    output.mixed_prefill_is_pp = True
    return output


def _result():
    return ModelRunnerOutput(req_ids=[], req_id_to_index={}, sampled_token_ids=[])


@pytest.fixture
def clock(monkeypatch):
    now = [10.0]
    monkeypatch.setattr(
        "vllm.v1.engine.mixed_prefill.time",
        SimpleNamespace(perf_counter=lambda: now[0]),
    )
    return now


def _engine(outputs):
    core = object.__new__(EngineCore)
    core.scheduler = Mock()
    core.scheduler.has_requests.return_value = True
    core.scheduler.schedule.side_effect = outputs
    core.scheduler.update_from_output.return_value = {}
    core.model_executor = Mock()
    core.log_error_detail = lambda _: nullcontext()
    core.log_iteration_details = lambda _: nullcontext()
    core._process_aborts_queue = Mock()
    core.async_scheduling = True
    core.is_pooling_model = False
    core.is_ec_consumer = True
    core.use_spec_decode = True
    core.batch_queue_size = 2
    core.batch_queue = deque(maxlen=2)
    core._sm70_async_cpu_trace_step = 0
    core.vllm_config = SimpleNamespace(
        observability_config=SimpleNamespace(
            runtime_trace=SimpleNamespace(async_cpu=False, async_every=1)
        )
    )
    return core


def test_pp_timing_is_bound_to_batch_even_when_results_are_consumed_late(clock):
    first, second = _batch(1), _batch(2, 64, 5)
    start_pp_mixed_prefill_timing(first)
    clock[0] = 10.2
    start_pp_mixed_prefill_timing(second)
    clock[0] = 10.8
    result = _result()
    finish_pp_mixed_prefill_timing(first, result)
    sample = result.mixed_prefill_timing
    assert sample.elapsed_ms == pytest.approx(800)
    assert (sample.prefill_tokens, sample.decode_tokens, sample.budget_tokens) == (
        32,
        0,
        32,
    )
    assert sample.is_pipeline
    assert first.mixed_prefill_started_at is None
    assert second.mixed_prefill_started_at == 10.2
    clock[0] = 10.9
    finish_pp_mixed_prefill_timing(second, result)
    assert result.mixed_prefill_timing.elapsed_ms == pytest.approx(700)
    assert result.mixed_prefill_timing.prefill_tokens == 64
    # Consuming the same output twice cannot overwrite another sample.
    finish_pp_mixed_prefill_timing(first, result)
    assert result.mixed_prefill_timing.prefill_tokens == 64


def test_pp1_and_pure_decode_do_not_read_host_clock_or_replace_gpu_sample(monkeypatch):
    monkeypatch.setattr(
        "vllm.v1.engine.mixed_prefill.time",
        SimpleNamespace(perf_counter=Mock(side_effect=AssertionError("unused clock"))),
    )
    gpu_sample = MixedPrefillTiming(32, 5, 32, 10)
    for output in (_batch(1, 0, 5), SchedulerOutput.make_empty()):
        result = _result()
        result.mixed_prefill_timing = gpu_sample
        start_pp_mixed_prefill_timing(output)
        finish_pp_mixed_prefill_timing(output, result)
        assert result.mixed_prefill_timing is gpu_sample


def test_standalone_prefill_is_observed_without_a_control_budget(clock):
    output = _batch(1, prefill=768)
    output.mixed_prefill_budget = 0
    start_pp_mixed_prefill_timing(output)
    clock[0] += 2.0
    result = _result()
    finish_pp_mixed_prefill_timing(output, result)
    assert result.mixed_prefill_timing == MixedPrefillTiming(
        768, 0, 0, 2000, is_pipeline=True
    )


@pytest.mark.parametrize("separate_sampling", [False, True])
def test_engine_step_records_execute_and_sampling_latency(clock, separate_sampling):
    batch = _batch(1)
    core = _engine([batch])
    result = _result()

    def execute(*args, **kwargs):
        assert batch.mixed_prefill_started_at == 10
        clock[0] = 10.4
        future: Future[ModelRunnerOutput | None] = Future()
        future.set_result(None if separate_sampling else result)
        return future

    def sample(*args, **kwargs):
        clock[0] = 10.7
        return result

    core.model_executor.execute_model.side_effect = execute
    core.model_executor.sample_tokens.side_effect = sample
    core.step()
    assert result.mixed_prefill_timing.elapsed_ms == pytest.approx(
        700 if separate_sampling else 400
    )
    core.scheduler.update_from_output.assert_called_once_with(batch, result)


@pytest.mark.parametrize("deferred", [False, True])
def test_engine_queue_preserves_batch_timing_and_deferred_mtp_feedback(clock, deferred):
    first, second = _batch(1), _batch(2, 64, 5)
    core = _engine([first, second])
    if deferred:
        core.async_scheduling = False
        second.pending_structured_output_tokens = True
    first_future: Future[ModelRunnerOutput] = Future()
    second_future: Future[ModelRunnerOutput] = Future()
    results = [_result(), _result()]
    core.model_executor.execute_model.side_effect = [Future(), Future()]
    core.model_executor.sample_tokens.side_effect = [first_future, second_future]
    # No response wait while there is room to fill the pipeline.
    assert core.step_with_batch_queue() == (None, True)
    assert first.mixed_prefill_started_at == 10
    clock[0] = 10.2
    first_future.set_result(results[0])
    core.step_with_batch_queue()
    assert results[0].mixed_prefill_timing.elapsed_ms == pytest.approx(200)
    assert results[0].mixed_prefill_timing.prefill_tokens == 32
    assert second.mixed_prefill_started_at == 10.2
    if deferred:
        core.scheduler.update_draft_token_ids_in_output.assert_called_once()
    # A completed future can wait in the queue before the engine consumes it.
    second_future.set_result(results[1])
    clock[0] = 10.9
    core.scheduler.has_requests.return_value = False
    core.step_with_batch_queue()
    assert results[1].mixed_prefill_timing.elapsed_ms == pytest.approx(700)
    assert results[1].mixed_prefill_timing.prefill_tokens == 64
    assert not core.batch_queue
