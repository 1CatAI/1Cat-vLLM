# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_PP_KEEP_BATCH, scheduler side and the shared decision.

With pipeline parallelism and synchronous scheduling a single running request is
in flight on every other engine step, so that step schedules nothing. With the
knob on, the scheduler keeps its previous-step record across such a step and
does not resend the request's whole token history on the next decode step.
With the knob off, the default path is unchanged.
"""

from types import SimpleNamespace

import pytest

from vllm.config import DeviceConfig
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.pp_keep_batch import (
    pp_keep_batch_decision,
    pp_keep_batch_requested,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus

from .utils import create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test

KNOB = "VLLM_PP_KEEP_BATCH"


@pytest.fixture
def two_devices(monkeypatch):
    # Scheduler bookkeeping is CPU-only; do not require two physical GPUs.
    monkeypatch.setattr("vllm.config.parallel.current_platform.device_count", lambda: 2)


def _knob(monkeypatch, value: str | None) -> None:
    if value is None:
        monkeypatch.delenv(KNOB, raising=False)
    else:
        monkeypatch.setenv(KNOB, value)


def _runner_output(
    scheduler_output: SchedulerOutput, token_id: int
) -> ModelRunnerOutput:
    req_ids = list(scheduler_output.num_scheduled_tokens.keys())
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
        sampled_token_ids=[[token_id] for _ in req_ids],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def _pp2_scheduler(monkeypatch, knob: str | None):
    _knob(monkeypatch, knob)
    scheduler = create_scheduler(
        pipeline_parallel_size=2, device_config=DeviceConfig(device="cpu")
    )
    assert not scheduler.use_v2_model_runner
    assert not scheduler.scheduler_config.async_scheduling
    return scheduler


def _decode_rounds(scheduler, request, num_rounds: int):
    """Drive the PP=2 batch-queue cadence: each round schedules the request,
    then runs one empty step while it is in flight, then applies its output.
    Returns the cached-request data of every scheduled step."""
    cached = []
    token_id = 100
    for _ in range(num_rounds):
        scheduled = scheduler.schedule()
        assert request.request_id in scheduled.num_scheduled_tokens
        cached.append(scheduled.scheduled_cached_reqs)
        in_flight = scheduler.schedule()
        assert in_flight.total_num_scheduled_tokens == 0
        assert not in_flight.num_scheduled_tokens
        scheduler.update_from_output(scheduled, _runner_output(scheduled, token_id))
        token_id += 1
    return cached


@pytest.mark.parametrize(
    ("knob", "history_sent"), [(None, True), ("0", True), ("1", False)]
)
def test_empty_pipeline_step_keeps_record_only_with_knob(
    two_devices, monkeypatch, knob, history_sent
):
    scheduler = _pp2_scheduler(monkeypatch, knob)
    assert scheduler.pp_keep_batch is (knob == "1")
    (request,) = create_requests(num_requests=1, num_tokens=40, max_tokens=8)
    scheduler.add_request(request)

    cached = _decode_rounds(scheduler, request, num_rounds=4)

    # Round 0 is the prefill (a new request, not cached data).
    assert cached[0].req_ids == []
    for round_data in cached[1:]:
        assert round_data.req_ids == [request.request_id]
        assert not round_data.resumed_req_ids
        # The sampled token is always sent to the first stage.
        assert len(round_data.new_token_ids) == 1
        assert len(round_data.new_token_ids[0]) == 1
        assert (request.request_id in round_data.all_token_ids) is history_sent
        if history_sent:
            history = round_data.all_token_ids[request.request_id]
            assert history == list(request.all_token_ids[: len(history)])
    if knob == "1":
        assert scheduler._pp_keep_batch_kept_logged
    else:
        assert not scheduler._pp_keep_batch_kept_logged


def test_new_token_ids_match_default_path(two_devices, monkeypatch):
    """The token ids sent to the first stage are identical with and without the
    knob; only the history resend differs."""
    sent = {}
    for knob in ("0", "1"):
        scheduler = _pp2_scheduler(monkeypatch, knob)
        (request,) = create_requests(num_requests=1, num_tokens=33, max_tokens=8)
        scheduler.add_request(request)
        cached = _decode_rounds(scheduler, request, num_rounds=5)
        sent[knob] = (
            [c.new_token_ids for c in cached],
            [c.num_computed_tokens for c in cached],
            [c.num_output_tokens for c in cached],
        )
    assert sent["0"] == sent["1"]


def test_finished_request_leaves_record_on_empty_step(two_devices, monkeypatch):
    scheduler = _pp2_scheduler(monkeypatch, "1")
    request_a, request_b = create_requests(num_requests=2, num_tokens=20)
    scheduler.add_request(request_a)
    scheduler.add_request(request_b)
    prefill = scheduler.schedule()
    assert set(prefill.num_scheduled_tokens) == {"0", "1"}
    assert scheduler.prev_step_scheduled_req_ids == {"0", "1"}

    scheduler.finish_requests("1", RequestStatus.FINISHED_ABORTED)
    in_flight = scheduler.schedule()
    assert in_flight.total_num_scheduled_tokens == 0
    # The aborted request is no longer running and leaves the record.
    assert scheduler.prev_step_scheduled_req_ids == {"0"}


def test_preempted_request_leaves_record_on_empty_step(two_devices, monkeypatch):
    scheduler = _pp2_scheduler(monkeypatch, "1")
    (request,) = create_requests(num_requests=1, num_tokens=20)
    scheduler.add_request(request)
    prefill = scheduler.schedule()
    scheduler.update_from_output(prefill, _runner_output(prefill, 7))
    assert scheduler.prev_step_scheduled_req_ids == {request.request_id}

    # Preempted outside schedule() (as reset_prefix_cache does, minus its own
    # record clear): the empty step must still drop it from the record.
    scheduler.running.remove(request)
    scheduler._preempt_request(request, 0.0)
    scheduler._keep_prev_step_record_on_empty_step()
    assert scheduler.prev_step_scheduled_req_ids == set()

    # The resumed request is sent with its history and is not treated as held
    # (Scheduler asserts that a resumed request did not run in the prior step).
    resumed = scheduler.schedule()
    cached = resumed.scheduled_cached_reqs
    assert cached.resumed_req_ids == {request.request_id}
    assert request.request_id in cached.all_token_ids


def test_reset_prefix_cache_with_running_requests_after_empty_step(
    two_devices, monkeypatch
):
    scheduler = _pp2_scheduler(monkeypatch, "1")
    (request,) = create_requests(num_requests=1, num_tokens=20)
    scheduler.add_request(request)
    prefill = scheduler.schedule()
    assert scheduler.schedule().total_num_scheduled_tokens == 0
    assert scheduler.prev_step_scheduled_req_ids == {request.request_id}
    scheduler.update_from_output(prefill, _runner_output(prefill, 7))

    assert scheduler.reset_prefix_cache(reset_running_requests=True)
    resumed = scheduler.schedule()
    cached = resumed.scheduled_cached_reqs
    assert cached.resumed_req_ids == {request.request_id}
    assert request.request_id in cached.all_token_ids


def test_non_empty_step_replaces_record_as_default(two_devices, monkeypatch):
    """Only a fully empty step keeps the record; a step that schedules any
    request replaces it exactly as the default path does."""
    scheduler = _pp2_scheduler(monkeypatch, "1")
    request_a, request_b = create_requests(num_requests=2, num_tokens=20)
    scheduler.add_request(request_a)
    first = scheduler.schedule()
    assert scheduler.prev_step_scheduled_req_ids == {"0"}
    scheduler.add_request(request_b)
    second = scheduler.schedule()
    assert set(second.num_scheduled_tokens) == {"1"}
    assert scheduler.prev_step_scheduled_req_ids == {"1"}
    scheduler.update_from_output(first, _runner_output(first, 5))
    third = scheduler.schedule()
    assert set(third.num_scheduled_tokens) == {"0"}
    # Request 0 skipped the second step, which was not empty: history resent.
    assert "0" in third.scheduled_cached_reqs.all_token_ids


@pytest.mark.parametrize("value", ["2", "true", "on", "yes"])
def test_invalid_knob_value_raises(monkeypatch, value):
    _knob(monkeypatch, value)
    with pytest.raises(ValueError, match=KNOB):
        _fake_config()


@pytest.mark.parametrize(
    ("value", "expected"), [(None, False), ("0", False), (" 1 ", True)]
)
def test_knob_values(monkeypatch, value, expected):
    _knob(monkeypatch, value)
    assert pp_keep_batch_requested(_fake_config()) is expected


def _fake_config(**overrides):
    from vllm.config import SchedulerConfig

    scheduler = SchedulerConfig(max_model_len=512, is_encoder_decoder=False)
    spec = overrides.pop("speculative_config", None)
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=overrides.pop("pp", 2),
            distributed_executor_backend=overrides.pop("backend", "mp"),
        ),
        scheduler_config=SimpleNamespace(
            pp_keep_batch=scheduler.pp_keep_batch,
            async_scheduling=overrides.pop("async_scheduling", False),
        ),
        model_config=SimpleNamespace(
            runner_type=overrides.pop("runner_type", "generate"),
            is_hybrid=overrides.pop("is_hybrid", False),
        ),
        speculative_config=spec,
        use_v2_model_runner=overrides.pop("v2", False),
        kv_transfer_config=overrides.pop("kv_transfer_config", None),
        ec_transfer_config=overrides.pop("ec_transfer_config", None),
    )
    assert not overrides
    return config


def _fake_spec(ngram_gpu=False, ddtree=False):
    return SimpleNamespace(supports_pp_row_retention=lambda: not (ngram_gpu or ddtree))


def test_decision_off_returns_no_reason(monkeypatch):
    _knob(monkeypatch, None)
    assert pp_keep_batch_decision(_fake_config()) is False
    _knob(monkeypatch, "0")
    # Even for shapes the knob would not cover: off means silent default.
    assert pp_keep_batch_decision(_fake_config(async_scheduling=True)) is False


def test_decision_on_for_covered_shape(monkeypatch):
    _knob(monkeypatch, "1")
    assert pp_keep_batch_decision(_fake_config(speculative_config=_fake_spec())) is True


@pytest.mark.parametrize(
    ("overrides", "needle"),
    [
        ({"pp": 1}, "pipeline_parallel_size is 1"),
        ({"v2": True}, "V2 model runner"),
        ({"async_scheduling": True}, "async scheduling"),
        ({"backend": "external_launcher"}, "external_launcher"),
        ({"kv_transfer_config": object()}, "KV connector"),
        ({"ec_transfer_config": object()}, "encoder-cache connector"),
        ({"runner_type": "pooling"}, "pooling"),
        ({"is_hybrid": True}, "hybrid"),
        (
            {"speculative_config": _fake_spec(ngram_gpu=True)},
            "speculative configuration",
        ),
        ({"speculative_config": _fake_spec(ddtree=True)}, "speculative configuration"),
    ],
)
def test_explicit_unsupported_decision_refuses(monkeypatch, overrides, needle):
    _knob(monkeypatch, "1")
    with pytest.raises(ValueError, match=KNOB) as exc:
        pp_keep_batch_decision(_fake_config(**overrides))
    assert needle in str(exc.value)


def test_scheduler_rejects_explicit_keep_without_pp(monkeypatch):
    _knob(monkeypatch, "1")
    with pytest.raises(ValueError, match="pipeline_parallel_size is 1"):
        create_scheduler(device_config=DeviceConfig(device="cpu"))


def test_env_registration():
    import vllm.envs as envs

    assert "VLLM_PP_KEEP_BATCH" in envs.environment_variables
    from vllm.config import SchedulerConfig
    from vllm.config.policy_defaults import runtime_compile_ignored_aliases

    config = SimpleNamespace(
        scheduler_config=SchedulerConfig(max_model_len=512, is_encoder_decoder=False)
    )
    assert KNOB in runtime_compile_ignored_aliases(config)
