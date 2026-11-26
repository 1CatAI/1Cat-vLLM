# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_V41_PP_EMPTY_AHEAD, VLLM_V41_PP_ANSWER_NO_SLEEP and VLLM_V41_PP_LOOP_STATS:
the engine core's loop between a decode step's answer and the request's next
step (synchronous scheduling, Model Runner V2, pipeline parallelism).

With the scheduler's spacing rule a request's next step is the pp_size-th call
after its last (tests/v1/core/test_pp_sampled_ring_cadence.py). What the loop
does in between was measured on a twelve-card V100 build (PP3 x TP4,
2026-11-17, profiler traces and the engine core's own counters):

- after a prompt read of one step: one empty step is submitted AFTER each
  answer, between two millisecond sleeps, before the request's next step (one
  empty step in between in 126 of 126 and 68 of 68 rounds);
- after a read of several steps: no empty step in between (67 of 67, 66 of 66);
  one sleep remains.

The tests drive the real loop (EngineCoreProc._process_engine_step,
EngineCore.step_with_batch_queue), the switches' own reading of the environment
(EngineCore._init_v41_loop_switches), the real scheduler with the spacing rule
and the executor's real FutureWrapper. Only the workers are replaced: a step
with tokens is answered when the engine waits for it, an empty step (no work on
the cards) at once. Expected values: the two measured states above, the
switches' stated effect on them, and PPHandler's ring replayed over the calls.
"""

import logging
import queue
from collections import deque
from concurrent.futures import Future

import pytest

from tests.v1.core.test_pp_sampled_ring_cadence import RecordingWorkers, _overtaken
from tests.v1.core.utils import create_requests, create_scheduler
from vllm.v1.engine.core import (
    V41_ANSWER_NO_SLEEP_KNOB,
    V41_EMPTY_AHEAD_KNOB,
    V41_LOOP_STATS_KNOB,
    EngineCoreProc,
)
from vllm.v1.executor.multiproc_executor import FutureWrapper
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput

pytestmark = pytest.mark.cpu_test

AHEAD = {V41_EMPTY_AHEAD_KNOB: "1"}
NO_SLEEP = {V41_ANSWER_NO_SLEEP_KNOB: "1"}
BOTH = {**AHEAD, **NO_SLEEP}
MAX_TOKENS = 10
# a prompt read in one step, and one read in three steps of 128 rows
ONE_STEP = dict(num_tokens=8, chunk=8192)
THREE_STEPS = dict(num_tokens=300, chunk=128)


class LoggingWorkers(RecordingWorkers):
    """RecordingWorkers that also log each submission (R: a step with tokens,
    E: an empty step), answer an empty step at once, and, as the model runner
    does, sample nothing for a prompt chunk that is not the prompt's last
    (PPHandler sends no packet for it either: compute_need_sampled_mask)."""

    def __init__(self, log: list[str], scheduler) -> None:
        super().__init__()
        self.log = log
        self.scheduler = scheduler

    def execute_model(self, scheduler_output, non_block: bool = False):
        assert non_block
        if not scheduler_output.total_num_scheduled_tokens:
            self.log.append("E")
            self.calls.append(())
            done: Future = Future()
            done.set_result(EMPTY_MODEL_RUNNER_OUTPUT)
            return done
        self.log.append("R")
        self.calls.append(
            tuple(
                req_id
                for req_id in scheduler_output.num_scheduled_tokens
                if not self.scheduler.requests[req_id].is_prefill_chunk
            )
        )
        self._step = scheduler_output
        return FutureWrapper(self.futures, lambda: None)

    def sample_tokens(self, grammar_output, non_block: bool = False):
        assert non_block
        req_ids = list(self._step.num_scheduled_tokens)
        sampling = set(self.calls[-1])
        output = ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
            sampled_token_ids=[[7] if r in sampling else [] for r in req_ids],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        )
        return FutureWrapper(self.futures, lambda: output)


def _engine(monkeypatch, pp_size: int, env: dict[str, str], chunk: int):
    from vllm.platforms import current_platform

    for knob in (V41_EMPTY_AHEAD_KNOB, V41_ANSWER_NO_SLEEP_KNOB, V41_LOOP_STATS_KNOB):
        monkeypatch.delenv(knob, raising=False)
    for knob, value in env.items():
        monkeypatch.setenv(knob, value)
    monkeypatch.setattr(
        "vllm.config.parallel.current_platform.device_count", lambda: pp_size
    )
    monkeypatch.setattr(
        current_platform,
        "is_device_capability",
        lambda cap, device_id=0: cap == (7, 0),
    )
    scheduler = create_scheduler(
        async_scheduling=False,
        pipeline_parallel_size=pp_size,
        max_num_batched_tokens=chunk,
        long_prefill_token_threshold=chunk if chunk < 8192 else 0,
    )
    scheduler.use_v2_model_runner = True
    log: list[str] = []
    update = scheduler.update_from_output

    def logged_update(scheduler_output, model_output):
        # a: the answer of a step with tokens is applied; e: an empty step's
        log.append("a" if scheduler_output.total_num_scheduled_tokens else "e")
        return update(scheduler_output, model_output)

    scheduler.update_from_output = logged_update
    monkeypatch.setattr(
        "vllm.v1.engine.core.time.sleep", lambda seconds: log.append("z")
    )
    core = EngineCoreProc.__new__(EngineCoreProc)
    core.vllm_config = scheduler.vllm_config
    core.scheduler = scheduler
    core.model_executor = LoggingWorkers(log, scheduler)
    core.batch_queue_size = pp_size
    core.batch_queue = deque(maxlen=pp_size)
    core._v41_depth = pp_size
    core._v41_skip = 0
    core._v41_micro = False
    core._sm70_async_cpu_trace_step = 0
    core.is_ec_consumer = True
    core.is_pooling_model = False
    core.use_spec_decode = False
    core.async_scheduling = False
    core._process_aborts_queue = lambda: None
    core.output_queue = queue.Queue()
    core.post_step = lambda model_executed: None
    core.step_fn = core.step_with_batch_queue
    core._init_v41_loop_switches()
    return core, log


def _play(monkeypatch, pp_size, env, num_tokens, chunk):
    core, log = _engine(monkeypatch, pp_size, env, chunk)
    (request,) = create_requests(
        num_requests=1, num_tokens=num_tokens, max_tokens=MAX_TOKENS
    )
    core.scheduler.add_request(request)
    iterations = 0
    while core.scheduler.has_requests() or core.batch_queue:
        core._process_engine_step()
        iterations += 1
        assert iterations < 1000, "the engine loop did not finish"
    return core, "".join(log)


def _between(log: str) -> list[str]:
    """For every decode round but the first: what the loop did between a token
    step's answer and the submission of the next step with tokens."""
    segments = []
    start = log.index("a")
    while True:
        nxt = log.find("R", start)
        if nxt < 0:
            break
        segments.append(log[start + 1 : nxt])
        start = log.find("a", nxt)
        if start < 0:
            break
    return segments[1:]


@pytest.mark.parametrize("pp_size", [3, 4])
@pytest.mark.parametrize("read", ["one", "three"])
@pytest.mark.parametrize(
    "env", [{}, AHEAD, NO_SLEEP, BOTH], ids=["off", "ahead", "nosleep", "both"]
)
def test_the_round_between_answer_and_next_step(monkeypatch, pp_size, read, env):
    core, log = _play(
        monkeypatch, pp_size, env, **(ONE_STEP if read == "one" else THREE_STEPS)
    )
    calls = core.model_executor.calls
    steps = [i for i, sampled in enumerate(calls) if sampled]
    assert len(steps) == MAX_TOKENS  # the prompt's last step and the decode steps
    decode = steps[1:]
    # The switches change no call: the ring is never overtaken and a lone
    # request's decode steps are exactly pp_size calls apart.
    assert _overtaken(calls, pp_size) == []
    assert [b - a for a, b in zip(decode, decode[1:])] == [pp_size] * (MAX_TOKENS - 2)
    rounds = _between(log)
    assert len(rounds) >= MAX_TOKENS - 3
    empties = {r.count("E") for r in rounds}
    sleeps = {r.count("z") for r in rounds}
    ahead = V41_EMPTY_AHEAD_KNOB in env
    no_sleep = V41_ANSWER_NO_SLEEP_KNOB in env
    if read == "one" and not ahead:
        # the measured first state: spacing steps submitted after the answer,
        # each followed by a sleep, plus the sleep after the answer
        assert empties == {pp_size - 2}
        assert sleeps == {pp_size - 2 + (0 if no_sleep else 1)}
    else:
        # the measured second state (and what EMPTY_AHEAD makes of the first):
        # every spacing step out before the answer
        assert empties == {0}
        assert sleeps == {0 if no_sleep else 1}


def test_off_is_the_measured_first_state_at_three_stages(monkeypatch):
    """The measured sequence after a one-step read at three stages: answer,
    sleep, empty step, its answer, sleep, the request's step."""
    _, log = _play(monkeypatch, 3, {}, **ONE_STEP)
    assert set(_between(log)) == {"zEez"}
    _, log = _play(monkeypatch, 3, BOTH, **ONE_STEP)
    assert set(_between(log)) == {""}


@pytest.mark.parametrize(
    "env,read,expected",
    [
        ({}, "one", "no empty step in between in 0 of 4, sleeps in between 8,"),
        ({}, "three", "no empty step in between in 4 of 4, sleeps in between 4,"),
        (BOTH, "one", "no empty step in between in 4 of 4, sleeps in between 0,"),
    ],
)
def test_loop_stats_count_the_rounds(monkeypatch, caplog, env, read, expected):
    env = {**env, V41_LOOP_STATS_KNOB: "4"}
    with caplog.at_level(logging.INFO, logger="vllm.v1.engine.core"):
        _play(monkeypatch, 3, env, **(ONE_STEP if read == "one" else THREE_STEPS))
    lines = [m for m in caplog.messages if m.startswith(V41_LOOP_STATS_KNOB + ":")]
    # MAX_TOKENS - 1 = 9 decode rounds follow an answer: two full lines of 4
    assert len(lines) == 2
    assert all(expected in line for line in lines)
    assert all("blocking draft takes in between 0;" in line for line in lines)


def test_switches_off_log_nothing(monkeypatch, caplog):
    with caplog.at_level(logging.INFO, logger="vllm.v1.engine.core"):
        _play(monkeypatch, 3, {}, **ONE_STEP)
    for knob in (V41_EMPTY_AHEAD_KNOB, V41_ANSWER_NO_SLEEP_KNOB, V41_LOOP_STATS_KNOB):
        assert knob not in caplog.text


def test_switches_log_once_when_they_act(monkeypatch, caplog):
    with caplog.at_level(logging.INFO, logger="vllm.v1.engine.core"):
        _play(monkeypatch, 3, BOTH, **ONE_STEP)
    assert caplog.text.count(f"{V41_EMPTY_AHEAD_KNOB}: first return") == 1
    assert caplog.text.count(f"{V41_ANSWER_NO_SLEEP_KNOB}: first loop") == 1


@pytest.mark.parametrize(
    "pp_size,env,async_scheduling,message",
    [
        (2, AHEAD, False, "needs three or more pipeline stages"),
        (3, AHEAD, True, "needs sync scheduling"),
        (3, NO_SLEEP, True, "needs sync scheduling"),
        (3, {V41_EMPTY_AHEAD_KNOB: "yes"}, False, "must be 0 or 1"),
        (3, {V41_ANSWER_NO_SLEEP_KNOB: "2"}, False, "must be 0 or 1"),
        (3, {V41_LOOP_STATS_KNOB: "-1"}, False, "must be 0 or a count"),
    ],
)
def test_a_switch_that_cannot_act_refuses(
    monkeypatch, pp_size, env, async_scheduling, message
):
    core, _ = _engine(monkeypatch, pp_size, {}, 8192)
    for knob, value in env.items():
        monkeypatch.setenv(knob, value)
    core.async_scheduling = async_scheduling
    with pytest.raises(ValueError, match=message):
        core._init_v41_loop_switches()


def test_empty_ahead_refuses_the_micro_batch_path(monkeypatch):
    core, _ = _engine(monkeypatch, 3, {}, 8192)
    monkeypatch.setenv(V41_EMPTY_AHEAD_KNOB, "1")
    core._v41_micro = True
    with pytest.raises(ValueError, match="does not combine"):
        core._init_v41_loop_switches()


def test_empty_ahead_refuses_data_parallelism(monkeypatch):
    """The data-parallel loop runs a dummy batch after a step that did not
    execute; an ahead-submitted empty step returns as one that did."""
    core, _ = _engine(monkeypatch, 3, {}, 8192)
    monkeypatch.setenv(V41_EMPTY_AHEAD_KNOB, "1")
    core.vllm_config.parallel_config.data_parallel_size = 2
    with pytest.raises(ValueError, match="data parallelism"):
        core._init_v41_loop_switches()


def test_no_batch_queue_refuses(monkeypatch):
    core, _ = _engine(monkeypatch, 3, {}, 8192)
    core.batch_queue = None
    for knob, value in ((V41_ANSWER_NO_SLEEP_KNOB, "1"), (V41_LOOP_STATS_KNOB, "8")):
        monkeypatch.setenv(knob, value)
        with pytest.raises(ValueError, match="needs pipeline parallelism"):
            core._init_v41_loop_switches()
        monkeypatch.delenv(knob)
