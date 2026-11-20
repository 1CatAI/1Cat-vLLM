# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A request's next step never overtakes its sampled tokens (Model Runner V2,
pipeline parallelism, synchronous scheduling).

Under pipeline parallelism a stage that is not the last learns a step's sampled
tokens, accepted count and next drafts from the last stage through PPHandler's
slot ring (vllm/v1/worker/gpu/pp_utils.py): ``deque([None] * pp.world_size)``;
every execute_model call pops the oldest slot and applies it, then a sampling
step stores its own packet in the newest slot. So a step's packet is applied at
the start of the pp_size-th execute_model call after it, and not before. If the
same request runs again earlier, its first stage embeds whatever the
request-state row held before.

Without a spacing rule, the engine's batch-queue loop
(EngineCore.step_with_batch_queue) and the executor's lazy futures give a lone
request the call sequence R E R E E R E E R ... at three stages: its first
decode step follows the prompt's step after two calls, every later one after
three. Two stages (R E R E R ...) are always on time. AsyncScheduler already
makes a sampling step's request ineligible for pp_size scheduling steps; the
synchronous Scheduler does the same in _update_after_schedule.

The tests drive the real loop, the real scheduler and the executor's real
FutureWrapper; only the worker processes are replaced, by an object that records
the calls and answers each step when the engine waits for it. The expected
spacing comes from the ring above, not from the scheduler.
"""

from collections import deque

import pytest

from tests.v1.core.utils import create_requests, create_scheduler
from vllm.v1.engine.core import EngineCore
from vllm.v1.executor.multiproc_executor import FutureWrapper
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput

pytestmark = pytest.mark.cpu_test

MAX_TOKENS = 12


class RecordingWorkers:
    """Stands where the worker processes stand. Each execute_model call is
    recorded in order (the order the workers run them in); each answer is
    produced when the engine waits for it, through the executor's own
    FutureWrapper queue (a future is done only once it, or a later one, has
    been waited for)."""

    def __init__(self) -> None:
        self.futures: deque[FutureWrapper] = deque()
        # per execute_model call: the requests it samples for
        self.calls: list[tuple[str, ...]] = []
        self._step = None

    def execute_model(self, scheduler_output, non_block: bool = False):
        assert non_block
        self.calls.append(tuple(scheduler_output.num_scheduled_tokens))
        if not scheduler_output.total_num_scheduled_tokens:
            return FutureWrapper(self.futures, lambda: EMPTY_MODEL_RUNNER_OUTPUT)
        self._step = scheduler_output
        return FutureWrapper(self.futures, lambda: None)

    def sample_tokens(self, grammar_output, non_block: bool = False):
        assert non_block
        req_ids = list(self._step.num_scheduled_tokens)
        output = ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
            sampled_token_ids=[[7] for _ in req_ids],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        )
        return FutureWrapper(self.futures, lambda: output)

    def take_draft_token_ids(self):
        return None


def _engine(monkeypatch, pp_size: int, num_requests: int):
    # One V100 (compute capability 7.0) per stage. Scheduling is CPU-only.
    from vllm.platforms import current_platform

    monkeypatch.setattr(
        "vllm.config.parallel.current_platform.device_count", lambda: pp_size
    )
    monkeypatch.setattr(
        current_platform,
        "is_device_capability",
        lambda cap, device_id=0: cap == (7, 0),
    )
    scheduler = create_scheduler(async_scheduling=False, pipeline_parallel_size=pp_size)
    scheduler.use_v2_model_runner = True
    requests = create_requests(
        num_requests=num_requests, num_tokens=8, max_tokens=MAX_TOKENS
    )
    core = EngineCore.__new__(EngineCore)
    core.vllm_config = scheduler.vllm_config
    core.scheduler = scheduler
    core.model_executor = RecordingWorkers()
    # MultiprocExecutor.max_concurrent_batches under pipeline parallelism
    core.batch_queue_size = pp_size
    core.batch_queue = deque(maxlen=pp_size)
    # EngineCore.__init__ defaults (VLLM_V41_PP_MICRO / PP_AHEAD off)
    core._v41_depth = pp_size
    core._v41_skip = 0
    core._v41_micro = False
    core._sm70_async_cpu_trace_step = 0
    core.is_ec_consumer = True
    core.is_pooling_model = False
    core.use_spec_decode = False
    core.async_scheduling = False
    core._process_aborts_queue = lambda: None
    return core, requests


def _run(core, requests, arrivals: dict[int, int]) -> list[tuple[str, ...]]:
    """Run the engine loop until every request has finished. arrivals: loop
    iteration -> index of the request that arrives before it."""
    iteration = 0
    while (
        iteration <= max(arrivals) or core.scheduler.has_requests() or core.batch_queue
    ):
        if iteration in arrivals:
            core.scheduler.add_request(requests[arrivals[iteration]])
        if core.scheduler.has_requests() or core.batch_queue:
            core.step_with_batch_queue()
        iteration += 1
        assert iteration < 2000, "the engine loop did not finish"
    return core.model_executor.calls


def _overtaken(calls: list[tuple[str, ...]], pp_size: int) -> list[tuple[int, str]]:
    """Replay PPHandler's ring over the recorded calls: (call index, request) for
    every step of a request that ran while the packet of that request's
    previous step was still in the ring."""
    ring: deque[tuple[str, ...] | None] = deque([None] * pp_size)
    waiting: set[str] = set()
    overtaken = []
    for index, sampled in enumerate(calls):
        applied = ring.popleft()
        ring.append(None)
        waiting -= set(applied or ())
        overtaken += [(index, req_id) for req_id in sampled if req_id in waiting]
        if sampled:
            ring[-1] = sampled
            waiting |= set(sampled)
    return overtaken


@pytest.mark.parametrize("pp_size", [2, 3, 4])
def test_lone_request_never_runs_before_its_sampled_tokens_arrive(monkeypatch, pp_size):
    core, requests = _engine(monkeypatch, pp_size, num_requests=1)
    calls = _run(core, requests, {0: 0})
    steps = [index for index, sampled in enumerate(calls) if sampled]
    # the prompt's step and MAX_TOKENS - 1 decode steps
    assert len(steps) == MAX_TOKENS
    assert _overtaken(calls, pp_size) == []
    # and no slower than the ring allows: a lone request runs every
    # pp_size-th call
    assert [b - a for a, b in zip(steps, steps[1:])] == [pp_size] * (MAX_TOKENS - 1)


@pytest.mark.parametrize("pp_size", [2, 3, 4])
@pytest.mark.parametrize("second_arrives", [1, 2, 3, 4, 5, 6, 7, 8])
def test_requests_arriving_at_any_point_never_overtake(
    monkeypatch, pp_size, second_arrives
):
    core, requests = _engine(monkeypatch, pp_size, num_requests=3)
    calls = _run(core, requests, {0: 0, second_arrives: 1, second_arrives + 5: 2})
    assert sum(len(sampled) for sampled in calls) == 3 * MAX_TOKENS
    assert _overtaken(calls, pp_size) == []


def test_two_stage_call_sequence_is_unchanged(monkeypatch):
    """Two stages: a lone request's calls were R E R E R ... without the rule
    and stay so."""
    core, requests = _engine(monkeypatch, 2, num_requests=1)
    calls = _run(core, requests, {0: 0})
    sequence = ["R" if sampled else "E" for sampled in calls[: 2 * MAX_TOKENS - 1]]
    assert sequence == ["R", "E"] * (MAX_TOKENS - 1) + ["R"]


def test_v1_model_runner_is_not_spaced(monkeypatch):
    """The V1 model runner gets each step's sampled tokens from the scheduler
    output, not through a ring, so its call sequence stays as it was."""
    core, requests = _engine(monkeypatch, 3, num_requests=1)
    core.scheduler.use_v2_model_runner = False
    calls = _run(core, requests, {0: 0})
    steps = [index for index, sampled in enumerate(calls) if sampled]
    assert len(steps) == MAX_TOKENS
    assert steps[1] - steps[0] == 2
