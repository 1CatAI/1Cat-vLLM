# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batch-queue draft cadence of the real EngineCore.step_with_batch_queue.

Only the engine loop is real. The scheduler and executor are scripted: the
executor's futures stay in flight until the engine blocks on them, and
the shared latest-draft RPC raises if the queue tries to use it. Expected values
come from the cadence itself: every token-bearing step's drafts must reach the
scheduler after that step's update_from_output and before the next schedule.
"""

import queue
from collections import deque
from types import SimpleNamespace

import pytest

from vllm.v1.engine.core import EngineCore
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput


class _InFlight:
    def __init__(self, value):
        self.value = value

    def done(self):
        return False

    def result(self):
        return self.value


class _Scheduler:
    def __init__(self, plan, log):
        self.plan = deque(plan)
        self.log = log
        self.step = 0

    def has_requests(self):
        return True

    def schedule(self):
        reqs = self.plan.popleft() if self.plan else ()
        self.step += 1
        self.log.append(("schedule", self.step, tuple(reqs)))
        return SimpleNamespace(
            step=self.step,
            reqs=tuple(reqs),
            total_num_scheduled_tokens=6 * len(reqs),
            pending_structured_output_tokens=False,
            has_structured_output_requests=False,
            scheduled_spec_decode_tokens={},
        )

    def get_grammar_bitmask(self, scheduler_output):
        return None

    def update_from_output(self, scheduler_output, model_output):
        self.log.append(("update", scheduler_output.step, scheduler_output.reqs))
        return {}

    def update_draft_token_ids(self, drafts):
        self.log.append(("drafts", drafts.step, tuple(drafts.req_ids)))

    def make_stats(self):
        return None


class _Executor:
    def __init__(self, attach):
        self.attach = attach
        self.pending = None

    @staticmethod
    def _drafts(scheduler_output):
        drafts = DraftTokenIds(
            list(scheduler_output.reqs), [[-1] * 5 for _ in scheduler_output.reqs]
        )
        drafts.step = scheduler_output.step
        return drafts

    def execute_model(self, scheduler_output, non_block=True):
        if not scheduler_output.total_num_scheduled_tokens:
            return _InFlight(ModelRunnerOutput(req_ids=[], req_id_to_index={}))
        self.pending = scheduler_output
        return _InFlight(None)

    def sample_tokens(self, grammar_output, non_block=True):
        pending = self.pending
        assert pending is not None
        drafts = self._drafts(pending) if self.attach else None
        return _InFlight(
            ModelRunnerOutput(
                req_ids=list(pending.reqs),
                req_id_to_index={req: i for i, req in enumerate(pending.reqs)},
                draft_token_ids=drafts,
            )
        )

    def take_draft_token_ids(self):
        raise AssertionError(
            "queued steps must never read the shared latest-draft slot"
        )


def _run(plan, *, synchronous, attach):
    log: list[tuple] = []
    engine = EngineCore.__new__(EngineCore)
    engine.vllm_config = SimpleNamespace(
        observability_config=SimpleNamespace(
            runtime_trace=SimpleNamespace(async_cpu=False, async_every=1),
            enable_logging_iteration_details=False,
        )
    )
    engine._sm70_async_cpu_trace_step = 0
    engine.batch_queue = deque()
    engine.batch_queue_size = 2
    # What __init__ sets with the DeepSeek-V4.1 PP switches off (9fbc11c6f).
    engine._v41_micro = False
    engine._v41_depth = engine.batch_queue_size
    engine._v41_skip = 0
    engine.scheduler = _Scheduler(plan, log)
    engine.model_executor = _Executor(attach)
    engine.is_ec_consumer = True
    engine.is_pooling_model = False
    engine.use_spec_decode = True
    engine.async_scheduling = not synchronous
    engine.aborts_queue = queue.Queue()
    for _ in range(len(plan)):
        engine.step_with_batch_queue()
    return log


def _late_or_missing(log):
    """Token-bearing updates whose own drafts did not arrive before the next
    schedule, or that received another step's drafts."""
    bad = []
    for i, event in enumerate(log):
        if event[0] != "update" or not event[2]:
            continue
        end = next(
            (j for j in range(i + 1, len(log)) if log[j][0] == "schedule"), len(log)
        )
        drafts = [e for e in log[i + 1 : end] if e[0] == "drafts"]
        if not drafts or any(e[1] != event[1] for e in drafts):
            bad.append(event)
    return bad


PLANS = {
    "two_interleaved": [("r1",), ("r2",)] * 20,
    "three_two_groups": [("r1", "r2"), ("r3",)] * 20,
    "one_left": [("r1",), ()] * 20,
}


@pytest.mark.parametrize("name", sorted(PLANS))
def test_synchronous_drafts_follow_their_own_step(name):
    log = _run(PLANS[name], synchronous=True, attach=True)
    assert [e for e in log if e[0] == "update" and e[2]]
    assert _late_or_missing(log) == []


def test_single_request_attaches_drafts_every_token_step():
    log = _run(PLANS["one_left"], synchronous=True, attach=True)
    assert [e for e in log if e[0] == "update" and e[2]]
    assert _late_or_missing(log) == []


def test_synchronous_step_without_attached_drafts_refuses():
    with pytest.raises(RuntimeError, match="PP synchronous speculative decoding"):
        _run(PLANS["two_interleaved"], synchronous=True, attach=False)


def test_async_queue_does_not_require_host_drafts():
    log = _run(PLANS["two_interleaved"], synchronous=False, attach=False)
    assert [e for e in log if e[0] == "update"]
    assert not [e for e in log if e[0] == "drafts"]


def test_output_snapshot_survives_serialization_and_later_worker_mutation():
    import pickle

    drafts = DraftTokenIds(["a"], [[10, 20]])
    first = ModelRunnerOutput(req_ids=["a"], req_id_to_index={"a": 0})
    first.set_draft_token_ids(drafts)
    drafts.req_ids[0] = "b"
    drafts.draft_token_ids[0][0] = 99
    restored = pickle.loads(pickle.dumps(first))
    assert restored.draft_token_ids.req_ids == ["a"]
    assert restored.draft_token_ids.draft_token_ids == [[10, 20]]


def test_output_refuses_missing_worker_drafts():
    output = ModelRunnerOutput(req_ids=["a"], req_id_to_index={"a": 0})
    with pytest.raises(RuntimeError, match="PP synchronous speculative decoding"):
        output.set_draft_token_ids(None)


@pytest.mark.parametrize("queued", [False, True])
def test_post_step_only_takes_drafts_without_a_queue(queued):
    drafts = DraftTokenIds(["a"], [[7]])
    calls = []

    def take():
        calls.append("take")
        return drafts

    engine = EngineCore.__new__(EngineCore)
    engine.async_scheduling = False
    engine.use_spec_decode = True
    engine.batch_queue = deque() if queued else None
    engine.model_executor = SimpleNamespace(take_draft_token_ids=take)
    engine.scheduler = SimpleNamespace(
        update_draft_token_ids=lambda value: calls.append(value)
    )
    engine.post_step(True)
    assert calls == ([] if queued else ["take", drafts])


def test_empty_sync_steps_do_not_require_proposals():
    log = _run([()] * 6, synchronous=True, attach=False)
    assert not [e for e in log if e[0] == "drafts"]
