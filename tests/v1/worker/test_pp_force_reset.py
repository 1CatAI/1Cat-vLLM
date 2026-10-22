# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Force-reset must invalidate delayed PP commits before physical block reuse.

The real runner execute/state-removal and PP feedback-generation paths run on
CPU. A small physical-state array stands in for Mamba GPU blocks; the forward
launch is deliberately stopped after request-state preparation.
"""

from collections import deque
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
from vllm.v1.worker.gpu import model_runner, pp_utils
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

pytestmark = pytest.mark.cpu_test


class _BeforeForward(Exception):
    pass


def _resumed_request() -> NewRequestData:
    return NewRequestData(
        req_id="a",
        prompt_token_ids=[1],
        mm_features=[],
        sampling_params=None,
        pooling_params=None,
        block_ids=([1],),
        num_computed_tokens=0,
        lora_request=None,
        prefill_token_ids=[1],
    )


@pytest.fixture
def worker(monkeypatch):
    runner: Any = object.__new__(GPUModelRunner)
    handler: Any = object.__new__(pp_utils.PPHandler)
    handler.req_idx_gen_np = np.zeros(2, dtype=np.int32)
    handler.main_stream = SimpleNamespace(wait_event=lambda _: None)
    handler.device = torch.device("cpu")
    runner.device = torch.device("cpu")
    runner.pp_handler = handler
    runner.cache_config = SimpleNamespace(mamba_cache_mode="align")
    runner.encoder_cache = None
    runner.prompt_logprobs_worker = None
    runner.lora_state = SimpleNamespace(remove_request=lambda _: None)
    live = {"a": 0, "b": 1}
    physical = torch.tensor([100, 200], dtype=torch.int64)
    blocks = {0: 0, 1: 1}
    committed: list[int] = []
    runner.req_states = SimpleNamespace(
        remove_request=lambda rid: live.pop(rid, None),
        draft_tokens=torch.empty((2, 4), dtype=torch.int64),
    )

    def commit(indices, sampled, *_):
        for row, req_idx in enumerate(indices.tolist()):
            if req_idx >= 0:
                physical[blocks[req_idx]] = sampled[row, 0]
                committed.append(req_idx)

    def add_requests(output):
        for request in output.scheduled_new_reqs:
            assert request.req_id == "a"
            # Match the real add_requests replacement lifecycle. Without the
            # reset notification, re-adding a still retires a, but not b.
            runner._remove_request(request.req_id)
            assert request.req_id not in live
            live[request.req_id] = 0
            # Reuse b's old physical block for a's replayed prompt. An old b
            # commit after this point would corrupt a's new recurrent state.
            blocks[0] = 1
            physical[1] = 1000

    def stop_before_forward():
        raise _BeforeForward

    runner.postprocess_sampled = commit
    runner.add_requests = add_requests
    runner.update_requests = lambda _: None
    runner.block_tables = SimpleNamespace(
        gather_block_tables=lambda *_args, **_kwargs: None,
        apply_staged_writes=stop_before_forward,
    )
    monkeypatch.setattr(model_runner, "scatter_draft_tokens", lambda *_: None)
    monkeypatch.setattr(
        pp_utils, "async_copy_to_gpu", lambda values, **_: torch.from_numpy(values)
    )

    def pending(indices, value=9999):
        indices_np = np.array(indices, dtype=np.int32)
        count = len(indices)
        event: Any = None  # No CUDA event is needed by the fake main stream.
        return pp_utils.PendingRecv(
            event=event,
            sampled_tokens=torch.full((count, 5), value, dtype=torch.int64),
            draft_tokens=torch.zeros((count, 4), dtype=torch.int64),
            num_sampled=torch.ones(count, dtype=torch.int32),
            num_rejected=torch.zeros(count, dtype=torch.int32),
            idx_mapping=torch.from_numpy(indices_np),
            idx_mapping_np=indices_np,
            need_sampled_mask=np.ones(count, dtype=np.bool_),
            gen_at_receive_np=handler.req_idx_gen_np[indices_np].copy(),
        )

    def execute(output):
        # Call the real method instead of reimplementing its critical ordering.
        with pytest.raises(_BeforeForward):
            runner.execute_model(output)

    return SimpleNamespace(
        runner=runner,
        handler=handler,
        live=live,
        physical=physical,
        committed=committed,
        pending=pending,
        execute=execute,
    )


@pytest.mark.parametrize("depth", [2, 4, 7])
def test_reset_drops_all_old_frames_when_only_one_request_resumes(worker, depth):
    # The first packet is already due, and later packets remain queued. All
    # must be masked, including b, which does not resume with a.
    worker.handler.queue = deque(
        [worker.pending([0])] + [None] * (depth - 2) + [worker.pending([1])]
    )
    reset = SchedulerOutput.make_empty()
    reset.prefix_cache_reset_req_ids = {"a", "b"}
    reset.scheduled_new_reqs = [_resumed_request()]
    worker.execute(reset)
    assert worker.live == {"a": 0}
    assert worker.handler.req_idx_gen_np.tolist() == [1, 1]
    assert not worker.committed
    assert worker.physical.tolist() == [100, 1000]

    # New a owns the same worker index, but only its new generation is valid.
    worker.handler.queue[-1] = worker.pending([0], value=1234)
    for _ in range(depth - 1):
        worker.execute(SchedulerOutput.make_empty())
        assert not worker.committed
        assert worker.physical.tolist() == [100, 1000]
    worker.execute(SchedulerOutput.make_empty())
    assert worker.committed == [0]
    assert worker.physical.tolist() == [100, 1234]


@pytest.mark.parametrize("depth", [2, 4, 7])
def test_repeated_reset_and_cancellation_invalidate_each_generation(worker, depth):
    worker.handler.queue = deque(
        [worker.pending([0])] + [None] * (depth - 2) + [worker.pending([1])]
    )
    reset = SchedulerOutput.make_empty()
    reset.prefix_cache_reset_req_ids = {"a", "b"}
    reset.finished_req_ids = {"b"}
    reset.scheduled_new_reqs = [_resumed_request()]
    worker.execute(reset)
    assert worker.handler.req_idx_gen_np.tolist() == [1, 1]
    worker.handler.queue[-1] = worker.pending([0], value=7777)

    repeated = SchedulerOutput.make_empty()
    repeated.prefix_cache_reset_req_ids = {"a", "b"}
    repeated.scheduled_new_reqs = [_resumed_request()]
    worker.execute(repeated)
    assert worker.handler.req_idx_gen_np.tolist() == [2, 1]
    for _ in range(depth):
        worker.execute(SchedulerOutput.make_empty())
    assert not worker.committed
    assert worker.physical.tolist() == [100, 1000]


def test_ordinary_preemption_keeps_due_output_and_retires_generation(worker):
    worker.handler.queue = deque([worker.pending([1]), None])
    output = SchedulerOutput.make_empty()
    output.preempted_req_ids = {"b"}
    worker.execute(output)
    # This path deliberately retains the existing ordinary-preemption order:
    # consume its due result, then retire the worker request generation.
    assert worker.committed == [1]
    assert worker.handler.req_idx_gen_np.tolist() == [0, 1]
    assert worker.live == {"a": 0}
    worker.execute(SchedulerOutput.make_empty())
    assert worker.committed == [1]
