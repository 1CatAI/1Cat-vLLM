# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest

from vllm.config import CUDAGraphMode
from vllm.v1.worker.gpu import model_runner as mrv2
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    _is_compatible,
    get_uniform_decode_token_count,
)
from vllm.v1.worker.gpu.model_runner import GPUModelRunner


@pytest.mark.parametrize("query_len", [1, 3, 4, 5, 7, 8, 16])
@pytest.mark.parametrize("phase", ["fresh", "tail", "mixed", "decode", "cached"])
def test_prefill_cannot_replay_decode_graph(query_len, phase):
    # Keep an unscheduled prefill at index 0 and schedule in reverse state order.
    # This also checks that request IDs, not batch positions, select state rows.
    prompt = 800 + query_len
    computed = [0, prompt, prompt]
    if phase == "fresh":
        computed[1:] = [0, 0]
    elif phase in ("tail", "mixed"):
        computed[1] = 800
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.req_states = SimpleNamespace(
        req_id_to_index={"unused": 0, "a": 1, "b": 2},
        num_computed_prefill_tokens=np.array(computed),
        prefill_len=SimpleNamespace(np=np.array([prompt] * 3)),
    )
    req_ids = ["b", "a"] if phase in ("mixed", "decode", "cached") else ["a"]
    schedule = SimpleNamespace(
        num_scheduled_tokens=dict.fromkeys(req_ids, query_len),
        total_num_scheduled_tokens=len(req_ids) * query_len,
    )
    uniform = runner._get_uniform_decode_token_count(schedule, False)
    expected = query_len if phase in ("decode", "cached") else None
    assert uniform == expected
    full = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.FULL,
        num_tokens=schedule.total_num_scheduled_tokens,
        num_reqs=len(req_ids),
        uniform_token_count=query_len,
    )
    assert _is_compatible(
        full, len(req_ids), schedule.total_num_scheduled_tokens, uniform
    ) == (expected is not None)


def test_dummy_and_nonuniform_batches_do_not_read_live_state():
    runner = GPUModelRunner.__new__(GPUModelRunner)
    schedule = SimpleNamespace(
        num_scheduled_tokens={"dummy": 4}, total_num_scheduled_tokens=4
    )
    assert runner._get_uniform_decode_token_count(schedule, True) == 4
    schedule.num_scheduled_tokens = {"a": 4, "b": 3}
    schedule.total_num_scheduled_tokens = 7
    assert runner._get_uniform_decode_token_count(schedule, False) is None


@pytest.mark.parametrize("has_prefill", [False, True])
def test_draft_phase_aware_predicate(has_prefill):
    assert get_uniform_decode_token_count(2, 8, 4, has_prefill) == (
        None if has_prefill else 4
    )
    assert get_uniform_decode_token_count(2, 7, 4, has_prefill) is None
    assert get_uniform_decode_token_count(0, 0, 0, has_prefill) is None


def test_execute_rejects_prefill_before_dp_graph_dispatch(monkeypatch):
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.req_states = SimpleNamespace(
        req_id_to_index={"tail": 0},
        num_computed_prefill_tokens=np.array([800]),
        prefill_len=SimpleNamespace(np=np.array([804])),
    )
    runner.update_pp_decode_requests = lambda: None
    for name in ("finish_requests", "free_states", "add_requests", "update_requests"):
        setattr(runner, name, lambda _: None)
    runner.block_tables = SimpleNamespace(apply_staged_writes=lambda: None)
    runner.is_encoder_decoder = False
    runner.cudagraph_manager = object()
    runner.dp_size, runner.dp_rank = 1, 0
    schedule = SimpleNamespace(
        num_scheduled_tokens={"tail": 4},
        total_num_scheduled_tokens=4,
        new_block_ids_to_zero=[],
    )

    class DispatchReached(Exception):
        pass

    def dispatch(manager, num_reqs, num_tokens, uniform_token_count, *args, **kwargs):
        assert (num_reqs, num_tokens) == (1, 4)
        assert uniform_token_count is None
        raise DispatchReached

    monkeypatch.setattr(mrv2, "dispatch_cg_and_sync_dp", dispatch)
    with pytest.raises(DispatchReached):
        runner.execute_model(schedule)
