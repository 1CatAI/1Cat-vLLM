# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Speculative decoding under pipeline parallelism: the round state the last
PP rank broadcasts and the non-last ranks rebuild from it.

Two CPU processes over gloo play the last and a non-last rank of a PP=2
deployment with async scheduling, so the wire format (shapes, padding,
ordering) and the receiving rank's bookkeeping are checked end to end without
a GPU.
"""

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.utils.network_utils import get_open_port

WORLD_SIZE = 2
LAST_RANK = 1
NUM_SPEC_TOKENS = 2
NUM_REQS = 3
REQ_IDS = ["r0", "r1", "r2"]
PROMPT_LEN = 4
# The sampler emitted one column less than num_spec_tokens + 1 this round.
SAMPLED = [[5, 7], [9, -1], [11, 12]]
SAMPLED_PADDED = [[5, 7, -1], [9, -1, -1], [11, 12, -1]]
DRAFTS = [[1, 2], [3, 4], [5, 6]]
DISCARDED = [False, True, False]


class _FakeRequest:
    def __init__(self, prompt_token_ids: list[int]) -> None:
        self.prompt_token_ids = prompt_token_ids
        self.num_prompt_tokens = len(prompt_token_ids)
        self.output_token_ids: list[int] = []

    def get_token_id(self, idx: int) -> int:
        if idx < self.num_prompt_tokens:
            return self.prompt_token_ids[idx]
        return self.output_token_ids[idx - self.num_prompt_tokens]


def _pp_group(rank: int) -> SimpleNamespace:
    return SimpleNamespace(
        rank=rank,
        last_rank=LAST_RANK,
        is_last_rank=rank == LAST_RANK,
        world_size=WORLD_SIZE,
        # The speculative round state travels over cpu_group; the plain
        # [num_reqs, 1] path keeps device_group. One gloo group plays both.
        cpu_group=dist.group.WORLD,
        device_group=dist.group.WORLD,
    )


def _make_runner(rank: int, num_spec_tokens: int):
    # Late import: the child process must not pay for it before the group
    # exists, and the module patches below must hit the child's copy.
    import vllm.v1.worker.gpu_model_runner as runner_module
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    runner_module.get_pp_group = lambda: _pp_group(rank)

    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.device = torch.device("cpu")
    runner.num_spec_tokens = num_spec_tokens
    runner.requests = {
        req_id: _FakeRequest([10 * (i + 1) + j for j in range(PROMPT_LEN)])
        for i, req_id in enumerate(REQ_IDS)
    }
    runner.input_batch = SimpleNamespace(
        num_reqs=NUM_REQS,
        req_ids=list(REQ_IDS),
        num_tokens_no_spec=np.full(NUM_REQS, PROMPT_LEN, dtype=np.int32),
        is_token_ids=np.zeros((NUM_REQS, 16), dtype=bool),
        prev_sampled_token_ids=None,
        prev_req_id_to_index=None,
    )
    runner.discard_request_mask = SimpleNamespace(
        np=np.array(DISCARDED), gpu=torch.tensor(DISCARDED)
    )
    runner.valid_sampled_token_count_event = object()
    runner._is_all_reqs_chunked_prefill = lambda: False
    runner._draft_token_ids = None
    runner._pp_nonlast_scheduler_output = None
    return runner


def _run_last_rank(runner) -> None:
    # Round A: speculative round with a narrower sampler output.
    runner._draft_token_ids = torch.tensor(DRAFTS, dtype=torch.int64)
    runner._pp_broadcast_prev_sampled_token_ids(
        torch.tensor(SAMPLED, dtype=torch.int64)
    )
    runner._pp_broadcast_draft_token_ids()
    # Round B: same payload; the receiver has no stashed scheduler_output.
    runner._pp_broadcast_prev_sampled_token_ids(
        torch.tensor(SAMPLED, dtype=torch.int64)
    )
    runner._pp_broadcast_draft_token_ids()
    # Round C: list-form drafts (ngram) travel as zeros.
    runner._draft_token_ids = [[1], [2], [3]]
    runner._pp_broadcast_prev_sampled_token_ids(
        torch.tensor(SAMPLED, dtype=torch.int64)
    )
    runner._pp_broadcast_draft_token_ids()
    # Round D: no speculative decoding, the plain [num_reqs, 1] path.
    runner.num_spec_tokens = 0
    runner._pp_broadcast_prev_sampled_token_ids(
        torch.tensor([[5], [9], [11]], dtype=torch.int32)
    )


def _run_non_last_rank(runner) -> None:
    recorded: dict[str, Any] = {}
    runner._copy_valid_sampled_token_count = lambda ids, counts: recorded.update(
        next_token_ids=ids.clone(), counts=counts.clone()
    )
    runner._update_states_after_model_execute = (
        lambda sampled, scheduler_output: recorded.update(
            sampled=sampled.clone(), scheduler_output=scheduler_output
        )
    )

    # Round A
    stash = object()
    runner._pp_nonlast_scheduler_output = stash
    runner._pp_receive_prev_sampled_token_ids_to_input_batch()
    assert torch.equal(
        recorded["sampled"], torch.tensor(SAMPLED_PADDED, dtype=torch.int32)
    )
    # The last contiguous token of each row and how many there are. Row 1
    # is discarded; the bookkeeping below leaves it out.
    assert recorded["next_token_ids"].tolist() == [7, 9, 12]
    assert recorded["counts"].tolist() == [2, 1, 2]
    assert torch.equal(runner._draft_token_ids, torch.tensor(DRAFTS, dtype=torch.int32))
    assert recorded["scheduler_output"] is stash
    assert runner._pp_nonlast_scheduler_output is None
    # Bookkeeping of the existing PP+async path still runs for the
    # non-discarded rows.
    assert runner.input_batch.prev_req_id_to_index == {"r0": 0, "r2": 2}
    assert runner.requests["r0"].output_token_ids == [-1]
    assert runner.requests["r1"].output_token_ids == []
    assert runner.input_batch.num_tokens_no_spec.tolist() == [5, 4, 5]
    assert runner.input_batch.is_token_ids[0, PROMPT_LEN]
    assert not runner.input_batch.is_token_ids[1, PROMPT_LEN]

    # Round B: no stash -> loud failure, after both broadcasts were consumed
    # (otherwise the last rank would hang on the next collective).
    with pytest.raises(RuntimeError, match="stashed scheduler_output"):
        runner._pp_receive_prev_sampled_token_ids_to_input_batch()

    # Round C
    runner._pp_nonlast_scheduler_output = object()
    runner._pp_receive_prev_sampled_token_ids_to_input_batch()
    assert torch.equal(
        runner._draft_token_ids,
        torch.zeros((NUM_REQS, NUM_SPEC_TOKENS), dtype=torch.int32),
    )

    # Round D
    runner.num_spec_tokens = 0
    runner._pp_receive_prev_sampled_token_ids_to_input_batch()
    assert runner.input_batch.prev_sampled_token_ids.tolist() == [[5], [9], [11]]


def _worker(rank: int, port: int) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=WORLD_SIZE,
    )
    try:
        runner = _make_runner(rank, NUM_SPEC_TOKENS)
        if rank == LAST_RANK:
            _run_last_rank(runner)
        else:
            _run_non_last_rank(runner)
    finally:
        dist.destroy_process_group()


def test_pp_spec_decode_state_round_trip() -> None:
    port = get_open_port()
    ctx = mp.get_context("spawn")
    procs = [
        ctx.Process(target=_worker, args=(rank, port)) for rank in range(WORLD_SIZE)
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(timeout=300)
    for proc in procs:
        if proc.is_alive():
            proc.kill()
    assert [proc.exitcode for proc in procs] == [0, 0]
