# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_PP_KEEP_BATCH, runner side.

The real scheduler drives the real ``GPUModelRunner._update_states`` on a CPU
``InputBatch`` through the PP=2 batch-queue cadence (scheduled step, empty step,
output). The runner is built without ``__init__`` so no GPU or process group is
needed; ``get_pp_group`` is replaced by a stub that names the rank.
"""

import dataclasses
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import vllm.v1.worker.gpu_model_runner as gpu_model_runner
from tests.v1.core.utils import create_requests, create_scheduler
from vllm.config import DeviceConfig
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
from vllm.v1.worker.gpu_input_batch import InputBatch
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

pytestmark = pytest.mark.cpu_test

KNOB = "VLLM_PP_KEEP_BATCH"
BLOCK_SIZE = 16
MAX_MODEL_LEN = 512


@pytest.fixture(autouse=True)
def two_devices(monkeypatch):
    # Scheduler bookkeeping is CPU-only; do not require two physical GPUs.
    monkeypatch.setattr("vllm.config.parallel.current_platform.device_count", lambda: 2)


def _runner(monkeypatch, keep: bool, is_last_rank: bool) -> GPUModelRunner:
    runner = object.__new__(GPUModelRunner)
    runner.pp_keep_batch = keep
    runner._pp_keep_batch_kept_logged = False
    runner.device = torch.device("cpu")
    runner.input_batch = InputBatch(
        max_num_reqs=4,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=MAX_MODEL_LEN,
        device=torch.device("cpu"),
        pin_memory=False,
        vocab_size=50272,
        block_sizes=[BLOCK_SIZE],
        kernel_block_sizes=[BLOCK_SIZE],
        num_spec_tokens=4,
    )
    runner.requests = {}
    runner.num_prompt_logprobs = {}
    runner.encoder_cache = {}
    runner.late_interaction_runner = Mock()
    runner._dynamic_draft_vocab_prefill_bootstrap = Mock()
    runner.speculative_config = None
    runner.use_async_spec_decode = False
    runner.sync_spec_decode_accept_counts = False
    runner.use_async_scheduling = False
    runner.is_pooling_model = False
    runner.uses_mrope = False
    runner.uses_xdrope_dim = 0
    runner.kv_cache_config = SimpleNamespace(kv_cache_groups=[])
    monkeypatch.setattr(
        gpu_model_runner,
        "get_pp_group",
        lambda: SimpleNamespace(
            is_last_rank=is_last_rank, is_first_rank=not is_last_rank
        ),
    )
    adds = []
    real_add = runner.input_batch.add_request

    def counting_add(request):
        adds.append(request.req_id)
        return real_add(request)

    runner.input_batch.add_request = counting_add
    runner.adds = adds
    return runner


def _row(runner: GPUModelRunner, req_id: str) -> dict:
    batch = runner.input_batch
    idx = batch.req_id_to_index[req_id]
    num_tokens = int(batch.num_tokens_no_spec[idx])
    table = batch.block_table.block_tables[0]
    return {
        "tokens": batch.token_ids_cpu[idx, :num_tokens].tolist(),
        "num_computed": int(batch.num_computed_tokens_cpu[idx]),
        "spec": list(batch.spec_token_ids[idx]),
        "blocks": table.block_table.np[idx, : table.num_blocks_per_row[idx]].tolist(),
        "outputs": list(runner.requests[req_id].output_token_ids),
        "req_num_computed": runner.requests[req_id].num_computed_tokens,
    }


def _runner_output(req_id: str, sampled: list[int]) -> ModelRunnerOutput:
    return ModelRunnerOutput(
        req_ids=[req_id],
        req_id_to_index={req_id: 0},
        sampled_token_ids=[sampled],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def _simulate(monkeypatch, knob: str, accepted: list[int]):
    """Run len(accepted) rounds of the PP=2 cadence with 3 drafts per decode
    step; ``accepted[i]`` drafts of round i are accepted. Returns the runner's
    row after every scheduled step, the cached data sent, and the runner."""
    monkeypatch.setenv(KNOB, knob)
    scheduler = create_scheduler(
        pipeline_parallel_size=2,
        device_config=DeviceConfig(device="cpu"),
        num_speculative_tokens=3,
        block_size=BLOCK_SIZE,
        max_model_len=MAX_MODEL_LEN,
    )
    assert scheduler.pp_keep_batch is (knob == "1")
    runner = _runner(monkeypatch, keep=scheduler.pp_keep_batch, is_last_rank=False)
    (request,) = create_requests(
        num_requests=1, num_tokens=37, max_tokens=200, block_size=BLOCK_SIZE
    )
    req_id = request.request_id
    scheduler.add_request(request)

    rows, sent = [], []
    token = 1000
    for num_accepted in accepted:
        scheduled = scheduler.schedule()
        assert req_id in scheduled.num_scheduled_tokens
        drafts = list(scheduled.scheduled_spec_decode_tokens.get(req_id, []))
        runner._update_states(scheduled)
        row = _row(runner, req_id)
        rows.append(row)
        sent.append(scheduled.scheduled_cached_reqs)

        # The first stage holds exactly the scheduler's history up to the
        # tokens it feeds the model, then this step's drafts.
        cached = scheduled.scheduled_cached_reqs
        if cached.req_ids:
            expected_len = cached.num_computed_tokens[0] + len(cached.new_token_ids[0])
        else:
            expected_len = request.num_prompt_tokens
        assert row["tokens"] == request.all_token_ids[:expected_len]
        assert row["spec"] == drafts

        in_flight = scheduler.schedule()
        assert in_flight.total_num_scheduled_tokens == 0
        runner._update_states(in_flight)

        assert num_accepted <= len(drafts)
        sampled = drafts[:num_accepted] + [token]
        scheduler.update_from_output(scheduled, _runner_output(req_id, sampled))
        scheduler.update_draft_token_ids(
            DraftTokenIds([req_id], [[token + 1, token + 2, token + 3]])
        )
        token += 10
    return rows, sent, runner


ACCEPTED = [0, 0, 3, 1, 0, 2, 3, 3, 0, 1, 2]


def test_first_stage_history_matches_default_path(monkeypatch):
    rows_off, sent_off, runner_off = _simulate(monkeypatch, "0", ACCEPTED)
    rows_on, sent_on, runner_on = _simulate(monkeypatch, "1", ACCEPTED)

    assert rows_on == rows_off
    # The default path re-adds the row on every decode round and resends the
    # whole history; the knob adds it once and never resends.
    assert runner_off.adds == ["0"] * len(ACCEPTED)
    assert runner_on.adds == ["0"]
    assert all("0" in c.all_token_ids for c in sent_off[1:])
    assert not any(c.all_token_ids for c in sent_on)
    assert runner_on._pp_keep_batch_kept_logged
    assert not runner_off._pp_keep_batch_kept_logged


def test_empty_step_keeps_rows_only_with_knob(monkeypatch):
    for knob, kept in (("0", False), ("1", True)):
        monkeypatch.setenv(KNOB, knob)
        scheduler = create_scheduler(
            pipeline_parallel_size=2, device_config=DeviceConfig(device="cpu")
        )
        runner = _runner(monkeypatch, keep=scheduler.pp_keep_batch, is_last_rank=True)
        (request,) = create_requests(num_requests=1, num_tokens=20)
        scheduler.add_request(request)
        runner._update_states(scheduler.schedule())
        before = _row(runner, request.request_id)
        in_flight = scheduler.schedule()
        assert in_flight.total_num_scheduled_tokens == 0
        runner._update_states(in_flight)
        assert (request.request_id in runner.input_batch.req_id_to_index) is kept
        # The cached request state survives either way.
        assert request.request_id in runner.requests
        if kept:
            assert _row(runner, request.request_id) == before


def test_empty_step_drops_preempted_and_finished_rows(monkeypatch):
    monkeypatch.setenv(KNOB, "1")
    scheduler = create_scheduler(
        pipeline_parallel_size=2, device_config=DeviceConfig(device="cpu")
    )
    runner = _runner(monkeypatch, keep=True, is_last_rank=True)
    request_a, request_b, request_c = create_requests(num_requests=3, num_tokens=20)
    for request in (request_a, request_b, request_c):
        scheduler.add_request(request)
    runner._update_states(scheduler.schedule())
    assert set(runner.input_batch.req_id_to_index) == {"0", "1", "2"}

    in_flight = scheduler.schedule()
    assert in_flight.total_num_scheduled_tokens == 0
    in_flight = dataclasses.replace(
        in_flight, preempted_req_ids={"1"}, finished_req_ids={"2"}
    )
    runner._update_states(in_flight)
    assert set(runner.input_batch.req_id_to_index) == {"0"}
    # A preempted request keeps its cached state; a finished one does not.
    assert set(runner.requests) == {"0", "1"}


def test_prepend_fails_closed_with_knob():
    """With the knob on, the scheduler omits the history for a request it
    believes the runner holds; a runner that cannot rebuild the accepted
    drafts must raise instead of corrupting the first stage's history."""
    fn = GPUModelRunner._pp_prepend_accepted_drafts
    req_state = SimpleNamespace(num_tokens=11)
    on = SimpleNamespace(
        pp_keep_batch=True, input_batch=SimpleNamespace(spec_token_ids=[[7, 8, 9]])
    )
    # Kept row: drafts come from the previous step, as on the default path.
    assert fn(on, "r", req_state, 0, 13, [42], {}) == [7, 8, 42]
    assert fn(on, "r", req_state, None, 13, [42], {"r": list(range(100, 114))}) == [
        111,
        112,
        42,
    ]
    with pytest.raises(RuntimeError, match="accepted draft token"):
        fn(on, "r", req_state, None, 13, [42], {})
    with pytest.raises(RuntimeError, match="accepted draft token"):
        fn(on, "r", req_state, 0, 15, [42], {})

    off = SimpleNamespace(
        pp_keep_batch=False, input_batch=SimpleNamespace(spec_token_ids=[[7, 8, 9]])
    )
    # The prerequisite correctness fix refuses missing history in both modes.
    with pytest.raises(RuntimeError, match="accepted draft token"):
        fn(off, "r", req_state, None, 13, [42], {})
