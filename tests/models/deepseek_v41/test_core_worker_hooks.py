# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P2 item 6: worker/runner hooks (PORT_DESIGN §2.1 rule 4, §3.5, §3.6).

* the legacy SM70 static PP path ([1,4,4096] fp16, PP2xTP4, B1) still triggers for its old schema;
* models declaring pp_static_schema/pp_send_schema get the schema-driven path for 1..8 tokens;
* the Engram step plan is built from SchedulerOutput alone; the runner layout carries the step id."""

from __future__ import annotations

import functools
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import vllm.envs as envs
from vllm.config import CUDAGraphMode
from vllm.models.deepseek_v41.common.contracts import EngramBatchLayout, EngramStepPlan
from vllm.models.deepseek_v41.common.topology import make_stage_plan
from vllm.sequence import IntermediateTensors
from vllm.v1.worker import gpu_worker
from vllm.v1.worker.gpu_worker import Worker, _EngramStepPlanner


def _vllm_config(pp: int = 2, tp: int = 4, max_num_seqs: int = 1, async_scheduling: bool = False):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(pipeline_parallel_size=pp, tensor_parallel_size=tp, enable_dbo=False,
                                        ubatch_size=1),
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs, async_scheduling=async_scheduling),
        speculative_config=None,
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.PIECEWISE,
                                           pass_config=SimpleNamespace(enable_sp=False)),
        model_config=SimpleNamespace(dtype=torch.float16),
    )


class _LegacyModel:
    """A V4/GLM-style model: only make_empty_intermediate_tensors with the [B,4,4096] fp16 hidden state."""

    def make_empty_intermediate_tensors(self, batch_size, dtype, device):
        return IntermediateTensors({"hidden_states": torch.zeros((batch_size, 4, 4096), dtype=dtype,
                                                                 device=device)})


def _v41_model(partition: list[int], rank: int, config):
    from vllm.models.deepseek_v41.sm70.model import DeepseekV41Model

    fake = SimpleNamespace(stage=make_stage_plan(config, rank, len(partition), partition),
                           pp_is_first=rank == 0, pp_is_last=rank == len(partition) - 1, kv20_crc_check=False)
    fake._schema = functools.partial(DeepseekV41Model._schema, fake)
    fake.pp_static_schema = functools.partial(DeepseekV41Model.pp_static_schema, fake)
    fake.pp_send_schema = functools.partial(DeepseekV41Model.pp_send_schema, fake)
    fake.make_empty_intermediate_tensors = functools.partial(DeepseekV41Model.make_empty_intermediate_tensors,
                                                             fake)
    return fake


def _worker(model, vllm_config, intermediate=None) -> Worker:
    worker = Worker.__new__(Worker)
    worker.vllm_config = vllm_config
    worker.model_runner = SimpleNamespace(model=model, get_model=lambda: model, intermediate_tensors=intermediate)
    return worker


@pytest.fixture
def sm70_platform(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_SM70_PP_STATIC_HIDDEN_TRANSFER", True, raising=False)
    monkeypatch.setattr(gpu_worker.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(gpu_worker.current_platform, "is_device_capability", lambda cap: cap == (7, 0))


def test_legacy_static_path_still_triggers(sm70_platform) -> None:
    worker = _worker(_LegacyModel(), _vllm_config())
    assert worker._use_sm70_static_pp_hidden_transfer(1)
    assert not worker._use_schema_static_pp_transfer(1)     # no schema declared -> legacy contract only
    assert worker._schema_static_pp_recv(1) is None
    # its exact conditions are unchanged
    for bad in (_vllm_config(pp=3), _vllm_config(tp=8), _vllm_config(max_num_seqs=2)):
        assert not _worker(_LegacyModel(), bad)._use_sm70_static_pp_hidden_transfer(1)
    assert not _worker(_LegacyModel(), _vllm_config())._use_sm70_static_pp_hidden_transfer(2)


@pytest.mark.parametrize("partition", [[20, 20], [14, 14, 12]])
def test_schema_path_gates(sm70_platform, ds41_text_config, partition) -> None:
    for rank in range(len(partition)):
        model = _v41_model(partition, rank, ds41_text_config)
        worker = _worker(model, _vllm_config(pp=len(partition), max_num_seqs=4))
        assert all(worker._use_schema_static_pp_transfer(t) for t in range(1, 9))
        assert not worker._use_schema_static_pp_transfer(9) and not worker._use_schema_static_pp_transfer(0)
        assert not worker._use_sm70_static_pp_hidden_transfer(1)   # V4.1 never matches the legacy schema
    spec = _vllm_config(pp=2)
    spec.speculative_config = object()
    assert not _worker(_v41_model([20, 20], 1, ds41_text_config), spec)._use_schema_static_pp_transfer(1)


def test_schema_path_off_without_flag(monkeypatch, sm70_platform, ds41_text_config) -> None:
    monkeypatch.setattr(envs, "VLLM_SM70_PP_STATIC_HIDDEN_TRANSFER", False, raising=False)
    worker = _worker(_v41_model([20, 20], 1, ds41_text_config), _vllm_config())
    assert not worker._use_schema_static_pp_transfer(1)


@pytest.mark.sm70
@pytest.mark.parametrize("partition, rank", [([20, 20], 1), ([14, 14, 12], 2)])
def test_schema_recv_and_send_buffers(sm70_platform, ds41_text_config, partition, rank) -> None:
    dev = torch.device("cuda")
    receiver = _v41_model(partition, rank, ds41_text_config)
    persistent = receiver.make_empty_intermediate_tensors(32, torch.float16, dev)
    worker = _worker(receiver, _vllm_config(pp=len(partition)), persistent)
    recv, runner_views = worker._schema_static_pp_recv(3)
    assert list(recv) == list(receiver.pp_static_schema(3))
    for key, view in recv.items():
        assert view.shape[0] == 3 and view.data_ptr() == persistent.tensors[key].data_ptr()
        assert runner_views[key] is persistent.tensors[key]
    sender = _v41_model(partition, rank - 1, ds41_text_config)
    send_worker = _worker(sender, _vllm_config(pp=len(partition)))
    out = IntermediateTensors({k: torch.zeros((8, *shape[1:]), dtype=dt, device=dev)
                               for k, (shape, dt) in sender.pp_send_schema(8).items()})
    send = send_worker._schema_static_pp_send(out, 3)
    assert {k: (tuple(v.shape), v.dtype) for k, v in send.items()} == {
        k: (tuple(v.shape), v.dtype) for k, v in recv.items()}
    bad = IntermediateTensors({k: v for k, v in list(out.tensors.items())[:1]})
    with pytest.raises(RuntimeError, match="keys"):
        send_worker._schema_static_pp_send(bad, 3)


# ---------------------------------------------------------------- Engram step plan
def _new_req(req_id, prompt, computed=0):
    return SimpleNamespace(req_id=req_id, prompt_token_ids=prompt, num_computed_tokens=computed)


def _sched(new=(), cached=None, num_scheduled=None, spec=None, finished=(), preempted=None):
    cached = cached or SimpleNamespace(req_ids=[], resumed_req_ids=set(), new_token_ids=[], all_token_ids={},
                                       num_computed_tokens=[])
    return SimpleNamespace(scheduled_new_reqs=list(new), scheduled_cached_reqs=cached,
                           num_scheduled_tokens=dict(num_scheduled or {}),
                           scheduled_spec_decode_tokens=dict(spec or {}), finished_req_ids=set(finished),
                           preempted_req_ids=preempted)


def test_planner_pp_sync_chunked_prefill_then_decode() -> None:
    planner = _EngramStepPlanner()
    prompt = list(range(100, 110))
    p0 = planner.plan(_sched(new=[_new_req("a", prompt)], num_scheduled={"a": 4}))
    assert isinstance(p0, EngramStepPlan) and p0.step_id == 0
    (s,) = p0.reqs
    assert (s.req_id, s.start_pos, s.num_tokens) == ("a", 0, 4)
    assert s.token_ids.tolist() == prompt[:4] and s.prompt_token_ids.tolist() == prompt
    cached = SimpleNamespace(req_ids=["a"], resumed_req_ids=set(), new_token_ids=[prompt[4:]], all_token_ids={},
                             num_computed_tokens=[4])
    (s,) = planner.plan(_sched(cached=cached, num_scheduled={"a": 6})).reqs
    assert s.start_pos == 4 and s.token_ids.tolist() == prompt[4:] and s.prompt_token_ids is None
    cached = SimpleNamespace(req_ids=["a"], resumed_req_ids=set(), new_token_ids=[[777]], all_token_ids={},
                             num_computed_tokens=[10])
    p2 = planner.plan(_sched(cached=cached, num_scheduled={"a": 1}, finished={"zz"}))
    assert p2.step_id == 2 and p2.reqs[0].token_ids.tolist() == [777] and p2.finished_req_ids == {"zz"}
    assert p2.reqs[0].token_ids.dtype == np.int32


def test_planner_without_pp_fills_prefill_from_prompt_and_leaves_decode_unknown() -> None:
    planner = _EngramStepPlanner()
    prompt = list(range(10))
    planner.plan(_sched(new=[_new_req("a", prompt)], num_scheduled={"a": 6}))
    cached = SimpleNamespace(req_ids=["a"], resumed_req_ids=set(), new_token_ids=[], all_token_ids={},
                             num_computed_tokens=[6])
    (s,) = planner.plan(_sched(cached=cached, num_scheduled={"a": 4})).reqs
    assert s.token_ids.tolist() == prompt[6:]
    cached.num_computed_tokens = [10]
    (s,) = planner.plan(_sched(cached=cached, num_scheduled={"a": 1})).reqs
    assert s.token_ids is None and s.num_tokens == 1


def test_planner_resume_and_errors() -> None:
    planner = _EngramStepPlanner()
    cached = SimpleNamespace(req_ids=["b"], resumed_req_ids={"b"}, new_token_ids=[[5, 6]],
                             all_token_ids={"b": [1, 2, 3, 4, 5, 6]}, num_computed_tokens=[4])
    (s,) = planner.plan(_sched(cached=cached, num_scheduled={"b": 2})).reqs
    assert s.prompt_token_ids.tolist() == [1, 2, 3, 4, 5, 6] and s.token_ids.tolist() == [5, 6]
    with pytest.raises(ValueError, match="prompt embeddings"):
        planner.plan(_sched(new=[_new_req("c", None)], num_scheduled={"c": 1}))
    with pytest.raises(ValueError, match="new_token_ids"):
        bad = SimpleNamespace(req_ids=["b"], resumed_req_ids=set(), new_token_ids=[[5]], all_token_ids={},
                              num_computed_tokens=[6])
        planner.plan(_sched(cached=bad, num_scheduled={"b": 2}))
    with pytest.raises(ValueError, match="beyond"):
        planner.plan(_sched(new=[_new_req("d", [1, 2])], num_scheduled={"d": 3}))


def test_worker_refuses_engram_with_async_scheduling() -> None:
    worker = _worker(SimpleNamespace(engram_service=object()), _vllm_config(async_scheduling=True))
    with pytest.raises(NotImplementedError, match="synchronous"):
        worker._engram_service()
    assert _worker(SimpleNamespace(), _vllm_config())._engram_service() is None


# ---------------------------------------------------------------- runner layout
def _runner(pp_world: int, monkeypatch):
    from vllm.v1.worker import gpu_model_runner
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    monkeypatch.setattr(gpu_model_runner, "get_pp_group", lambda: SimpleNamespace(world_size=pp_world))
    runner = GPUModelRunner.__new__(GPUModelRunner)
    tokens = np.zeros((4, 16), dtype=np.int64)
    tokens[0, :5] = [11, 12, 13, 14, 15]
    tokens[1, :3] = [21, 22, 23]
    runner.input_batch = SimpleNamespace(req_ids=["r0", "r1", None, None], token_ids_cpu=tokens,
                                         num_computed_tokens_cpu=np.array([4, 2, 0, 0]))
    runner.query_start_loc = SimpleNamespace(np=np.array([0, 1, 2, 2, 2], dtype=np.int32))
    return runner


def test_runner_layout_pp(monkeypatch) -> None:
    runner = _runner(2, monkeypatch)
    with pytest.raises(RuntimeError, match="begin_step"):
        runner._engram_batch_layout(2, 2, 4)
    runner._engram_step_id = 7
    layout = runner._engram_batch_layout(2, 2, 4)
    assert isinstance(layout, EngramBatchLayout)
    assert (layout.step_id, layout.req_order, layout.num_tokens, layout.num_tokens_padded) == (7, ("r0", "r1"), 2, 4)
    assert layout.query_start_loc.tolist() == [0, 1, 2] and layout.sampled_fill is None
    assert runner._engram_step_id is None                     # consumed: one bind per begin


@pytest.mark.sm70
def test_runner_layout_sampling_rank_fills_tokens(monkeypatch) -> None:
    runner = _runner(1, monkeypatch)
    runner._engram_step_id = 3
    layout = runner._engram_batch_layout(2, 2, 2)
    fill, event = layout.sampled_fill
    assert fill.is_pinned() and fill.dtype == torch.int32 and fill.tolist() == [15, 23]
    event.synchronize()


@pytest.mark.parametrize("mode, refused", [(CUDAGraphMode.FULL, True), (CUDAGraphMode.FULL_AND_PIECEWISE, True),
                                           (CUDAGraphMode.FULL_DECODE_ONLY, True), (CUDAGraphMode.PIECEWISE, False),
                                           (CUDAGraphMode.NONE, False)])
def test_full_cudagraphs_refused_on_engram_stage(mode, refused) -> None:
    """MC-CORE F4 / AM-11c: FULL graph modes are refused when the stage owns Engram layers (after mode resolution)."""
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    for service, expect in ((object(), refused), (None, False)):
        runner = GPUModelRunner.__new__(GPUModelRunner)
        runner._engram_bind_service_cache = service
        runner.compilation_config = SimpleNamespace(cudagraph_mode=mode)
        if expect:
            with pytest.raises(ValueError, match="Engram"):
                runner._refuse_full_cudagraphs_with_engram()
        else:
            runner._refuse_full_cudagraphs_with_engram()


def test_refusal_runs_after_mode_resolution() -> None:
    import inspect

    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    src = inspect.getsource(GPUModelRunner)
    resolve = src.index("self._check_and_update_cudagraph_mode(")
    assert src.index("self._refuse_full_cudagraphs_with_engram()", resolve) > resolve


# ---------------------------------------------------------------- D11 ENGRAM F2: end_step
class _RecordingEngram:
    def __init__(self, log: list) -> None:
        self.log = log

    def begin_step(self, plan) -> None:
        self.log.append(("begin", plan.step_id))

    def end_step(self, step_id: int) -> None:
        self.log.append(("end", step_id))


def _engram_worker(monkeypatch, log: list, forward):
    import contextlib

    monkeypatch.setattr(gpu_worker, "get_pp_group", lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True))
    worker = Worker.__new__(Worker)
    worker.vllm_config = _vllm_config(pp=1)
    worker._pp_send_work = []
    worker.use_v2_model_runner = False
    worker.annotate_profile = lambda so: contextlib.nullcontext()
    service = _RecordingEngram(log)
    worker._engram_service = lambda: service
    worker._engram_planner = SimpleNamespace(plan=lambda so: SimpleNamespace(step_id=7))

    def execute_model(so, it):
        log.append(("forward", worker.model_runner._engram_step_id))
        return forward()

    worker.model_runner = SimpleNamespace(execute_model=execute_model, _engram_step_id=-1)
    return worker


def test_end_step_after_each_real_forward(monkeypatch) -> None:
    log: list = []
    worker = _engram_worker(monkeypatch, log, lambda: None)
    assert Worker.execute_model.__wrapped__(worker, SimpleNamespace(total_num_scheduled_tokens=3)) is None
    assert log == [("begin", 7), ("forward", 7), ("end", 7)]


def test_end_step_runs_when_forward_raises(monkeypatch) -> None:
    log: list = []

    def boom():
        raise RuntimeError("forward failed")

    worker = _engram_worker(monkeypatch, log, boom)
    with pytest.raises(RuntimeError, match="forward failed"):
        Worker.execute_model.__wrapped__(worker, SimpleNamespace(total_num_scheduled_tokens=3))
    assert log == [("begin", 7), ("forward", 7), ("end", 7)]


def test_no_engram_step_without_scheduled_tokens(monkeypatch) -> None:
    log: list = []
    worker = _engram_worker(monkeypatch, log, lambda: None)
    Worker.execute_model.__wrapped__(worker, SimpleNamespace(total_num_scheduled_tokens=0))
    assert log == [("forward", -1)]
