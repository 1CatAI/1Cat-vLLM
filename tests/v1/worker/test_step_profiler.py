# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pickle
from types import SimpleNamespace

import pytest
import torch

from vllm import envs
from vllm.config.kernel import KernelConfig
from vllm.config.observability import ObservabilityConfig
from vllm.config.sm70_runtime import Sm70RuntimeConfig, StepProfilerConfig
from vllm.model_executor.warmup.plan import (
    WarmupTask,
    run_warmup_tasks,
    warmup_boolean,
    warmup_unconditional,
)
from vllm.v1.spec_decode.profiling import create_step_profiler
from vllm.v1.worker.runtime.profiling import StepProfiler


def make_profiler(role, *, interval=2, enabled=True, messages=None):
    if messages is None:
        messages = []
    config = SimpleNamespace(
        observability_config=ObservabilityConfig(
            step_profiler=StepProfilerConfig(enabled=enabled, interval=interval)
        ),
        speculative_config=SimpleNamespace(method="mtp"),
    )
    result = create_step_profiler(
        config,
        torch.device("cuda"),
        role=role,
        logger=SimpleNamespace(info=lambda msg, *args: messages.append(msg % args)),
    )
    result.report_rank = lambda: True
    return result


class Event:
    def __init__(self, elapsed=0.0):
        self.elapsed = elapsed
        self.synchronizations = 0

    def synchronize(self):
        self.synchronizations += 1

    def elapsed_time(self, end):
        return self.elapsed


def test_report_totals_interval_order_and_step_counts():
    messages: list[str] = []
    profiler = make_profiler("runner", messages=messages)
    end = Event()
    ctx = dict(
        events=[
            ("target_forward", Event(2), Event()),
            ("target_forward", Event(3), end),
        ],
        cpu_ms={"draft_wall_cpu": 7},
        has_spec_decode_metadata=True,
        num_tokens=5,
        num_reqs=1,
    )
    profiler.report(ctx)
    assert messages == [
        (
            "SM70 spec runner profile avg_ms calls=1 spec_steps=1 "
            "num_tokens=5 num_reqs=1 target_forward=5.000 draft_wall_cpu=7.000"
        ),
        (
            "SM70 spec runner profile interval_avg_ms calls=1 interval_calls=1 "
            "interval_spec_steps=1 num_tokens=5 num_reqs=1 "
            "target_forward=5.000 draft_wall_cpu=7.000"
        ),
    ]
    ctx["has_spec_decode_metadata"] = False
    profiler.report(ctx)
    profiler.report(ctx)
    assert end.synchronizations == 3
    assert profiler.totals == {"target_forward": 15, "draft_wall_cpu": 21}
    assert profiler.calls == 3 and profiler.last_calls == 2
    assert len(messages) == 4
    profiler.report_rank = lambda: False
    profiler.report(ctx)
    assert profiler.calls == 4 and profiler.last_calls == 2
    assert len(messages) == 4


def test_disabled_context_never_allocates_events(monkeypatch):
    def forbidden(**kwargs):
        raise AssertionError("disabled profiling allocated CUDA event")

    monkeypatch.setattr(torch.cuda, "Event", forbidden)
    profiler = make_profiler("runner_v2", enabled=False)
    assert profiler.start(None) is None
    assert profiler.start_context(None) is None
    profiler.finish(None, "unused", None)
    profiler.finish_context(None, "unused", None)
    profiler.add_cpu_context(None, "unused", 0)
    profiler.report(None)
    assert profiler.calls == 0 and not profiler.totals


@pytest.mark.parametrize("reverse", [False, True])
def test_config_snapshot_priority_serialization_and_hash(monkeypatch, reverse):
    monkeypatch.setenv("VLLM_SM70_MTP_PROFILE", "1")
    monkeypatch.setenv("VLLM_SM70_MTP_PROFILE_INTERVAL", "0")
    monkeypatch.setenv("VLLM_SM70_AUX_KERNEL_WARMUP", "0")
    monkeypatch.setattr(envs, "VLLM_SM70_AUX_KERNEL_WARMUP", True)
    first = lambda: (KernelConfig(), ObservabilityConfig())
    second = lambda: (
        KernelConfig(sm70_runtime=Sm70RuntimeConfig(auxiliary_warmup=True)),
        ObservabilityConfig(
            step_profiler=StepProfilerConfig(enabled=False, interval=5)
        ),
    )
    a, b = (second(), first()) if reverse else (first(), second())
    if reverse:
        a, b = b, a
    assert not a[0].sm70_runtime.auxiliary_warmup
    assert b[0].sm70_runtime.auxiliary_warmup
    assert a[1].step_profiler.enabled and a[1].step_profiler.interval == 1
    assert not b[1].step_profiler.enabled and b[1].step_profiler.interval == 5
    assert b[0].sm70_runtime.sources["auxiliary_warmup"] == "typed"
    assert a[0].compute_hash() == b[0].compute_hash()
    assert a[1].compute_hash() == b[1].compute_hash()
    monkeypatch.setenv("VLLM_SM70_MTP_PROFILE", "0")
    monkeypatch.setenv("VLLM_SM70_AUX_KERNEL_WARMUP", "1")
    restored = pickle.loads(pickle.dumps(a))
    assert restored[1].step_profiler.enabled
    assert not restored[0].sm70_runtime.auxiliary_warmup
    assert restored[0].sm70_runtime.sources == a[0].sm70_runtime.sources


def test_warmup_order_result_and_exception_boundary():
    calls: list[str] = []

    def record_missing():
        calls.append("missing")
        return False

    def record_draft():
        calls.append("draft")
        return ("draft1", "draft2")

    tasks = [
        warmup_boolean("missing", record_missing),
        warmup_unconditional("slot", lambda: calls.append("slot")),
        WarmupTask("draft", record_draft),
    ]
    assert run_warmup_tasks(tasks) == ("slot", "draft1", "draft2")
    assert calls == ["missing", "slot", "draft"]

    def fail():
        calls.append("error")
        raise RuntimeError("warmup failed")

    calls.clear()
    with pytest.raises(RuntimeError, match="warmup failed"):
        run_warmup_tasks([tasks[0], WarmupTask("failure", fail), tasks[1]])
    assert calls == ["missing", "error"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA lifecycle test")
def test_profile_events_and_changed_input_graph_replay():
    profiler = make_profiler("proposer")
    x = torch.randn(8, 32, device="cuda")
    weight = torch.randn(32, 16, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            output = torch.relu(x @ weight)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        assert StepProfiler.start(None) is None
        output = torch.relu(x @ weight)
    for _ in range(3):
        x.add_(0.125)
        events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
        start = profiler.start(events)
        graph.replay()
        profiler.finish(events, "total_gpu", start)
        profiler.report_proposal(events, {}, 1, 8)
        assert torch.equal(output, torch.relu(x @ weight))
    assert profiler.calls == 3 and profiler.totals["total_gpu"] > 0


@pytest.mark.parametrize("legacy", ["0", "1"])
@pytest.mark.parametrize("channels", ["", "mtp"])
def test_unified_debug_precedence_and_source(monkeypatch, legacy, channels):
    monkeypatch.setenv("VLLM_SM70_MTP_PROFILE", legacy)
    monkeypatch.setenv("VLLM_SM70_DEBUG", channels)
    policy = StepProfilerConfig()
    assert policy.enabled == bool(channels)
    assert policy.sources["enabled"] == "VLLM_SM70_DEBUG"
    assert not StepProfilerConfig(enabled=False).enabled
