# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.config.kernel import KernelConfig
from vllm.config.observability import ObservabilityConfig
from vllm.config.sm70_runtime import RuntimeTraceConfig, Sm70RuntimeConfig
from vllm.platforms.sm70 import runner_hooks
from vllm.utils.staged_copy import StagedCopyOwner
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.runtime.input_transfer import InputTransferSession


@pytest.mark.parametrize("event_present", [False, True])
@pytest.mark.parametrize("staged", [False, True])
@pytest.mark.parametrize("failure", [None, "body", "sync", "record"])
def test_transfer_order_and_restore(event_present, staged, failure):
    calls = []

    def operation(name):
        calls.append(name)
        if failure == name:
            raise RuntimeError(name)

    session = InputTransferSession(synchronize=lambda event, label: operation("sync"))
    if event_present:
        session.event = SimpleNamespace(record=lambda: operation("record"))
    expected_failure = (
        failure == "body"
        or event_present
        and (failure == "record" or failure == "sync" and not staged)
    )

    def prepare():
        with session.prepare(skip_sync=staged):
            assert session.active is staged
            operation("body")

    if expected_failure:
        with pytest.raises(RuntimeError, match=failure):
            prepare()
    else:
        prepare()
    assert not session.active
    expected = ["sync"] if event_present and not staged else []
    if not (event_present and not staged and failure == "sync"):
        expected.append("body")
        if event_present:
            expected.append("record")
    assert calls == expected


def test_nested_transfer_and_per_engine_state():
    first, second = InputTransferSession(), InputTransferSession()
    with first.prepare(skip_sync=True):
        assert first.active and not second.active
        with first.prepare(skip_sync=False):
            assert not first.active
        assert first.active
    assert not first.active


@pytest.mark.parametrize(
    "change",
    [
        dict(num_tokens=0),
        dict(num_tokens=2),
        dict(num_reqs=2),
        dict(has_previous_sample=False),
        dict(has_encoder_inputs=True),
        dict(has_draft_tokens=True),
        dict(has_accepted_event=True),
    ],
)
def test_dynamic_admission_does_not_expand(change):
    session = InputTransferSession(eligible=True)
    step = dict(
        num_tokens=1,
        num_reqs=1,
        has_previous_sample=True,
        has_encoder_inputs=False,
        has_draft_tokens=False,
        has_accepted_event=False,
    )
    assert session.can_stage(**step)
    assert not session.can_stage(**(step | change))


@pytest.mark.parametrize(
    "reject", [None, "disabled", "cpu", "capability", "sync", "spec", "encoder"]
)
def test_static_admission_is_captured(monkeypatch, reject):
    monkeypatch.setattr(
        runner_hooks.current_platform,
        "is_device_capability",
        lambda cap: reject != "capability",
    )
    config = SimpleNamespace(
        kernel_config=KernelConfig(
            sm70_runtime=Sm70RuntimeConfig(staged_input=reject != "disabled")
        ),
        observability_config=ObservabilityConfig(
            runtime_trace=RuntimeTraceConfig(async_cpu=False, events=False)
        ),
        scheduler_config=SimpleNamespace(async_scheduling=reject != "sync"),
        speculative_config=object() if reject == "spec" else None,
        num_speculative_tokens=0,
        model_config=SimpleNamespace(is_encoder_decoder=reject == "encoder"),
    )
    session = runner_hooks.create_input_transfer(
        config, torch.device("cpu" if reject == "cpu" else "cuda"), logger=Mock()
    )
    assert session.eligible is (reject is None)
    monkeypatch.setenv("VLLM_SM70_ASYNC_STAGED_INPUT_PREP", "0")
    assert session.eligible is (reject is None)


def test_staged_sources_survive_failed_capacity_sync(monkeypatch):
    owner = StagedCopyOwner()
    event = Mock()
    event.query.return_value = False
    event.synchronize.side_effect = RuntimeError("copy still pending")
    retained = torch.ones(1)
    owner.pending = [(event, retained)] * 64
    destination = SimpleNamespace(device=torch.device("cuda"))
    with pytest.raises(RuntimeError, match="copy still pending"):
        owner.copy(torch.zeros(1), destination)
    assert len(owner.pending) == 64
    assert owner.pending[0][1] is retained
    event.synchronize.assert_called_once_with()


def test_completed_sources_can_be_released_independently():
    first, second = StagedCopyOwner(), StagedCopyOwner()
    first.pending = [(SimpleNamespace(query=lambda: True), torch.ones(1))]
    second.pending = [(SimpleNamespace(query=lambda: False), torch.ones(1))]
    first.prune()
    second.prune()
    assert first.pending == []
    assert len(second.pending) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA copy/capture lifecycle")
def test_staged_source_mutation_and_graph_replay():
    buffer = CpuGpuBuffer(
        8, 32, dtype=torch.float32, device=torch.device("cuda"), pin_memory=True
    )
    session = InputTransferSession(eligible=True)
    session.event = torch.cuda.Event()
    weight = torch.randn(32, 16, device="cuda")
    state = torch.zeros(8, 16, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            state.add_(torch.relu(buffer.gpu @ weight))
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        state.add_(torch.relu(buffer.gpu @ weight))
    state.zero_()
    expected = torch.zeros_like(state)
    for value in (1.0, -2.0, 3.0):
        buffer.cpu.fill_(value)
        with session.prepare(skip_sync=True):
            session.copy_buffer(buffer)
        # A reusable pinned source can change immediately after preparation.
        buffer.cpu.fill_(99)
        graph.replay()
        expected.add_(torch.relu(torch.full_like(buffer.gpu, value) @ weight))
        torch.accelerator.synchronize()
        assert torch.equal(state, expected)
    buffer._prune_staged_gpu_copies()
    assert not buffer._staged_gpu_copies


@pytest.mark.parametrize("failure", ["copy", "record"])
def test_failed_enqueue_keeps_source_until_stream_completes(monkeypatch, failure):
    source, staging, stream, event = Mock(), Mock(), Mock(), Mock()
    source.shape, source.dtype = (1,), torch.float32
    source.is_pinned.return_value = True
    stream.query.return_value = False
    destination = Mock(device=torch.device("cuda"))
    if failure == "copy":
        destination.copy_.side_effect = RuntimeError("copy")
    else:
        event.record.side_effect = RuntimeError("record")
    monkeypatch.setattr(torch, "empty", lambda *args, **kwargs: staging)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: stream)
    monkeypatch.setattr(torch.cuda, "Event", lambda: event)
    owner = StagedCopyOwner()
    with pytest.raises(RuntimeError, match=failure):
        owner.copy(source, destination)
    assert owner.pending == [(stream, staging)]
    owner.prune()
    assert owner.pending == [(stream, staging)]
    stream.query.return_value = True
    owner.prune()
    assert owner.pending == []


@pytest.mark.parametrize("order", [False, True])
def test_runtime_policy_legacy_priority_and_isolation(monkeypatch, order):
    from vllm import envs
    from vllm.sm70_decode_trace import DecodeEventTracer

    configs = []
    # Exercise the process-wide legacy getter cache used by initialized workers.
    envs.enable_envs_cache()
    try:
        for enabled in (order, not order):
            monkeypatch.setenv("VLLM_SM70_DEBUG", "events" if enabled else "")
            monkeypatch.setenv("VLLM_SM70_DECODE_EVENT_TRACE", str(int(not enabled)))
            monkeypatch.setenv("VLLM_SM70_ASYNC_CPU_TRACE_EVERY", "0")
            monkeypatch.setenv("VLLM_SM70_ASYNC_STAGED_INPUT_PREP", str(int(enabled)))
            configs.append((RuntimeTraceConfig(), Sm70RuntimeConfig()))
        for (trace, runtime), enabled in zip(configs, (order, not order)):
            assert trace.events is enabled
            assert trace.sources["events"] == "VLLM_SM70_DEBUG"
            assert trace.async_every == 1
            assert runtime.staged_input is enabled
        explicit = RuntimeTraceConfig(events=True, async_every=7)
        assert explicit.events and explicit.async_every == 7
        assert explicit.sources["events"] == "typed"
        first, second = DecodeEventTracer(explicit), DecodeEventTracer(explicit)
        for _ in range(5):
            first._should_log("copy", 10.0)
        assert first.counts["copy"] == 5 and not second.counts
    finally:
        envs.disable_envs_cache()
