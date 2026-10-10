# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm.config.sm70_runtime import RuntimeTraceConfig
from vllm.diagnostics import bind_event_tracer, diagnostics_for
from vllm.runtime_resources import release_runtime_resources


def config(events, every=8):
    return SimpleNamespace(
        observability_config=SimpleNamespace(
            runtime_trace=RuntimeTraceConfig(
                events=events, event_every=every, event_threshold_ms=0
            )
        )
    )


@pytest.mark.parametrize("order", [(False, True), (True, False)])
def test_event_counts_and_policy_are_owned_and_frozen(monkeypatch, order):
    from vllm import sm70_decode_trace as trace

    configs = [config(enabled) for enabled in order]
    tracers = [bind_event_tracer(cfg) for cfg in configs]
    for cfg, tracer in zip(configs, tracers):
        assert bind_event_tracer(cfg) is tracer
        assert tracer.counts is diagnostics_for(cfg).counters
    monkeypatch.setenv("VLLM_SM70_DECODE_EVENT_TRACE", "invalid-after-init")
    monkeypatch.setattr(
        trace,
        "sm70_decode_event_trace_enabled",
        Mock(side_effect=AssertionError("legacy getter")),
    )
    # Event tracing only queries CUDA for its existing NVTX range, never a new
    # synchronization. Use the real threshold/budget helper without a GPU.
    from contextlib import contextmanager

    @contextmanager
    def record(label, should_log):
        yield
        should_log(label, 1.0)

    monkeypatch.setattr(trace, "_trace_range", record)
    event = Mock()
    for _ in range(5):
        for tracer in tracers:
            assert tracer.call("graph.replay", lambda: 7) == 7
            tracer.synchronize(event, "runner.event")
    assert event.synchronize.call_count == 10
    for enabled, tracer in zip(order, tracers):
        assert tracer.counts == (
            {"graph.replay": 5, "runner.event": 5} if enabled else {}
        )
    release_runtime_resources(configs[0])
    assert not tracers[0].counts
    assert tracers[1].counts == (
        {"graph.replay": 5, "runner.event": 5} if order[1] else {}
    )


def test_disabled_event_tracer_does_not_enter_timing_or_cuda(monkeypatch):
    from vllm import sm70_decode_trace as trace

    tracer = bind_event_tracer(config(False))
    fail = Mock(side_effect=AssertionError("disabled trace entered timing"))
    monkeypatch.setattr(trace, "_trace_range", fail)
    event = Mock()
    assert tracer.call("replay", lambda: 9) == 9
    tracer.synchronize(event, "event")
    event.synchronize.assert_called_once_with()
    fail.assert_not_called()
