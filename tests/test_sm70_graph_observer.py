# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import msgspec
import pytest

from vllm.sm70_graph_observer import CPUStageRecorder


def test_disabled_observer_preserves_results_and_records_nothing():
    owner = SimpleNamespace(f=lambda x: x + 1)
    rec = CPUStageRecorder(0)
    rec.wrap(owner, "f", "test", advances_step=True)
    assert owner.f(3) == 4
    assert rec.step == 0 and not rec.events


def test_enabled_observer_keeps_metadata_and_serializes_without_callables():
    owner = SimpleNamespace(f=lambda x: x + 1)
    rec = CPUStageRecorder(2)
    rec.enabled = True
    rec.wrap(owner, "f", "test", metadata=lambda x: {"tokens": x}, advances_step=True)
    assert owner.f(3) == 4
    row = rec.read()
    event = row["events"][0]
    assert event["step"] == 1 and event["tokens"] == 3
    assert event["start_ns"] <= event["end_ns"]
    assert msgspec.msgpack.decode(msgspec.msgpack.encode(row))["rank"] == 2
    owner.f(5)
    assert len(rec.events) == 1


def test_observer_records_exceptions_without_swallowing_them():
    def fail():
        raise ValueError("original failure")

    owner = SimpleNamespace(f=fail)
    rec = CPUStageRecorder(0)
    rec.enabled = True
    rec.wrap(owner, "f", "failure")
    with pytest.raises(ValueError, match="original failure"):
        owner.f()
    assert len(rec.events) == 1 and rec.events[0]["label"] == "failure"


def test_bounded_observer_reports_dropped_events():
    rec = CPUStageRecorder(0, limit=1)
    rec.enabled = True
    with rec.stage("first"):
        pass
    with rec.stage("second"):
        pass
    assert rec.read()["dropped"] == 1


def test_target_replay_marks_actual_graph_and_excludes_draft():
    class Graph:
        def replay(self):
            return "result"

    graph = Graph()
    manager = SimpleNamespace(run_fullgraph=lambda desc: graph.replay())
    desc = SimpleNamespace(num_tokens=5, num_reqs=1, uniform_token_count=5)
    rec = CPUStageRecorder(0)
    rec.wrap_target_replay(manager, Graph)
    assert manager.run_fullgraph(desc) == "result"
    assert not rec.events
    rec.enabled = True
    assert manager.run_fullgraph(desc) == "result"
    assert graph.replay() == "result"
    assert [e["label"] for e in rec.events] == ["target.replay", "target.manager"]
    assert rec.events[0]["tokens"] == 5
    assert rec.events[1]["start_ns"] <= rec.events[0]["start_ns"]
    assert rec.events[0]["end_ns"] <= rec.events[1]["end_ns"]


def test_failed_manager_clears_target_context():
    class Graph:
        def replay(self):
            return None

    def fail(desc):
        raise ValueError("manager failure")

    rec = CPUStageRecorder(0)
    rec.enabled = True
    desc = SimpleNamespace(num_tokens=5, num_reqs=1, uniform_token_count=5)
    # Call the wrapper retained on the manager, then verify an unrelated graph
    # is not mistaken for a target replay after the exception.
    manager = SimpleNamespace(run_fullgraph=fail)
    rec.wrap_target_replay(manager, Graph)
    with pytest.raises(ValueError, match="manager failure"):
        manager.run_fullgraph(desc)
    before = len(rec.events)
    Graph().replay()
    assert len(rec.events) == before
