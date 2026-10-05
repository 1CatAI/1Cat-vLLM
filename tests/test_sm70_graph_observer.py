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
