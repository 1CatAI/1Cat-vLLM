# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy

import pytest

from benchmarks.analyze_sm70_mtp4_phases import analyze, rank_phases


def worker(rank):
    records = []
    for step in range(5):

        def event(label, start, duration, step=step):
            records.append(
                {
                    "label": label,
                    "gpu_start_ms": rank * 1000 + step * 10 + start,
                    "gpu_duration_ms": duration,
                    "host_start_ns": step * 10000 + int(start * 1000),
                    "host_end_ns": step * 10000 + int((start + duration) * 1000),
                }
            )

        event("target_execute", 0, 6)
        event("target_forward/M5", 1, 5)
        event("sample_handoff_draft", 6, 3.8)
        event("target_head_sample", 6.1, 0.5)
        event("draft_all", 6.7, 3)
        for i in range(4):
            event(f"draft_step/{i}/M{5 if i == 0 else 1}", 6.8 + i * 0.7, 0.7)
    return {"rank": rank, "records": records}


def report():
    control = {
        "id": "fixed8k",
        "repeat": 0,
        "token_ids": [1, 2, 3],
        "metrics": {"decode_time": 0.03, "steady_decode_tokens": 3},
        "spec_decoding": {"num_drafts": 3},
    }
    case = copy.deepcopy(control)
    case.update(complete_round_ms=10, phase_events=[worker(i) for i in range(4)])
    return {"source": "test", "cases": [case], "phase_event_controls": [control]}


def test_nested_ranges_close_without_double_counting_or_cross_rank_clocks():
    result = analyze(report(), 0.02)
    assert result["calibrated"]
    ranks = result["cases"][0]["ranks"]
    for rank in ranks:
        phases = rank["ms"]
        assert phases["complete_round"]["mean"] == 10
        assert phases["target_outside_forward"]["mean"] == 1
        assert sum(
            v["mean"] for k, v in phases.items() if k != "complete_round"
        ) == pytest.approx(10)
    assert ranks[0]["ms"] == ranks[3]["ms"]


@pytest.mark.parametrize("failure", ["missing_control", "different_tokens", "overhead"])
def test_uncalibrated_observer_cannot_pass(failure):
    data = report()
    if failure == "missing_control":
        data.pop("phase_event_controls")
    elif failure == "different_tokens":
        data["cases"][0]["token_ids"] = [1, 3, 2]
    else:
        data["cases"][0]["complete_round_ms"] = 10.3
    assert not analyze(data, 0.02)["calibrated"]


def test_reject_overlapping_nested_ranges():
    data = worker(0)
    for event in data["records"]:
        if event["label"] == "draft_all":
            event["gpu_duration_ms"] = 4
    with pytest.raises(ValueError, match="overlap"):
        rank_phases(data)
