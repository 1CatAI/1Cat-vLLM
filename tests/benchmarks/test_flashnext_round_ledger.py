# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from benchmarks.analyze_flashnext_graph_nodes import select_target_ranges
from benchmarks.analyze_flashnext_round_ledger import (
    classify,
    clipped,
    exclusive_activity_ns,
    idle_edges,
    union_ns,
)


@pytest.mark.parametrize(
    "intervals,expected",
    [
        ([], 0),
        ([(10, 20)], 10),
        ([(10, 20), (12, 17)], 10),
        ([(20, 30), (10, 23), (35, 40)], 25),
        ([(10, 20), (20, 30)], 20),
    ],
)
def test_concurrent_activity_is_counted_once(intervals, expected):
    assert union_ns(intervals) == expected


def test_round_clipping_closes_busy_and_gap_time():
    activity = [(5, 13), (12, 17), (22, 30), (31, 35)]
    round_start, round_end = 10, 25
    busy = union_ns(clipped(activity, round_start, round_end))
    assert busy == 10
    assert round_end - round_start - busy == 5


def test_nested_stream_activity_closes_without_double_counting():
    result = exclusive_activity_ns(
        {
            "target": [(10, 15)],
            "draft": [(14, 18)],
            "copies": [(12, 19)],
        },
        10,
        20,
    )
    assert result == {"target": 5, "draft": 3, "copies": 1, "no activity": 1}


def test_idle_edge_uses_last_finishing_event_across_streams():
    activities = [
        (8, 15, "target", "long"),
        (11, 13, "copies", "short"),
        (18, 22, "draft", "next"),
    ]
    gaps = idle_edges(activities, 10, 25)
    assert gaps == [
        (3, ("target", "long"), ("draft", "next")),
        (3, ("draft", "next"), None),
    ]
    assert sum(row[0] for row in gaps) == 15 - union_ns(
        clipped([(a, b) for a, b, _, _ in activities], 10, 25)
    )


def test_idle_window_without_activity_is_explicit():
    assert idle_edges([], 10, 25) == [(15, None, None)]


def test_shape_selection_uses_worker_ordinal_not_clock_proximity():
    workers = [
        {
            "pid": 7,
            "events": [
                {"label": "target.replay", "start_ns": 90, "tokens": 4, "requests": 4},
                {
                    "label": "target.replay",
                    "start_ns": 110,
                    "tokens": 20,
                    "requests": 4,
                },
                {
                    "label": "target.replay",
                    "start_ns": 1000,
                    "tokens": 15,
                    "requests": 3,
                },
            ],
        }
    ]
    tid = (7 << 24) | 3
    ranges = {tid: [(2000, 2010), (2020, 2030), (2040, 2050)]}
    assert select_target_ranges(workers, ranges, 20, 4) == {tid: [(2020, 2030)]}
    with pytest.raises(AssertionError):
        select_target_ranges(workers, {tid: ranges[tid][:2]}, 20, 4)


@pytest.mark.parametrize(
    "name,expected",
    [
        ("void <unnamed>::dense_mv<(int)8>(Segs)", "GGUF dense MMA"),
        ("void <unnamed>::swiglu_mv<(int)8>(SwArgs)", "Shared expert gate/up"),
        ("void <unnamed>::gate_up(turbomind::gemm::StridedPtr*)", "Expert gate/up"),
        ("quantize_q8(Q8_1*)", "Activation quantization"),
    ],
)
def test_specialized_kernels_are_not_classified_by_parameter_types(name, expected):
    assert classify(name) == expected
