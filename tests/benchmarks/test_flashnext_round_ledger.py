# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from benchmarks.analyze_flashnext_round_ledger import (
    classify,
    clipped,
    exclusive_activity_ns,
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
