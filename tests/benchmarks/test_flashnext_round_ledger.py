# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from benchmarks.analyze_flashnext_round_ledger import clipped, union_ns


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
