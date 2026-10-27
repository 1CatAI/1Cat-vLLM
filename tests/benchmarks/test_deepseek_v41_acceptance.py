# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-end evidence must agree with all private state boundaries."""

import copy

import pytest

from benchmarks.deepseek_v41.acceptance import validate_commits


def _records():
    return [
        dict(
            rank=rank,
            commits=[
                dict(
                    rows=128,
                    accepted_inputs=1,
                    prefill=True,
                    start=0,
                    hash_history=[2, 3, 4, 128],
                    hash_metadata=[0, 128, 0],
                    c2_positions=[128] * 3,
                ),
                dict(
                    rows=6,
                    accepted_inputs=3,
                    prefill=False,
                    start=128,
                    hash_history=[5, 6, 7, 131],
                    hash_metadata=[128, 6, 0],
                    c2_positions=[131] * 3,
                ),
            ],
        )
        for rank in range(4)
    ]


def test_final_state_matches_emitted_prefix_including_eos_or_length_stop():
    records = _records()
    validate_commits(records, expected_end=128 + 4 - 1)
    # A self-consistent state can still commit an unaccepted candidate after
    # the frontend stopped. Check against the observed emitted-token count.
    with pytest.raises(AssertionError):
        validate_commits(records, expected_end=130)


@pytest.mark.parametrize("failure", ["hash", "c2", "pending", "count", "missing"])
def test_one_rank_cannot_report_a_different_private_boundary(failure):
    records = copy.deepcopy(_records())
    row = records[2]["commits"][-1]
    if failure == "hash":
        row["hash_history"][3] += 1
    elif failure == "c2":
        row["c2_positions"][1] -= 1
    elif failure == "pending":
        row["hash_metadata"][2] = 1
    elif failure == "count":
        row["accepted_inputs"] = 7
    else:
        records[2]["commits"] = []
    with pytest.raises(AssertionError):
        validate_commits(records, expected_end=131)
