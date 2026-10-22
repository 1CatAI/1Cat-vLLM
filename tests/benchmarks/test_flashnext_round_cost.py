# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from benchmarks.analyze_flashnext_round_cost import (
    draft_groups,
    expert_reads,
    projection_bytes,
)


def test_repeated_routes_are_not_compulsory_weight_reads():
    cost = expert_reads([{"ordinal": 0, "rows": 2, "ids": [[1, 2], [2, 3]]}], 110)
    assert cost["mean_compulsory_bytes"] == 330
    assert cost["mean_route_issued_bytes"] == 440
    assert cost["samples"][0]["tokens_per_expert"] == {1: 1, 2: 2, 3: 1}


def test_bank_selection_preserves_canonical_padding_and_excludes_retained_copy():
    bank = {
        "raw_weights": {"shape": [4, 2, 110], "logical_bytes": 880, "name": "raw"},
        "weights": {"shape": [4, 128], "logical_bytes": 512, "name": "codes"},
        "stats": {"shape": [4, 16], "logical_bytes": 64, "name": "stats"},
    }
    assert projection_bytes(bank, True) == (220, ["raw"])
    assert projection_bytes(bank, False) == (144, ["codes", "stats"])


def test_unwritten_route_cannot_contribute_to_floor():
    with pytest.raises(ValueError, match="Unwritten"):
        expert_reads([{"ordinal": 0, "rows": 1, "ids": [[-1]]}], 110)


def test_bootstrap_calls_are_not_charged_to_steady_draft_rounds():
    rows = [1, 1, 5, 1, 1, 1, 5, 1, 1, 1]
    groups = draft_groups([{"rows": row} for row in rows], 5)
    assert len(groups) == 2
    assert [record["rows"] for record in groups[0]] == [5, 1, 1, 1]
    with pytest.raises(ValueError, match="complete"):
        draft_groups([{"rows": row} for row in [5, 1, 1]], 5)
