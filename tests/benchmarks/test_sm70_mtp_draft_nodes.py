# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from benchmarks.analyze_sm70_mtp_draft_nodes import summarize_draft_nodes


def test_multidimensional_launch_geometry_and_no_double_count():
    def node(grid, block, ms):
        return dict(
            rank=0,
            phase="mtp15.draft_step/0/M5",
            grid=grid,
            block=block,
            calls_per_round=1,
            service_ms_per_round=ms,
            kernel="fixture",
        )

    data = {
        "kernels": [
            node([1, 4, 1], [16, 2, 1], 2),
            node([1, 5, 1], [128, 1, 1], 3),
            node([1, 1, 1], [512, 1, 1], 4),
        ]
    }
    result = summarize_draft_nodes(data)
    assert result["rank_selection"] == "largest summed draft service"
    assert result["flagged_calls"] == 2
    assert result["flagged_service_ms"] == 6
    assert result["nodes"][1]["ctas"] == 4
    assert result["nodes"][1]["at_most_one_warp"]
    with pytest.raises(ValueError, match="no draft nodes"):
        summarize_draft_nodes(data, 1)
