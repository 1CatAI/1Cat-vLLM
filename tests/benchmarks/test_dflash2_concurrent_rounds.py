# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
from pathlib import Path

import pytest

_PATH = (
    Path(__file__).resolve().parents[2]
    / "benchmarks/benchmark_dflash2_concurrent_rounds.py"
)
_SPEC = importlib.util.spec_from_file_location("concurrent_rounds", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def cohort(times, index=0):
    return {
        "request_index": index,
        "events": [(timestamp, [1, 2, 3]) for timestamp in times],
        "ttft_ms": 5000,
        "usage": {"completion_tokens": len(times) * 3},
    }


def test_common_window_excludes_prefill_and_partial_intervals():
    # The slower prefill determines the start. Neither early decode from the
    # first request nor the partially intersecting interval counts as a round.
    result = _MODULE.common_window(
        [cohort([0, 1, 2, 3, 4, 5, 6]), cohort([1.5, 2.5, 3.5, 4.5, 5.5, 6.5], 1)],
        trim=1,
    )
    assert result["begin_wall_s"] == 2.5
    assert result["end_wall_s"] == 5
    assert [r["intervals"] for r in result["requests"]] == [2, 2]
    assert all(r["stream_interval_ms_mean"] == 1000 for r in result["requests"])
    assert all(r["tokens_per_stream_chunk"] == 3 for r in result["requests"])


def test_common_window_rejects_nonoverlapping_requests():
    with pytest.raises(ValueError, match="no common"):
        _MODULE.common_window(
            [cohort([0, 1, 2, 3, 4]), cohort([10, 11, 12, 13, 14], 1)], trim=1
        )
