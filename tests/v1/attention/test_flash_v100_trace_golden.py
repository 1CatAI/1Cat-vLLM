# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Immutable #1060 behavioral trace and dependency ceilings for Phase A3."""

import json
import types
from pathlib import Path

import pytest

from tools.sm70.flash_v100_audit import audit
from tools.sm70.flash_v100_trace import BASELINE, cases, run_case, strict_shim

pytestmark = pytest.mark.cpu_test
FIXTURES = Path(__file__).with_name("fixtures")
GOLDEN = [
    json.loads(line)
    for line in (FIXTURES / "flash_v100_trace_golden.jsonl").read_text().splitlines()
]


@pytest.mark.parametrize("start", range(1, len(GOLDEN), 128))
def test_forward_trace_matches_1060(start):
    # Group cases to avoid repeating the repository's expensive engine cleanup
    # fixture for every case. Each constructs an independent real backend.
    for expected in GOLDEN[start : start + 128]:
        result = run_case(expected["case"])
        assert result["events"] == expected["events"], expected["case"]
        buffers = [event[1] for event in result["events"] if event[0] == "buffers"]
        if buffers:
            assert len(buffers) == 2
            assert buffers[0] == buffers[1], expected["case"]


def test_matrix_and_coverage_are_complete():
    assert GOLDEN[0] == {"baseline": BASELINE, "schema": 1}
    assert list(cases()) == [row["case"] for row in GOLDEN[1:]]
    coverage = json.loads((FIXTURES / "flash_v100_coverage.json").read_text())
    assert coverage["baseline"] == BASELINE
    for function in coverage["functions"].values():
        assert function["required_sites"]
        assert function["missing_sites"] == []


def test_dependency_and_coupling_ratchet():
    baseline = json.loads(
        (FIXTURES / "flash_v100_dependency_baseline.json").read_text()
    )
    current = audit()
    for metric, value in current["metrics"].items():
        assert value <= baseline["metrics"][metric], (metric, value)
    assert {tuple(c) for c in current["cycles"]} <= {
        tuple(c) for c in baseline["cycles"]
    }
    assert set(map(tuple, current["forbidden_edges"])) <= set(
        map(tuple, baseline["forbidden_edges"])
    )


def test_strict_shim_rejects_orphan_even_if_local_attribute_exists():
    with strict_shim() as legacy:
        types.ModuleType.__setattr__(legacy, "_orphan_test_op", lambda: None)
        try:
            with pytest.raises(AttributeError, match="no owner"):
                legacy._orphan_test_op = lambda: 1
            with pytest.raises(AttributeError, match="no owner"):
                del legacy._orphan_test_op
        finally:
            types.ModuleType.__delattr__(legacy, "_orphan_test_op")


def test_decline_keeps_native_attempt_before_fallback():
    traces = [row for row in GOLDEN[1:] if row["case"].get("decline")]
    assert traces
    for row in traces:
        events = row["events"]
        attempted = next(
            i
            for i, e in enumerate(events)
            if e[0] == "op" and e[1].startswith("splitd_")
        )
        declined = next(
            i
            for i, e in enumerate(events[attempted:], attempted)
            if e[:2] == ["return", "_try_sm70_fa2_d256_prefill"] and e[2] is None
        )
        fallback = next(
            i
            for i, e in enumerate(events[declined:], declined)
            if e[0] == "op" and e[1] in ("dense", "paged")
        )
        assert attempted < declined < fallback


def test_capture_disables_diagnostic_dense_decode():
    traces = [
        row
        for row in GOLDEN[1:]
        if row["case"]["capture"]
        and row["case"]["stage"] == "decode"
        and row["case"].get("env")
    ]
    assert traces
    for row in traces:
        assert not any(
            e[0] == "route"
            and e[1]
            in ("decode_dense_cache", "decode_dense_reference", "decode_paged_prefill")
            for e in row["events"]
        )
