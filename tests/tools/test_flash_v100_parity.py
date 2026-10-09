# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Negative controls for A3's acceptance tools, without requiring a GPU."""

from copy import deepcopy
from typing import Any

import pytest
import torch

from tools.sm70.op_parity import compare
from tools.sm70.parity_common import compare_routes, require_routes

pytestmark = pytest.mark.cpu_test


def operator_report():
    cases = [
        dict(name="decode", performance_role="decode_xqa"),
        dict(name="prefill", performance_role="prefill_75t"),
    ]
    return dict(
        contract=dict(cases=cases, native_sha256={"native": "fixed"}),
        cases={
            c["name"]: dict(
                output=torch.ones(2, dtype=torch.float16),
                replay_output=torch.ones(2, dtype=torch.float16),
                routes={c["name"]: 1},
                median_ms=1.0,
                persistent_pointers_stable=True,
            )
            for c in cases
        },
    )


def test_operator_exact_parity_and_performance():
    report = operator_report()
    assert all(
        row["max_abs"] == 0 for row in compare(report, deepcopy(report), True).values()
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("output", torch.tensor([1.0, 1.001], dtype=torch.float16)),
        ("output", torch.tensor([1.0, float("nan")], dtype=torch.float16)),
        ("output", torch.ones(2, dtype=torch.float32)),
        ("replay_output", torch.zeros(2, dtype=torch.float16)),
        ("routes", {"fallback": 1}),
        ("persistent_pointers_stable", False),
    ],
)
def test_operator_rejects_changed_result(field, value):
    original = operator_report()
    candidate = deepcopy(original)
    candidate["cases"]["decode"][field] = value
    with pytest.raises(AssertionError):
        compare(original, candidate)


def test_operator_rejects_native_change_and_missing_case():
    original = operator_report()
    candidate = deepcopy(original)
    candidate["contract"]["native_sha256"]["native"] = "changed"
    with pytest.raises(ValueError, match="contracts differ"):
        compare(original, candidate)
    candidate = deepcopy(original)
    del candidate["cases"]["decode"]
    with pytest.raises(AssertionError, match="Case sets"):
        compare(original, candidate)


@pytest.mark.parametrize("time", [0.97, 1.03])
def test_operator_timing_gate_is_separate(time):
    original = operator_report()
    candidate = deepcopy(original)
    candidate["cases"]["prefill"]["median_ms"] = time
    assert compare(original, candidate)
    with pytest.raises(AssertionError, match="exceeds"):
        compare(original, candidate, performance=True)


def test_empty_or_nonfinite_evidence_is_rejected():
    empty: dict[str, Any] = dict(contract=dict(cases=[]), cases={})
    with pytest.raises(AssertionError, match="Empty"):
        compare(empty, deepcopy(empty))
    report = operator_report()
    report["cases"]["decode"]["median_ms"] = float("nan")
    with pytest.raises(AssertionError, match="timing"):
        compare(report, deepcopy(report))
    report = route_report()
    report["requests"] = []
    with pytest.raises(AssertionError, match="Empty"):
        compare_routes(report, deepcopy(report))


def route_report():
    return dict(
        contract=dict(model="fixed", graph=True),
        requests=[dict(token_ids=[7, 9], finish_reason="stop")],
        startup=[dict(rank=0, routes={"decode": 1}, host_kv={})],
        after=[dict(rank=0, routes={"decode": 4}, host_kv={})],
    )


def test_route_and_token_parity():
    report = route_report()
    assert compare_routes(report, deepcopy(report))["equal"]
    require_routes(report["after"], ["decode"])
    with pytest.raises(AssertionError, match="not observed"):
        require_routes(report["after"], ["prefill"])


@pytest.mark.parametrize("field", ["contract", "requests", "startup", "after"])
def test_route_rejects_changes(field):
    report = route_report()
    candidate = deepcopy(report)
    candidate[field] = {} if field == "contract" else []
    with pytest.raises((AssertionError, ValueError)):
        compare_routes(report, candidate)
