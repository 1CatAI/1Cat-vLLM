# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fail before any model import when the endpoint contract is incomplete."""

import json
from types import SimpleNamespace

import pytest

from benchmarks import benchmark_sm70_qwen38_quality as endpoint


@pytest.fixture
def launch(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"tensor": "weights.safetensors"}})
    )
    (model / "weights.safetensors").write_bytes(b"fixture")
    include = tmp_path / "include"
    include.mkdir()
    (include / "Python.h").write_text("fixture")
    monkeypatch.setattr(endpoint.sys, "version_info", (3, 12, 14))
    monkeypatch.setattr(endpoint.sysconfig, "get_path", lambda _: str(include))
    monkeypatch.setattr(
        endpoint.shutil, "disk_usage", lambda _: SimpleNamespace(free=16 * 1024**3)
    )
    monkeypatch.setattr(endpoint.subprocess, "check_output", lambda *a, **kw: "")
    return SimpleNamespace(
        model=model,
        output=tmp_path / "result.json",
        min_free_gib=8,
        timing_only=True,
        cases=None,
        case_id=None,
    )


def test_complete_contract_passes_without_model_import(launch):
    report = endpoint.preflight(launch)
    assert report["gpu_idle"] and report["spawn_guard"] and report["request_metrics"]


@pytest.mark.parametrize("failure", ["header", "disk", "metrics", "gpu", "shard"])
def test_launch_failures_are_detected_before_model_loading(
    launch, monkeypatch, failure
):
    if failure == "header":
        include = endpoint.sysconfig.get_path("include")
        endpoint.Path(include, "Python.h").unlink()
    elif failure == "disk":
        monkeypatch.setattr(
            endpoint.shutil, "disk_usage", lambda _: SimpleNamespace(free=1024)
        )
    elif failure == "metrics":
        monkeypatch.setattr(endpoint, "REQUEST_METRICS_ENABLED", False)
    elif failure == "gpu":
        monkeypatch.setattr(
            endpoint.subprocess, "check_output", lambda *a, **kw: "12345\n"
        )
    else:
        (launch.model / "weights.safetensors").unlink()
    with pytest.raises(RuntimeError):
        endpoint.preflight(launch)


def test_quality_cases_are_validated_before_loading(launch):
    launch.timing_only = False
    with pytest.raises(RuntimeError, match="frozen cases"):
        endpoint.preflight(launch)


def test_estimate_requires_matching_contract_and_marks_large_deviation():
    baseline = {
        "runtime": "fixture",
        "torch": "fixture",
        "cuda": "fixture",
        "source_native": {"_C": "sha"},
        "contract": {"tp": 4},
        "complete": True,
        "median_tpot_ms": 11.0,
    }
    candidate = {**baseline, "median_tpot_ms": 10.4}
    assert not endpoint.compare_timing(baseline, candidate, 0.63)[
        "calibration_required"
    ]
    candidate["median_tpot_ms"] = 10.9
    assert endpoint.compare_timing(baseline, candidate, 0.63)["calibration_required"]
    candidate["contract"] = {"tp": 2}
    with pytest.raises(ValueError, match="contract"):
        endpoint.compare_timing(baseline, candidate, 0.63)


def test_unknown_triage_case_is_rejected_before_loading(launch):
    launch.timing_only = False
    launch.cases = launch.output.parent / "cases.json"
    launch.cases.write_text(json.dumps({"cases": [{"id": "known"}]}))
    launch.case_id = ["unknown"]
    with pytest.raises(RuntimeError, match="Unknown quality case"):
        endpoint.preflight(launch)
