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


def test_route_record_freezes_nested_requested_engine_options(monkeypatch, tmp_path):
    from types import SimpleNamespace

    import vllm
    from tools.sm70 import route_parity
    from tools.sm70.parity_common import digest, read_json, write_json

    options = dict(
        model="target",
        tensor_parallel_size=1,
        dtype="half",
        kv_cache_dtype="auto",
        max_model_len=128,
        max_num_batched_tokens=32,
        enforce_eager=False,
        speculative_config=dict(
            method="dflash", model="draft", num_speculative_tokens=7
        ),
        compilation_config=dict(cudagraph_capture_sizes=[8]),
    )

    class Engine:
        def __init__(self, **received):
            # Reproduce runtime enrichment with an object JSON cannot encode,
            # plus an in-place mutation below a second nested container.
            received["speculative_config"]["draft_model_config"] = object()
            received["compilation_config"]["cudagraph_capture_sizes"].append(16)

        def collective_rpc(self, name):
            if name == "parity_native_provenance":
                return [dict(rank=0, libraries={"native": dict(sha256="fixed")})]
            assert name == "parity_snapshot"
            return [dict(rank=0, routes={"decode_scalar_paged": 1}, host_kv={})]

        def generate(self, prompts, sampling, *, use_tqdm):
            assert prompts == ["fixed prompt"] and not use_tqdm
            assert sampling.temperature == 0.0 and not sampling.ignore_eos
            return [
                SimpleNamespace(
                    prompt_token_ids=[3],
                    outputs=[
                        SimpleNamespace(
                            token_ids=[7, 9], text="answer", finish_reason="stop"
                        )
                    ],
                )
            ]

    monkeypatch.setattr(vllm, "LLM", Engine)
    monkeypatch.setattr(
        vllm, "SamplingParams", lambda **kwargs: SimpleNamespace(**kwargs)
    )
    monkeypatch.setattr(route_parity, "runtime", lambda: {"test": True})
    monkeypatch.setattr(route_parity, "provenance", lambda sha: dict(source_sha=sha))
    monkeypatch.setenv("VLLM_NO_USAGE_STATS", "1")
    monkeypatch.setenv("VLLM_FLASH_V100_ROUTE_SUMMARY", "1")
    model = tmp_path / "model.bin"
    model.write_bytes(b"fixed model identity")
    args = SimpleNamespace(
        engine_args=tmp_path / "engine.json",
        prompts=tmp_path / "prompts.json",
        model_identity=tmp_path / "identity.json",
        output=tmp_path / "result.json",
        source_sha="a" * 40,
        max_tokens=8,
        require_route=["decode_scalar_paged"],
        require_host_fp8=False,
    )
    write_json(args.engine_args, options)
    write_json(args.prompts, ["fixed prompt"])
    write_json(args.model_identity, {"files": {str(model): digest(model)}})
    route_parity.record(args)
    result = read_json(args.output)
    recorded = result["contract"]["engine"]
    assert recorded["speculative_config"] == options["speculative_config"]
    assert recorded["compilation_config"] == options["compilation_config"]
    assert result["requests"][0]["token_ids"] == [7, 9]
    assert result["contract"]["native_sha256"] == [{"native": "fixed"}]
