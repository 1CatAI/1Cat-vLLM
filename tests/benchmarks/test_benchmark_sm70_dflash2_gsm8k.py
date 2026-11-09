# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
from argparse import Namespace
from pathlib import Path

import pytest


@pytest.fixture
def request_seeds_parser(monkeypatch):
    benchmark_dir = Path(__file__).resolve().parents[2] / "benchmarks"
    monkeypatch.syspath_prepend(str(benchmark_dir))
    module = importlib.import_module("benchmark_sm70_dflash2_gsm8k")
    return module._request_seeds


@pytest.fixture
def answer_parser(monkeypatch):
    benchmark_dir = Path(__file__).resolve().parents[2] / "benchmarks"
    monkeypatch.syspath_prepend(str(benchmark_dir))
    module = importlib.import_module("benchmark_sm70_dflash2_gsm8k")
    return module._answer_value, module.INVALID_ANSWER


@pytest.fixture
def acceptance_gate(monkeypatch):
    benchmark_dir = Path(__file__).resolve().parents[2] / "benchmarks"
    monkeypatch.syspath_prepend(str(benchmark_dir))
    module = importlib.import_module("benchmark_sm70_dflash2_gsm8k")
    return module._acceptance_gate


@pytest.fixture
def tracked_env(monkeypatch):
    benchmark_dir = Path(__file__).resolve().parents[2] / "benchmarks"
    monkeypatch.syspath_prepend(str(benchmark_dir))
    module = importlib.import_module("benchmark_sm70_dflash2_gsm8k")
    return module._tracked_env


@pytest.mark.parametrize(
    ("request_seed", "request_seeds", "expected"),
    [
        (0, None, [0]),
        (-1, None, [None]),
        (0, "11, 22,33", [11, 22, 33]),
    ],
)
def test_request_seeds_parsing(
    request_seeds_parser, request_seed, request_seeds, expected
):
    args = Namespace(request_seed=request_seed, request_seeds=request_seeds)

    assert request_seeds_parser(args) == expected


@pytest.mark.parametrize(
    ("request_seed", "request_seeds"),
    [
        (-2, None),
        (0, ""),
        (0, "1,-2"),
        (0, "7,7"),
        (0, "1,invalid"),
    ],
)
def test_request_seeds_rejects_ambiguous_contracts(
    request_seeds_parser, request_seed, request_seeds
):
    args = Namespace(request_seed=request_seed, request_seeds=request_seeds)

    with pytest.raises(ValueError):
        request_seeds_parser(args)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (r"The answer is \boxed{10} inches after 3 weeks.", 10),
        (r"Primary: \boxed{106}. Compound alternative: 106.12.", 106),
        (r"Nested formatting: \boxed{\text{-1,024}}.", -1024),
        ("Fallback answer: 12.0", 12),
    ],
)
def test_answer_value_prefers_integral_boxed_answer(answer_parser, text, expected):
    parse, _invalid = answer_parser

    assert parse(text) == expected


@pytest.mark.parametrize("text", [r"\boxed{12.5}", "no numeric answer"])
def test_answer_value_rejects_missing_or_non_integral_answer(answer_parser, text):
    parse, invalid = answer_parser

    assert parse(text) == invalid


@pytest.mark.parametrize(
    ("observed", "minimum", "passed"),
    [
        (5.78, 5.78, True),
        (5.7629, 5.78, False),
        (None, 5.78, None),
    ],
)
def test_acceptance_gate_uses_its_named_metric(
    acceptance_gate, observed, minimum, passed
):
    gate = acceptance_gate(observed, minimum, "per_request_completion")

    assert gate == {
        "metric": "per_request_completion",
        "minimum": minimum,
        "observed": observed,
        "passed": passed,
    }


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("VLLM_GLM53_PP_MHC_MATERIALIZE", "1"),
        ("VLLM_PP_LAYER_PARTITION", "25,20"),
    ],
)
def test_runtime_contract_tracks_dflash_environment(
    tracked_env, monkeypatch, name, value
):
    monkeypatch.setenv(name, value)

    assert tracked_env()[name] == value
