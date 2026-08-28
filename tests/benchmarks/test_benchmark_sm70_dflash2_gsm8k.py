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
