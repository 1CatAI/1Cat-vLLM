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
