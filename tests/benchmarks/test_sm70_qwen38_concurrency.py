# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from benchmarks.benchmark_sm70_qwen38_concurrency import (
    compare_tokens,
    finalize_measurements,
)


@pytest.mark.parametrize(
    "actual,expected,difference",
    [
        ([1, 2], [1, 2], None),
        ([1, 3], [1, 2], 1),
        ([1], [1, 2], 1),
        ([1, 2], [1], 1),
        ([], [1], 0),
    ],
)
def test_first_token_difference(actual, expected, difference):
    assert compare_tokens([{"token_ids": actual}], [{"token_ids": expected}]) == [
        difference
    ]


def test_request_count_is_not_silently_truncated():
    with pytest.raises(ValueError):
        compare_tokens([], [{"token_ids": [1]}])


@pytest.mark.parametrize("key", ["tokens_match_reference", "tokens_match_first_repeat"])
def test_failed_case_is_not_accepted_after_later_passes(key):
    report = {"complete": False, "cases": [{key: [True, False]}, {key: [True]}]}
    with pytest.raises(RuntimeError, match="not an accepted"):
        finalize_measurements(report)
    assert report["measurements_complete"] is True
    assert report["complete"] is False
    assert report["token_parity_passed"] is False
    assert report["cases"] == [{key: [True, False]}, {key: [True]}]


@pytest.mark.parametrize("checks,expected", [([], None), ([True], True)])
def test_complete_does_not_invent_unperformed_parity(checks, expected):
    report = {"complete": False, "cases": [{"tokens_match_reference": checks}]}
    finalize_measurements(report)
    assert report["complete"] is True
    assert report["measurements_complete"] is True
    assert report["token_parity_passed"] is expected


def test_unaccepted_c2_admission_remains_opt_in(monkeypatch):
    from vllm import envs

    monkeypatch.delenv("VLLM_SM70_TP4_PUSH_ALLREDUCE_SUM2_M2", raising=False)
    assert envs.environment_variables["VLLM_SM70_TP4_PUSH_ALLREDUCE_SUM2_M2"]() is False
    monkeypatch.setenv("VLLM_SM70_TP4_PUSH_ALLREDUCE_SUM2_M2", "1")
    assert envs.environment_variables["VLLM_SM70_TP4_PUSH_ALLREDUCE_SUM2_M2"]() is True
