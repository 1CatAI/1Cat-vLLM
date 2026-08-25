# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from copy import deepcopy

from benchmarks.benchmark_dsv4_gsm8k_api import _last_integer
from benchmarks.compare_dsv4_quality_results import (
    _compare_gsm8k,
    _compare_humaneval,
    _compare_needle,
)


def test_last_integer_normalizes_signed_integral_decimal() -> None:
    assert _last_integer("work... #### -1,024") == -1024
    assert _last_integer("answer: 12.0") == 12
    assert _last_integer("answer: 12.5") is None
    assert _last_integer("no numeric answer") is None


def _gsm8k_artifact(predictions: list[int | None]) -> dict:
    expected = [13, 40]
    rows = [
        {
            "index": index,
            "question": f"question-{index}",
            "expected": answer,
            "predicted": prediction,
            "correct": prediction is not None and prediction == answer,
            "invalid": prediction is None,
        }
        for index, (answer, prediction) in enumerate(zip(expected, predictions))
    ]
    return {
        "contract": {
            "questions": 2,
            "few_shot": 5,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 42,
            "max_tokens": 256,
            "strictly_sequential": True,
            "train_selection": "first_n",
            "test_selection": "first_n",
            "prompt_format": "gsm8k_question_answer_v1",
            "answer_normalization": "last_signed_integral_decimal_v1",
            "stop_sequences": ["Question"],
        },
        "input_manifest": {
            "train": {"sha256": "train-hash"},
            "test": {"sha256": "test-hash"},
        },
        "correct": sum(bool(row["correct"]) for row in rows),
        "invalid": sum(bool(row["invalid"]) for row in rows),
        "rows": rows,
    }


def test_gsm8k_gate_allows_balanced_answer_flips_without_greedy_identity() -> None:
    reference = _gsm8k_artifact([13, 42])
    candidate = _gsm8k_artifact([12, 40])

    result = _compare_gsm8k(reference, candidate)

    assert result["passed"]
    assert result["correct_delta"] == 0
    assert result["regressions"] == [0]
    assert result["improvements"] == [1]
    assert result["exact_prediction_matches"] == 0


def test_gsm8k_gate_rejects_aggregate_drop_contract_drift_and_bad_summary() -> None:
    reference = _gsm8k_artifact([13, 40])
    candidate = _gsm8k_artifact([12, 40])
    assert not _compare_gsm8k(reference, candidate)["passed"]

    candidate = deepcopy(reference)
    candidate["contract"]["seed"] = 7
    assert not _compare_gsm8k(reference, candidate)["passed"]

    candidate = deepcopy(reference)
    candidate["correct"] = 0
    assert not _compare_gsm8k(reference, candidate)["passed"]


def test_humaneval_gate_uses_aggregate_quality_and_reports_flips() -> None:
    reference = {
        "human_eval": {
            "passed": 1,
            "records": [
                {"task_id": "a", "passed": True, "response": "a0"},
                {"task_id": "b", "passed": False, "response": "b0"},
            ],
        }
    }
    candidate = {
        "human_eval": {
            "passed": 1,
            "records": [
                {"task_id": "a", "passed": False, "response": "a1"},
                {"task_id": "b", "passed": True, "response": "b1"},
            ],
        }
    }

    result = _compare_humaneval(reference, candidate)

    assert result["passed"]
    assert result["regressions"] == ["a"]
    assert result["improvements"] == ["b"]


def test_needle_gate_uses_aggregate_hits_and_reports_flips() -> None:
    reference = [
        {
            "sample_id": "a",
            "target_tokens": 100,
            "depth": 0.5,
            "code": "x",
            "hit": True,
            "hit_anywhere": True,
        },
        {
            "sample_id": "b",
            "target_tokens": 100,
            "depth": 0.5,
            "code": "y",
            "hit": False,
            "hit_anywhere": False,
        },
    ]
    candidate = [
        {**reference[0], "hit": False, "hit_anywhere": False},
        {**reference[1], "hit": True, "hit_anywhere": True},
    ]

    result = _compare_needle(reference, candidate)

    assert result["passed"]
    assert result["final_hit_regressions"] == ["a"]
    assert result["anywhere_hit_regressions"] == ["a"]
