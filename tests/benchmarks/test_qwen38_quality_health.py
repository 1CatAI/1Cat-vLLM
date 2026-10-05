# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A passing task score must not hide an unhealthy generation."""

import copy

import pytest

from benchmarks.benchmark_sm70_qwen38_quality import compare_quality, health_failures


def test_scored_code_at_token_limit_is_unhealthy():
    record = {
        "id": "code-case",
        "score": {"passed": True},
        "health": {
            "natural_eos": False,
            "nonempty_final": True,
            "replacement_characters": 0,
            "line_repetition": 11,
        },
    }
    assert health_failures([record]) == [
        {
            "id": "code-case",
            "reasons": ["not_natural_eos", "repeated_final_answer_line"],
        }
    ]


def test_healthy_output_can_repeat_a_line_once():
    record = {
        "id": "healthy-case",
        "health": {
            "natural_eos": True,
            "nonempty_final": True,
            "replacement_characters": 0,
            "line_repetition": 2,
        },
    }
    assert health_failures([record]) == []


def test_empty_or_corrupt_final_answer_requires_review():
    records = [
        {
            "id": "empty-case",
            "health": {
                "natural_eos": True,
                "nonempty_final": False,
                "replacement_characters": 0,
                "line_repetition": 0,
            },
        },
        {
            "id": "corrupt-case",
            "health": {
                "natural_eos": True,
                "nonempty_final": True,
                "replacement_characters": 1,
                "line_repetition": 0,
            },
        },
    ]
    assert health_failures(records) == [
        {"id": "empty-case", "reasons": ["empty_final_answer"]},
        {"id": "corrupt-case", "reasons": ["replacement_characters"]},
    ]


def three_seed_report():
    records = []
    for case, category, length in (
        ("code", "mbpp", 200),
        ("needle-128", "needle", 131000),
        ("needle-258", "needle", 258000),
    ):
        for seed in (4201, 5201, 6201):
            records.append(
                {
                    "id": case,
                    "category": category,
                    "seed": seed,
                    "input_tokens": length,
                    "prompt_token_sha256": case,
                    "score": {"passed": True},
                    "health": {
                        "natural_eos": True,
                        "nonempty_final": True,
                        "replacement_characters": 0,
                        "line_repetition": 0,
                    },
                }
            )
    return {
        "complete": True,
        "quality_evaluated": True,
        "contract": {},
        "sampling": {},
        "suite_sha256": "frozen",
        "quality_seed_bases": [4201, 5201, 6201],
        "quality": records,
    }


def test_three_seed_health_compares_against_baseline():
    baseline = three_seed_report()
    baseline["quality"][0]["health"]["natural_eos"] = False
    candidate = copy.deepcopy(baseline)
    assert compare_quality(baseline, candidate)["passed"]
    candidate["quality"][1]["health"]["natural_eos"] = False
    assert not compare_quality(baseline, candidate)["passed"]


def test_three_seed_score_regression_is_rejected():
    baseline = three_seed_report()
    candidate = copy.deepcopy(baseline)
    candidate["quality"][0]["score"]["passed"] = False
    assert not compare_quality(baseline, candidate)["passed"]


def test_quality_requires_matched_prefixes_and_long_needles():
    baseline = three_seed_report()
    candidate = copy.deepcopy(baseline)
    candidate["quality"][0]["prompt_token_sha256"] = "changed"
    with pytest.raises(ValueError, match="prefix differs"):
        compare_quality(baseline, candidate)
    baseline["quality"] = [r for r in baseline["quality"] if r["id"] != "needle-258"]
    with pytest.raises(ValueError, match="128K and 258K"):
        compare_quality(baseline, copy.deepcopy(baseline))
