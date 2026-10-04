# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A passing task score must not hide an unhealthy generation."""

from benchmarks.benchmark_sm70_qwen38_quality import health_failures


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
