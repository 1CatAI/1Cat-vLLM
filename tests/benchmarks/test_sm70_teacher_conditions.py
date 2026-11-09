# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from benchmarks.sm70_teacher_conditions import prefix_digest, teacher_conditions


def reference():
    return {
        "rows": [
            {"id": "story", "prompt_token_ids": [1, 2], "output_token_ids": [3, 4, 5]}
        ]
    }


def test_chained_reference_retains_original_teacher_prefix():
    report = reference()
    original = list(teacher_conditions(report, 2))
    report["teacher_forcing"] = {
        "rows": [
            {
                "key": key,
                "prefix_token_ids": prefix,
                "prefix_sha256": prefix_digest(prefix),
                "position": len(prefix),
                "forced": forced,
            }
            for key, prefix, forced in original
        ]
    }
    report["rows"][0]["output_token_ids"] = [8, 9, 10]
    assert list(teacher_conditions(report, 2)) == original


def test_legacy_changed_condition_fails_instead_of_comparing_other_logits():
    report = reference()
    report["teacher_forcing"] = {
        "rows": [
            {
                "key": "story-001",
                "prefix_sha256": prefix_digest([1, 2, 8]),
                "position": 3,
                "forced": 9,
            }
        ]
    }
    with pytest.raises(RuntimeError, match="not recoverable"):
        list(teacher_conditions(report, 2))


def test_legacy_identical_reference_is_recoverable():
    report = reference()
    report["teacher_forcing"] = {
        "rows": [
            {
                "key": "story-001",
                "prefix_sha256": prefix_digest([1, 2, 3]),
                "position": 3,
                "forced": 4,
            }
        ]
    }
    assert list(teacher_conditions(report, 2))[1] == ("story-001", [1, 2, 3], 4)
