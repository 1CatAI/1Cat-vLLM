# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import hashlib
import json

import pytest

from benchmarks.sm70_teacher_forcing_metrics import validate_manifest


def tape():
    ids = [1] * 8192 + [2] * 192
    return {
        "id": "one",
        "token_ids": ids,
        "prompt_length": 8192,
        "output_length": 160,
        "prompt_sha256": hashlib.sha256(json.dumps(ids[:8192]).encode()).hexdigest(),
    }


def test_valid_padded_frozen_tape():
    case = tape()
    assert validate_manifest([case], 3) == [case]


@pytest.mark.parametrize("bad", ["input_ids", "attention_mask", True, -1, 3])
def test_bad_tokens_rejected_before_model_initialization(bad):
    case = tape()
    case["token_ids"][0] = bad
    with pytest.raises(ValueError, match="token"):
        validate_manifest([case], 3)


def test_prompt_identity_and_duplicate_case_rejected():
    case = tape()
    case["prompt_sha256"] = "incorrect"
    with pytest.raises(ValueError, match="SHA"):
        validate_manifest([case], 3)
    with pytest.raises(ValueError, match="Duplicate"):
        validate_manifest([tape(), tape()], 3)
