# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from benchmarks.benchmark_sm70_qwen38_concurrency import (
    finalize_measurements,
    generate_cohort,
    long_quality_prompt_ids,
)


def make_llm():
    events = Mock()
    llm = SimpleNamespace(
        llm_engine=SimpleNamespace(
            engine_core=SimpleNamespace(call_utility=events.rpc)
        ),
        enqueue=events.enqueue,
        generate=events.generate,
        wait_for_completion=events.wait,
    )
    return llm, events


class CharacterTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": True,
        }
        return "<user>" + messages[0]["content"] + "</user><assistant><think>"

    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        return list(map(ord, text))


@pytest.mark.parametrize("length", [131072 - 513, 262144 - 513, 262144 - 1])
def test_long_quality_prompt_preserves_exact_length_and_middle_record(length):
    tokenizer = CharacterTokenizer()
    ids = long_quality_prompt_ids(tokenizer, length)
    assert len(ids) == length
    text = "".join(map(chr, ids))
    assert text.count("Archive code: MAPLE-8261.") == 1
    assert abs(text.index("Archive code:") - length // 2) < 128
    assert text.startswith("<user>")
    assert text.endswith("</user><assistant><think>")
    assert "finish with RESULT=<code>." in text


def test_long_quality_prompt_rejects_insufficient_room():
    with pytest.raises(ValueError, match="needs room"):
        long_quality_prompt_ids(CharacterTokenizer(), 1)


def test_streaming_cohort_preserves_generate():
    llm, events = make_llm()
    result = generate_cohort(llm, ["prompt"], "sampling")
    assert result is events.generate.return_value
    assert events.mock_calls == [call.generate(["prompt"], "sampling", use_tqdm=False)]


def test_atomic_cohort_resumes_only_after_enqueue():
    llm, events = make_llm()
    result = generate_cohort(llm, ["a", "b"], "sampling", atomic=True)
    assert result is events.wait.return_value
    assert events.mock_calls == [
        call.rpc("pause_scheduler", "keep", False),
        call.enqueue(["a", "b"], "sampling", use_tqdm=False),
        call.rpc("resume_scheduler"),
        call.wait(use_tqdm=False),
    ]


def test_atomic_cohort_resumes_on_enqueue_failure():
    llm, events = make_llm()
    events.enqueue.side_effect = ValueError("invalid prompt")
    with pytest.raises(ValueError, match="invalid prompt"):
        generate_cohort(llm, ["prompt"], "sampling", atomic=True)
    assert events.mock_calls[-1] == call.rpc("resume_scheduler")
    events.wait.assert_not_called()


def test_atomic_cohort_does_not_enqueue_after_pause_failure():
    llm, events = make_llm()
    events.rpc.side_effect = RuntimeError("pause failed")
    with pytest.raises(RuntimeError, match="pause failed"):
        generate_cohort(llm, ["prompt"], "sampling", atomic=True)
    assert events.mock_calls == [call.rpc("pause_scheduler", "keep", False)]


@pytest.mark.parametrize(
    "extra",
    [
        {"baseline_runs": [{"matches_reference": False}]},
        {"reference_accepted": False},
        {"cases": [{"tokens_match_reference": [True, False]}]},
        {"cases": [{"tokens_match_first_repeat": [False]}]},
    ],
)
def test_parity_failure_keeps_measurements_but_rejects_result(extra):
    report = {
        "cases": [{"tokens_match_first_repeat": [True]}],
        "complete": True,
        **extra,
    }
    with pytest.raises(RuntimeError, match="Token parity failed"):
        finalize_measurements(report)
    assert report["measurements_complete"] is True
    assert report["token_parity_passed"] is False
    assert report["complete"] is False


@pytest.mark.parametrize("extra", [{}, {"reference_accepted": True}])
def test_no_observed_parity_cannot_pass(extra):
    report = {"cases": [{}], **extra}
    with pytest.raises(RuntimeError, match="Token parity was not checked"):
        finalize_measurements(report)
    assert report["measurements_complete"] is True
    assert report["token_parity_passed"] is None
    assert report["complete"] is False


@pytest.mark.parametrize(
    "extra",
    [
        {"cases": [{"tokens_match_first_repeat": [True, True]}]},
        {"cases": [{"tokens_match_reference": [True]}], "reference_accepted": True},
        {"baseline_runs": [{"matches_reference": True}]},
    ],
)
def test_passing_observed_parity_accepts_result(extra):
    report = {"cases": [{}], **extra}
    finalize_measurements(report)
    assert report["measurements_complete"] is True
    assert report["token_parity_passed"] is True
    assert report["complete"] is True
