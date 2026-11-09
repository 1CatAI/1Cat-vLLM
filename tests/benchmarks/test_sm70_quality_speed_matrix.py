# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import pytest

from benchmarks.run_sm70_quality_speed_matrix import (
    _make_chat_prompt,
    _quality_metrics,
)


class RecordingTokenizer:
    def __init__(self, *, rejects_enable_thinking: bool = False) -> None:
        self.rejects_enable_thinking = rejects_enable_thinking
        self.calls: list[dict[str, Any]] = []

    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        self.calls.append(kwargs)
        if self.rejects_enable_thinking and "enable_thinking" in kwargs:
            raise TypeError("enable_thinking is unsupported")
        return f"{messages[0]['content']}<think>"


def test_make_chat_prompt_forwards_reasoning_effort():
    tokenizer = RecordingTokenizer()

    prompt = _make_chat_prompt(
        tokenizer,
        "question",
        enable_thinking=True,
        reasoning_effort="low",
    )

    assert prompt == "question<think>"
    assert tokenizer.calls == [
        {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": True,
            "reasoning_effort": "low",
        }
    ]


def test_make_chat_prompt_fallback_preserves_reasoning_effort():
    tokenizer = RecordingTokenizer(rejects_enable_thinking=True)

    _make_chat_prompt(
        tokenizer,
        "question",
        enable_thinking=False,
        reasoning_effort="high",
    )

    assert tokenizer.calls[-1]["reasoning_effort"] == "high"
    assert "enable_thinking" not in tokenizer.calls[-1]


def test_quality_metrics_reject_unclosed_reasoning():
    metrics = _quality_metrics(
        "still reasoning",
        list(range(64)),
        finish_reason="length",
        chat_prompt="question<think>",
    )

    assert metrics["reasoning_closed"] is False
    assert metrics["visible_final_chars"] == 0
    assert "unclosed_reasoning" in metrics["failures"]


def test_quality_metrics_reject_missing_final_answer():
    metrics = _quality_metrics(
        "reasoning</think>\n",
        list(range(64)),
        finish_reason="stop",
        chat_prompt="question<think>",
    )

    assert metrics["reasoning_closed"] is True
    assert metrics["visible_final_chars"] == 0
    assert "missing_final_answer" in metrics["failures"]


def test_quality_metrics_accept_short_natural_stop():
    metrics = _quality_metrics(
        "</think>\n```python\ndef stable_unique(items):\n    return list(items)\n```",
        list(range(16)),
        finish_reason="stop",
        chat_prompt="question<think>",
    )

    assert metrics["visible_final_chars"] > 0
    assert "too_few_output_tokens" not in metrics["failures"]
    assert metrics["passed"]


def test_quality_metrics_reject_short_length_cutoff():
    metrics = _quality_metrics(
        "incomplete output",
        list(range(16)),
        finish_reason="length",
    )

    assert "too_few_output_tokens" in metrics["failures"]


@pytest.mark.parametrize("text", ["", " \n\t"])
def test_quality_metrics_reject_empty_natural_stop(text):
    metrics = _quality_metrics(text, [], finish_reason="stop")

    assert not metrics["passed"]
    assert "missing_final_answer" in metrics["failures"]
