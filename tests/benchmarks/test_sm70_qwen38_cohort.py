# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from benchmarks.benchmark_sm70_qwen38_concurrency import generate_cohort


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
