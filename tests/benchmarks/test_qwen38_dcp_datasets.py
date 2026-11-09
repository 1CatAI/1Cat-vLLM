# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from benchmarks.qwen38_dcp_datasets import (
    INVALID_ANSWER,
    _answer_value,
    evaluate_datasets,
    prepare_datasets,
    summarize_dataset,
)

pytestmark = pytest.mark.cpu_test


def test_dataset_import_does_not_inject_source_paths():
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; before = sys.path[:]; "
                "import benchmarks.qwen38_dcp_datasets; "
                "assert sys.path == before, (before, sys.path)"
            ),
        ],
        check=True,
    )


@pytest.mark.parametrize(
    ("text", "answer"),
    [
        (r"reason 42 then \\boxed{1,234}", 1234),
        (r"\\boxed{-7}", -7),
        ("reference reasoning\n#### 437", 437),
        (r"\\boxed{1.5}", INVALID_ANSWER),
        ("unfinished without answer", INVALID_ANSWER),
    ],
)
def test_answer_extractor_keeps_existing_integer_contract(text, answer):
    assert _answer_value(text) == answer


def test_dataset_preflight_selection_and_scoring(tmp_path):
    data = tmp_path / "math.jsonl"
    data.write_text(
        "\n".join(
            json.dumps(
                {"question": f"Question {i}", "answer": "reference reasoning\n#### 437"}
            )
            for i in range(4)
        )
    )
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps(
            {
                "selection_seed": 123,
                "generation_seed": 7,
                "batch_size": 2,
                "gsm8k": {"path": str(data), "count": 3, "max_tokens": 2048},
            }
        )
    )

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["return_dict"] is False
            assert kwargs["enable_thinking"] is True
            assert "437" not in messages[0]["content"]
            return [1, 2, 3]

    cases, metrics, manifest = prepare_datasets(spec, Tokenizer())
    assert prepare_datasets(spec, Tokenizer())[2] == manifest
    assert len(cases) == 3 and len({case["index"] for case in cases}) == 3
    assert [case["seed"] for case in cases] == [7, 8, 9]

    class LLM:
        def generate(self, prompts, sampling, **kwargs):
            assert len(prompts) <= 2
            assert all(p.temperature == 1.0 and not p.ignore_eos for p in sampling)
            return [
                SimpleNamespace(
                    metrics=SimpleNamespace(is_corrupted=False),
                    outputs=[
                        SimpleNamespace(
                            text="reasoning</think>\\boxed{437}",
                            token_ids=[1, 2],
                            finish_reason="stop",
                        )
                    ],
                )
                for _ in prompts
            ]

    report: dict[str, Any] = {}
    saves = []
    evaluate_datasets(
        LLM(),
        cases,
        metrics,
        manifest,
        {"temperature": 1.0, "top_p": 0.95, "top_k": 20},
        report,
        lambda: saves.append(True),
    )
    assert report["dataset"]["complete"]
    assert report["dataset"]["summary"]["gsm8k"]["score"] == 100
    assert len(saves) == 2


@pytest.mark.parametrize(
    ("score", "reason", "accepted"),
    [(1.0, "stop", True), (0.0, "stop", False), (1.0, "length", False)],
)
def test_dataset_gate_preserves_score_and_health(score, reason, accepted):
    control = {
        "id": "gsm8k/0",
        "dataset": "gsm8k",
        "score": 1.0,
        "finish_reason": "stop",
        "answer": "437",
        "corrupted": False,
    }
    candidate = control | {"score": score, "finish_reason": reason}
    summary = summarize_dataset([candidate], [control])["gsm8k"]
    assert summary["observed_no_regression"] is accepted
    assert summary["losses"] == (score < 1.0)
