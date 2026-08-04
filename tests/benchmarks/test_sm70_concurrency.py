# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for the SM70 exact-batch concurrency benchmark helpers."""

import importlib.util
import math
from pathlib import Path


def _load_benchmark_module():
    path = Path(__file__).parents[2] / "benchmarks" / "benchmark_sm70_concurrency.py"
    spec = importlib.util.spec_from_file_location("sm70_concurrency_benchmark", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Tokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        if "Request identifier:" in text:
            return [91, ord(text[-2])]
        return [11, 12, 13]


def test_prompt_ids_keep_exact_length_and_per_request_suffix():
    benchmark = _load_benchmark_module()

    first = benchmark._make_prompt_token_ids(_Tokenizer(), "base", 12, 0)
    second = benchmark._make_prompt_token_ids(_Tokenizer(), "base", 12, 1)

    assert len(first) == 12
    assert len(second) == 12
    assert first != second


def test_spec_metric_delta_and_batch_summary():
    benchmark = _load_benchmark_module()
    before = {
        "num_drafts": 2,
        "num_draft_tokens": 8,
        "num_accepted_tokens": 5,
        "per_pos_accepted": [2, 2, 1],
    }
    after = {
        "num_drafts": 5,
        "num_draft_tokens": 20,
        "num_accepted_tokens": 14,
        "per_pos_accepted": [5, 5, 3, 1],
    }

    delta = benchmark._diff_spec_metrics(before, after)
    assert delta == {
        "num_drafts": 3,
        "num_draft_tokens": 12,
        "num_accepted_tokens": 9,
        "mean_acceptance_length": 4.0,
        "draft_acceptance_rate": 0.75,
        "per_pos_accepted": [3, 3, 2, 1],
        "per_position_acceptance_rate": [1.0, 1.0, 2 / 3, 1 / 3],
    }

    records = [
        {
            "output_tokens": 5,
            "request_metrics": {
                "steady_decode_tokens": 4,
                "tpot_s": 0.01,
                "raw": {
                    "first_token_ts": 10.0,
                    "last_token_ts": 10.04,
                    "is_corrupted": False,
                },
            },
        },
        {
            "output_tokens": 4,
            "request_metrics": {
                "steady_decode_tokens": 3,
                "tpot_s": 0.02,
                "raw": {
                    "first_token_ts": 10.0,
                    "last_token_ts": 10.06,
                    "is_corrupted": True,
                },
            },
        },
    ]

    summary = benchmark._summarize_records(records, elapsed_s=0.1)
    assert summary["total_output_tokens"] == 9
    assert summary["steady_decode_tokens"] == 7
    assert math.isclose(summary["aggregate_steady_decode_tps"], 7 / 0.06)
    assert summary["per_request_tpot_ms"]["p50"] == 15.0
    assert summary["corrupted_request_count"] == 1
