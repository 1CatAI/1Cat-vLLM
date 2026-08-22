# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure an SGLang DFlash2 server on the local GSM8K corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import requests
from benchmark_sm70_dflash2_gsm8k import (
    GSM8K_PROMPT_SUFFIX,
    _answer_value,
    _distribution,
    _load_rows,
)
from transformers import AutoTokenizer


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18082")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--num-questions", type=int, default=64)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--dataset-order",
        choices=("sequential", "zlab-shuffle42"),
        default="zlab-shuffle42",
    )
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--warmup-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--request-seed", type=int, default=-1)
    parser.add_argument("--timeout-seconds", type=float, default=3600.0)
    return parser.parse_args()


def _finish_reason(meta: dict[str, Any]) -> str | None:
    reason = meta.get("finish_reason")
    if isinstance(reason, dict):
        value = reason.get("type")
        return str(value) if value is not None else None
    return str(reason) if reason is not None else None


def main() -> int:
    args = _parse_args()
    rows = _load_rows(args)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model))
    prompts = [
        tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": row["question"] + GSM8K_PROMPT_SUFFIX,
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
            reasoning_effort="xhigh",
        )
        for _, row in rows
    ]

    session = requests.Session()

    def generate(prompt: str, max_tokens: int) -> dict[str, Any]:
        sampling_params: dict[str, Any] = {
            "temperature": args.temperature,
            "top_p": 0.95,
            "top_k": 20,
            "max_new_tokens": max_tokens,
        }
        if args.request_seed >= 0:
            sampling_params["seed"] = args.request_seed
        response = session.post(
            f"{args.base_url.rstrip('/')}/generate",
            json={"text": prompt, "sampling_params": sampling_params},
            timeout=args.timeout_seconds,
        )
        response.raise_for_status()
        output = response.json()
        if not isinstance(output, dict):
            raise TypeError(f"Expected an object response, got {type(output).__name__}")
        return output

    generate(prompts[0], args.warmup_tokens)
    cases: list[dict[str, Any]] = []
    started = time.perf_counter()
    for (dataset_index, row), prompt in zip(rows, prompts, strict=True):
        request_started = time.perf_counter()
        output = generate(prompt, args.max_tokens)
        request_elapsed = time.perf_counter() - request_started
        meta = output.get("meta_info") or {}
        text = str(output.get("text") or "")
        expected_answer = _answer_value(row["answer"])
        predicted_answer = _answer_value(text)
        cases.append(
            {
                "dataset_index": dataset_index,
                "question": row["question"],
                "expected_answer": expected_answer,
                "predicted_answer": predicted_answer,
                "correct": predicted_answer == expected_answer,
                "text": text,
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "request_elapsed_seconds": request_elapsed,
                "prompt_tokens": int(meta.get("prompt_tokens", 0)),
                "completion_tokens": int(meta.get("completion_tokens", 0)),
                "finish_reason": _finish_reason(meta),
                "spec_accept_length": float(meta["spec_accept_length"]),
                "spec_verify_ct": int(meta["spec_verify_ct"]),
                "spec_num_correct_drafts": int(meta.get("spec_num_correct_drafts", 0)),
                "spec_num_proposed_drafts": int(
                    meta.get("spec_num_proposed_drafts", 0)
                ),
                "spec_correct_drafts_histogram": meta.get(
                    "spec_correct_drafts_histogram"
                ),
            }
        )
    elapsed = time.perf_counter() - started

    completion_tokens = sum(case["completion_tokens"] for case in cases)
    verify_steps = sum(case["spec_verify_ct"] for case in cases)
    result = {
        "contract": {
            "backend": "sglang-v100",
            "base_url": args.base_url,
            "model": str(args.model),
            "dataset": str(args.dataset),
            "dataset_order": args.dataset_order,
            "num_questions": args.num_questions,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": 0.95,
            "top_k": 20,
            "reasoning_effort": "xhigh",
            "request_seed": None if args.request_seed < 0 else args.request_seed,
        },
        "results": {
            "elapsed_seconds": elapsed,
            "total_completion_tokens": completion_tokens,
            "aggregate_output_tokens_per_second": completion_tokens / elapsed,
            "accuracy": sum(case["correct"] for case in cases) / len(cases),
            "finish_reasons": dict(Counter(case["finish_reason"] for case in cases)),
            "per_request_acceptance_length": _distribution(
                [case["spec_accept_length"] for case in cases]
            ),
            "pooled_completion_tokens_per_verify": (
                completion_tokens / verify_steps if verify_steps else None
            ),
            "request_elapsed_seconds": _distribution(
                [case["request_elapsed_seconds"] for case in cases]
            ),
        },
        "cases": cases,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["results"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
