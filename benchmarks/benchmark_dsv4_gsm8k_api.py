# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run a deterministic, sequential GSM8K gate against a vLLM API server."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import regex as re

_NUMBER_RE = re.compile(r"(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![\w.])")
_STOP_SEQUENCES = ["Question", "Assistant:", "<|separator|>"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]


def _last_integer(text: str) -> int | None:
    numbers = _NUMBER_RE.findall(text)
    if not numbers:
        return None
    try:
        value = Decimal(numbers[-1].replace(",", ""))
    except InvalidOperation:
        return None
    if not value.is_finite() or value != value.to_integral_value():
        return None
    return int(value)


def _post_completion(
    *,
    host: str,
    port: int,
    timeout: int,
    payload: dict[str, Any],
) -> tuple[int, dict[str, Any], float]:
    connection = http.client.HTTPConnection(host, port, timeout=timeout)
    started = time.perf_counter()
    try:
        connection.request(
            "POST",
            "/v1/completions",
            body=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        body = response.read()
    finally:
        connection.close()
    elapsed = time.perf_counter() - started
    try:
        decoded = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"completion returned non-JSON HTTP {response.status}: "
            f"{body.decode('utf-8', errors='replace')}"
        ) from exc
    return response.status, decoded, elapsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--few-shot", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--min-correct", type=int, default=0)
    parser.add_argument("--max-invalid", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.limit <= 0 or args.few_shot < 0:
        raise ValueError("--limit must be positive and --few-shot non-negative")
    if args.min_correct < 0 or args.max_invalid < 0:
        raise ValueError("quality thresholds must be non-negative")

    parsed = urlparse(args.base_url)
    if parsed.scheme != "http" or not parsed.hostname:
        raise ValueError("--base-url must be an http URL")
    host = parsed.hostname
    port = parsed.port or 80

    train = _load_jsonl(args.train)
    test = _load_jsonl(args.test)[: args.limit]
    if len(train) < args.few_shot:
        raise RuntimeError(
            f"training set has {len(train)} rows, fewer than {args.few_shot} shots"
        )
    if len(test) < args.limit:
        raise RuntimeError(f"test set has only {len(test)} rows, need {args.limit}")

    few_shot = "".join(
        f"Question: {row['question']}\nAnswer: {row['answer']}\n\n"
        for row in train[: args.few_shot]
    )
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, row in enumerate(test):
        prompt = few_shot + f"Question: {row['question']}\nAnswer:"
        request = {
            "model": args.model,
            "prompt": prompt,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": args.max_tokens,
            "stop": _STOP_SEQUENCES,
            "seed": args.seed,
        }
        status, response, elapsed = _post_completion(
            host=host,
            port=port,
            timeout=args.timeout,
            payload=request,
        )
        choices = response.get("choices") or []
        choice = choices[0] if choices else {}
        output = choice.get("text") or ""
        expected = _last_integer(row["answer"])
        if expected is None:
            raise ValueError(f"GSM8K item {index} has no integral reference answer")
        predicted = _last_integer(output)
        rows.append(
            {
                "index": index,
                "question": row["question"],
                "expected": expected,
                "predicted": predicted,
                "correct": predicted is not None and predicted == expected,
                "invalid": predicted is None,
                "status": status,
                "elapsed_seconds": elapsed,
                "finish_reason": choice.get("finish_reason"),
                "usage": response.get("usage"),
                "output": output,
            }
        )
        if status != 200:
            raise RuntimeError(f"GSM8K item {index} returned HTTP {status}: {response}")

    wall_seconds = time.perf_counter() - started
    correct = sum(bool(row["correct"]) for row in rows)
    invalid = sum(bool(row["invalid"]) for row in rows)
    completion_tokens = sum(
        int((row.get("usage") or {}).get("completion_tokens") or 0) for row in rows
    )
    passed = correct >= args.min_correct and invalid <= args.max_invalid
    result = {
        "contract": {
            "base_url": args.base_url,
            "model": args.model,
            "questions": args.limit,
            "few_shot": args.few_shot,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": args.seed,
            "max_tokens": args.max_tokens,
            "strictly_sequential": True,
            "train_selection": "first_n",
            "test_selection": "first_n",
            "prompt_format": "gsm8k_question_answer_v1",
            "answer_normalization": "last_signed_integral_decimal_v1",
            "stop_sequences": _STOP_SEQUENCES,
            "min_correct": args.min_correct,
            "max_invalid": args.max_invalid,
        },
        "input_manifest": {
            "train": {"path": str(args.train), "sha256": _sha256(args.train)},
            "test": {"path": str(args.test), "sha256": _sha256(args.test)},
        },
        "correct": correct,
        "accuracy": correct / len(rows),
        "invalid": invalid,
        "wall_seconds": wall_seconds,
        "completion_tokens": completion_tokens,
        "aggregate_output_tokens_per_second": (
            completion_tokens / wall_seconds if wall_seconds else None
        ),
        "passed": passed,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "correct": correct,
                "samples": len(rows),
                "invalid": invalid,
                "passed": passed,
            }
        ),
        flush=True,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
