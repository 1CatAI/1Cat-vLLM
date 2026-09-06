# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Concurrent BFCL/JSONSchemaBench API gate using the existing task scorers.

Run identical cases/seeds against control and candidate servers separately.
Client concurrency is NOT proof of a GPU batch width or a throughput metric:
confirm actual batch dispatch from worker logs/traces. No tools are executed.
"""

import argparse
import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import jsonschema

from benchmarks.benchmark_sm70_tool_protocol import (
    _adapt_openai_json_schema,
    _bfcl_tools,
    _read_jsonl,
    _stream_chat,
    _validate_bfcl_calls,
)


def load_cases(bfcl_dir, schema_dir, per_category, schema_limit, max_schema_bytes):
    cases = []
    sources = {}
    for category in ("simple_python", "parallel", "multiple", "irrelevance"):
        path = bfcl_dir / f"BFCL_v4_{category}.json"
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        truth = {}
        if category != "irrelevance":
            answers = bfcl_dir / "possible_answer" / path.name
            sources[str(answers)] = hashlib.sha256(answers.read_bytes()).hexdigest()
            truth = {row["id"]: row["ground_truth"] for row in _read_jsonl(answers)}
        for entry in _read_jsonl(path)[:per_category]:
            cases.append(
                {
                    "id": entry["id"],
                    "suite": f"bfcl/{category}",
                    "entry": entry,
                    "ground_truth": (
                        None if category == "irrelevance" else truth[entry["id"]]
                    ),
                    "irrelevance": category == "irrelevance",
                    "request": {
                        "messages": entry["question"][0],
                        "tools": _bfcl_tools(entry["function"]),
                        "tool_choice": "auto",
                        "parallel_tool_calls": True,
                    },
                }
            )
    paths = sorted(
        (p for p in schema_dir.glob("*.json") if p.stat().st_size <= max_schema_bytes),
        key=lambda p: (p.stat().st_size, p.name),
    )
    if len(paths) > schema_limit:
        paths = (
            [paths[len(paths) // 2]]
            if schema_limit == 1
            else [
                paths[round(i * (len(paths) - 1) / (schema_limit - 1))]
                for i in range(schema_limit)
            ]
        )
    if not paths:
        raise ValueError("No JSONSchemaBench cases selected")
    for path in paths:
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        schema = _adapt_openai_json_schema(json.loads(path.read_text()))
        jsonschema.validators.validator_for(schema).check_schema(schema)
        cases.append(
            {
                "id": path.stem,
                "suite": "json_schema",
                "schema": schema,
                "request": {
                    "messages": [
                        {
                            "role": "system",
                            "content": "Generate a JSON object matching the schema.",
                        },
                        {"role": "user", "content": json.dumps(schema)},
                    ],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": path.stem,
                            "strict": True,
                            "schema": schema,
                        },
                    },
                },
            }
        )
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Duplicate case IDs")
    return cases, sources


def score_case(case, response):
    errors = []
    if not response.get("ok"):
        return ["request failed"]
    if response.get("finish_reason") not in ("stop", "tool_calls"):
        errors.append(f"incomplete output: {response.get('finish_reason')!r}")
    if response.get("tool_calls") and response.get("finish_reason") != "tool_calls":
        errors.append("tool calls have an incorrect finish_reason")
    if case["suite"] == "json_schema":
        try:
            jsonschema.validate(json.loads(response["content"]), case["schema"])
        except (KeyError, json.JSONDecodeError, jsonschema.ValidationError) as exc:
            errors.append(str(exc))
    else:
        errors.extend(
            _validate_bfcl_calls(
                response,
                case["entry"],
                case["ground_truth"],
                irrelevance=case["irrelevance"],
            )
        )
    return errors


def run_cases(cases, base_url, common, concurrency, request=_stream_chat):
    lock = threading.Lock()
    active = peak = 0
    started = time.perf_counter()

    def run(item):
        nonlocal active, peak
        index, case = item
        payload = {**common, **case["request"], "seed": common["seed"] + index}
        with lock:
            active += 1
            peak = max(peak, active)
        start = time.perf_counter() - started
        try:
            response = request(base_url, payload)
        except Exception as exc:
            # Retain transport/parser failures as failed cases, never silently
            # drop them or retry into a different quality sample.
            response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        finally:
            end = time.perf_counter() - started
            with lock:
                active -= 1
        return {
            "id": case["id"],
            "suite": case["suite"],
            "payload": payload,
            "client_start_seconds": start,
            "client_end_seconds": end,
            "response": response,
            "errors": score_case(case, response),
        }

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(run, enumerate(cases)))
    suites = {}
    for row in results:
        counts = suites.setdefault(row["suite"], {"correct": 0, "total": 0})
        counts["total"] += 1
        counts["correct"] += not row["errors"]
    return {
        "requested_client_concurrency": concurrency,
        "peak_inflight_client_requests": peak,
        "elapsed_seconds": time.perf_counter() - started,
        "note": "Fixed dataset subset, not official leaderboard or speed results.",
        "suites": suites,
        "cases": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--bfcl-dir", type=Path, required=True)
    parser.add_argument("--schema-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--per-category", type=int, default=16)
    parser.add_argument("--schema-limit", type=int, default=16)
    parser.add_argument("--max-schema-bytes", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if min(args.concurrency, args.per_category, args.schema_limit, args.max_tokens) < 1:
        parser.error("Concurrency, counts and max-tokens must be positive")
    cases, sources = load_cases(
        args.bfcl_dir,
        args.schema_dir,
        args.per_category,
        args.schema_limit,
        args.max_schema_bytes,
    )
    common = {
        "model": args.model,
        "stream": True,
        "return_token_ids": True,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "chat_template_kwargs": {"enable_thinking": args.enable_thinking},
    }
    result = (
        {"dry_run": True, "selected_cases": cases}
        if args.dry_run
        else run_cases(cases, args.base_url, common, args.concurrency)
    )
    result.update(sources=sources, common_payload=common)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result.get("suites", {"selected": len(cases)})))


if __name__ == "__main__":
    main()
