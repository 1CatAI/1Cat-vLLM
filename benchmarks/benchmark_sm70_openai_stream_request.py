# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Issue one timed OpenAI completions request for SM70 Nsight captures."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path


def _post(url: str, payload: dict[str, object] | None = None):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return urllib.request.urlopen(request, timeout=900)


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18080")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=4111)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--profile", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.profile:
        with _post(f"{args.base_url}/start_profile") as response:
            response.read()

    payload = {
        "model": args.model,
        "prompt": args.prompt_file.read_text(),
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    started = time.perf_counter()
    token_times: list[float] = []
    output_parts: list[str] = []
    usage = None
    finish_reason = None
    with _post(f"{args.base_url}/v1/completions", payload) as response:
        for raw_line in response:
            line = raw_line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                text = choice.get("text", "")
                if text:
                    token_times.append(time.perf_counter())
                    output_parts.append(text)
                if choice.get("finish_reason") is not None:
                    finish_reason = choice["finish_reason"]
    finished = time.perf_counter()

    intervals_ms = [
        (right - left) * 1000
        for left, right in zip(token_times, token_times[1:], strict=False)
    ]
    result = {
        "contract": {
            "max_output_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
        },
        "usage": usage,
        "stream_chunks": len(token_times),
        "finish_reason": finish_reason,
        "generation_wall_ms": (finished - started) * 1000,
        "ttft_ms": (token_times[0] - started) * 1000 if token_times else None,
        "trace_tpot_mean_ms": (statistics.mean(intervals_ms) if intervals_ms else None),
        "trace_tpot_p50_ms": _percentile(intervals_ms, 0.50),
        "trace_tpot_p90_ms": _percentile(intervals_ms, 0.90),
        "trace_tpot_p99_ms": _percentile(intervals_ms, 0.99),
        "output": "".join(output_parts),
        "profile_stop_result": None,
    }
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2))

    if args.profile:
        try:
            with _post(f"{args.base_url}/stop_profile") as response:
                result["profile_stop_result"] = response.read().decode()
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            result["profile_stop_result"] = f"{type(exc).__name__}: {exc}"
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
