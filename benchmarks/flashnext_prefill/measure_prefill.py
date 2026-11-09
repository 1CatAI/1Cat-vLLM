# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Matched, cold-prefix prefill requests against an exclusively owned server.

Use the same --seed/--words/--warmup/--trials for control and candidate. Word
count is not token count: throughput uses the server's reported prompt_tokens.
--profile wraps only measured requests in the server's profiler window; do not
use profiled timing as an unprofiled end-to-end result.
"""

import argparse
import hashlib
import json
import random
import statistics
import time
import urllib.request
from pathlib import Path

WORDS = "alpha river stone cloud engine paper signal forest copper lantern orbit meadow"
METRICS = (
    "vllm:request_prefill_time_seconds",
    "vllm:time_to_first_token_seconds",
    "vllm:request_queue_time_seconds",
)


def request(base, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        base + path, data, {"Content-Type": "application/json"}
    )
    return urllib.request.urlopen(req, timeout=1800)


def metrics(base):
    with request(base, "/metrics") as response:
        lines = response.read().decode().splitlines()
    result = {}
    for line in lines:
        if not line or line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name in {m + suffix for m in METRICS for suffix in ("_sum", "_count")}:
            result[name] = result.get(name, 0) + float(line.rsplit(" ", 1)[1])
    return result


def metric_deltas(base, before):
    deadline = time.monotonic() + 8
    while True:
        after = metrics(base)
        counts = {
            m: after.get(m + "_count", 0) - before.get(m + "_count", 0) for m in METRICS
        }
        if all(n >= 1 for n in counts.values()) or time.monotonic() >= deadline:
            break
        time.sleep(0.2)
    # More than one request means this measurement cannot be attributed.
    return {
        m.removeprefix("vllm:"): {
            "count": counts[m],
            "seconds": (
                after.get(m + "_sum", 0) - before.get(m + "_sum", 0)
                if counts[m] == 1
                else None
            ),
        }
        for m in METRICS
    }


def run_one(args, index):
    rng = random.Random(args.seed + index)
    # An early request-specific nonce also avoids sharing warmup prefixes.
    prompt = f"[{rng.randrange(10**9)}] " + " ".join(
        rng.choice(WORDS.split()) for _ in range(args.words)
    )
    body = {
        "model": args.model,
        "prompt": prompt,
        "temperature": 0,
        "max_tokens": 1,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    # This endpoint is intended only for a server owned by this benchmark.
    with request(args.base_url, "/reset_prefix_cache", {}) as response:
        response.read()
    before = metrics(args.base_url)
    started_unix = time.time()
    start = time.perf_counter()
    first_choice = first_text = None
    usage, choices = None, []
    with request(args.base_url, "/v1/completions", body) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            data = line[6:].strip()
            if data == b"[DONE]":
                break
            event = json.loads(data)
            if "error" in event:
                raise RuntimeError(event["error"])
            if event.get("choices"):
                choices.extend(event["choices"])
                elapsed = time.perf_counter() - start
                if first_choice is None:
                    first_choice = elapsed
                if first_text is None and any(c.get("text") for c in event["choices"]):
                    first_text = elapsed
            if event.get("usage"):
                usage = event["usage"]
    total = time.perf_counter() - start
    if first_choice is None or usage is None:
        raise RuntimeError("Missing completion choice or final usage")
    return {
        "index": index,
        "client_started_unix": started_unix,
        "client_finished_unix": time.time(),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "usage": usage,
        "first_choice_seconds": first_choice,
        "first_text_seconds": first_text,
        "response_seconds": total,
        "prompt_tokens_per_first_choice_second": usage["prompt_tokens"] / first_choice,
        "choices": choices,
        "server_metrics": metric_deltas(args.base_url, before),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8002")
    parser.add_argument("--model", default="flash-next")
    parser.add_argument("--words", type=int, default=32000)
    parser.add_argument("--seed", type=int, default=9683)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.words < 1 or args.warmup < 0 or args.trials < 1:
        parser.error("words/trials must be positive and warmup nonnegative")
    report = {"config": vars(args) | {"output": str(args.output)}, "requests": []}
    for i in range(args.warmup + args.trials):
        profiling = args.profile and i >= args.warmup
        if profiling:
            with request(args.base_url, "/start_profile", {}) as response:
                response.read()
        try:
            result = run_one(args, i)
        finally:
            if profiling:
                with request(args.base_url, "/stop_profile", {}) as response:
                    response.read()
        result["warmup"] = i < args.warmup
        report["requests"].append(result)
        print(json.dumps(result), flush=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    measured = [r for r in report["requests"] if not r["warmup"]]
    report["summary"] = {
        "profiled": args.profile,
        "median_first_choice_seconds": statistics.median(
            r["first_choice_seconds"] for r in measured
        ),
        "median_prompt_tokens_per_first_choice_second": statistics.median(
            r["prompt_tokens_per_first_choice_second"] for r in measured
        ),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"]), flush=True)


if __name__ == "__main__":
    main()
