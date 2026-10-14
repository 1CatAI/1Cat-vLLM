# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Matched concurrent MTP requests, with an explicit common decode interval.

Token timestamps are HTTP arrival times, not GPU verifier timers. Warm the same
prompts first; use the worker step profiler separately for phase attribution.
"""

import argparse
import hashlib
import json
import os
import statistics
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import regex as re


def request(base, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"}
    if key := os.environ.get("VLLM_API_KEY"):
        headers["Authorization"] = "Bearer " + key
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return opener.open(urllib.request.Request(base + path, data, headers), timeout=900)


def counters(base):
    with request(base, "/metrics") as response:
        text = response.read().decode()
    return {
        name: sum(
            float(value)
            for value in re.findall(
                rf"^vllm:spec_decode_num_{name}_total\{{[^}}]*\}} ([0-9.e+]+)$",
                text,
                re.M,
            )
        )
        for name in ("drafts", "draft_tokens", "accepted_tokens")
    }


def run_one(base, case, output_tokens, barrier, seed):
    body = {
        "model": "flash-next-abliterated",
        "messages": [{"role": "user", "content": case["prompt"]}],
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "seed": seed,
        "max_tokens": output_tokens,
        "chat_template_kwargs": {"enable_thinking": True},
        "stream": True,
        "stream_options": {"include_usage": True},
        "return_token_ids": True,
    }
    barrier.wait()
    started = time.perf_counter()
    events = []
    content = []
    reasoning = []
    usage = None
    finish = None
    done = False
    with request(base, "/v1/chat/completions", body) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            if line == "data: [DONE]":
                done = True
                continue
            obj = json.loads(line[5:])
            if obj.get("usage"):
                usage = obj["usage"]
            for choice in obj.get("choices", []):
                finish = choice.get("finish_reason") or finish
                delta = choice.get("delta", {})
                ids = choice.get("token_ids") or delta.get("token_ids") or []
                if ids:
                    events.append({"at": time.perf_counter(), "token_ids": ids})
                content.append(delta.get("content") or "")
                reasoning.append(delta.get("reasoning") or "")
    ended = time.perf_counter()
    if not done or usage is None or not events:
        raise RuntimeError("Incomplete stream or missing emitted token IDs")
    emitted = sum(len(event["token_ids"]) for event in events)
    if emitted != usage["completion_tokens"]:
        raise RuntimeError(f"Token accounting differs: {emitted} versus {usage}")
    first, last = events[0]["at"], events[-1]["at"]
    decode_tokens = emitted - len(events[0]["token_ids"])
    return {
        "seed": seed,
        "prompt_sha256": hashlib.sha256(case["prompt"].encode()).hexdigest(),
        "usage": usage,
        "finish_reason": finish,
        "ttft_s": first - started,
        "total_s": ended - started,
        "decode_s": last - first,
        "decode_tokens": decode_tokens,
        "output_token_ms": 1000 * (last - first) / max(decode_tokens, 1),
        "events": events,
        "content": "".join(content),
        "reasoning": "".join(reasoning),
    }


def run_batch(base, case, concurrency, output_tokens, repetition, independent_prefix):
    before = counters(base)
    barrier = threading.Barrier(concurrency)
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = []
        for i in range(concurrency):
            worker_case = dict(case)
            if independent_prefix:
                worker_case["prompt"] = f"[Independent request {i}]\n" + case["prompt"]
            futures.append(
                pool.submit(
                    run_one, base, worker_case, output_tokens, barrier, 20261014 + i
                )
            )
        rows = [future.result() for future in futures]
    time.sleep(0.2)  # Let the final asynchronous scheduler counters settle.
    after = counters(base)
    # Exclude every request's first and last emission. All requests must still
    # be decoding in the shared interval; this avoids calling tail drain C8.
    common_start = max(row["events"][0]["at"] for row in rows)
    common_end = min(row["events"][-1]["at"] for row in rows)
    seconds = common_end - common_start
    common_tokens = sum(
        len(event["token_ids"])
        for row in rows
        for event in row["events"]
        if common_start < event["at"] < common_end
    )
    valid_overlap = seconds > 0 and common_tokens >= concurrency * 16
    if repetition >= 0 and not valid_overlap:
        raise RuntimeError(
            "Insufficient common decode interval; increase output length"
        )
    delta = {key: after[key] - before[key] for key in before}
    return {
        "concurrency": concurrency,
        "prefix_mode": "independent" if independent_prefix else "shared",
        "repetition": repetition,
        "batch_elapsed_s": time.perf_counter() - start,
        "mean_ttft_s": statistics.mean(row["ttft_s"] for row in rows),
        "mean_output_token_ms": statistics.mean(row["output_token_ms"] for row in rows),
        "common_decode_s": seconds if valid_overlap else None,
        "common_decode_tokens": common_tokens,
        "common_aggregate_tokens_per_s": common_tokens / seconds
        if valid_overlap
        else None,
        "common_per_request_token_ms": (
            1000 * seconds * concurrency / common_tokens if valid_overlap else None
        ),
        "accepted_length": 1 + delta["accepted_tokens"] / max(delta["drafts"], 1),
        "mtp_metrics_delta": delta,
        "requests": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18041")
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--case", default="i32768")
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 2, 4, 6, 8])
    parser.add_argument("--output-tokens", type=int, default=384)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--independent-prefix", action="store_true")
    parser.add_argument("--service-log", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    case = next(
        row for row in json.loads(args.cases.read_text()) if row["name"] == args.case
    )
    results = []
    for concurrency in args.concurrency:
        for repetition in range(-args.warmups, args.repetitions):
            log_start = args.service_log.stat().st_size if args.service_log else None
            row = run_batch(
                args.base_url,
                case,
                concurrency,
                args.output_tokens,
                repetition,
                args.independent_prefix,
            )
            results.append(row)
            if args.service_log:
                row.update(
                    profile_byte_start=log_start,
                    profile_byte_end=args.service_log.stat().st_size,
                )
            args.out.write_text(json.dumps(results, indent=2))
            print(
                json.dumps({k: v for k, v in row.items() if k != "requests"}),
                flush=True,
            )


if __name__ == "__main__":
    main()
