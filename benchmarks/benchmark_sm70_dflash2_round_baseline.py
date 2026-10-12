# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Retained C1 streaming-round protocol using only normal serving endpoints.

This timing fixture ignores EOS and is not a natural-output quality gate.
Stream intervals, engine decode/round metrics and emitted-token cost remain
separate. No target-event or profiled service time is reported as a full round.
"""

import argparse
import hashlib
import json
import statistics
import time
import urllib.request
from pathlib import Path

SAMPLING = {"temperature": 0.7, "top_p": 0.9, "top_k": 20, "seed": 123}


def steady_rounds(events: list[tuple[float, list[int]]], warmup: int = 20) -> dict:
    steady = events[warmup:]
    if len(steady) < 2:
        raise ValueError("Insufficient complete stream intervals after warmup")
    gaps = [1000 * (b[0] - a[0]) for a, b in zip(steady, steady[1:])]
    if any(gap <= 0 for gap in gaps):
        raise ValueError("Stream arrivals must be strictly ordered")
    counts = [len(ids) for _, ids in steady[1:]]
    if not all(counts):
        raise ValueError("Empty metadata chunks are not speculative rounds")
    ordered = sorted(gaps)
    return {
        "stream_complete_round_mean_ms": statistics.mean(gaps),
        "stream_complete_round_p50_ms": statistics.median(gaps),
        "stream_complete_round_p90_ms": ordered[int(0.9 * (len(ordered) - 1))],
        "stream_complete_round_p99_ms": ordered[int(0.99 * (len(ordered) - 1))],
        "steady_emitted_tokens_per_round": statistics.mean(counts),
        "steady_emitted_tokens_per_second": sum(counts)
        / (steady[-1][0] - steady[0][0]),
        "steady_rounds": len(gaps),
    }


def engine_cost(delta: dict, output_tokens: int) -> dict:
    rounds = delta.get("vllm:spec_decode_num_drafts_total", 0)
    accepted = delta.get("vllm:spec_decode_num_accepted_tokens_total", 0)
    proposed = delta.get("vllm:spec_decode_num_draft_tokens_total", 0)
    decode = delta.get("vllm:request_decode_time_seconds_sum", 0)
    if rounds and decode <= 0:
        raise ValueError("Missing or incomplete engine decode timing")
    if accepted > proposed:
        raise ValueError("Accepted drafts exceed proposed tokens")
    return {
        "engine_complete_round_mean_ms": 1000 * decode / rounds if rounds else None,
        "accepted_drafts_per_round": accepted / rounds if rounds else None,
        "accepted_over_proposed": accepted / proposed if proposed else None,
        "engine_decode_ms_per_emitted_token": 1000 * decode / (output_tokens - 1)
        if output_tokens > 1 and decode
        else None,
        "engine_prefill_s": delta.get("vllm:request_prefill_time_seconds_sum"),
        "engine_rounds": rounds,
        "metric_deltas": delta,
    }


def metrics(base: str) -> dict:
    from prometheus_client.parser import text_string_to_metric_families

    with urllib.request.urlopen(base + "/metrics", timeout=30) as response:
        result = {}
        for family in text_string_to_metric_families(response.read().decode()):
            for sample in family.samples:
                if "spec_decode" in sample.name or sample.name.endswith(
                    ("_time_seconds_sum", "_latency_seconds_sum")
                ):
                    key = sample.name
                    if "position" in sample.labels:
                        key += ":position=" + sample.labels["position"]
                    result[key] = result.get(key, 0.0) + sample.value
        return result


def request(base: str, model: str, prompt: str, output_tokens: int) -> dict:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        **SAMPLING,
        "max_tokens": output_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "return_token_ids": True,
        "ignore_eos": True,
        "chat_template_kwargs": {"enable_thinking": False},
        "cache_salt": f"round-baseline-{time.time_ns()}",
    }
    before = metrics(base)
    events, usage = [], None
    started = time.monotonic()
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as response:
        for raw in response:
            at = time.monotonic()
            line = raw.decode().strip()
            if not line.startswith("data: ") or line[6:] == "[DONE]":
                continue
            chunk = json.loads(line[6:])
            if chunk.get("error"):
                raise RuntimeError(chunk["error"])
            usage = chunk.get("usage") or usage
            for choice in chunk.get("choices", []):
                ids = choice.get("token_ids") or []
                if ids:
                    events.append((at, ids))
    after = metrics(base)
    emitted = sum(len(ids) for _, ids in events)
    if not usage or emitted != output_tokens or usage["completion_tokens"] != emitted:
        raise ValueError("Incomplete timing fixture or mismatched stream token counts")
    delta = {key: after.get(key, 0) - before.get(key, 0) for key in before | after}
    if any(value < 0 for value in delta.values()):
        raise ValueError("Engine counters reset during the request")
    return {
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "usage": usage,
        "ttft_ms": 1000 * (events[0][0] - started),
        **steady_rounds(events),
        **engine_cost(delta, emitted),
        "stream_steps": [
            {"arrival_ms": 1000 * (at - started), "token_ids": ids}
            for at, ids in events
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--fixtures", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--lengths", nargs="+", type=int, default=[1024, 8192])
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; preserve previous evidence")
    raw = args.fixtures.read_bytes()
    fixtures = json.loads(raw)
    report = {
        "scope": "Unprofiled C1 timing fixture; no output-quality admission",
        "sampling": SAMPLING,
        "output_tokens": 600,
        "warmup_stream_events": 20,
        "repeats_per_length": 8,
        "fixture_sha256": hashlib.sha256(raw).hexdigest(),
        "profiler": False,
        "cases": [],
        "complete": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    save()
    for length in args.lengths:
        prompts = fixtures[str(length)]
        if len(prompts) != 8 or not all(isinstance(p, str) and p for p in prompts):
            raise ValueError(
                "C1 protocol requires eight frozen text prompts per length"
            )
        request(args.base_url, args.model, prompts[0], 600)
        for index, prompt in enumerate(prompts):
            row = request(args.base_url, args.model, prompt, 600)
            row.update(input_fixture_length=length, repeat=index)
            report["cases"].append(row)
            save()
    report["complete"] = True
    save()


if __name__ == "__main__":
    main()
