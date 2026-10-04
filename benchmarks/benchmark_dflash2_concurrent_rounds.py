# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure streaming cadence in the common decode window of a request cohort.

CUDA worker timing is the authoritative complete-round measurement. Streaming
cadence is reported separately: frontend buffering can combine or delay chunks.
The fixed-length speed fixture deliberately ignores EOS; use natural completions
separately for quality checks.
"""

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import httpx


def common_window(requests: list[dict], trim: int) -> dict:
    if any(len(request["events"]) <= 2 * trim + 1 for request in requests):
        raise ValueError("Not enough decode events after trimming both ends")
    begin = max(request["events"][trim][0] for request in requests)
    end = min(request["events"][-trim - 1][0] for request in requests)
    if end <= begin:
        raise ValueError("Requests have no common steady decode window")
    results = []
    for request in requests:
        intervals = [
            (right[0] - left[0], len(right[1]))
            for left, right in zip(request["events"], request["events"][1:])
            if left[0] >= begin and right[0] <= end
        ]
        if not intervals:
            raise ValueError("No complete interval inside the common window")
        seconds = sum(gap for gap, _ in intervals)
        results.append(
            {
                "request_index": request["request_index"],
                "intervals": len(intervals),
                "stream_interval_ms_mean": seconds * 1000 / len(intervals),
                "stream_interval_ms_median": statistics.median(
                    gap * 1000 for gap, _ in intervals
                ),
                "tokens_per_stream_chunk": statistics.mean(
                    count for _, count in intervals
                ),
                "decode_tokens_per_second": sum(n for _, n in intervals) / seconds,
                "ttft_ms": request["ttft_ms"],
                "usage": request["usage"],
            }
        )
    return {"begin_wall_s": begin, "end_wall_s": end, "requests": results}


async def request(client, args, prompt, index, barrier):
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "seed": args.seed + index,
        "max_tokens": args.output_tokens,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "return_token_ids": True,
        "chat_template_kwargs": {"enable_thinking": False},
        "cache_salt": f"concurrent-rounds-{time.time_ns()}-{index}",
    }
    await barrier.wait()
    start = time.perf_counter()
    events = []
    usage = None
    ttft_ms = None
    async with client.stream(
        "POST", args.base_url + "/v1/chat/completions", json=body
    ) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data: ") or line[6:] == "[DONE]":
                continue
            chunk = json.loads(line[6:])
            if chunk.get("error"):
                raise RuntimeError(chunk["error"])
            usage = chunk.get("usage") or usage
            for choice in chunk.get("choices", []):
                token_ids = choice.get("token_ids") or []
                if token_ids:
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - start) * 1000
                    events.append((time.time(), token_ids))
    emitted = sum(len(ids) for _, ids in events)
    if emitted != args.output_tokens:
        raise ValueError(f"Request {index}: emitted {emitted}, expected fixed length")
    if not usage or usage["completion_tokens"] != emitted:
        raise ValueError(f"Request {index}: missing or inconsistent usage")
    return {
        "request_index": index,
        "events": events,
        "usage": usage,
        "ttft_ms": ttft_ms,
        "total_wall_s": time.perf_counter() - start,
    }


async def run(args):
    fixture = json.loads(args.inputs.read_text())
    prompts = fixture[str(args.input_tokens)]
    if len(prompts) < args.concurrency:
        raise ValueError("Fixture must contain a distinct prompt per request")
    barrier = asyncio.Barrier(args.concurrency)
    async with httpx.AsyncClient(timeout=3600, trust_env=False) as client:
        requests = await asyncio.gather(
            *(
                request(client, args, prompts[index], index, barrier)
                for index in range(args.concurrency)
            )
        )
    result = {
        "contract": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "common_decode_window": common_window(requests, args.trim_chunks),
        "raw_requests": requests,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["common_decode_window"], indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--input-tokens", type=int, choices=(8192, 32768), required=True
    )
    parser.add_argument("--concurrency", type=int, choices=(1, 2, 4, 8), required=True)
    parser.add_argument("--output-tokens", type=int, default=600)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trim-chunks", type=int, default=20)
    args = parser.parse_args()
    if args.trim_chunks < 1:
        parser.error("--trim-chunks must be positive")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
