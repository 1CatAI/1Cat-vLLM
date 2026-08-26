# SPDX-License-Identifier: Apache-2.0

"""Measure one streamed completions request with separate TTFT and TPOT."""

import argparse
import json
import statistics
import time
import urllib.request
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:18082")
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=4201)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(round((len(ordered) - 1) * fraction), len(ordered) - 1)
    return ordered[index]


def main() -> None:
    args = parse_args()
    payload = {
        "model": "deepseek-v4-flash",
        "prompt": args.prompt_file.read_text(),
        "max_tokens": args.max_tokens,
        "temperature": 1.0,
        "top_p": 1.0,
        "seed": args.seed,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        f"{args.url}/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )

    start = time.perf_counter()
    chunk_times: list[float] = []
    text_parts: list[str] = []
    token_ids: list[int] = []
    usage = None
    finish_reason = None
    chunks = 0
    with urllib.request.urlopen(request, timeout=1800) as response:
        for raw_line in response:
            now = time.perf_counter()
            line = raw_line.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line.removeprefix("data:").strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("usage") is not None:
                usage = chunk["usage"]
            choices = chunk.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            finish_reason = choice.get("finish_reason") or finish_reason
            text = choice.get("text") or ""
            ids = choice.get("token_ids") or []
            if text or ids:
                chunks += 1
                text_parts.append(text)
                token_ids.extend(ids)
                chunk_times.append(now)
    end = time.perf_counter()

    if not chunk_times:
        raise RuntimeError("stream produced no output chunks")
    completion_tokens = (
        int(usage["completion_tokens"]) if usage is not None else len(token_ids)
    )
    ttft_ms = (chunk_times[0] - start) * 1000.0
    decode_ms = (end - chunk_times[0]) * 1000.0
    chunk_intervals_ms = [
        (right - left) * 1000.0 for left, right in zip(chunk_times, chunk_times[1:])
    ]
    decode_tps = (
        (completion_tokens - 1) / (decode_ms / 1000.0)
        if completion_tokens > 1 and decode_ms > 0.0
        else 0.0
    )
    result = {
        "contract": {
            "max_output_tokens": args.max_tokens,
            "temperature": 1.0,
            "top_p": 1.0,
            "seed": args.seed,
        },
        "usage": usage,
        "stream_chunks": chunks,
        "token_ids_complete": len(token_ids) == completion_tokens,
        "finish_reason": finish_reason,
        "generation_wall_ms": (end - start) * 1000.0,
        "ttft_ms": ttft_ms,
        "decode_ms": decode_ms,
        "decode_tps": decode_tps,
        # SSE chunks may contain multiple tokens. These are transport chunk
        # intervals, not per-token latency measurements.
        "stream_chunk_interval_mean_ms": (
            statistics.fmean(chunk_intervals_ms) if chunk_intervals_ms else 0.0
        ),
        "stream_chunk_interval_p50_ms": percentile(chunk_intervals_ms, 0.50),
        "stream_chunk_interval_p90_ms": percentile(chunk_intervals_ms, 0.90),
        "stream_chunk_interval_p99_ms": percentile(chunk_intervals_ms, 0.99),
        "output": "".join(text_parts),
        "token_ids": token_ids,
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
