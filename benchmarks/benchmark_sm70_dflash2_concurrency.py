# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure concurrent DFlash2 requests during their shared steady decode window."""

import argparse
import copy
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from benchmark_sm70_dflash2_c1 import run


def measure(args, prompts: list[str], length: int) -> dict:
    barrier = threading.Barrier(args.concurrency)

    def request(index: int) -> dict:
        request_args = copy.copy(args)
        request_args.input_len = length
        barrier.wait()
        return run(request_args, prompts[index % len(prompts)], 0.7)

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        requests = list(executor.map(request, range(args.concurrency)))
    arrivals = [
        [
            result["request_start_monotonic_s"] + step["arrival_ms"] / 1000
            for step in result["steps"]
        ]
        for result in requests
    ]
    begin = max(times[20] for times in arrivals)
    end = min(times[-1] for times in arrivals)
    if end <= begin:
        raise RuntimeError("Requests have no shared steady decode window")
    tokens = sum(
        result["steps"][index]["token_count"]
        for result, times in zip(requests, arrivals)
        for index in range(21, len(times))
        if begin <= times[index - 1] and times[index] <= end
    )
    return {
        "input_fixture_length": length,
        "concurrency": args.concurrency,
        "shared_steady_decode_seconds": end - begin,
        "tokens_in_complete_shared_intervals": tokens,
        "shared_steady_tokens_per_second": tokens / (end - begin),
        "requests": requests,
        "contract": (
            "Temperature 0.7, top-p 0.9, thinking off, retained speed fixture. "
            "Exclude each request's first 20 events, prefill, and partial edge "
            "intervals; stop the shared window when the first request ends."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--input-lengths", type=int, nargs="+", default=[1024, 8192])
    parser.add_argument("--max-tokens", type=int, default=600)
    args = parser.parse_args()
    prompts = json.loads(args.inputs.read_text())
    results = []
    for length in args.input_lengths:
        results.append(measure(args, prompts[str(length)], length))
        args.output.write_text(json.dumps(results, indent=2))
    print(
        json.dumps(
            [
                {key: value for key, value in result.items() if key != "requests"}
                for result in results
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
