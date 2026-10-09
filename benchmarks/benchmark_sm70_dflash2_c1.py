# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Record streamed DFlash2 rounds, then capture a short steady Torch profile."""

import argparse
import json
import statistics
import threading
import time
from pathlib import Path

import httpx


def run(args, text: str, temperature: float, profile: bool = False) -> dict:
    body = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": text},
        ],
        "temperature": temperature,
        "top_p": 0.9,
        "max_tokens": args.max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "return_token_ids": True,
        # Matches the retained speed fixture; quality runs must use natural EOS.
        "ignore_eos": True,
        "chat_template_kwargs": {"enable_thinking": False},
        "cache_salt": f"c1-{time.time_ns()}",
    }
    seed = getattr(args, "seed", None)
    if seed is not None:
        body["seed"] = seed
    events = []
    usage = None
    profile_thread = None
    profile_errors = []
    started = time.monotonic()

    def start_profile() -> None:
        try:
            response = httpx.post(
                args.base_url + "/start_profile", timeout=120, trust_env=False
            )
            response.raise_for_status()
        except Exception as error:
            profile_errors.append(str(error))

    with (
        httpx.Client(timeout=3600, trust_env=False) as client,
        client.stream(
            "POST", args.base_url + "/v1/chat/completions", json=body
        ) as response,
    ):
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith("data: ") or line[6:] == "[DONE]":
                continue
            chunk = json.loads(line[6:])
            if chunk.get("error"):
                raise RuntimeError(chunk["error"])
            usage = chunk.get("usage") or usage
            for choice in chunk.get("choices", []):
                ids = choice.get("token_ids") or []
                if ids:
                    events.append((time.monotonic(), ids))
                    if profile and profile_thread is None and len(events) >= 60:
                        profile_thread = threading.Thread(target=start_profile)
                        profile_thread.start()
    if profile_thread is not None:
        profile_thread.join(timeout=125)
        if profile_thread.is_alive() or profile_errors:
            raise RuntimeError(f"Profile start failed: {profile_errors}")
    elif profile:
        raise RuntimeError("Output ended before profiling could start")

    steady = events[20:]
    if len(steady) < 2:
        raise RuntimeError("Insufficient streamed rounds after warmup")
    gaps = [(right[0] - left[0]) * 1000 for left, right in zip(steady, steady[1:])]
    counts = [len(ids) for _, ids in steady[1:]]
    ordered = sorted(gaps)
    return {
        "temp": temperature,
        "seed": seed,
        "profiled_request": profile,
        "input_fixture_length": args.input_len,
        "request_start_monotonic_s": started,
        "usage": usage,
        "ttft_ms": (events[0][0] - started) * 1000,
        "tok_s": sum(counts) / (steady[-1][0] - steady[0][0]),
        "tokens_per_step": statistics.mean(counts),
        "step_ms_median": statistics.median(gaps),
        "step_ms_mean": statistics.mean(gaps),
        "step_ms_p90": ordered[int(0.9 * (len(ordered) - 1))],
        "step_ms_p99": ordered[int(0.99 * (len(ordered) - 1))],
        "steps": [
            {
                "step": index,
                "arrival_ms": (arrival - started) * 1000,
                "interval_ms": (
                    None if index == 0 else (arrival - events[index - 1][0]) * 1000
                ),
                "token_count": len(ids),
                "token_ids": ids,
                "included_in_steady_stats": index > 20,
            }
            for index, (arrival, ids) in enumerate(events)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-len", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=600)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--seed", type=int, help="Fix request RNG for paired comparisons"
    )
    parser.add_argument("--skip-greedy", action="store_true")
    parser.add_argument("--skip-profile", action="store_true")
    args = parser.parse_args()
    prompts = json.loads(args.inputs.read_text())[str(args.input_len)]
    run(args, prompts[0], 0.7)
    results = []
    for index in range(args.repeats):
        results.append(run(args, prompts[index % len(prompts)], 0.7))
        if not args.skip_greedy:
            results.append(run(args, prompts[index % len(prompts)], 0.0))
    # Save unprofiled evidence before any profiling failure.
    args.output.write_text(json.dumps(results, indent=2))
    if not args.skip_profile:
        results.append(run(args, prompts[0], 0.7, profile=True))
        time.sleep(2)
        response = httpx.post(
            args.base_url + "/stop_profile", timeout=900, trust_env=False
        )
        response.raise_for_status()
        args.output.write_text(json.dumps(results, indent=2))
    print(
        json.dumps(
            [
                {key: value for key, value in result.items() if key != "steps"}
                for result in results
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
