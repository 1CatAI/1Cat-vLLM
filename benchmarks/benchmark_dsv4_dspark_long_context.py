# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure DSpark verification and output health across long contexts.

The benchmark targets an already-running OpenAI-compatible server. It records
stream timestamps together with vLLM speculative metrics so decode speed,
verification-round cost, and draft acceptance can be compared per context
bucket instead of being conflated with prefill latency.
"""

from __future__ import annotations

import argparse
import http.client
import json
import statistics
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import regex as re
from transformers import AutoTokenizer

_PROFILE_INTERVAL_MARKER = "SM70 spec runner profile interval_avg_ms "
_PROFILE_FIELD_PATTERN = re.compile(r"([a-z_]+)=([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")


def _parse_lengths(raw: str) -> list[int]:
    values = sorted({int(value.strip()) for value in raw.split(",") if value.strip()})
    if not values or values[0] <= 0:
        raise ValueError("--lengths must contain positive integers")
    return values


def _template_token_ids(tokenizer: Any, messages: list[dict[str, str]]) -> list[int]:
    try:
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except (TypeError, ValueError):
        # DeepSeek V4's custom tokenizer mode supplies the template in the
        # server rather than tokenizer_config.json. The constant allowance is
        # only for fitting below the target; server-reported usage remains the
        # authoritative prompt length in every result row.
        encoded = [0] * 32
        for message in messages:
            encoded.extend(
                tokenizer.encode(message["content"], add_special_tokens=False)
            )
    if isinstance(encoded, dict):
        encoded = encoded["input_ids"]
    elif hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return list(encoded)


def _filler_unit(index: int) -> str:
    return (
        f"Record {index:06d}: unrelated inventory values "
        f"{index * 17 % 9973}, {index * 31 % 7919}, {index * 43 % 6151}. "
        "This record is only deterministic long-context filler and contains "
        "no answer to the final instruction.\n"
    )


def _build_messages(unit_count: int, marker: str) -> list[dict[str, str]]:
    filler = "".join(_filler_unit(index) for index in range(unit_count))
    instruction = (
        "\nFINAL INSTRUCTION: Ignore the inventory records. Begin the response "
        f"with the exact marker {marker} on its own line, then write a concise "
        "technical paragraph explaining why bounded CUDA-graph context buckets "
        "reduce long-context decode latency. Do not repeat the marker."
    )
    return [
        {
            "role": "system",
            "content": (
                f"This request is isolated by {marker}. Follow the final user "
                "instruction exactly. Long repeated context must not cause "
                "repetition, garbling, or loss of the suffix."
            ),
        },
        {"role": "user", "content": filler + instruction},
    ]


def _fit_messages(
    tokenizer: Any,
    target_tokens: int,
    marker: str,
) -> tuple[list[dict[str, str]], int, int]:
    sample_count = 64
    sample_tokens = len(
        _template_token_ids(tokenizer, _build_messages(sample_count, marker))
    )
    base_tokens = len(_template_token_ids(tokenizer, _build_messages(0, marker)))
    tokens_per_unit = max(1.0, (sample_tokens - base_tokens) / sample_count)
    low = 0
    high = max(1, int(target_tokens / tokens_per_unit * 1.2) + 32)
    best_messages = _build_messages(0, marker)
    best_tokens = len(_template_token_ids(tokenizer, best_messages))
    best_units = 0
    for _ in range(24):
        if low > high:
            break
        middle = (low + high) // 2
        messages = _build_messages(middle, marker)
        prompt_tokens = len(_template_token_ids(tokenizer, messages))
        if prompt_tokens <= target_tokens:
            best_messages = messages
            best_tokens = prompt_tokens
            best_units = middle
            low = middle + 1
        else:
            high = middle - 1
    return best_messages, best_tokens, best_units


def _metric_total(text: str, name: str) -> float:
    return sum(
        float(line.rsplit(" ", 1)[-1])
        for line in text.splitlines()
        if line.startswith(name + "{") or line.startswith(name + " ")
    )


def _metrics_snapshot(host: str, port: int, timeout: int) -> dict[str, Any]:
    connection = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        connection.request("GET", "/metrics")
        response = connection.getresponse()
        raw = response.read()
    finally:
        connection.close()
    if response.status != 200:
        raise RuntimeError(f"metrics endpoint returned HTTP {response.status}")
    text = raw.decode("utf-8", errors="replace")
    positions: dict[int, float] = {}
    prefix = "vllm:spec_decode_num_accepted_tokens_per_pos_total{"
    for line in text.splitlines():
        if not line.startswith(prefix):
            continue
        match = re.search(r'position="(\d+)"', line)
        if match:
            position = int(match.group(1))
            positions[position] = positions.get(position, 0.0) + float(
                line.rsplit(" ", 1)[-1]
            )
    return {
        "rounds": _metric_total(text, "vllm:spec_decode_num_drafts_total"),
        "proposed": _metric_total(text, "vllm:spec_decode_num_draft_tokens_total"),
        "accepted": _metric_total(text, "vllm:spec_decode_num_accepted_tokens_total"),
        "prompt_tokens": _metric_total(text, "vllm:prompt_tokens_total"),
        "prefill_seconds": _metric_total(text, "vllm:request_prefill_time_seconds_sum"),
        "positions": positions,
    }


def _longest_same_token_run(token_ids: list[int]) -> int:
    longest = 0
    current = 0
    previous: int | None = None
    for token_id in token_ids:
        if token_id == previous:
            current += 1
        else:
            previous = token_id
            current = 1
        longest = max(longest, current)
    return longest


def _profile_log_offset(profile_log: Path | None) -> int | None:
    if profile_log is None:
        return None
    try:
        return profile_log.stat().st_size
    except FileNotFoundError:
        return 0


def _read_verifier_profile(
    profile_log: Path | None,
    offset: int | None,
    verifier_rows: int,
) -> dict[str, Any]:
    if profile_log is None or offset is None:
        return {"available": False, "reason": "profile log was not requested"}
    try:
        with profile_log.open("rb") as profile_file:
            profile_file.seek(offset)
            text = profile_file.read().decode("utf-8", errors="replace")
    except FileNotFoundError:
        return {"available": False, "reason": "profile log does not exist"}

    intervals: list[dict[str, Any]] = []
    for line in text.splitlines():
        if _PROFILE_INTERVAL_MARKER not in line:
            continue
        fields = {
            key: float(value)
            for key, value in _PROFILE_FIELD_PATTERN.findall(
                line.split(_PROFILE_INTERVAL_MARKER, 1)[1]
            )
        }
        if int(fields.get("num_tokens", -1)) != verifier_rows:
            continue
        required = ("target_forward", "target_logits", "target_rejection_sample")
        if any(name not in fields for name in required):
            continue
        interval = dict(fields)
        for name in (
            "calls",
            "interval_calls",
            "interval_spec_steps",
            "num_tokens",
            "num_reqs",
        ):
            if name in interval:
                interval[name] = int(interval[name])
        interval["complete_verifier_ms"] = sum(fields[name] for name in required)
        intervals.append(interval)

    # The first/last profiler window can straddle prefill or request teardown.
    # Long sweeps run with a short profile interval and retain only complete
    # interior windows when enough samples are available.
    if len(intervals) >= 4:
        steady_intervals = intervals[1:-1]
    elif len(intervals) == 3:
        steady_intervals = intervals[1:]
    else:
        steady_intervals = intervals
    median_names = (
        "target_forward",
        "target_logits",
        "target_rejection_sample",
        "complete_verifier_ms",
        "draft_total",
    )
    medians = {
        name: statistics.median(
            float(interval[name]) for interval in steady_intervals if name in interval
        )
        for name in median_names
        if any(name in interval for interval in steady_intervals)
    }
    return {
        "available": bool(intervals),
        "profile_log": str(profile_log),
        "verifier_rows": verifier_rows,
        "raw_interval_count": len(intervals),
        "steady_interval_count": len(steady_intervals),
        "medians_ms": medians,
        "intervals": intervals,
    }


def _stream_request(
    *,
    host: str,
    port: int,
    model: str,
    messages: list[dict[str, str]],
    output_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    timeout: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": output_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    connection = http.client.HTTPConnection(host, port, timeout=timeout)
    encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    started_ns = time.perf_counter_ns()
    first_token_ns: int | None = None
    last_token_ns: int | None = None
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    usage: dict[str, Any] = {}
    finish_reason: str | None = None
    try:
        connection.request(
            "POST",
            "/v1/chat/completions",
            body=encoded,
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        if response.status != 200:
            raw = response.read()
            raise RuntimeError(
                f"request returned HTTP {response.status}: "
                f"{raw.decode('utf-8', errors='replace')}"
            )
        for raw_line in response:
            arrival_ns = time.perf_counter_ns()
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            event = json.loads(data)
            if event.get("usage"):
                usage = event["usage"]
            choices = event.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            if choice.get("finish_reason") is not None:
                finish_reason = choice["finish_reason"]
            delta = choice.get("delta") or {}
            content = delta.get("content") or ""
            reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
            if content or reasoning:
                first_token_ns = first_token_ns or arrival_ns
                last_token_ns = arrival_ns
                if content:
                    content_parts.append(content)
                if reasoning:
                    reasoning_parts.append(reasoning)
    finally:
        finished_ns = time.perf_counter_ns()
        connection.close()

    completion_tokens = int(usage.get("completion_tokens") or 0)
    decode_seconds = (
        (last_token_ns - first_token_ns) / 1_000_000_000
        if first_token_ns is not None and last_token_ns is not None
        else None
    )
    return {
        "request": payload,
        "usage": usage,
        "finish_reason": finish_reason,
        "wall_seconds": (finished_ns - started_ns) / 1_000_000_000,
        "ttft_seconds": (
            (first_token_ns - started_ns) / 1_000_000_000
            if first_token_ns is not None
            else None
        ),
        "pure_decode_seconds": decode_seconds,
        "pure_decode_tokens_per_second": (
            (completion_tokens - 1) / decode_seconds
            if completion_tokens > 1 and decode_seconds
            else None
        ),
        "content": "".join(content_parts),
        "reasoning": "".join(reasoning_parts),
    }


def _run_case(
    *,
    tokenizer: Any,
    host: str,
    port: int,
    model: str,
    target_tokens: int,
    output_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    timeout: int,
    num_speculative_tokens: int,
    profile_log: Path | None,
) -> dict[str, Any]:
    marker = f"DSV4-CONTEXT-{target_tokens}-OK"
    messages, local_prompt_tokens, unit_count = _fit_messages(
        tokenizer, target_tokens, marker
    )
    profile_offset = _profile_log_offset(profile_log)
    before = _metrics_snapshot(host, port, timeout)
    result = _stream_request(
        host=host,
        port=port,
        model=model,
        messages=messages,
        output_tokens=output_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        timeout=timeout,
    )
    after = _metrics_snapshot(host, port, timeout)
    rounds = after["rounds"] - before["rounds"]
    proposed = after["proposed"] - before["proposed"]
    accepted = after["accepted"] - before["accepted"]
    computed_prompt_tokens = after["prompt_tokens"] - before["prompt_tokens"]
    prefill_seconds = after["prefill_seconds"] - before["prefill_seconds"]
    accepted_per_position = [
        after["positions"].get(position, 0.0) - before["positions"].get(position, 0.0)
        for position in range(num_speculative_tokens)
    ]
    verifier_profile = (
        _read_verifier_profile(
            profile_log,
            profile_offset,
            num_speculative_tokens + 1,
        )
        if num_speculative_tokens
        else {"available": False, "reason": "speculative decoding is disabled"}
    )
    output_text = f"{result['reasoning']}\n{result['content']}"
    output_ids = tokenizer.encode(output_text, add_special_tokens=False)
    completion_tokens = int(result["usage"].get("completion_tokens") or 0)
    result.update(
        {
            "target_prompt_tokens": target_tokens,
            "local_prompt_tokens": local_prompt_tokens,
            "unit_count": unit_count,
            "marker": marker,
            "output_health": {
                "marker_in_content": marker in result["content"],
                "replacement_characters": output_text.count("\ufffd"),
                "unique_output_tokens": len(set(output_ids)),
                "longest_same_token_run": _longest_same_token_run(output_ids),
            },
            "speculative_metrics": {
                "rounds": rounds,
                "proposed_tokens": proposed,
                "accepted_tokens": accepted,
                "accepted_fraction_of_proposed": (
                    accepted / proposed if proposed else None
                ),
                "mean_emitted_tokens_per_round": (
                    1.0 + accepted / rounds if rounds else None
                ),
                "round_wall_milliseconds": (
                    result["pure_decode_seconds"] * 1000.0 / rounds
                    if result["pure_decode_seconds"] and rounds
                    else None
                ),
                "unconditional_acceptance_per_position": [
                    value / rounds if rounds else None
                    for value in accepted_per_position
                ],
            },
            "prefill_metrics": {
                "computed_prompt_tokens": computed_prompt_tokens,
                "prefill_seconds": prefill_seconds,
                "tokens_per_second": (
                    computed_prompt_tokens / prefill_seconds
                    if prefill_seconds > 0
                    else None
                ),
            },
            "verifier_profile": verifier_profile,
        }
    )
    result["passed"] = bool(
        completion_tokens == output_tokens
        and result["output_health"]["marker_in_content"]
        and result["output_health"]["replacement_characters"] == 0
        and result["output_health"]["unique_output_tokens"] >= 8
        and result["output_health"]["longest_same_token_run"] < 16
    )
    return result


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    speeds = [
        float(record["pure_decode_tokens_per_second"])
        for record in records
        if record["pure_decode_tokens_per_second"] is not None
    ]
    round_costs = [
        float(record["speculative_metrics"]["round_wall_milliseconds"])
        for record in records
        if record["speculative_metrics"]["round_wall_milliseconds"] is not None
    ]
    verifier_costs = [
        float(record["verifier_profile"]["medians_ms"]["complete_verifier_ms"])
        for record in records
        if record["verifier_profile"].get("medians_ms", {}).get("complete_verifier_ms")
        is not None
    ]
    return {
        "cases": len(records),
        "passed": all(record["passed"] for record in records),
        "pure_decode_tps_min": min(speeds) if speeds else None,
        "pure_decode_tps_median": statistics.median(speeds) if speeds else None,
        "round_wall_ms_max": max(round_costs) if round_costs else None,
        "complete_verifier_ms_median": (
            statistics.median(verifier_costs) if verifier_costs else None
        ),
        "complete_verifier_ms_max": max(verifier_costs) if verifier_costs else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--lengths", default="1024,4096,16384,65536,131072,252000")
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--num-speculative-tokens", type=int, default=7)
    parser.add_argument("--profile-log", type=Path)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if not 0 <= args.num_speculative_tokens <= 7:
        raise ValueError("--num-speculative-tokens must be in [0, 7]")

    parsed = urlparse(args.base_url)
    if parsed.scheme != "http" or not parsed.hostname:
        raise ValueError("--base-url must be an http URL")
    host = parsed.hostname
    port = parsed.port or 80
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_path),
        local_files_only=True,
        trust_remote_code=True,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for target_tokens in _parse_lengths(args.lengths):
        record = _run_case(
            tokenizer=tokenizer,
            host=host,
            port=port,
            model=args.model,
            target_tokens=target_tokens,
            output_tokens=args.output_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
            timeout=args.timeout,
            num_speculative_tokens=args.num_speculative_tokens,
            profile_log=args.profile_log,
        )
        records.append(record)
        contract = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        }
        payload = {
            "contract": contract,
            "records": records,
            "summary": _summary(records),
        }
        args.out.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "target_prompt_tokens": target_tokens,
                    "usage": record["usage"],
                    "pure_decode_tokens_per_second": record[
                        "pure_decode_tokens_per_second"
                    ],
                    "speculative_metrics": record["speculative_metrics"],
                    "verifier_profile": record["verifier_profile"],
                    "output_health": record["output_health"],
                    "passed": record["passed"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    return 0 if all(record["passed"] for record in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
