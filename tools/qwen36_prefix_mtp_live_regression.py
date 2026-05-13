#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Live Qwen3.6 prefix-cache + MTP serving regression.

This script talks to an already running OpenAI-compatible vLLM server.  It is
intended for V100 Qwen3.6 27B AWQ regression runs where the server is launched
with prefix caching and MTP enabled.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import statistics
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


@dataclass(frozen=True)
class Case:
    case_id: str
    category: str
    messages: list[dict[str, Any]]
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    stream: bool = False
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] = "none"
    expected_contains: list[str] | None = None
    banned_contains: list[str] | None = None
    expected_tool: str | None = None
    expected_tool_args: dict[str, Any] | None = None


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen3.6-27b-awq-mtp")
    parser.add_argument("--tokenizer", default="/home/ymzx/models/Qwen3.6-27B-AWQ")
    parser.add_argument("--lengths", default="512,4096,8192,32768")
    parser.add_argument("--quality-lengths", default="4096,32768")
    parser.add_argument("--decode-tokens", type=int, default=256)
    parser.add_argument("--quality-output-tokens", type=int, default=512)
    parser.add_argument("--request-timeout", type=int, default=2400)
    parser.add_argument("--seed", type=int, default=20260512)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--skip-speed", action="store_true")
    parser.add_argument("--skip-quality", action="store_true")
    return parser.parse_args()


def output_dir() -> Path:
    return Path("bench_results/qwen36_prefix_mtp_live") / time.strftime(
        "%Y%m%d-%H%M%S"
    )


def request_json(
    url: str,
    payload: dict[str, Any] | None = None,
    timeout: int = 60,
) -> dict[str, Any]:
    if payload is None:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def parse_metric_labels(line: str) -> dict[str, str]:
    if "{" not in line or "}" not in line:
        return {}
    labels = line[line.index("{") + 1 : line.index("}")]
    result: dict[str, str] = {}
    for part in re.findall(r'([^=,\s]+)="([^"]*)"', labels):
        result[part[0]] = part[1]
    return result


def fetch_metrics(base_url: str) -> dict[str, Any]:
    with urllib.request.urlopen(f"{base_url}/metrics", timeout=30) as response:
        text = response.read().decode("utf-8", errors="replace")
    values: dict[str, float] = collections.defaultdict(float)
    per_pos: dict[int, float] = collections.defaultdict(float)
    source_tokens: dict[str, float] = collections.defaultdict(float)
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            value = float(parts[-1])
        except ValueError:
            continue
        metric = line.split("{", 1)[0].split()[0]
        labels = parse_metric_labels(line)
        if metric == "vllm:spec_decode_num_accepted_tokens_per_pos_total":
            pos = labels.get("position")
            if pos is not None:
                per_pos[int(pos)] += value
            continue
        if metric == "vllm:prompt_tokens_by_source_total":
            source = labels.get("source")
            if source:
                source_tokens[source] += value
            continue
        if metric.startswith("vllm:"):
            values[metric] += value
    return {
        "values": dict(values),
        "spec_accepted_per_pos": dict(per_pos),
        "prompt_tokens_by_source": dict(source_tokens),
    }


def metric_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    before_values = before.get("values", {})
    after_values = after.get("values", {})
    keys = set(before_values) | set(after_values)
    values = {
        key: after_values.get(key, 0.0) - before_values.get(key, 0.0)
        for key in sorted(keys)
    }
    before_pos = before.get("spec_accepted_per_pos", {})
    after_pos = after.get("spec_accepted_per_pos", {})
    per_pos = {
        int(pos): after_pos.get(pos, 0.0) - before_pos.get(pos, 0.0)
        for pos in sorted(set(before_pos) | set(after_pos))
    }
    before_sources = before.get("prompt_tokens_by_source", {})
    after_sources = after.get("prompt_tokens_by_source", {})
    sources = {
        source: after_sources.get(source, 0.0) - before_sources.get(source, 0.0)
        for source in sorted(set(before_sources) | set(after_sources))
    }
    drafts = values.get("vllm:spec_decode_num_drafts_total", 0.0)
    draft_tokens = values.get("vllm:spec_decode_num_draft_tokens_total", 0.0)
    accepted = values.get("vllm:spec_decode_num_accepted_tokens_total", 0.0)
    prefix_queries = values.get("vllm:prefix_cache_queries_total", 0.0)
    prefix_hits = values.get("vllm:prefix_cache_hits_total", 0.0)
    prompt_compute = sources.get("local_compute", 0.0)
    prompt_hit = sources.get("local_cache_hit", 0.0)
    return {
        "values": values,
        "spec_accepted_per_pos": per_pos,
        "prompt_tokens_by_source": sources,
        "spec_acceptance_rate": accepted / draft_tokens if draft_tokens else None,
        "spec_acceptance_length": 1.0 + accepted / drafts if drafts else None,
        "prefix_hit_ratio": prefix_hits / prefix_queries if prefix_queries else None,
        "prompt_local_cache_ratio": (
            prompt_hit / (prompt_hit + prompt_compute)
            if prompt_hit + prompt_compute
            else None
        ),
    }


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def max_run(values: list[Any]) -> int:
    best = 0
    last = object()
    run = 0
    for value in values:
        if value == last:
            run += 1
        else:
            last = value
            run = 1
        best = max(best, run)
    return best


def max_ngram_count(values: list[Any], n: int) -> int:
    if len(values) < n:
        return 0
    grams = [tuple(values[i : i + n]) for i in range(len(values) - n + 1)]
    return max(collections.Counter(grams).values(), default=0)


def anomaly_metrics(text: str) -> dict[str, Any]:
    chars = [ch for ch in text if not ch.isspace()]
    exclaims = sum(1 for ch in chars if ch in {"!", "！"})
    punct = sum(1 for ch in chars if ch in set("!！?？。.,，;；:：、"))
    unique_ratio = len(set(chars)) / len(chars) if chars else 1.0
    metrics = {
        "char_count": len(chars),
        "exclamation_count": exclaims,
        "exclamation_ratio": exclaims / max(1, len(chars)),
        "punctuation_ratio": punct / max(1, len(chars)),
        "max_char_run": max_run(chars),
        "max_4gram_count": max_ngram_count(chars, 4),
        "unique_char_ratio": unique_ratio,
    }
    metrics["flagged"] = bool(
        metrics["exclamation_ratio"] >= 0.25
        or metrics["max_char_run"] >= 20
        or metrics["max_4gram_count"] >= 48
        or (len(chars) >= 256 and unique_ratio <= 0.02)
    )
    return metrics


def base_unit(index: int, tag: str = "") -> str:
    return (
        f"Segment {index:06d}: prefix cache and MTP audit context {tag}. "
        f"Numbers {index * 17 % 9973}, {index * 31 % 7919}, "
        f"{index * 43 % 6151}. This paragraph is distractor text only.\n"
    )


def chat_token_len(tokenizer: Any, messages: list[dict[str, str]]) -> int:
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return len(tokenizer.encode(rendered, add_special_tokens=False))


def fit_context(
    tokenizer: Any,
    target_tokens: int,
    tail: str,
    tag: str = "",
) -> tuple[str, int]:
    head = (
        "Read the audit context. It is synthetic and may contain distractors.\n\n"
        f"Audit tag: {tag}\n\n"
        "<context>\n"
    )
    foot = "\n</context>\n\n" + tail

    def make(count: int) -> str:
        return head + "".join(base_unit(i, tag) for i in range(count)) + foot

    def count_tokens(text: str) -> int:
        return chat_token_len(tokenizer, [{"role": "user", "content": text}])

    unit_tokens = max(
        1, len(tokenizer.encode(base_unit(0, tag), add_special_tokens=False))
    )
    hi = max(4, int(target_tokens / unit_tokens * 1.25) + 8)
    while count_tokens(make(hi)) <= target_tokens:
        hi *= 2
    lo = 0
    best = make(0)
    best_len = count_tokens(best)
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = make(mid)
        length = count_tokens(candidate)
        if length <= target_tokens:
            best = candidate
            best_len = length
            lo = mid + 1
        else:
            hi = mid - 1
    return best, best_len


def stream_chat(
    *,
    base_url: str,
    payload: dict[str, Any],
    timeout: int,
) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    first_content = None
    last_content = None
    content_parts: list[str] = []
    tool_calls: dict[int, dict[str, Any]] = {}
    usage: dict[str, Any] = {}
    finish_reason = None
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or line.startswith(":") or not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            obj = json.loads(data)
            now = time.perf_counter()
            if obj.get("usage") is not None:
                usage = obj["usage"]
            for choice in obj.get("choices") or []:
                finish_reason = choice.get("finish_reason") or finish_reason
                delta = choice.get("delta") or {}
                piece = delta.get("content") or ""
                if piece:
                    if first_content is None:
                        first_content = now
                    last_content = now
                    content_parts.append(piece)
                for tool_call in delta.get("tool_calls") or []:
                    idx = int(tool_call.get("index", 0))
                    acc = tool_calls.setdefault(
                        idx,
                        {
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        },
                    )
                    if tool_call.get("id"):
                        acc["id"] += tool_call["id"]
                    fn = tool_call.get("function") or {}
                    if fn.get("name"):
                        acc["function"]["name"] += fn["name"]
                    if fn.get("arguments"):
                        acc["function"]["arguments"] += fn["arguments"]
                        if first_content is None:
                            first_content = now
                        last_content = now
    end = time.perf_counter()
    completion_tokens = int(usage.get("completion_tokens") or 0)
    ttft_ms = (first_content - start) * 1000.0 if first_content else None
    decode_tps = None
    if (
        first_content is not None
        and last_content is not None
        and last_content > first_content
        and completion_tokens > 1
    ):
        decode_tps = (completion_tokens - 1) / (last_content - first_content)
    return {
        "content": "".join(content_parts),
        "tool_calls": [tool_calls[i] for i in sorted(tool_calls)],
        "usage": usage,
        "finish_reason": finish_reason,
        "latency_sec": end - start,
        "ttft_ms": ttft_ms,
        "decode_tps": decode_tps,
        "output_tps": completion_tokens / (end - start) if end > start else None,
    }


def nonstream_chat(
    *,
    base_url: str,
    payload: dict[str, Any],
    timeout: int,
) -> dict[str, Any]:
    start = time.perf_counter()
    obj = request_json(
        f"{base_url}/v1/chat/completions",
        payload=payload,
        timeout=timeout,
    )
    elapsed = time.perf_counter() - start
    choice = obj["choices"][0]
    message = choice.get("message") or {}
    usage = obj.get("usage") or {}
    completion_tokens = int(usage.get("completion_tokens") or 0)
    return {
        "content": message.get("content") or "",
        "tool_calls": message.get("tool_calls") or [],
        "usage": usage,
        "finish_reason": choice.get("finish_reason"),
        "latency_sec": elapsed,
        "ttft_ms": None,
        "decode_tps": None,
        "output_tps": completion_tokens / elapsed if elapsed > 0 else None,
    }


def run_payload(
    *,
    base_url: str,
    payload: dict[str, Any],
    timeout: int,
) -> dict[str, Any]:
    before = fetch_metrics(base_url)
    if payload.get("stream"):
        response = stream_chat(base_url=base_url, payload=payload, timeout=timeout)
    else:
        response = nonstream_chat(base_url=base_url, payload=payload, timeout=timeout)
    after = fetch_metrics(base_url)
    response["metric_delta"] = metric_delta(before, after)
    response["metrics"] = anomaly_metrics(response.get("content") or "")
    response["content_prefix"] = normalize(response.get("content") or "")[:600]
    return response


def weather_tool() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                        },
                    },
                    "required": ["city"],
                },
            },
        }
    ]


def calculator_tool() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "Evaluate a basic arithmetic expression.",
                "parameters": {
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                    "required": ["expression"],
                },
            },
        }
    ]


def parse_tool_args(tool_call: dict[str, Any]) -> Any:
    args = (tool_call.get("function") or {}).get("arguments")
    if isinstance(args, str):
        try:
            return json.loads(args)
        except json.JSONDecodeError:
            return args
    return args


def arg_matches(expected: dict[str, Any], observed: Any) -> bool:
    if not isinstance(observed, dict):
        return False
    for key, expected_value in expected.items():
        if key not in observed:
            return False
        if str(expected_value).replace(" ", "") != str(observed[key]).replace(" ", ""):
            return False
    return True


def grade_case(case: Case, result: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    text = result.get("content") or ""
    normalized = normalize(text)
    for expected in case.expected_contains or []:
        if expected not in text and expected not in normalized:
            failures.append(f"missing expected substring: {expected}")
    for banned in case.banned_contains or []:
        if banned in text:
            failures.append(f"banned substring present: {banned}")
    tool_calls = result.get("tool_calls") or []
    if case.expected_tool is not None:
        if not tool_calls:
            failures.append("missing tool_calls")
        else:
            fn = tool_calls[0].get("function") or {}
            if fn.get("name") != case.expected_tool:
                failures.append(
                    f"tool name mismatch: expected {case.expected_tool}, got {fn.get('name')}"
                )
            if case.expected_tool_args and not arg_matches(
                case.expected_tool_args, parse_tool_args(tool_calls[0])
            ):
                failures.append(
                    f"tool args mismatch: expected {case.expected_tool_args}, "
                    f"got {parse_tool_args(tool_calls[0])}"
                )
    elif case.category == "tool_negative" and tool_calls:
        failures.append(f"unexpected tool_calls: {tool_calls}")
    if result.get("metrics", {}).get("flagged"):
        failures.append(f"anomaly metrics flagged: {result['metrics']}")
    corrupted = result.get("metric_delta", {}).get("values", {}).get(
        "vllm:corrupted_requests_total", 0.0
    )
    if corrupted:
        failures.append(f"corrupted request metric increased by {corrupted}")
    return {"passed": not failures, "failures": failures}


def build_quality_cases(
    tokenizer: Any,
    quality_lengths: list[int],
    quality_output_tokens: int,
) -> list[Case]:
    cases = [
        Case(
            case_id="greedy_exact_ok",
            category="greedy",
            messages=[{"role": "user", "content": "只输出 OK，不要输出其它字符。"}],
            max_tokens=8,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            expected_contains=["OK"],
            banned_contains=["!", "！", "<think>"],
        ),
        Case(
            case_id="greedy_json_contract",
            category="greedy",
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Return JSON only, no markdown: "
                        '{"audit":"qwen36-prefix-mtp","status":"ok","count":3}'
                    ),
                }
            ],
            max_tokens=128,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            expected_contains=['"audit"', '"status"', '"count"'],
            banned_contains=["```", "<think>", "</think>"],
        ),
        Case(
            case_id="official_zh_stability",
            category="official_sampling",
            messages=[
                {
                    "role": "user",
                    "content": (
                        "请用克制、技术化的中文解释：为什么开启前缀缓存后仍然"
                        "必须保证投机解码不改变最终输出分布？不要使用感叹号，"
                        "也不要重复同一句话。"
                    ),
                }
            ],
            max_tokens=quality_output_tokens,
            temperature=1.0,
            top_p=0.95,
            top_k=20,
            banned_contains=["<|im_start|>", "<tool_call>"],
        ),
        Case(
            case_id="official_code_stability",
            category="official_sampling",
            messages=[
                {
                    "role": "user",
                    "content": (
                        "写一个 Python 函数 summarize_regression(rows)，返回"
                        "prefix_cache_hit_rate、mtp_acceptance_length 和异常输出数量。"
                        "代码后用两句话说明边界情况。"
                    ),
                }
            ],
            max_tokens=quality_output_tokens,
            temperature=1.0,
            top_p=0.95,
            top_k=20,
            banned_contains=["<|im_start|>", "<tool_call>"],
        ),
        Case(
            case_id="tool_weather_nonstream",
            category="tool_call",
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Call the available weather function for city Beijing "
                        "with unit celsius. Do not answer in natural language."
                    ),
                }
            ],
            max_tokens=192,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            tools=weather_tool(),
            tool_choice="auto",
            expected_tool="get_weather",
            expected_tool_args={"city": "Beijing", "unit": "celsius"},
        ),
        Case(
            case_id="tool_calculator_stream",
            category="tool_call_stream",
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Use the calculator tool for expression 137*29+418. "
                        "Do not solve it in natural language."
                    ),
                }
            ],
            max_tokens=192,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            stream=True,
            tools=calculator_tool(),
            tool_choice="auto",
            expected_tool="calculator",
            expected_tool_args={"expression": "137*29+418"},
        ),
        Case(
            case_id="tool_negative",
            category="tool_negative",
            messages=[
                {
                    "role": "user",
                    "content": (
                        "No tool is needed. Answer in one short sentence: "
                        "what is the purpose of a regression test?"
                    ),
                }
            ],
            max_tokens=96,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            tools=weather_tool(),
            tool_choice="auto",
            banned_contains=["<tool_call>"],
        ),
    ]
    for length in quality_lengths:
        code = f"Q36-PREFIX-MTP-{length}"
        prompt, prompt_tokens = fit_context(
            tokenizer,
            length,
            (
                f"\nNeedle: the exact audit code is {code}. "
                f"Question: reply with only the exact audit code {code}."
            ),
            tag=f"quality-{length}",
        )
        cases.append(
            Case(
                case_id=f"long_needle_{length}",
                category="long_context",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=64,
                temperature=0.0,
                top_p=1.0,
                top_k=0,
                expected_contains=[code],
                banned_contains=["<think>", "</think>", "<tool_call>"],
            )
        )
    return cases


def make_payload(args: argparse.Namespace, case: Case) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": args.model,
        "messages": case.messages,
        "max_tokens": case.max_tokens,
        "temperature": case.temperature,
        "top_p": case.top_p,
        "top_k": case.top_k,
        "seed": args.seed,
        "stream": case.stream,
        "tool_choice": case.tool_choice,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if case.stream:
        payload["stream_options"] = {"include_usage": True}
    if case.tools is not None:
        payload["tools"] = case.tools
    return payload


def run_quality(args: argparse.Namespace, tokenizer: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in build_quality_cases(
        tokenizer, parse_int_list(args.quality_lengths), args.quality_output_tokens
    ):
        payload = make_payload(args, case)
        result = run_payload(
            base_url=args.base_url, payload=payload, timeout=args.request_timeout
        )
        result["case_id"] = case.case_id
        result["category"] = case.category
        result["grade"] = grade_case(case, result)
        rows.append(result)
        print(
            json.dumps(
                {
                    "phase": "quality",
                    "case": case.case_id,
                    "passed": result["grade"]["passed"],
                    "latency_sec": round(result["latency_sec"], 3),
                    "completion_tokens": result.get("usage", {}).get(
                        "completion_tokens"
                    ),
                    "accept_len": result["metric_delta"].get(
                        "spec_acceptance_length"
                    ),
                    "prefix_hit_ratio": result["metric_delta"].get(
                        "prefix_hit_ratio"
                    ),
                    "failures": result["grade"]["failures"],
                    "prefix": result["content_prefix"][:120],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return rows


def speed_messages(context: str, question: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": (
                context
                + "\n\n"
                + question
                + "\nWrite a detailed technical answer. Avoid ending early."
            ),
        }
    ]


def run_speed(args: argparse.Namespace, tokenizer: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for length in parse_int_list(args.lengths):
        base_context, prompt_tokens = fit_context(
            tokenizer,
            length,
            "Question: explain why this context exists for a serving speed test.",
            tag=f"speed-{int(time.time())}-{length}",
        )
        variants = [
            (
                "miss",
                speed_messages(base_context, "First request. Discuss prefill."),
            ),
            (
                "full_repeat_hit",
                speed_messages(base_context, "First request. Discuss prefill."),
            ),
            (
                "shared_prefix_hit",
                speed_messages(base_context, "Second request. Discuss decode."),
            ),
        ]
        for variant, messages in variants:
            case = Case(
                case_id=f"speed_{length}_{variant}",
                category="speed",
                messages=messages,
                max_tokens=args.decode_tokens,
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                stream=True,
            )
            payload = make_payload(args, case)
            result = run_payload(
                base_url=args.base_url,
                payload=payload,
                timeout=args.request_timeout,
            )
            result["case_id"] = case.case_id
            result["target_prompt_tokens"] = length
            result["actual_prompt_tokens_estimate"] = prompt_tokens
            result["variant"] = variant
            rows.append(result)
            print(
                json.dumps(
                    {
                        "phase": "speed",
                        "case": case.case_id,
                        "prompt_tokens": result.get("usage", {}).get(
                            "prompt_tokens"
                        ),
                        "completion_tokens": result.get("usage", {}).get(
                            "completion_tokens"
                        ),
                        "ttft_ms": (
                            round(result["ttft_ms"], 1)
                            if result["ttft_ms"] is not None
                            else None
                        ),
                        "decode_tps": (
                            round(result["decode_tps"], 2)
                            if result["decode_tps"] is not None
                            else None
                        ),
                        "output_tps": (
                            round(result["output_tps"], 2)
                            if result["output_tps"] is not None
                            else None
                        ),
                        "prefix_hit_ratio": result["metric_delta"].get(
                            "prefix_hit_ratio"
                        ),
                        "prompt_cache_ratio": result["metric_delta"].get(
                            "prompt_local_cache_ratio"
                        ),
                        "accept_len": result["metric_delta"].get(
                            "spec_acceptance_length"
                        ),
                        "flagged": result["metrics"]["flagged"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    return rows


def summarize(speed_rows: list[dict[str, Any]], quality_rows: list[dict[str, Any]]) -> dict[str, Any]:
    failed_quality = [
        row["case_id"]
        for row in quality_rows
        if not row.get("grade", {}).get("passed", False)
    ]
    flagged_speed = [
        row["case_id"] for row in speed_rows if row.get("metrics", {}).get("flagged")
    ]
    speed_by_length: dict[str, dict[str, Any]] = {}
    for row in speed_rows:
        length = str(row["target_prompt_tokens"])
        speed_by_length.setdefault(length, {})[row["variant"]] = {
            "ttft_ms": row.get("ttft_ms"),
            "decode_tps": row.get("decode_tps"),
            "output_tps": row.get("output_tps"),
            "completion_tokens": row.get("usage", {}).get("completion_tokens"),
            "prefix_hit_ratio": row.get("metric_delta", {}).get("prefix_hit_ratio"),
            "prompt_cache_ratio": row.get("metric_delta", {}).get(
                "prompt_local_cache_ratio"
            ),
            "acceptance_length": row.get("metric_delta", {}).get(
                "spec_acceptance_length"
            ),
        }
    accept_lens = [
        row.get("metric_delta", {}).get("spec_acceptance_length")
        for row in speed_rows + quality_rows
        if row.get("metric_delta", {}).get("spec_acceptance_length") is not None
    ]
    return {
        "passed": not failed_quality and not flagged_speed,
        "failed_quality": failed_quality,
        "flagged_speed": flagged_speed,
        "median_acceptance_length": (
            statistics.median(accept_lens) if accept_lens else None
        ),
        "speed_by_length": speed_by_length,
    }


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir or output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    args_dict = vars(args).copy()
    args_dict["output_dir"] = str(out_dir)
    model_info = request_json(f"{args.base_url}/v1/models", timeout=30)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=True, local_files_only=True
    )
    speed_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    if not args.skip_speed:
        speed_rows = run_speed(args, tokenizer)
    if not args.skip_quality:
        quality_rows = run_quality(args, tokenizer)
    summary = {
        "args": args_dict,
        "model_info": model_info,
        "summary": summarize(speed_rows, quality_rows),
        "speed_rows": speed_rows,
        "quality_rows": quality_rows,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"result_dir": str(out_dir), **summary["summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
