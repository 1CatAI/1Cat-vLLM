#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure exact-length OpenAI completion streaming latency."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from pathlib import Path


def _post_json(url: str, payload: dict[str, object]):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(request, timeout=600)


def _tokenize(base_url: str, model: str, prompt: str) -> list[int]:
    with _post_json(
        f"{base_url}/tokenize",
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "add_generation_prompt": True,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    ) as response:
        return json.loads(response.read())["tokens"]


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = min(len(ordered) - 1, int(fraction * len(ordered)))
    return ordered[position]


def _build_prompt_ids(base_url: str, model: str, input_len: int) -> list[int]:
    prefix = (
        "你是一名负责大模型推理性能的工程师。请阅读材料，最后给出严谨的"
        "瓶颈分析和下一步优化建议。\n\n材料：\n"
    )
    paragraph = (
        "在单请求 decode 中，需要分别观察矩阵乘、注意力、跨卡通信和调度"
        "等待。任何优化都必须保持相同采样参数和数值语义，并通过完整输出质量"
        "检查。性能结论应区分端到端 TPOT、CUDA Graph 墙钟、kernel service "
        "time 以及 profiler 本身的开销。\n"
    )
    suffix = (
        "\n问题：请基于以上材料总结三个最重要的瓶颈，解释判断依据，并提出"
        "不会牺牲输出质量的优化顺序。"
    )
    # Keep the model's user/assistant delimiters while filling the exact-length
    # context. A bare completion prompt can terminate immediately at long input.
    skeleton = _tokenize(base_url, model, prefix + suffix)
    full = _tokenize(base_url, model, prefix + paragraph * 80 + suffix)
    left = 0
    while left < len(skeleton) and skeleton[left] == full[left]:
        left += 1
    right = 0
    while right < len(skeleton) - left and skeleton[-1 - right] == full[-1 - right]:
        right += 1
    if not right:
        raise ValueError("Chat template has no stable generation suffix")
    prefix_ids = full[:left]
    suffix_ids = full[-right:]
    context_ids = full[left:-right]
    context_len = input_len - len(prefix_ids) - len(suffix_ids)
    if context_len < 0 or not context_ids:
        raise ValueError("Unable to construct the requested input length")
    context_ids *= (context_len + len(context_ids) - 1) // len(context_ids)
    prompt_ids = prefix_ids + context_ids[:context_len] + suffix_ids
    if len(prompt_ids) != input_len:
        raise AssertionError(
            f"Expected {input_len} prompt tokens, got {len(prompt_ids)}"
        )
    return prompt_ids


def _latencies(token_events: list[tuple[float, int]]) -> dict[str, object]:
    # DFlash2 emits several tokens per chunk. These are client availability
    # times, not separate GPU completion times; co-emitted tokens share a time.
    token_times = [at for at, count in token_events for _ in range(count)]
    intervals_ms = [
        (right - left) * 1000
        for left, right in zip(token_times, token_times[1:], strict=False)
    ]
    chunk_intervals_ms = [
        (right[0] - left[0]) * 1000
        for left, right in zip(token_events, token_events[1:], strict=False)
    ]
    return {
        "latency_basis": "client token availability; co-emitted tokens share a time",
        "tpot_mean_ms": statistics.mean(intervals_ms) if intervals_ms else None,
        "tpot_p50_ms": _percentile(intervals_ms, 0.50) if intervals_ms else None,
        "tpot_p95_ms": _percentile(intervals_ms, 0.95) if intervals_ms else None,
        "tpot_p90_ms": _percentile(intervals_ms, 0.90) if intervals_ms else None,
        "tpot_p99_ms": _percentile(intervals_ms, 0.99) if intervals_ms else None,
        "chunk_interval_mean_ms": (
            statistics.mean(chunk_intervals_ms) if chunk_intervals_ms else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-len", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=4201)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    prompt_ids = _build_prompt_ids(args.base_url, args.model, args.input_len)
    payload = {
        "model": args.model,
        "prompt": prompt_ids,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "stream": True,
        "stream_options": {"include_usage": True},
        "return_token_ids": True,
    }
    if args.top_k is not None:
        payload["top_k"] = args.top_k

    started = time.perf_counter()
    token_events = []
    token_ids = []
    text_parts = []
    usage = None
    finish_reason = None
    with _post_json(f"{args.base_url}/v1/completions", payload) as response:
        for raw_line in response:
            line = raw_line.decode().strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("error"):
                raise RuntimeError(chunk["error"])
            if chunk.get("usage") is not None:
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                delta = choice.get("text", "")
                ids = choice.get("token_ids") or []
                if ids:
                    token_events.append((time.perf_counter(), len(ids)))
                    token_ids.extend(ids)
                text_parts.append(delta)
                if choice.get("finish_reason") is not None:
                    finish_reason = choice["finish_reason"]
    finished = time.perf_counter()

    if usage is None or usage["completion_tokens"] != len(token_ids):
        raise RuntimeError("Streaming token IDs are missing or disagree with usage")
    result = {
        "contract": {
            "input_len": args.input_len,
            "max_output_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "seed": args.seed,
            "ignore_eos": False,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        "usage": usage,
        "stream_chunks": len(token_events),
        "token_ids": token_ids,
        "token_events": token_events,
        "finish_reason": finish_reason,
        "generation_wall_ms": (finished - started) * 1000,
        "ttft_ms": (token_events[0][0] - started) * 1000 if token_events else None,
        **_latencies(token_events),
        "output": "".join(text_parts),
    }
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
