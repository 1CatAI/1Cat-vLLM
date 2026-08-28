# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark cumulative coding chats through an OpenAI-compatible endpoint."""

from __future__ import annotations

import argparse
import contextlib
import json
import time
import urllib.request
from pathlib import Path
from typing import Any

SPEC_METRICS = (
    "vllm:spec_decode_num_drafts",
    "vllm:spec_decode_num_draft_tokens",
    "vllm:spec_decode_num_accepted_tokens",
)

DIRECT_CODING_SYSTEM_PROMPT = (
    "You are editing a single-file Python application in a benchmark. "
    "No tools, filesystem, shell, or external project are available. Never emit "
    "tool calls and never ask to inspect files. Answer each request directly. "
    "When asked to build or update app.py, emit the complete current app.py in "
    "one Python code block so later turns can modify the text from conversation "
    "history."
)


def _load_prompts(path: Path) -> list[dict[str, str]]:
    raw = json.loads(path.read_text())
    if not isinstance(raw, list) or not raw:
        raise ValueError("--prompts-json must contain a non-empty JSON list")
    prompts: list[dict[str, str]] = []
    for index, item in enumerate(raw):
        if isinstance(item, str):
            prompts.append({"label": f"turn-{index + 1:02d}", "prompt": item})
        elif isinstance(item, dict) and isinstance(item.get("prompt"), str):
            prompts.append(
                {
                    "label": str(item.get("label") or f"turn-{index + 1:02d}"),
                    "prompt": item["prompt"],
                }
            )
        else:
            raise TypeError(f"invalid prompt record at index {index}: {item!r}")
    return prompts


def _get_json(url: str, timeout: float = 10.0) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read())


def _metric_snapshot(base_url: str) -> dict[str, float]:
    try:
        with urllib.request.urlopen(f"{base_url}/metrics", timeout=10.0) as response:
            text = response.read().decode()
    except Exception:
        return {}
    totals = {name: 0.0 for name in SPEC_METRICS}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        metric_name = line.split("{", 1)[0].split(" ", 1)[0]
        for name in SPEC_METRICS:
            if metric_name in (name, f"{name}_total"):
                with contextlib.suppress(ValueError):
                    totals[name] += float(line.rsplit(None, 1)[-1])
    return totals


def _metric_delta(
    before: dict[str, float], after: dict[str, float]
) -> dict[str, float]:
    return {name: after.get(name, 0.0) - before.get(name, 0.0) for name in SPEC_METRICS}


def _stream_turn(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    timeout: float,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    first_token_at: float | None = None
    finished_at = started
    text_parts: list[str] = []
    usage: dict[str, Any] = {}
    finish_reason: str | None = None
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw_line in response:
            line = raw_line.decode().strip()
            if not line.startswith("data: "):
                continue
            body = line[6:]
            if body == "[DONE]":
                break
            event = json.loads(body)
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices") or []:
                delta = choice.get("delta") or {}
                content = delta.get("content") or ""
                if content:
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    text_parts.append(content)
                if choice.get("finish_reason") is not None:
                    finish_reason = choice["finish_reason"]
            finished_at = time.perf_counter()
    if first_token_at is None:
        first_token_at = finished_at
    completion_tokens = int(usage.get("completion_tokens") or 0)
    decode_seconds = max(finished_at - first_token_at, 1e-9)
    return {
        "text": "".join(text_parts),
        "usage": usage,
        "finish_reason": finish_reason,
        "ttft_seconds": first_token_at - started,
        "decode_seconds": decode_seconds,
        "wall_seconds": finished_at - started,
        "steady_decode_tps": (
            max(completion_tokens - 1, 0) / decode_seconds if completion_tokens else 0.0
        ),
    }


def _write_result(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18020")
    parser.add_argument("--model")
    parser.add_argument("--prompts-json", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-turns", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument(
        "--system-prompt",
        default=DIRECT_CODING_SYSTEM_PROMPT,
        help="System contract. Pass an empty string to omit the system message.",
    )
    parser.add_argument("--timeout", type=float, default=1800.0)
    args = parser.parse_args()

    prompts = _load_prompts(args.prompts_json)
    if args.max_turns > 0:
        prompts = prompts[: args.max_turns]
    models = _get_json(f"{args.base_url}/v1/models").get("data") or []
    model = args.model or (models[0].get("id") if models else None)
    if not model:
        raise RuntimeError("could not resolve the served model ID")

    result: dict[str, Any] = {
        "base_url": args.base_url,
        "model": model,
        "prompts_json": str(args.prompts_json),
        "settings": {
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "system_prompt": args.system_prompt,
        },
        "records": [],
    }
    messages: list[dict[str, str]] = []
    if args.system_prompt:
        messages.append({"role": "system", "content": args.system_prompt})
    for index, prompt in enumerate(prompts):
        messages.append({"role": "user", "content": prompt["prompt"]})
        before = _metric_snapshot(args.base_url)
        turn = _stream_turn(
            args.base_url,
            model,
            messages,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            timeout=args.timeout,
        )
        after = _metric_snapshot(args.base_url)
        delta = _metric_delta(before, after)
        drafts = delta["vllm:spec_decode_num_drafts"]
        accepted = delta["vllm:spec_decode_num_accepted_tokens"]
        draft_tokens = delta["vllm:spec_decode_num_draft_tokens"]
        turn["spec_decoding"] = {
            "num_drafts": drafts,
            "num_draft_tokens": draft_tokens,
            "num_accepted_tokens": accepted,
            "mean_acceptance_length": 1.0 + accepted / drafts if drafts else None,
            "draft_acceptance_rate": accepted / draft_tokens if draft_tokens else None,
        }
        record = {"index": index + 1, "label": prompt["label"], **turn}
        result["records"].append(record)
        messages.append({"role": "assistant", "content": turn["text"]})
        _write_result(args.out, result)
        usage = turn["usage"]
        print(
            f"turn={index + 1:02d} label={prompt['label']!r} "
            f"prompt={usage.get('prompt_tokens', 0)} "
            f"output={usage.get('completion_tokens', 0)} "
            f"decode={turn['steady_decode_tps']:.2f} tok/s "
            f"accept={turn['spec_decoding']['mean_acceptance_length']}",
            flush=True,
        )

    records = result["records"]
    output_tokens = sum(int(r["usage"].get("completion_tokens") or 0) for r in records)
    decode_tokens = sum(
        max(int(r["usage"].get("completion_tokens") or 0) - 1, 0) for r in records
    )
    decode_seconds = sum(float(r["decode_seconds"]) for r in records)
    drafts = sum(float(r["spec_decoding"]["num_drafts"]) for r in records)
    accepted = sum(float(r["spec_decoding"]["num_accepted_tokens"]) for r in records)
    result["aggregate"] = {
        "turns": len(records),
        "output_tokens": output_tokens,
        "steady_decode_tps": decode_tokens / decode_seconds if decode_seconds else 0.0,
        "mean_acceptance_length": 1.0 + accepted / drafts if drafts else None,
        "total_wall_seconds": sum(float(r["wall_seconds"]) for r in records),
    }
    _write_result(args.out, result)
    print(json.dumps(result["aggregate"], indent=2))


if __name__ == "__main__":
    main()
