#!/usr/bin/env python3
"""NeedleBench-style long-context quality probe for OpenAI-compatible servers."""

from __future__ import annotations

import argparse
import json
import random
import string
import sys
import time
from pathlib import Path
from typing import Any

import requests
from transformers import AutoTokenizer


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def get_model_limit(base_url: str, model: str) -> int | None:
    try:
        response = requests.get(f"{base_url.rstrip('/')}/models", timeout=10)
        response.raise_for_status()
        for item in response.json().get("data", []):
            if item.get("id") == model:
                value = item.get("max_model_len")
                return int(value) if value is not None else None
    except Exception:
        return None
    return None


def load_tokenizer(model_path: str):
    return AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )


def count_prompt_tokens(tokenizer: Any, messages: list[dict[str, str]]) -> int:
    try:
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        if isinstance(encoded, dict):
            encoded = encoded.get("input_ids", encoded)
        elif hasattr(encoded, "input_ids"):
            encoded = encoded.input_ids
        if isinstance(encoded, list) and encoded and isinstance(encoded[0], list):
            encoded = encoded[0]
        return len(encoded)
    except Exception:
        return sum(len(tokenizer.encode(m["content"], add_special_tokens=False))
                   for m in messages) + 32


def make_units(count: int) -> list[str]:
    units = []
    topics = [
        "crimson archive",
        "green ledger",
        "silver transit map",
        "quiet observatory",
        "northern index",
        "amber protocol",
    ]
    for i in range(count):
        topic = topics[i % len(topics)]
        units.append(
            f"Paragraph {i:06d}: This is distractor material from the {topic}. "
            f"It records routine inventory, calendar notes, and unrelated "
            f"numbers {i * 17 % 9973}, {i * 31 % 7919}, {i * 43 % 6151}. "
            "It does not contain the requested secret code.\n"
        )
    return units


def build_messages(units: list[str], depth: float, code: str) -> list[dict[str, str]]:
    idx = max(0, min(len(units), int(round(len(units) * depth))))
    needle = (
        f"\nIMPORTANT NEEDLE: The unique verification code is {code}. "
        f"If asked for the verification code, answer exactly {code}.\n\n"
    )
    context = "".join(units[:idx]) + needle + "".join(units[idx:])
    user = (
        "Read the long context below. It contains exactly one verification code.\n"
        "You may think briefly, but the final answer must contain only the code.\n\n"
        "<context>\n"
        f"{context}"
        "</context>\n\n"
        "Question: What is the unique verification code? Reply with only the code."
    )
    return [
        {
            "role": "system",
            "content": (
                "You are a long-context retrieval evaluator. Use only the "
                "provided context and return the requested code exactly."
            ),
        },
        {"role": "user", "content": user},
    ]


def fit_units_for_target(
    tokenizer: Any,
    target_tokens: int,
    depth: float,
    code: str,
) -> tuple[list[dict[str, str]], int, int]:
    sample_units = make_units(128)
    sample_messages = build_messages(sample_units, depth, code)
    sample_tokens = max(1, count_prompt_tokens(tokenizer, sample_messages))
    per_unit = max(1.0, (sample_tokens - 128) / 128.0)
    high = max(1, int(target_tokens / per_unit * 1.25) + 64)
    low = 1
    best_messages = build_messages(make_units(low), depth, code)
    best_tokens = count_prompt_tokens(tokenizer, best_messages)

    for _ in range(18):
        mid = (low + high) // 2
        messages = build_messages(make_units(mid), depth, code)
        tokens = count_prompt_tokens(tokenizer, messages)
        if tokens <= target_tokens:
            best_messages = messages
            best_tokens = tokens
            low = mid + 1
        else:
            high = mid - 1
    return best_messages, best_tokens, max(0, low - 1)


def extract_reasoning(message: dict[str, Any]) -> str:
    for key in ("reasoning_content", "reasoning"):
        value = message.get(key)
        if isinstance(value, str):
            return value
    return ""


def has_bad_bang(text: str, threshold: int) -> bool:
    return "!" * threshold in text or "！" * threshold in text


def run_one(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout: int,
) -> tuple[dict[str, Any], float]:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    start = time.time()
    response = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        timeout=timeout,
    )
    elapsed = time.time() - start
    response.raise_for_status()
    return response.json(), elapsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="Qwen3.5-27B-AWQ")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--lengths", default="128000,252000")
    parser.add_argument("--depths", default="0.0,0.25,0.5,0.75,0.95")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--bang-threshold", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260502)
    parser.add_argument("--output", default="needle_quality_probe.jsonl")
    args = parser.parse_args()

    random.seed(args.seed)
    tokenizer = load_tokenizer(args.model_path)
    lengths = parse_csv_ints(args.lengths)
    depths = parse_csv_floats(args.depths)
    model_limit = get_model_limit(args.base_url, args.model)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        json.dumps(
            {
                "event": "start",
                "model": args.model,
                "model_limit": model_limit,
                "lengths": lengths,
                "depths": depths,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "output": str(out_path),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    with out_path.open("w", encoding="utf-8") as fh:
        for target in lengths:
            safe_target = target
            if model_limit is not None:
                safe_target = min(target, model_limit - args.max_tokens - 256)
            if safe_target <= 0:
                print(
                    json.dumps(
                        {"event": "skip", "target_tokens": target,
                         "reason": "target exceeds model limit"},
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                continue
            for depth in depths:
                suffix = "".join(random.choices(
                    string.ascii_uppercase + string.digits, k=8))
                code = f"NB{safe_target}-{int(depth * 100):02d}-{suffix}"
                messages, prompt_tokens, unit_count = fit_units_for_target(
                    tokenizer, safe_target, depth, code)
                sample_id = f"len{safe_target}_depth{depth:.2f}"
                print(
                    json.dumps(
                        {
                            "event": "sample_start",
                            "sample_id": sample_id,
                            "target_tokens": target,
                            "prompt_tokens": prompt_tokens,
                            "depth": depth,
                            "unit_count": unit_count,
                            "code": code,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                try:
                    result, elapsed = run_one(
                        args.base_url,
                        args.model,
                        messages,
                        args.max_tokens,
                        args.temperature,
                        args.top_p,
                        args.timeout,
                    )
                    choice = result.get("choices", [{}])[0]
                    message = choice.get("message") or {}
                    reasoning = extract_reasoning(message)
                    content = message.get("content") or ""
                    joined = f"{reasoning}\n{content}"
                    row = {
                        "sample_id": sample_id,
                        "target_tokens": target,
                        "prompt_tokens_local": prompt_tokens,
                        "depth": depth,
                        "code": code,
                        "elapsed_s": round(elapsed, 3),
                        "finish_reason": choice.get("finish_reason"),
                        "reasoning_len": len(reasoning),
                        "content_len": len(content),
                        "bad_bang": has_bad_bang(joined, args.bang_threshold),
                        "hit": code in content,
                        "hit_anywhere": code in joined,
                        "content_head": content[:200],
                        "reasoning_head": reasoning[:200],
                        "usage": result.get("usage"),
                    }
                except Exception as exc:
                    row = {
                        "sample_id": sample_id,
                        "target_tokens": target,
                        "prompt_tokens_local": prompt_tokens,
                        "depth": depth,
                        "code": code,
                        "error": repr(exc),
                    }
                rows.append(row)
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                print(
                    json.dumps({"event": "sample_done", **row},
                               ensure_ascii=False),
                    flush=True,
                )

    total = len(rows)
    errors = sum(1 for r in rows if "error" in r)
    bang = sum(1 for r in rows if r.get("bad_bang"))
    hit = sum(1 for r in rows if r.get("hit"))
    empty = sum(1 for r in rows if not r.get("content_len") and "error" not in r)
    length_stop = sum(1 for r in rows if r.get("finish_reason") == "length")
    summary = {
        "event": "summary",
        "total": total,
        "errors": errors,
        "bad_bang": bang,
        "hit": hit,
        "hit_rate": round(hit / total, 4) if total else 0,
        "empty_content": empty,
        "length_stop": length_stop,
        "output": str(out_path),
    }
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
