#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""End-to-end model output quality audit for Qwen3.5 serving surfaces."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from pathlib import Path
from typing import Any

from fa2_model_quality_audit import (
    apply_chat_template,
    fit_needle_prompt,
    normalize_text,
    output_has_nonfinite,
    serialize_output,
    token_count,
)


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def build_tool_case_prompt(tokenizer: Any, args: argparse.Namespace) -> str:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["city"],
                },
            },
        }
    ]
    messages = [
        {
            "role": "user",
            "content": (
                "Call the available weather function for city Beijing and unit "
                "celsius. Do not answer in natural language."
            ),
        }
    ]
    return tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=args.enable_thinking,
    )


def build_cases(tokenizer: Any, args: argparse.Namespace) -> list[dict[str, Any]]:
    raw_cases = [
        {
            "id": "exact_zh_phrase_no_leak",
            "category": "template_instruction",
            "prompt": "请只输出：V100-QUALITY-SENTINEL。不要解释，不要输出其它字符。",
            "grader": {
                "type": "contains_without",
                "expected": ["V100-QUALITY-SENTINEL"],
                "banned": ["<think>", "</think>", "<|im_start|>", "<tool_call>"],
            },
        },
        {
            "id": "json_contract_stable",
            "category": "format",
            "prompt": (
                "Return JSON only. The JSON object must be exactly equivalent to "
                '{"audit":"model-output","ok":true,"count":3}.'
            ),
            "grader": {
                "type": "json_fields",
                "fields": {"audit": "model-output", "ok": True, "count": 3},
                "banned": ["```", "<think>", "</think>", "<|im_start|>"],
            },
        },
        {
            "id": "distractor_resistance",
            "category": "instruction",
            "prompt": (
                "Wrong candidates: ALPHA-001, BETA-002, GAMMA-003. "
                "The final answer is PRIMARY-7788. Reply only with the final answer."
            ),
            "grader": {
                "type": "contains_without",
                "expected": ["PRIMARY-7788"],
                "banned": ["ALPHA-001", "BETA-002", "GAMMA-003", "<think>"],
            },
        },
        {
            "id": "simple_code_reasoning",
            "category": "reasoning",
            "prompt": (
                "What is the output of this Python expression?\n"
                "sum(i * 3 for i in range(1, 5))\n"
                "Reply only with the integer."
            ),
            "grader": {
                "type": "contains_without",
                "expected": ["30"],
                "banned": ["<think>", "</think>", "```"],
            },
        },
    ]

    cases = []
    for item in raw_cases:
        prompt = apply_chat_template(tokenizer, item["prompt"], args)
        cases.append({
            **item,
            "prompt": prompt,
            "prompt_token_count_estimate": token_count(tokenizer, prompt),
            "target_tokens": None,
        })

    tool_prompt = build_tool_case_prompt(tokenizer, args)
    cases.append({
        "id": "official_tool_template_xml_call",
        "category": "tool_template",
        "prompt": tool_prompt,
        "prompt_token_count_estimate": token_count(tokenizer, tool_prompt),
        "target_tokens": None,
        "grader": {
            "type": "regex_all",
            "patterns": [
                r"<tool_call>",
                r"<function=get_weather>",
                r"<parameter=city>\s*Beijing\s*</parameter>",
                r"<parameter=unit>\s*celsius\s*</parameter>",
                r"</function>\s*</tool_call>",
            ],
            "banned": ["<|im_start|>", "<|im_end|>"],
        },
    })

    for target_tokens in parse_csv_ints(args.target_lengths):
        for depth in parse_csv_floats(args.needle_depths):
            code = f"MQA{target_tokens}-{int(depth * 100):02d}-V100"
            prompt, prompt_tokens, unit_count = fit_needle_prompt(
                tokenizer, args, target_tokens, depth, code)
            cases.append({
                "id": f"needle_len{target_tokens}_depth{depth:.2f}",
                "category": "long_context",
                "prompt": prompt,
                "prompt_token_count_estimate": prompt_tokens,
                "unit_count": unit_count,
                "target_tokens": target_tokens,
                "needle_depth": depth,
                "grader": {
                    "type": "contains_without",
                    "expected": [code],
                    "banned": ["<think>", "</think>", "<|im_start|>"],
                },
            })
    return cases


def find_json_object(text: str) -> Any | None:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except Exception:
        return None


def grade_output(text: str, grader: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_text(text)
    banned = grader.get("banned", [])
    banned_hits = [item for item in banned if item in text]
    kind = grader["type"]
    if kind == "contains_without":
        missing = [item for item in grader["expected"] if item not in text]
        return {
            "passed": not missing and not banned_hits,
            "missing": missing,
            "banned_hits": banned_hits,
            "observed": normalized[:512],
        }
    if kind == "json_fields":
        parsed = find_json_object(text)
        fields = grader["fields"]
        field_ok = isinstance(parsed, dict) and all(
            parsed.get(key) == value for key, value in fields.items())
        return {
            "passed": bool(field_ok) and not banned_hits,
            "expected": fields,
            "observed": parsed if parsed is not None else normalized[:512],
            "banned_hits": banned_hits,
        }
    if kind == "regex_all":
        missing = [
            pattern for pattern in grader["patterns"]
            if re.search(pattern, text, flags=re.DOTALL) is None
        ]
        return {
            "passed": not missing and not banned_hits,
            "missing": missing,
            "banned_hits": banned_hits,
            "observed": normalized[:512],
        }
    raise ValueError(f"unknown grader type: {kind}")


def main() -> int:
    from transformers import AutoTokenizer

    from vllm import LLM, SamplingParams

    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.offline,
    )
    cases = build_cases(tokenizer, args)
    llm_kwargs = {}
    if args.disable_mm:
        llm_kwargs.update(
            limit_mm_per_prompt={"image": 0, "video": 0},
            mm_processor_cache_gb=0,
            skip_mm_profiling=True,
        )

    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        kv_cache_dtype=args.kv_cache_dtype,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        disable_custom_all_reduce=args.disable_custom_all_reduce,
        trust_remote_code=args.trust_remote_code,
        seed=args.seed,
        enable_prefix_caching=False,
        attention_config={"backend": args.backend},
        compilation_config={
            "cudagraph_mode": "full_and_piecewise",
            "cudagraph_capture_sizes": [1, 2, 4, 8],
        },
        disable_log_stats=True,
        **llm_kwargs,
    )
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        logprobs=args.logprobs if args.logprobs > 0 else None,
        seed=args.seed,
    )
    if args.warmup:
        llm.generate(
            ["model output quality warmup"],
            SamplingParams(temperature=0, max_tokens=1, ignore_eos=True),
            use_tqdm=False,
        )

    rows = []
    for case in cases:
        start = time.perf_counter()
        output = llm.generate([case["prompt"]], sampling, use_tqdm=False)[0]
        latency = time.perf_counter() - start
        serialized = serialize_output(output, include_prompt_token_ids=False)
        grade = grade_output(serialized["output_text"], case["grader"])
        rows.append({
            "id": case["id"],
            "category": case["category"],
            "target_tokens": case.get("target_tokens"),
            "needle_depth": case.get("needle_depth"),
            "prompt_token_count_estimate": case["prompt_token_count_estimate"],
            "latency_sec": latency,
            "grader": case["grader"],
            "grade": grade,
            "nonfinite_logprob": output_has_nonfinite(serialized),
            "output": serialized,
        })
        print(json.dumps({
            "case": case["id"],
            "passed": grade["passed"],
            "prompt_tokens": serialized["prompt_token_count"],
            "latency_sec": latency,
            "output": normalize_text(serialized["output_text"])[:220],
        }, ensure_ascii=False), flush=True)

    passed = sum(1 for row in rows if row["grade"]["passed"])
    payload = {
        "model": args.model,
        "backend": args.backend,
        "dtype": args.dtype,
        "kv_cache_dtype": args.kv_cache_dtype,
        "tensor_parallel_size": args.tensor_parallel_size,
        "enable_thinking": args.enable_thinking,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "summary": {
            "passed": passed,
            "total": len(rows),
            "pass_rate": passed / len(rows),
            "failed_cases": [row["id"] for row in rows if not row["grade"]["passed"]],
            "nonfinite_cases": [row["id"] for row in rows if row["nonfinite_logprob"]],
            "median_latency_sec": statistics.median(row["latency_sec"] for row in rows),
        },
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    print(f"wrote {args.output}")
    print(json.dumps(payload["summary"], ensure_ascii=False))
    return 0 if passed == len(rows) else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/ymzx/models/Qwen3.5-35B-A3B-AWQ")
    parser.add_argument("--backend", default="FLASH_ATTN_V100")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--kv-cache-dtype", default="auto")
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--disable-custom-all-reduce", action="store_true")
    parser.add_argument("--prompt-style", choices=("raw", "qwen35-chat"),
                        default="qwen35-chat")
    parser.add_argument("--disable-thinking", action="store_false",
                        dest="enable_thinking")
    parser.add_argument("--disable-mm", action="store_true")
    parser.add_argument("--target-lengths", default="32768")
    parser.add_argument("--needle-depths", default="0.10,0.50,0.90")
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--logprobs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260508)
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--no-trust-remote-code", action="store_false",
                        dest="trust_remote_code")
    parser.add_argument("--offline", action="store_true", default=True)
    parser.add_argument("--online", action="store_false", dest="offline")
    parser.add_argument("--warmup", action="store_true", default=True)
    parser.add_argument("--no-warmup", action="store_false", dest="warmup")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
