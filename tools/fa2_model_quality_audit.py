#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""End-to-end model quality audit for FLASH_ATTN_V100.

The audit uses objective, locally graded tasks so backend output drift is not
confused with quality loss. Run from outside the source checkout when possible.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import statistics
import string
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_BACKENDS = ("FLASH_ATTN_V100", "TRITON_ATTN")


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def base_noise_unit(index: int) -> str:
    topics = [
        "V100 scheduling notes",
        "Qwen3.5 routing telemetry",
        "long-context retrieval diary",
        "attention backend validation",
        "warehouse inventory digest",
    ]
    topic = topics[index % len(topics)]
    return (
        f"Paragraph {index:06d}: This {topic} paragraph is distractor text. "
        f"It lists unrelated values {index * 17 % 9973}, "
        f"{index * 31 % 7919}, and {index * 43 % 6151}. "
        "It does not contain the requested audit answer.\n"
    )


def apply_chat_template(tokenizer: Any, prompt: str, args: argparse.Namespace) -> str:
    if args.prompt_style == "raw":
        return prompt
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=args.enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def token_count(tokenizer: Any, prompt: str) -> int:
    return len(tokenizer.encode(prompt, add_special_tokens=False))


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def build_needle_prompt(units: list[str], depth: float, code: str) -> str:
    insert_at = max(0, min(len(units), int(round(len(units) * depth))))
    needle = (
        f"\nIMPORTANT AUDIT NEEDLE: The exact verification code is {code}. "
        f"If asked for the verification code, answer exactly {code}.\n\n"
    )
    context = "".join(units[:insert_at]) + needle + "".join(units[insert_at:])
    return (
        "Read the long context below. It contains exactly one verification code. "
        "Use only the provided context. Reply with only the code.\n\n"
        "<context>\n"
        f"{context}"
        "</context>\n\n"
        "Question: What is the exact verification code?"
    )


def fit_needle_prompt(
    tokenizer: Any,
    args: argparse.Namespace,
    target_tokens: int,
    depth: float,
    code: str,
) -> tuple[str, int, int]:
    sample_units = [base_noise_unit(i) for i in range(128)]
    sample_prompt = apply_chat_template(
        tokenizer, build_needle_prompt(sample_units, depth, code), args)
    sample_tokens = max(1, token_count(tokenizer, sample_prompt))
    per_unit = max(1.0, (sample_tokens - 128) / 128.0)
    high = max(1, int(target_tokens / per_unit * 1.30) + 64)
    low = 1
    best_prompt = apply_chat_template(
        tokenizer, build_needle_prompt([base_noise_unit(0)], depth, code), args)
    best_tokens = token_count(tokenizer, best_prompt)
    best_units = 1

    for _ in range(20):
        mid = (low + high) // 2
        units = [base_noise_unit(i) for i in range(mid)]
        prompt = apply_chat_template(
            tokenizer, build_needle_prompt(units, depth, code), args)
        count = token_count(tokenizer, prompt)
        if count <= target_tokens:
            best_prompt = prompt
            best_tokens = count
            best_units = mid
            low = mid + 1
        else:
            high = mid - 1
    return best_prompt, best_tokens, best_units


def short_cases(tokenizer: Any, args: argparse.Namespace) -> list[dict[str, Any]]:
    raw_cases = [
        {
            "id": "arith_integer_4391",
            "category": "math",
            "prompt": "Compute 137 * 29 + 418. Reply with only the integer.",
            "grader": {"type": "contains", "expected": "4391"},
        },
        {
            "id": "arith_integer_55",
            "category": "code_reasoning",
            "prompt": (
                "What is the output of this Python expression?\n"
                "sum(i*i for i in range(1, 6))\n"
                "Reply with only the integer."
            ),
            "grader": {"type": "contains", "expected": "55"},
        },
        {
            "id": "json_contract",
            "category": "format",
            "prompt": (
                "Return a JSON object only, no markdown. It must have exactly "
                "these values: backend='FA2', status='ok', tokens=262144."
            ),
            "grader": {
                "type": "json_fields",
                "fields": {"backend": "FA2", "status": "ok", "tokens": 262144},
            },
        },
        {
            "id": "zh_instruction_keyword",
            "category": "instruction",
            "prompt": (
                "请只输出下面这段校验短语，不要解释："
                "长上下文审计通过-蓝色"
            ),
            "grader": {
                "type": "contains",
                "expected": "长上下文审计通过-蓝色",
            },
        },
        {
            "id": "ignore_distractor_code",
            "category": "instruction",
            "prompt": (
                "Several wrong answers are RED-111, BLUE-222, GREEN-333. "
                "Ignore them. The correct final answer is SAFE-V100-42. "
                "Reply with only the correct final answer."
            ),
            "grader": {"type": "contains", "expected": "SAFE-V100-42"},
        },
    ]
    cases = []
    for item in raw_cases:
        prompt = apply_chat_template(tokenizer, item["prompt"], args)
        cases.append({
            **item,
            "prompt": prompt,
            "target_tokens": None,
            "prompt_token_count_estimate": token_count(tokenizer, prompt),
        })
    return cases


def long_needle_cases(tokenizer: Any, args: argparse.Namespace) -> list[dict[str, Any]]:
    cases = []
    rng = random.Random(args.seed)
    for target_tokens in parse_csv_ints(args.target_lengths):
        for depth in parse_csv_floats(args.needle_depths):
            suffix = "".join(rng.choices(string.ascii_uppercase + string.digits, k=8))
            code = f"QA{target_tokens}-{int(depth * 100):02d}-{suffix}"
            prompt, prompt_tokens, unit_count = fit_needle_prompt(
                tokenizer, args, target_tokens, depth, code)
            cases.append({
                "id": f"needle_len{target_tokens}_depth{depth:.2f}",
                "category": "long_needle",
                "prompt": prompt,
                "target_tokens": target_tokens,
                "prompt_token_count_estimate": prompt_tokens,
                "unit_count": unit_count,
                "needle_depth": depth,
                "grader": {"type": "contains", "expected": code},
            })
    return cases


def build_cases(tokenizer: Any, args: argparse.Namespace) -> list[dict[str, Any]]:
    cases = []
    if args.include_short:
        cases.extend(short_cases(tokenizer, args))
    cases.extend(long_needle_cases(tokenizer, args))
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
    kind = grader["type"]
    if kind == "contains":
        expected = str(grader["expected"])
        return {
            "passed": expected in text or expected in normalized,
            "expected": expected,
            "observed": normalized[:512],
        }
    if kind == "json_fields":
        parsed = find_json_object(text)
        expected_fields = grader["fields"]
        passed = isinstance(parsed, dict) and all(
            parsed.get(key) == value for key, value in expected_fields.items())
        return {
            "passed": bool(passed),
            "expected": expected_fields,
            "observed": parsed if parsed is not None else normalized[:512],
        }
    raise ValueError(f"unknown grader type: {kind}")


def logprob_obj_to_dict(obj: Any) -> dict[str, Any]:
    return {
        "logprob": float(getattr(obj, "logprob")),
        "rank": getattr(obj, "rank", None),
        "decoded_token": getattr(obj, "decoded_token", None),
    }


def serialize_logprob_map(logprob_map: Any, limit: int = 8) -> list[dict[str, Any]]:
    if not logprob_map:
        return []
    items = []
    for token_id, value in logprob_map.items():
        row = {"token_id": int(token_id)}
        row.update(logprob_obj_to_dict(value))
        items.append(row)
    items.sort(key=lambda row: (
        row["rank"] if row["rank"] is not None else 1_000_000,
        -row["logprob"],
    ))
    return items[:limit]


def selected_logprob(logprob_map: Any, token_id: int) -> float | None:
    if not logprob_map or token_id not in logprob_map:
        return None
    return float(getattr(logprob_map[token_id], "logprob"))


def serialize_output(output: Any, include_prompt_token_ids: bool) -> dict[str, Any]:
    completion = output.outputs[0]
    prompt_token_ids = [int(token_id) for token_id in output.prompt_token_ids]
    output_token_ids = [int(token_id) for token_id in completion.token_ids]
    generated_logprobs = []
    for idx, token_id in enumerate(output_token_ids):
        logprob_map = None
        if completion.logprobs is not None and idx < len(completion.logprobs):
            logprob_map = completion.logprobs[idx]
        generated_logprobs.append({
            "token_id": token_id,
            "selected_logprob": selected_logprob(logprob_map, token_id),
            "top": serialize_logprob_map(logprob_map),
        })
    return {
        "prompt_token_ids": prompt_token_ids if include_prompt_token_ids else [],
        "prompt_token_count": len(prompt_token_ids),
        "output_text": completion.text,
        "output_token_ids": output_token_ids,
        "finish_reason": completion.finish_reason,
        "stop_reason": completion.stop_reason,
        "cumulative_logprob": (
            None if completion.cumulative_logprob is None
            else float(completion.cumulative_logprob)
        ),
        "generated_logprobs": generated_logprobs,
    }


def output_has_nonfinite(output: dict[str, Any]) -> bool:
    values = []
    if output.get("cumulative_logprob") is not None:
        values.append(output["cumulative_logprob"])
    for row in output.get("generated_logprobs", []):
        if row.get("selected_logprob") is not None:
            values.append(row["selected_logprob"])
        for top in row.get("top", []):
            values.append(top["logprob"])
    return any(not math.isfinite(float(value)) for value in values)


def child_main(args: argparse.Namespace) -> int:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.offline,
    )
    cases = build_cases(tokenizer, args)
    if not cases:
        raise RuntimeError("no audit cases were built")

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
        enforce_eager=args.enforce_eager,
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

    warmup_sampling = SamplingParams(
        temperature=0,
        max_tokens=1,
        ignore_eos=True,
        seed=args.seed,
    )
    if args.warmup:
        llm.generate(["quality audit warmup"], warmup_sampling, use_tqdm=False)

    rows = []
    for case in cases:
        start = time.perf_counter()
        output = llm.generate([case["prompt"]], sampling, use_tqdm=False)[0]
        latency = time.perf_counter() - start
        serialized = serialize_output(
            output, include_prompt_token_ids=not args.omit_prompt_token_ids)
        grade = grade_output(serialized["output_text"], case["grader"])
        rows.append({
            "id": case["id"],
            "category": case["category"],
            "target_tokens": case.get("target_tokens"),
            "needle_depth": case.get("needle_depth"),
            "prompt_token_count_estimate": case.get(
                "prompt_token_count_estimate"),
            "unit_count": case.get("unit_count"),
            "grader": case["grader"],
            "grade": grade,
            "latency_sec": latency,
            "nonfinite_logprob": output_has_nonfinite(serialized),
            "output": serialized,
        })
        print(
            json.dumps({
                "backend": args.backend,
                "case": case["id"],
                "passed": grade["passed"],
                "latency_sec": latency,
                "prompt_tokens": serialized["prompt_token_count"],
                "output_prefix": normalize_text(serialized["output_text"])[:160],
            }, ensure_ascii=False),
            flush=True,
        )

    passed = sum(1 for row in rows if row["grade"]["passed"])
    payload = {
        "backend": args.backend,
        "model": args.model,
        "dtype": args.dtype,
        "kv_cache_dtype": args.kv_cache_dtype,
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "prompt_style": args.prompt_style,
        "enable_thinking": args.enable_thinking,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "summary": {
            "passed": passed,
            "total": len(rows),
            "pass_rate": passed / len(rows),
            "nonfinite_cases": [
                row["id"] for row in rows if row["nonfinite_logprob"]
            ],
            "median_latency_sec": statistics.median(
                row["latency_sec"] for row in rows),
        },
        "cases": rows,
    }
    args.child_output.parent.mkdir(parents=True, exist_ok=True)
    args.child_output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0 if passed == len(rows) else 2


def run_backend(args: argparse.Namespace, backend: str,
                output_dir: Path) -> dict[str, Any]:
    child_json = output_dir / f"{backend}.json"
    child_log = output_dir / f"{backend}.log"
    env = os.environ.copy()
    env.update({
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": args.cuda_visible_devices,
        "HF_HUB_OFFLINE": "1" if args.offline else env.get("HF_HUB_OFFLINE", "0"),
        "TRANSFORMERS_OFFLINE": "1" if args.offline else env.get(
            "TRANSFORMERS_OFFLINE", "0"),
        "VLLM_USE_V1": "1",
        "VLLM_ATTENTION_BACKEND": backend,
        "VLLM_SM70_ENABLE_LM_HEAD_FASTPATH": "1",
    })
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--backend",
        backend,
        "--model",
        args.model,
        "--dtype",
        args.dtype,
        "--kv-cache-dtype",
        args.kv_cache_dtype,
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--prompt-style",
        args.prompt_style,
        "--target-lengths",
        args.target_lengths,
        "--needle-depths",
        args.needle_depths,
        "--max-tokens",
        str(args.max_tokens),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--logprobs",
        str(args.logprobs),
        "--seed",
        str(args.seed),
        "--child-output",
        str(child_json),
    ]
    if args.include_short:
        cmd.append("--include-short")
    if args.disable_mm:
        cmd.append("--disable-mm")
    if args.disable_custom_all_reduce:
        cmd.append("--disable-custom-all-reduce")
    if not args.enable_thinking:
        cmd.append("--disable-thinking")
    if args.omit_prompt_token_ids:
        cmd.append("--omit-prompt-token-ids")
    if args.enforce_eager:
        cmd.append("--enforce-eager")
    if not args.warmup:
        cmd.append("--no-warmup")
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")
    if not args.offline:
        cmd.append("--online")

    with child_log.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            cmd,
            cwd="/tmp",
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if not child_json.exists():
        raise RuntimeError(
            f"{backend} did not write {child_json}; exit={proc.returncode}; "
            f"see {child_log}")
    data = json.loads(child_json.read_text(encoding="utf-8"))
    data["log_path"] = str(child_log)
    data["exit_code"] = proc.returncode
    return data


def compare_results(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    backend_names = list(results)
    by_backend = {}
    for backend, payload in results.items():
        by_backend[backend] = {
            row["id"]: row for row in payload["cases"]
        }

    comparison: dict[str, Any] = {
        "backend_summaries": {
            backend: payload["summary"] for backend, payload in results.items()
        },
        "case_comparisons": {},
    }
    if len(backend_names) < 2:
        return comparison

    reference = backend_names[0]
    candidate = backend_names[1]
    ref_cases = by_backend[reference]
    cand_cases = by_backend[candidate]
    fa2_unique_failures = []
    reference_unique_failures = []
    both_pass_diverged = []
    for case_id, ref_row in ref_cases.items():
        cand_row = cand_cases[case_id]
        ref_pass = ref_row["grade"]["passed"]
        cand_pass = cand_row["grade"]["passed"]
        ref_text = ref_row["output"]["output_text"]
        cand_text = cand_row["output"]["output_text"]
        exact_text = ref_text == cand_text
        exact_tokens = (
            ref_row["output"]["output_token_ids"]
            == cand_row["output"]["output_token_ids"]
        )
        row = {
            reference: ref_pass,
            candidate: cand_pass,
            "text_exact": exact_text,
            "token_exact": exact_tokens,
            "reference_prefix": normalize_text(ref_text)[:240],
            "candidate_prefix": normalize_text(cand_text)[:240],
            "reference_latency_sec": ref_row["latency_sec"],
            "candidate_latency_sec": cand_row["latency_sec"],
        }
        comparison["case_comparisons"][case_id] = row
        if reference == "FLASH_ATTN_V100" and not ref_pass and cand_pass:
            fa2_unique_failures.append(case_id)
        elif candidate == "FLASH_ATTN_V100" and not cand_pass and ref_pass:
            fa2_unique_failures.append(case_id)
        if ref_pass and not cand_pass:
            reference_unique_failures.append(case_id)
        if ref_pass and cand_pass and not exact_tokens:
            both_pass_diverged.append(case_id)
    comparison["fa2_unique_failures"] = fa2_unique_failures
    comparison["reference_unique_failures"] = reference_unique_failures
    comparison["both_pass_but_token_diverged"] = both_pass_diverged
    comparison["passed"] = not fa2_unique_failures
    return comparison


def parent_main(args: argparse.Namespace) -> int:
    output_dir = args.output_dir or Path(
        f"/tmp/fa2_model_quality_audit_{time.strftime('%Y%m%d_%H%M%S')}")
    output_dir.mkdir(parents=True, exist_ok=True)
    backends = parse_csv_strings(args.backends)
    results = {}
    for backend in backends:
        print(f"running quality audit backend {backend} ...", flush=True)
        results[backend] = run_backend(args, backend, output_dir)
    comparison = compare_results(results)
    combined = {
        "model": args.model,
        "cuda_visible_devices": args.cuda_visible_devices,
        "backends": backends,
        "results": results,
        "comparison": comparison,
    }
    combined_json = output_dir / "combined.json"
    combined_json.write_text(
        json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {combined_json}")
    for backend in backends:
        summary = results[backend]["summary"]
        print(
            f"{backend}: {summary['passed']}/{summary['total']} "
            f"pass_rate={summary['pass_rate']:.3f} "
            f"nonfinite={summary['nonfinite_cases']}")
    if "fa2_unique_failures" in comparison:
        print(f"fa2_unique_failures={comparison['fa2_unique_failures']}")
        print(
            "both_pass_but_token_diverged="
            f"{comparison['both_pass_but_token_diverged']}")
    return 0 if comparison.get("passed", True) else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--backend", choices=DEFAULT_BACKENDS)
    parser.add_argument("--backends", default="FLASH_ATTN_V100,TRITON_ATTN")
    parser.add_argument("--model", default="/home/ymzx/models/Qwen3.5-35B-A3B-AWQ")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--kv-cache-dtype", default="auto")
    parser.add_argument("--cuda-visible-devices", default="1,2,3,4")
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--disable-custom-all-reduce", action="store_true")
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--prompt-style", choices=("raw", "qwen35-chat"),
                        default="qwen35-chat")
    parser.add_argument("--disable-thinking", action="store_false",
                        dest="enable_thinking")
    parser.add_argument("--disable-mm", action="store_true")
    parser.add_argument("--include-short", action="store_true")
    parser.add_argument("--target-lengths", default="8192,32768")
    parser.add_argument("--needle-depths", default="0.10,0.50,0.90")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--logprobs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260508)
    parser.add_argument("--omit-prompt-token-ids", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--no-trust-remote-code", action="store_false",
                        dest="trust_remote_code")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--warmup", action="store_true", default=True)
    parser.add_argument("--no-warmup", action="store_false", dest="warmup")
    parser.add_argument("--offline", action="store_true", default=True)
    parser.add_argument("--online", action="store_false", dest="offline")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--child-output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.child:
        if args.backend is None or args.child_output is None:
            raise SystemExit("--child requires --backend and --child-output")
        return child_main(args)
    return parent_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
