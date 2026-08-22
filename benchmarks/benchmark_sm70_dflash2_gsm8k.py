# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure target-only or DFlash2 on a fixed local GSM8K subset."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import random
import re
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

from benchmark_sm70_decode import (
    _diff_spec_metrics,
    _hash_ids,
    _json_safe,
    _module_file,
    _module_realpath,
    _request_metrics_dict,
    _spec_metrics_snapshot,
    _tracked_env,
)

INVALID_ANSWER = -9_999_999
GSM8K_PROMPT_SUFFIX = (
    "\nPlease reason step by step, and put your final answer within \\boxed{}."
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--draft-model", type=Path)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("target-only", "dflash"), required=True)
    parser.add_argument("--num-questions", type=int, default=64)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--dataset-order",
        choices=("sequential", "zlab-shuffle42"),
        default="sequential",
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--warmup-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-batched-tokens", type=int, default=512)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--sequential", action="store_true")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument(
        "--target-kv-cache-dtype",
        choices=("auto", "fp8_e5m2"),
        default="fp8_e5m2",
    )
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--draft-sample-method",
        choices=("greedy", "probabilistic"),
        default="greedy",
    )
    parser.add_argument(
        "--draft-attention-backend",
        choices=("FLASH_ATTN_V100", "TRITON_ATTN"),
        default="FLASH_ATTN_V100",
        help="Draft-only attention backend; the target remains on FLASH_ATTN_V100.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--request-seed",
        type=int,
        default=0,
        help="Sampling seed for every request; use -1 for server-style random seeds.",
    )
    parser.add_argument(
        "--cuda-profiler-capture",
        action="store_true",
        help="Wrap the measured generation in cudaProfilerStart/Stop for nsys.",
    )
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _answer_value(text: str) -> int:
    numbers = re.findall(r"\d+", text.replace(",", ""))
    if not numbers:
        return INVALID_ANSWER
    try:
        return int(ast.literal_eval(numbers[-1]))
    except (SyntaxError, ValueError):
        return INVALID_ANSWER


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": sum(values) / len(values) if values else None,
        "p50": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
        "p99": _percentile(values, 0.99),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def _load_rows(args: argparse.Namespace) -> list[tuple[int, dict[str, str]]]:
    rows = [
        json.loads(line)
        for line in args.dataset.read_text(encoding="utf-8").splitlines()
        if line
    ]
    indices = list(range(len(rows)))
    if args.dataset_order == "zlab-shuffle42":
        random.Random(42).shuffle(indices)
    selected_indices = indices[args.start_index : args.start_index + args.num_questions]
    if len(selected_indices) != args.num_questions:
        raise ValueError(
            f"Requested {args.num_questions} rows at index {args.start_index}, "
            f"but the dataset provided {len(selected_indices)}."
        )
    return [(index, rows[index]) for index in selected_indices]


def _summarize_requests(cases: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = (
        "queued_time",
        "first_token_latency",
        "prefill_time",
        "decode_time",
        "steady_decode_tps",
        "tpot_seconds",
    )
    summary = {}
    for metric_name in metric_names:
        values = [
            float(case["request_metrics"][metric_name])
            for case in cases
            if case["request_metrics"] is not None
            and case["request_metrics"].get(metric_name) is not None
        ]
        summary[metric_name] = _distribution(values)

    prefill_tps = [
        case["prompt_tokens"] / case["request_metrics"]["prefill_time"]
        for case in cases
        if case["request_metrics"] is not None
        and case["request_metrics"].get("prefill_time")
    ]
    summary["prefill_tokens_per_second"] = _distribution(prefill_tps)
    summary["prompt_tokens"] = _distribution(
        [float(case["prompt_tokens"]) for case in cases]
    )
    summary["output_tokens"] = _distribution(
        [float(case["output_tokens"]) for case in cases]
    )
    return summary


def main() -> int:
    args = _parse_args()
    if args.num_questions <= 0:
        raise ValueError("--num-questions must be positive")
    if args.request_seed < -1:
        raise ValueError("--request-seed must be -1 or a non-negative integer")
    if args.mode == "dflash" and args.draft_model is None:
        raise ValueError("--draft-model is required for --mode dflash")

    import torch
    import vllm._C as vllm_c
    import vllm._C_stable_libtorch as vllm_c_stable
    from transformers import AutoTokenizer

    import vllm
    from vllm import LLM, SamplingParams

    rows = _load_rows(args)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model))
    prompts = [
        tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": row["question"] + GSM8K_PROMPT_SUFFIX,
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
            reasoning_effort="xhigh",
        )
        for _, row in rows
    ]
    prompt_token_counts = [len(tokenizer.encode(prompt)) for prompt in prompts]

    speculative_config = None
    if args.mode == "dflash":
        speculative_config = {
            "method": "dflash",
            "model": str(args.draft_model),
            "revision": "dedf8df68adfb1afeaf7b7480c0a0243108177b4",
            "num_speculative_tokens": 7,
            "kv_cache_dtype": "auto",
            "attention_backend": args.draft_attention_backend,
            "draft_sample_method": args.draft_sample_method,
            "enforce_eager": args.enforce_eager,
        }

    engine_kwargs = {
        "model": str(args.model),
        "tensor_parallel_size": args.tensor_parallel_size,
        "dtype": "half",
        "kv_cache_dtype": args.target_kv_cache_dtype,
        "attention_backend": "FLASH_ATTN_V100",
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_seqs": args.max_num_seqs,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enable_prefix_caching": False,
        "disable_log_stats": False,
        "enforce_eager": args.enforce_eager,
        "seed": args.seed,
        "speculative_config": speculative_config,
    }
    if args.cuda_profiler_capture:
        engine_kwargs["profiler_config"] = {"profiler": "cuda"}

    load_started = time.perf_counter()
    llm = LLM(**engine_kwargs)
    load_seconds = time.perf_counter() - load_started

    request_seed = None if args.request_seed == -1 else args.request_seed
    warmup_sampling = SamplingParams(
        temperature=args.temperature,
        top_p=0.95,
        top_k=20,
        max_tokens=args.warmup_tokens,
        seed=request_seed,
        skip_special_tokens=False,
    )
    llm.generate([prompts[0]], warmup_sampling, use_tqdm=False)

    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=0.95,
        top_k=20,
        max_tokens=args.max_tokens,
        seed=request_seed,
        skip_special_tokens=False,
    )
    spec_before = _spec_metrics_snapshot(llm)
    if args.cuda_profiler_capture:
        llm.start_profile()
    started = time.perf_counter()
    if args.sequential:
        outputs = []
        request_spec_metrics = []
        for prompt in prompts:
            request_spec_before = _spec_metrics_snapshot(llm)
            outputs.append(llm.generate([prompt], sampling, use_tqdm=False)[0])
            request_spec_after = _spec_metrics_snapshot(llm)
            request_spec_metrics.append(
                _diff_spec_metrics(request_spec_before, request_spec_after)
            )
    else:
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        request_spec_metrics = [None] * len(outputs)
    elapsed_seconds = time.perf_counter() - started
    if args.cuda_profiler_capture:
        llm.stop_profile()
    spec_after = _spec_metrics_snapshot(llm)

    cases = []
    for (dataset_index, row), prompt_tokens, output, request_spec in zip(
        rows, prompt_token_counts, outputs, request_spec_metrics
    ):
        result = output.outputs[0]
        token_ids = list(result.token_ids)
        prediction = _answer_value(result.text)
        expected = _answer_value(row["answer"])
        cases.append(
            {
                "dataset_index": dataset_index,
                "question": row["question"],
                "expected_answer": expected,
                "predicted_answer": prediction,
                "correct": prediction == expected,
                "prompt_tokens": prompt_tokens,
                "output_tokens": len(token_ids),
                "finish_reason": result.finish_reason,
                "stop_reason": result.stop_reason,
                "text": result.text,
                "token_ids": token_ids,
                "token_hash": _hash_ids(token_ids),
                "spec_decode_metrics": request_spec,
                "request_metrics": _request_metrics_dict(
                    output.metrics,
                    len(token_ids),
                ),
            }
        )

    total_output_tokens = sum(case["output_tokens"] for case in cases)
    correct = sum(case["correct"] for case in cases)
    invalid = sum(case["predicted_answer"] == INVALID_ANSWER for case in cases)
    per_request_acceptance_lengths = [
        float(case["spec_decode_metrics"]["acceptance_length"])
        for case in cases
        if case["spec_decode_metrics"] is not None
        and case["spec_decode_metrics"].get("acceptance_length") is not None
    ]
    per_request_completion_tokens_per_verification_step = [
        case["output_tokens"] / case["spec_decode_metrics"]["num_drafts"]
        for case in cases
        if case["spec_decode_metrics"] is not None
        and case["spec_decode_metrics"].get("num_drafts", 0) > 0
    ]
    c_extension = Path(vllm_c.__file__).resolve()
    c_stable_extension = Path(vllm_c_stable.__file__).resolve()
    payload = {
        "contract": {
            "source_sha": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "mode": args.mode,
            "dataset": str(args.dataset),
            "dataset_sha256": _sha256_file(args.dataset),
            "start_index": args.start_index,
            "num_questions": args.num_questions,
            "dataset_order": args.dataset_order,
            "model": str(args.model),
            "draft_model": str(args.draft_model) if args.draft_model else None,
            "graph": not args.enforce_eager,
            "sequential": args.sequential,
            "sampling": {
                "temperature": args.temperature,
                "top_p": 0.95,
                "top_k": 20,
                "max_tokens": args.max_tokens,
                "seed": request_seed,
                "ignore_eos": False,
                "thinking": True,
                "reasoning_effort": "xhigh",
                "prompt_suffix": GSM8K_PROMPT_SUFFIX,
            },
            "engine_kwargs": engine_kwargs,
        },
        "runtime": {
            "vllm_version": getattr(vllm, "__version__", None),
            "vllm_file": getattr(vllm, "__file__", None),
            "torch_version": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_device_count": torch.accelerator.device_count(),
            "device_capabilities": [
                list(torch.cuda.get_device_capability(index))
                for index in range(torch.accelerator.device_count())
            ],
            "c_extension": str(c_extension),
            "c_extension_sha256": _sha256_file(c_extension),
            "c_stable_extension": str(c_stable_extension),
            "c_stable_extension_sha256": _sha256_file(c_stable_extension),
            "flash_attn_v100_python": _module_file("flash_attn_v100"),
            "flash_attn_v100_cuda": _module_realpath("flash_attn_v100_cuda"),
            "tracked_env": _tracked_env(),
            "load_seconds": load_seconds,
        },
        "results": {
            "elapsed_seconds": elapsed_seconds,
            "total_output_tokens": total_output_tokens,
            "aggregate_output_tokens_per_second": (
                total_output_tokens / elapsed_seconds
            ),
            "questions_per_second": len(cases) / elapsed_seconds,
            "accuracy": correct / len(cases),
            "invalid_answer_rate": invalid / len(cases),
            "finish_reasons": dict(
                Counter(str(case["finish_reason"]) for case in cases)
            ),
            "request_metrics": _summarize_requests(cases),
            "spec_decode_metrics": _diff_spec_metrics(spec_before, spec_after),
            "per_request_acceptance_length": _distribution(
                per_request_acceptance_lengths
            ),
            "per_request_completion_tokens_per_verification_step": _distribution(
                per_request_completion_tokens_per_verification_step
            ),
        },
        "cases": cases,
    }
    payload = _json_safe(payload)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    printable = {"contract": payload["contract"], "results": payload["results"]}
    print(json.dumps(printable, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
