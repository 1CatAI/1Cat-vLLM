# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure target-only or DFlash2 on a fixed local quality subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import time
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import regex as re

if __package__:
    from benchmarks.benchmark_sm70_decode import (
        _diff_spec_metrics,
        _hash_ids,
        _json_safe,
        _module_file,
        _module_realpath,
        _parse_extra_engine_args,
        _request_metrics_dict,
        _spec_metrics_snapshot,
        _tracked_env,
    )
else:
    from benchmark_sm70_decode import (
        _diff_spec_metrics,
        _hash_ids,
        _json_safe,
        _module_file,
        _module_realpath,
        _parse_extra_engine_args,
        _request_metrics_dict,
        _spec_metrics_snapshot,
        _tracked_env,
    )

INVALID_ANSWER = -9_999_999
IMPLEMENTATION_MIN_ACCEPTANCE_LENGTH = 4.85
RELEASE_CARD_GSM8K_MIN_ACCEPTANCE_LENGTH = 5.78
_NUMBER_RE = re.compile(r"(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![\w.])")
_BOXED_RE = re.compile(r"\\boxed\{(?P<value>(?:[^{}]+|\{(?&value)\})*)\}")
GSM8K_PROMPT_SUFFIX = (
    "\nPlease reason step by step, and put your final answer within \\boxed{}."
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--draft-model", type=Path)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--dataset-format",
        choices=("gsm8k", "turns"),
        default="gsm8k",
        help=(
            "Input schema. 'gsm8k' scores question/answer rows; 'turns' uses "
            "the first preformatted turn and leaves task scoring to a separate "
            "evaluator."
        ),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("target-only", "dflash"), required=True)
    parser.add_argument("--num-questions", type=int, default=60)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--dataset-order",
        choices=("sequential", "zlab-shuffle42"),
        default="sequential",
    )
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--warmup-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-batched-tokens", type=int, default=512)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--sequential", action="store_true")
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument(
        "--mamba-cache-mode",
        choices=("all", "align", "none"),
        help="Override the engine Mamba cache mode for practical-contract runs.",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument(
        "--target-kv-cache-dtype",
        choices=("auto", "fp8_e4m3", "fp8_e5m2"),
        default="fp8_e4m3",
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
        help=(
            "Draft-only attention backend. The target backend is selected "
            "independently so MLA models keep their compatible fast path."
        ),
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("low", "high", "max"),
        default="max",
    )
    parser.add_argument("--seed", type=int, default=0)
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument(
        "--request-seed",
        type=int,
        default=0,
        help="Sampling seed for every request; use -1 for server-style random seeds.",
    )
    seed_group.add_argument(
        "--request-seeds",
        help=(
            "Comma-separated non-negative sampling seeds. Each seed replays the "
            "selected prompt set without reloading the model."
        ),
    )
    parser.add_argument(
        "--request-seed-mode",
        choices=("fixed", "index"),
        default="fixed",
        help="Use each seed as-is or add the original dataset index per request.",
    )
    parser.add_argument(
        "--cuda-profiler-capture",
        action="store_true",
        help="Wrap the measured generation in cudaProfilerStart/Stop for nsys.",
    )
    parser.add_argument(
        "--engine-arg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Additional vLLM engine argument. May be repeated; JSON scalar "
            "values are decoded consistently with benchmark_sm70_decode.py."
        ),
    )
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_sha(path: Path) -> str | None:
    """Return repository provenance without making benchmark data disposable."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _append_partial_case(
    path: Path,
    *,
    dataset_index: int,
    request_seed: int | None,
    output: Any,
) -> None:
    """Durably retain a completed case before a long quality run finishes."""
    result = output.outputs[0]
    record = {
        "dataset_index": dataset_index,
        "request_seed": request_seed,
        "output_tokens": len(result.token_ids),
        "finish_reason": result.finish_reason,
        "stop_reason": result.stop_reason,
        "text": result.text,
        "token_ids": list(result.token_ids),
    }
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, sort_keys=True) + "\n")
        file.flush()
        os.fsync(file.fileno())


def _request_seeds(args: argparse.Namespace) -> list[int | None]:
    if args.request_seeds is None:
        if args.request_seed < -1:
            raise ValueError("--request-seed must be -1 or a non-negative integer")
        return [None if args.request_seed == -1 else args.request_seed]

    try:
        seeds = [int(value.strip()) for value in args.request_seeds.split(",")]
    except ValueError as exc:
        raise ValueError("--request-seeds must be comma-separated integers") from exc
    if not seeds or any(seed < 0 for seed in seeds):
        raise ValueError("--request-seeds must contain non-negative integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("--request-seeds must not contain duplicates")
    return seeds


def _model_weight_files(model: Path) -> dict[str, str]:
    index = model / "model.safetensors.index.json"
    if not index.is_file():
        weight = model / "model.safetensors"
        return {weight.name: str(weight.resolve())}
    weight_map = json.loads(index.read_text())["weight_map"]
    return {
        name: str((model / name).resolve()) for name in sorted(set(weight_map.values()))
    }


def _answer_value(text: str) -> int:
    boxed = list(_BOXED_RE.finditer(text))
    answer_text = boxed[-1].group("value") if boxed else text
    numbers = _NUMBER_RE.findall(answer_text)
    if not numbers:
        return INVALID_ANSWER
    try:
        value = Decimal(numbers[-1].replace(",", ""))
    except InvalidOperation:
        return INVALID_ANSWER
    if not value.is_finite() or value != value.to_integral_value():
        return INVALID_ANSWER
    return int(value)


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


def _acceptance_gate(
    observed: float | int | None,
    minimum: float,
    metric: str,
) -> dict[str, float | str | bool | None]:
    return {
        "metric": metric,
        "minimum": minimum,
        "observed": observed,
        "passed": float(observed) >= minimum if observed is not None else None,
    }


def _load_rows(args: argparse.Namespace) -> list[tuple[int, dict[str, Any]]]:
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


def _prompt_content(row: dict[str, Any], dataset_format: str) -> str:
    if dataset_format == "gsm8k":
        return str(row["question"]) + GSM8K_PROMPT_SUFFIX
    turns = row.get("turns")
    if not isinstance(turns, list) or not turns or not isinstance(turns[0], str):
        raise ValueError("turns dataset rows must contain a non-empty string list")
    return turns[0]


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
    if args.mode == "dflash" and args.draft_model is None:
        raise ValueError("--draft-model is required for --mode dflash")
    if args.request_seed_mode == "index" and not args.sequential:
        raise ValueError("--request-seed-mode index requires --sequential")
    request_seeds = _request_seeds(args)

    import torch
    import vllm._C as vllm_c
    import vllm._C_stable_libtorch as vllm_c_stable
    from transformers import AutoTokenizer

    import vllm
    from vllm import LLM, SamplingParams

    rows = _load_rows(args)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model))
    prompt_contents = [_prompt_content(row, args.dataset_format) for _, row in rows]
    prompts = [
        tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": prompt_content,
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
            reasoning_effort=args.reasoning_effort,
        )
        for prompt_content in prompt_contents
    ]
    prompt_token_counts = [len(tokenizer.encode(prompt)) for prompt in prompts]

    speculative_config = None
    if args.mode == "dflash":
        speculative_config = {
            "method": "dflash",
            "model": str(args.draft_model),
            "num_speculative_tokens": 7,
            "kv_cache_dtype": "auto",
            "attention_backend": args.draft_attention_backend,
            "draft_sample_method": args.draft_sample_method,
            "enforce_eager": args.enforce_eager,
        }

    engine_kwargs = {
        "model": str(args.model),
        "tensor_parallel_size": args.tensor_parallel_size,
        "pipeline_parallel_size": args.pipeline_parallel_size,
        "dtype": "half",
        "kv_cache_dtype": args.target_kv_cache_dtype,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_seqs": args.max_num_seqs,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enable_prefix_caching": args.enable_prefix_caching,
        "disable_log_stats": False,
        "enforce_eager": args.enforce_eager,
        "seed": args.seed,
        "speculative_config": speculative_config,
    }
    engine_kwargs.update(_parse_extra_engine_args(args.engine_arg))
    if args.mamba_cache_mode is not None:
        engine_kwargs["mamba_cache_mode"] = args.mamba_cache_mode
    if args.cuda_profiler_capture:
        engine_kwargs["profiler_config"] = {"profiler": "cuda"}

    load_started = time.perf_counter()
    llm = LLM(**engine_kwargs)
    load_seconds = time.perf_counter() - load_started

    warmup_seed = request_seeds[0]
    if warmup_seed is not None and args.request_seed_mode == "index":
        warmup_seed += rows[0][0]
    warmup_sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.warmup_tokens,
        seed=warmup_seed,
        skip_special_tokens=False,
    )
    llm.generate([prompts[0]], warmup_sampling, use_tqdm=False)

    spec_before = _spec_metrics_snapshot(llm)
    if args.cuda_profiler_capture:
        llm.start_profile()
    started = time.perf_counter()
    outputs = []
    request_spec_metrics = []
    case_inputs = []
    partial_path = args.out.with_suffix(args.out.suffix + ".partial.jsonl")
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path.write_text("", encoding="utf-8")
    for seed_base in request_seeds:
        if args.sequential:
            seed_outputs = []
            seed_spec_metrics = []
            actual_seeds = []
            for (dataset_index, _row), prompt in zip(rows, prompts, strict=True):
                request_seed = seed_base
                if request_seed is not None and args.request_seed_mode == "index":
                    request_seed += dataset_index
                sampling = SamplingParams(
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    max_tokens=args.max_tokens,
                    seed=request_seed,
                    skip_special_tokens=False,
                )
                request_spec_before = _spec_metrics_snapshot(llm)
                generated = llm.generate([prompt], sampling, use_tqdm=False)[0]
                seed_outputs.append(generated)
                request_spec_after = _spec_metrics_snapshot(llm)
                seed_spec_metrics.append(
                    _diff_spec_metrics(request_spec_before, request_spec_after)
                )
                actual_seeds.append(request_seed)
                _append_partial_case(
                    partial_path,
                    dataset_index=dataset_index,
                    request_seed=request_seed,
                    output=generated,
                )
        else:
            sampling = SamplingParams(
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_tokens=args.max_tokens,
                seed=seed_base,
                skip_special_tokens=False,
            )
            seed_outputs = llm.generate(prompts, sampling, use_tqdm=False)
            seed_spec_metrics = [None] * len(seed_outputs)
            actual_seeds = [seed_base] * len(seed_outputs)
            for (dataset_index, _row), generated in zip(
                rows, seed_outputs, strict=True
            ):
                _append_partial_case(
                    partial_path,
                    dataset_index=dataset_index,
                    request_seed=seed_base,
                    output=generated,
                )
        outputs.extend(seed_outputs)
        request_spec_metrics.extend(seed_spec_metrics)
        case_inputs.extend(
            (
                seed_base,
                request_seed,
                dataset_index,
                row,
                prompt_content,
                prompt_tokens,
            )
            for request_seed, (
                dataset_index,
                row,
            ), prompt_content, prompt_tokens in zip(
                actual_seeds,
                rows,
                prompt_contents,
                prompt_token_counts,
                strict=True,
            )
        )
    elapsed_seconds = time.perf_counter() - started
    if args.cuda_profiler_capture:
        llm.stop_profile()
    spec_after = _spec_metrics_snapshot(llm)

    cases = []
    for (
        request_seed_base,
        request_seed,
        dataset_index,
        row,
        prompt_content,
        prompt_tokens,
    ), output, request_spec in zip(
        case_inputs,
        outputs,
        request_spec_metrics,
        strict=True,
    ):
        result = output.outputs[0]
        token_ids = list(result.token_ids)
        if args.dataset_format == "gsm8k":
            prediction = _answer_value(result.text)
            expected = _answer_value(row["answer"])
            correct = prediction == expected
        else:
            prediction = None
            expected = None
            correct = None
        cases.append(
            {
                "request_seed_base": request_seed_base,
                "request_seed": request_seed,
                "dataset_index": dataset_index,
                "suite": row.get("_suite"),
                "suite_index": row.get("_suite_index"),
                "question": row.get("question"),
                "prompt_content": prompt_content,
                "expected_answer": expected,
                "predicted_answer": prediction,
                "correct": correct,
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
    scored_cases = [case for case in cases if case["correct"] is not None]
    correct = sum(bool(case["correct"]) for case in scored_cases)
    invalid = sum(case["predicted_answer"] == INVALID_ANSWER for case in scored_cases)
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
    per_request_acceptance_summary = _distribution(per_request_acceptance_lengths)
    per_request_completion_summary = _distribution(
        per_request_completion_tokens_per_verification_step
    )
    c_extension = Path(vllm_c.__file__).resolve()
    c_stable_extension = Path(vllm_c_stable.__file__).resolve()
    aggregate_spec_metrics = _diff_spec_metrics(spec_before, spec_after)
    observed_acceptance_length = (
        aggregate_spec_metrics.get("acceptance_length")
        if aggregate_spec_metrics is not None
        else None
    )
    payload = {
        "contract": {
            "source_sha": _git_sha(Path(__file__).resolve().parents[1]),
            "partial_result_file": str(partial_path),
            "mode": args.mode,
            "dataset": str(args.dataset),
            "dataset_sha256": _sha256_file(args.dataset),
            "dataset_format": args.dataset_format,
            "start_index": args.start_index,
            "num_questions": args.num_questions,
            "num_samples": len(cases),
            "dataset_order": args.dataset_order,
            "model": str(args.model),
            "model_config_sha256": _sha256_file(args.model / "config.json"),
            "model_index_sha256": (
                _sha256_file(args.model / "model.safetensors.index.json")
                if (args.model / "model.safetensors.index.json").is_file()
                else None
            ),
            "model_weight_files": _model_weight_files(args.model),
            "model_weights_realpath": str((args.model / "model.safetensors").resolve()),
            "draft_model": str(args.draft_model) if args.draft_model else None,
            "draft_model_config_sha256": (
                _sha256_file(args.draft_model / "config.json")
                if args.draft_model is not None
                else None
            ),
            "draft_model_weight_sha256": (
                _sha256_file(args.draft_model / "model.safetensors")
                if args.draft_model is not None
                else None
            ),
            "graph": not args.enforce_eager,
            "sequential": args.sequential,
            "sampling": {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "max_tokens": args.max_tokens,
                "seed": (
                    request_seeds[0]
                    if len(request_seeds) == 1 and args.request_seed_mode == "fixed"
                    else None
                ),
                "seeds": (request_seeds if args.request_seed_mode == "fixed" else None),
                "seed_bases": request_seeds,
                "seed_mode": args.request_seed_mode,
                "ignore_eos": False,
                "thinking": True,
                "reasoning_effort": args.reasoning_effort,
                "prompt_suffix": (
                    GSM8K_PROMPT_SUFFIX if args.dataset_format == "gsm8k" else None
                ),
            },
            "official_sglang_acceptance_reference": {
                "workload": "5-shot GSM8K, temperature 0",
                "parallel_1_questions": 60,
                "parallel_1_accuracy": 0.95,
                "parallel_1_token_weighted_acceptance_length": 4.85,
                "parallel_32_questions": 200,
                "parallel_32_accuracy": 0.91,
                "parallel_32_token_weighted_acceptance_length": 4.86,
                "hard_min_acceptance_length": IMPLEMENTATION_MIN_ACCEPTANCE_LENGTH,
            },
            "release_card_acceptance_reference": {
                "workload": "GSM8K, temperature 1.0, top-p 0.95, Max reasoning",
                "dataset_order": "zlab-shuffle42",
                "explicit_request_seed": False,
                "questions": 128,
                "max_new_tokens": 4096,
                "metric": (
                    "mean per-request completion tokens divided by verification steps"
                ),
                "minimum": RELEASE_CARD_GSM8K_MIN_ACCEPTANCE_LENGTH,
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
            "accuracy": correct / len(scored_cases) if scored_cases else None,
            "invalid_answer_rate": (
                invalid / len(scored_cases) if scored_cases else None
            ),
            "finish_reasons": dict(
                Counter(str(case["finish_reason"]) for case in cases)
            ),
            "request_metrics": _summarize_requests(cases),
            "spec_decode_metrics": aggregate_spec_metrics,
            "implementation_acceptance_gate": _acceptance_gate(
                observed_acceptance_length,
                IMPLEMENTATION_MIN_ACCEPTANCE_LENGTH,
                "token_weighted_mean_acceptance_length",
            ),
            "release_card_acceptance_gate": _acceptance_gate(
                per_request_completion_summary["mean"],
                RELEASE_CARD_GSM8K_MIN_ACCEPTANCE_LENGTH,
                "mean_per_request_completion_tokens_per_verification_step",
            ),
            "per_request_acceptance_length": per_request_acceptance_summary,
            "per_request_completion_tokens_per_verification_step": (
                per_request_completion_summary
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
