#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure one exact SM70 decode batch with official Qwen sampling.

This is an offline, synchronized-batch harness. Every request is submitted in
one ``LLM.generate`` call, so ``--concurrency`` is the active GPU batch rather
than an arrival-rate approximation. It is intentionally separate from
``vllm bench serve``: use this first to identify M-dependent kernel and graph
route changes, then confirm the winning configuration through the API server.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("none", "null"):
        return None
    if value.startswith(("{", "[")):
        return json.loads(value)
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _parse_engine_args(values: list[str]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected KEY=VALUE for --engine-arg, got {value!r}")
        key, raw = value.split("=", 1)
        parsed[key.replace("-", "_")] = _parse_scalar(raw)
    return parsed


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return repr(value)


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _tracked_env() -> dict[str, str]:
    prefixes = ("VLLM_", "CUDA_", "TORCH_", "TRITON_")
    return {
        key: value
        for key, value in sorted(os.environ.items())
        if key.startswith(prefixes)
    }


def _hash_ids(token_ids: list[int]) -> str:
    encoded = ",".join(str(token_id) for token_id in token_ids).encode()
    return hashlib.sha256(encoded).hexdigest()


def _make_prompt_token_ids(
    tokenizer: Any,
    prompt_base: str,
    input_len: int,
    request_index: int,
) -> list[int]:
    if input_len <= 0:
        raise ValueError("--input-len must be positive")
    base_ids = tokenizer.encode(prompt_base, add_special_tokens=False)
    if not base_ids:
        raise ValueError("--prompt-base produced no tokens")
    prompt_ids = (base_ids * ((input_len + len(base_ids) - 1) // len(base_ids)))[
        :input_len
    ]
    suffix_ids = tokenizer.encode(
        f"\nRequest identifier: {request_index}\n", add_special_tokens=False
    )
    suffix_count = min(len(suffix_ids), input_len)
    if suffix_count:
        prompt_ids[-suffix_count:] = suffix_ids[-suffix_count:]
    return prompt_ids


def _safe_delta(end: float, start: float) -> float | None:
    return end - start if end > 0.0 and start > 0.0 else None


def _request_metrics(metrics: Any, output_tokens: int) -> dict[str, Any] | None:
    if metrics is None:
        return None
    queued_time = _safe_delta(metrics.scheduled_ts, metrics.queued_ts)
    prefill_time = _safe_delta(metrics.first_token_ts, metrics.scheduled_ts)
    decode_time = _safe_delta(metrics.last_token_ts, metrics.first_token_ts)
    steady_tokens = max(output_tokens - 1, 0)
    tpot_seconds = (
        decode_time / steady_tokens
        if decode_time is not None and steady_tokens > 0
        else None
    )
    return {
        "queued_time_s": queued_time,
        "prefill_time_s": prefill_time,
        "decode_time_s": decode_time,
        "steady_decode_tokens": steady_tokens,
        "steady_decode_tps": (
            steady_tokens / decode_time
            if decode_time is not None and steady_tokens > 0
            else None
        ),
        "tpot_s": tpot_seconds,
        "raw": {
            "scheduled_ts": metrics.scheduled_ts,
            "first_token_ts": metrics.first_token_ts,
            "last_token_ts": metrics.last_token_ts,
            "is_corrupted": metrics.is_corrupted,
        },
    }


def _spec_metrics_snapshot(llm: Any) -> dict[str, Any] | None:
    try:
        metrics = llm.get_metrics()
    except (AssertionError, AttributeError):
        return None
    raw: dict[str, Any] = {
        "num_drafts": 0,
        "num_draft_tokens": 0,
        "num_accepted_tokens": 0,
        "per_pos_accepted": [],
    }
    found = False
    for metric in metrics:
        name = getattr(metric, "name", "")
        if name == "vllm:spec_decode_num_drafts" and hasattr(metric, "value"):
            raw["num_drafts"] += int(metric.value)
            found = True
        elif name == "vllm:spec_decode_num_draft_tokens" and hasattr(metric, "value"):
            raw["num_draft_tokens"] += int(metric.value)
            found = True
        elif name == "vllm:spec_decode_num_accepted_tokens" and hasattr(
            metric, "value"
        ):
            raw["num_accepted_tokens"] += int(metric.value)
            found = True
        elif name == "vllm:spec_decode_num_accepted_tokens_per_pos" and hasattr(
            metric, "values"
        ):
            values = [int(value) for value in metric.values]
            if len(raw["per_pos_accepted"]) < len(values):
                raw["per_pos_accepted"].extend(
                    [0] * (len(values) - len(raw["per_pos_accepted"]))
                )
            for index, value in enumerate(values):
                raw["per_pos_accepted"][index] += value
            found = True
    return raw if found else None


def _diff_spec_metrics(
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if after is None:
        return None
    before = before or {}
    per_pos_before = list(before.get("per_pos_accepted", []))
    per_pos_after = list(after.get("per_pos_accepted", []))
    per_pos = [
        (per_pos_after[index] if index < len(per_pos_after) else 0)
        - (per_pos_before[index] if index < len(per_pos_before) else 0)
        for index in range(max(len(per_pos_before), len(per_pos_after)))
    ]
    num_drafts = int(after.get("num_drafts", 0)) - int(before.get("num_drafts", 0))
    num_draft_tokens = int(after.get("num_draft_tokens", 0)) - int(
        before.get("num_draft_tokens", 0)
    )
    num_accepted_tokens = int(after.get("num_accepted_tokens", 0)) - int(
        before.get("num_accepted_tokens", 0)
    )
    if num_drafts <= 0:
        return None
    return {
        "num_drafts": num_drafts,
        "num_draft_tokens": num_draft_tokens,
        "num_accepted_tokens": num_accepted_tokens,
        "mean_acceptance_length": 1.0 + num_accepted_tokens / num_drafts,
        "draft_acceptance_rate": (
            num_accepted_tokens / num_draft_tokens if num_draft_tokens else None
        ),
        "per_pos_accepted": per_pos,
        "per_position_acceptance_rate": [value / num_drafts for value in per_pos],
    }


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _summarize_records(
    records: list[dict[str, Any]], elapsed_s: float
) -> dict[str, Any]:
    total_output_tokens = sum(record["output_tokens"] for record in records)
    steady_tokens = sum(
        record["request_metrics"]["steady_decode_tokens"]
        for record in records
        if record["request_metrics"] is not None
    )
    first_token_times = [
        record["request_metrics"]["raw"]["first_token_ts"]
        for record in records
        if record["request_metrics"] is not None
        and record["request_metrics"]["raw"]["first_token_ts"] > 0.0
    ]
    last_token_times = [
        record["request_metrics"]["raw"]["last_token_ts"]
        for record in records
        if record["request_metrics"] is not None
        and record["request_metrics"]["raw"]["last_token_ts"] > 0.0
    ]
    batch_decode_window_s = (
        max(last_token_times) - min(first_token_times)
        if first_token_times and last_token_times
        else None
    )
    tpot_values = [
        record["request_metrics"]["tpot_s"]
        for record in records
        if record["request_metrics"] is not None
        and record["request_metrics"]["tpot_s"] is not None
    ]
    return {
        "wall_seconds": elapsed_s,
        "total_output_tokens": total_output_tokens,
        "output_tps_including_prefill": (
            total_output_tokens / elapsed_s if elapsed_s > 0.0 else None
        ),
        "steady_decode_tokens": steady_tokens,
        "batch_decode_window_s": batch_decode_window_s,
        "aggregate_steady_decode_tps": (
            steady_tokens / batch_decode_window_s
            if batch_decode_window_s is not None and batch_decode_window_s > 0.0
            else None
        ),
        "per_request_tpot_ms": {
            "mean": statistics.mean(tpot_values) * 1000.0 if tpot_values else None,
            "p50": _percentile(tpot_values, 0.50) * 1000.0 if tpot_values else None,
            "p90": _percentile(tpot_values, 0.90) * 1000.0 if tpot_values else None,
            "p99": _percentile(tpot_values, 0.99) * 1000.0 if tpot_values else None,
        },
        "corrupted_request_count": sum(
            1
            for record in records
            if record["request_metrics"] is not None
            and record["request_metrics"]["raw"]["is_corrupted"]
        ),
    }


def _run_once(
    llm: Any,
    prompts: list[dict[str, list[int]]],
    sampling: Any,
) -> dict[str, Any]:
    import torch

    spec_before = _spec_metrics_snapshot(llm)
    torch.accelerator.synchronize()
    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling, use_tqdm=False)
    torch.accelerator.synchronize()
    elapsed_s = time.perf_counter() - start
    spec_after = _spec_metrics_snapshot(llm)
    records = []
    for request_index, output in enumerate(outputs):
        completion = output.outputs[0]
        token_ids = list(completion.token_ids)
        records.append(
            {
                "request_index": request_index,
                "output_tokens": len(token_ids),
                "token_hash": _hash_ids(token_ids),
                "finish_reason": completion.finish_reason,
                "stop_reason": completion.stop_reason,
                "request_metrics": _request_metrics(output.metrics, len(token_ids)),
            }
        )
    return {
        "summary": _summarize_records(records, elapsed_s),
        "spec_decode_metrics": _diff_spec_metrics(spec_before, spec_after),
        "records": records,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--input-len", type=int, default=1024)
    parser.add_argument("--output-len", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument(
        "--prompt-base",
        default=(
            "Implement the requested behavior carefully, explain the important "
            "tradeoffs, and return a complete answer. "
        ),
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--sampling-seed", type=int)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--quantization")
    parser.add_argument("--kv-cache-dtype")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        help="Defaults to --concurrency so graph and scheduler capacity are exact.",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--attention-backend", default="FLASH_ATTN_V100")
    parser.add_argument("--mamba-cache-mode", default="align")
    parser.add_argument("--speculative-tokens", type=int, default=0)
    parser.add_argument("--draft-attention-backend", default="FLASH_ATTN_V100")
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--disable-custom-all-reduce", action="store_true")
    parser.add_argument("--engine-arg", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.concurrency <= 0:
        raise ValueError("--concurrency must be positive")
    if args.warmup < 0 or args.repeat <= 0:
        raise ValueError("--warmup must be non-negative and --repeat must be positive")
    if args.output_len <= 0:
        raise ValueError("--output-len must be positive")

    import torch
    from transformers import AutoTokenizer

    import vllm
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model), trust_remote_code=args.trust_remote_code
    )
    prompts = [
        {
            "prompt_token_ids": _make_prompt_token_ids(
                tokenizer, args.prompt_base, args.input_len, request_index
            )
        }
        for request_index in range(args.concurrency)
    ]
    max_num_seqs = args.max_num_seqs or args.concurrency
    llm_kwargs: dict[str, Any] = {
        "model": str(args.model),
        "trust_remote_code": args.trust_remote_code,
        "tensor_parallel_size": args.tensor_parallel_size,
        "dtype": args.dtype,
        "quantization": args.quantization,
        "kv_cache_dtype": args.kv_cache_dtype,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_seqs": max_num_seqs,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "attention_backend": args.attention_backend,
        "mamba_cache_mode": args.mamba_cache_mode,
        "enable_prefix_caching": args.enable_prefix_caching,
        "disable_custom_all_reduce": args.disable_custom_all_reduce,
        "disable_log_stats": False,
        "enforce_eager": False,
    }
    if args.speculative_tokens > 0:
        llm_kwargs["speculative_config"] = {
            "method": "mtp",
            "num_speculative_tokens": args.speculative_tokens,
            "draft_sample_method": "probabilistic",
            "use_local_argmax_reduction": True,
            "attention_backend": args.draft_attention_backend,
        }
    llm_kwargs.update(_parse_engine_args(args.engine_arg))
    llm_kwargs = {key: value for key, value in llm_kwargs.items() if value is not None}
    sampling = SamplingParams(
        max_tokens=args.output_len,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        seed=args.sampling_seed,
        ignore_eos=False,
        skip_special_tokens=False,
    )
    payload: dict[str, Any] = {
        "source_sha": _git_sha(),
        "runtime": {
            "vllm_version": getattr(vllm, "__version__", None),
            "vllm_file": getattr(vllm, "__file__", None),
            "torch_version": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_capabilities": [
                list(torch.cuda.get_device_capability(index))
                for index in range(torch.cuda.device_count())
            ],
        },
        "model": str(args.model),
        "concurrency": args.concurrency,
        "prompt_input_len": args.input_len,
        "prompt_hashes": [_hash_ids(prompt["prompt_token_ids"]) for prompt in prompts],
        "engine_kwargs": llm_kwargs,
        "sampling": {
            "max_tokens": args.output_len,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "min_p": args.min_p,
            "seed": args.sampling_seed,
            "ignore_eos": False,
        },
        "env": _tracked_env(),
        "warmup": [],
        "measurements": [],
    }
    try:
        load_start = time.perf_counter()
        llm = LLM(**llm_kwargs)
        payload["load_seconds"] = time.perf_counter() - load_start
        for _ in range(args.warmup):
            payload["warmup"].append(_run_once(llm, prompts, sampling))
        for _ in range(args.repeat):
            payload["measurements"].append(_run_once(llm, prompts, sampling))
    except Exception as exc:
        payload["error"] = {"type": type(exc).__name__, "message": str(exc)}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(_json_safe(payload), indent=2) + "\n")
        raise

    aggregate_tps = [
        measurement["summary"]["aggregate_steady_decode_tps"]
        for measurement in payload["measurements"]
        if measurement["summary"]["aggregate_steady_decode_tps"] is not None
    ]
    payload["summary"] = {
        "aggregate_steady_decode_tps": {
            "mean": statistics.mean(aggregate_tps) if aggregate_tps else None,
            "p50": _percentile(aggregate_tps, 0.50),
            "p90": _percentile(aggregate_tps, 0.90),
            "p99": _percentile(aggregate_tps, 0.99),
        },
        "note": (
            "This is exact synchronous batch throughput. Natural EOS is enabled; "
            "inspect per-run output lengths before comparing a workload with "
            "substantial batch shrinkage."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(_json_safe(payload), indent=2) + "\n")
    print(json.dumps(payload["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
