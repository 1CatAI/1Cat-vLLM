# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Quality harness for SM70 TurboQuant KV-cache routes.

TurboQuant is a lossy KV-cache format, so exact greedy token equality is too
strict for acceptance. This harness compares the next-token top-logprob
distribution for identical long contexts, then runs a small long-output canary
to catch obvious decode degradation.
"""

import argparse
import gc
import hashlib
import json
import math
import os
import time
from pathlib import Path
from statistics import mean
from typing import Any

from benchmark_sm70_model_tokens import _parse_extra_engine_args, _tracked_env

DEFAULT_PROMPT_BASES = [
    "Explain why deterministic GPU validation needs fixed seeds and stable inputs. ",
    "Summarize the tradeoff between throughput, latency, and output quality. ",
    "Write a Python function that validates a list of token ids without mutation. ",
    "A model serving stack should preserve numerical stability while changing "
    "cache format. ",
]
DEFAULT_CONTEXT_LENS = [128, 512, 1024]
DEFAULT_CANARY_PROMPTS = [
    (
        "Write a concise technical note explaining how KV cache quantization "
        "can affect long-context generation. Include three numbered points "
        "and one caveat."
    ),
    (
        "Implement a small Python function that groups token ids into fixed-size "
        "blocks, then describe the edge cases in comments."
    ),
]


def _hash_ids(token_ids: list[int]) -> str:
    encoded = ",".join(str(token_id) for token_id in token_ids).encode()
    return hashlib.sha256(encoded).hexdigest()


def _make_prompt_token_ids(
    tokenizer: Any,
    prompt_base: str,
    input_len: int,
) -> list[int]:
    if input_len <= 0:
        raise ValueError("context lengths must be positive")

    chunk = tokenizer.encode(prompt_base, add_special_tokens=False)
    if not chunk:
        raise ValueError("prompt base produced no tokens")

    token_ids: list[int] = []
    while len(token_ids) < input_len:
        token_ids.extend(chunk)
    return token_ids[:input_len]


def _load_probe_prompts(args: argparse.Namespace) -> list[dict[str, Any]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model),
        trust_remote_code=args.trust_remote_code,
    )
    bases = args.prompt_base or DEFAULT_PROMPT_BASES
    context_lens = args.context_len or DEFAULT_CONTEXT_LENS

    prompts: list[dict[str, Any]] = []
    for input_len in context_lens:
        for base_index, prompt_base in enumerate(bases):
            token_ids = _make_prompt_token_ids(tokenizer, prompt_base, input_len)
            prompts.append(
                {
                    "prompt_token_ids": token_ids,
                    "meta": {
                        "base_index": base_index,
                        "context_len": input_len,
                        "prompt_hash": _hash_ids(token_ids),
                    },
                }
            )
    return prompts[: args.max_probe_prompts]


def _load_canary_prompts(args: argparse.Namespace) -> list[dict[str, Any]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model),
        trust_remote_code=args.trust_remote_code,
    )
    prompts = args.canary_prompt or DEFAULT_CANARY_PROMPTS
    encoded_prompts = []
    for index, prompt in enumerate(prompts[: args.max_canary_prompts]):
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        encoded_prompts.append(
            {
                "prompt_token_ids": token_ids,
                "meta": {
                    "canary_index": index,
                    "context_len": len(token_ids),
                    "prompt_hash": _hash_ids(token_ids),
                },
            }
        )
    return encoded_prompts


def _logprob_map(step: Any) -> dict[str, float]:
    if not isinstance(step, dict):
        return {}
    values: dict[str, float] = {}
    for token_id, entry in step.items():
        logprob = getattr(entry, "logprob", None)
        if logprob is None and isinstance(entry, dict):
            logprob = entry.get("logprob")
        if logprob is not None:
            values[str(token_id)] = float(logprob)
    return values


def _serialize_logprob_map(step: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(step, dict):
        return {}
    values: dict[str, dict[str, Any]] = {}
    for token_id, entry in step.items():
        logprob = getattr(entry, "logprob", None)
        if logprob is None and isinstance(entry, dict):
            logprob = entry.get("logprob")
        if logprob is None:
            continue
        serialized: dict[str, Any] = {"logprob": float(logprob)}
        rank = getattr(entry, "rank", None)
        if rank is not None:
            serialized["rank"] = int(rank)
        decoded_token = getattr(entry, "decoded_token", None)
        if decoded_token is not None:
            serialized["decoded_token"] = decoded_token
        values[str(token_id)] = serialized
    return values


def _logsumexp(values: list[float]) -> float:
    if not values:
        return float("-inf")
    max_value = max(values)
    if math.isinf(max_value):
        return max_value
    return max_value + math.log(sum(math.exp(value - max_value) for value in values))


def _prob_vector(
    logprobs: dict[str, float],
    keys: list[str],
    missing_floor: float,
) -> list[float]:
    values = [logprobs.get(key, missing_floor) for key in keys]
    normalizer = _logsumexp(values)
    if math.isinf(normalizer):
        return [0.0 for _ in values]
    return [math.exp(value - normalizer) for value in values]


def _kl_div(left: list[float], right: list[float]) -> float:
    total = 0.0
    for left_prob, right_prob in zip(left, right, strict=False):
        if left_prob <= 0.0:
            continue
        if right_prob <= 0.0:
            return float("inf")
        total += left_prob * math.log(left_prob / right_prob)
    return total


def _js_divergence(
    left: dict[str, float],
    right: dict[str, float],
    missing_floor: float,
) -> float | None:
    keys = sorted(set(left) | set(right))
    if not keys:
        return None
    left_probs = _prob_vector(left, keys, missing_floor)
    right_probs = _prob_vector(right, keys, missing_floor)
    midpoint = [
        (lprob + rprob) * 0.5
        for lprob, rprob in zip(left_probs, right_probs, strict=False)
    ]
    return 0.5 * _kl_div(left_probs, midpoint) + 0.5 * _kl_div(right_probs, midpoint)


def _sorted_tokens(logprobs: dict[str, float]) -> list[tuple[str, float]]:
    return sorted(logprobs.items(), key=lambda item: item[1], reverse=True)


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    if pct <= 0.0:
        return min(values)
    if pct >= 100.0:
        return max(values)
    ordered = sorted(values)
    position = (len(ordered) - 1) * pct / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _longest_run(token_ids: list[int]) -> int:
    if not token_ids:
        return 0
    longest = 1
    current = 1
    for prev, token_id in zip(token_ids, token_ids[1:], strict=False):
        if token_id == prev:
            current += 1
        else:
            longest = max(longest, current)
            current = 1
    return max(longest, current)


def _make_sampling_params(
    args: argparse.Namespace, max_tokens: int, logprobs: int | None
):
    from vllm import SamplingParams

    return SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        seed=args.seed,
        max_tokens=max_tokens,
        ignore_eos=args.ignore_eos,
        logprobs=logprobs,
        detokenize=False,
        skip_special_tokens=False,
    )


def _llm_kwargs(args: argparse.Namespace, kv_cache_dtype: str) -> dict[str, Any]:
    kwargs = {
        "model": str(args.model),
        "dtype": args.dtype,
        "kv_cache_dtype": kv_cache_dtype,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "trust_remote_code": args.trust_remote_code,
        "enforce_eager": args.enforce_eager,
    }
    kwargs.update(_parse_extra_engine_args(args.engine_arg))
    return kwargs


def _records_from_outputs(
    prompts: list[dict[str, Any]],
    outputs: list[Any],
) -> list[dict[str, Any]]:
    records = []
    for prompt_meta, request_output in zip(prompts, outputs, strict=False):
        completion = request_output.outputs[0]
        token_ids = list(completion.token_ids)
        first_step = None
        if completion.logprobs:
            first_step = completion.logprobs[0]
        records.append(
            {
                "meta": prompt_meta["meta"],
                "prompt_token_hash": _hash_ids(
                    list(request_output.prompt_token_ids or [])
                ),
                "output_token_ids": token_ids,
                "finish_reason": completion.finish_reason,
                "stop_reason": completion.stop_reason,
                "cumulative_logprob": completion.cumulative_logprob,
                "first_step_top_logprobs": _serialize_logprob_map(first_step),
                "first_step_top_logprob_values": _logprob_map(first_step),
            }
        )
    return records


def _generate_records(
    llm: Any,
    args: argparse.Namespace,
    prompts: list[dict[str, Any]],
    max_tokens: int,
    logprobs: int | None,
) -> dict[str, Any]:
    import torch

    prompt_inputs = [{"prompt_token_ids": item["prompt_token_ids"]} for item in prompts]
    sampling_params = _make_sampling_params(args, max_tokens, logprobs)
    generate_start = time.perf_counter()
    outputs = llm.generate(prompt_inputs, sampling_params, use_tqdm=False)
    if torch.cuda.is_available():
        torch.accelerator.synchronize()
    generate_seconds = time.perf_counter() - generate_start
    return {
        "generate_seconds": generate_seconds,
        "record_count": len(outputs),
        "records": _records_from_outputs(prompts, outputs),
    }


def _run_case(
    args: argparse.Namespace,
    label: str,
    kv_cache_dtype: str,
    probe_prompts: list[dict[str, Any]],
    canary_prompts: list[dict[str, Any]],
) -> dict[str, Any]:
    import torch

    import vllm
    from vllm import LLM

    kwargs = _llm_kwargs(args, kv_cache_dtype)

    load_start = time.perf_counter()
    llm = LLM(**kwargs)
    if torch.cuda.is_available():
        torch.accelerator.synchronize()
    load_seconds = time.perf_counter() - load_start

    probe = _generate_records(
        llm,
        args,
        probe_prompts,
        max_tokens=1,
        logprobs=args.top_logprobs,
    )
    canary = _generate_records(
        llm,
        args,
        canary_prompts,
        max_tokens=args.canary_output_len,
        logprobs=None,
    )

    result = {
        "label": label,
        "kv_cache_dtype": kv_cache_dtype,
        "engine_kwargs": kwargs,
        "load_seconds": load_seconds,
        "probe": probe,
        "canary": canary,
        "vllm": {
            "version": getattr(vllm, "__version__", None),
            "file": getattr(vllm, "__file__", None),
        },
        "torch": {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
        },
    }

    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.accelerator.empty_cache()
        torch.accelerator.synchronize()
    return result


def _compare_probe_records(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    comparisons = []
    js_values: list[float] = []
    overlaps: list[float] = []
    baseline_top1_deltas: list[float] = []
    baseline_top1_missing = 0
    high_margin = 0
    high_margin_mismatch = 0
    token_mismatch = 0

    for base_record, cand_record in zip(baseline, candidate, strict=False):
        base_map = base_record["first_step_top_logprob_values"]
        cand_map = cand_record["first_step_top_logprob_values"]
        base_tokens = base_record["output_token_ids"]
        cand_tokens = cand_record["output_token_ids"]
        base_token = str(base_tokens[0]) if base_tokens else None
        cand_token = str(cand_tokens[0]) if cand_tokens else None

        if base_token != cand_token:
            token_mismatch += 1

        common = set(base_map) & set(cand_map)
        union = set(base_map) | set(cand_map)
        overlap = (len(common) / len(union)) if union else None
        if overlap is not None:
            overlaps.append(overlap)

        jsd = _js_divergence(base_map, cand_map, args.missing_logprob_floor)
        if jsd is not None and math.isfinite(jsd):
            js_values.append(jsd)

        base_sorted = _sorted_tokens(base_map)
        margin = None
        if len(base_sorted) >= 2:
            margin = base_sorted[0][1] - base_sorted[1][1]
        if margin is not None and margin >= args.high_margin_threshold:
            high_margin += 1
            if base_token != cand_token:
                high_margin_mismatch += 1

        base_top1_delta = None
        if base_token is not None and base_token in base_map:
            cand_logprob = cand_map.get(base_token)
            if cand_logprob is None:
                baseline_top1_missing += 1
                cand_logprob = args.missing_logprob_floor
            base_top1_delta = base_map[base_token] - cand_logprob
            baseline_top1_deltas.append(base_top1_delta)

        comparisons.append(
            {
                "meta": base_record["meta"],
                "same_prompt_hash": (
                    base_record["prompt_token_hash"] == cand_record["prompt_token_hash"]
                ),
                "baseline_token_id": base_token,
                "candidate_token_id": cand_token,
                "token_match": base_token == cand_token,
                "top_logprob_overlap": overlap,
                "top_logprob_jsd": jsd,
                "baseline_top1_margin": margin,
                "baseline_top1_nll_delta": base_top1_delta,
                "baseline_top1_missing_in_candidate_topk": (
                    base_token is not None and base_token not in cand_map
                ),
            }
        )

    count = len(comparisons)
    return {
        "count": count,
        "token_mismatch_count": token_mismatch,
        "token_mismatch_rate": token_mismatch / count if count else None,
        "mean_top_logprob_overlap": mean(overlaps) if overlaps else None,
        "mean_top_logprob_jsd": mean(js_values) if js_values else None,
        "p95_top_logprob_jsd": _percentile(js_values, 95.0),
        "max_top_logprob_jsd": max(js_values) if js_values else None,
        "mean_baseline_top1_nll_delta": (
            mean(baseline_top1_deltas) if baseline_top1_deltas else None
        ),
        "max_baseline_top1_nll_delta": (
            max(baseline_top1_deltas) if baseline_top1_deltas else None
        ),
        "baseline_top1_missing_count": baseline_top1_missing,
        "baseline_top1_missing_rate": (
            baseline_top1_missing / count if count else None
        ),
        "high_margin_count": high_margin,
        "high_margin_mismatch_count": high_margin_mismatch,
        "high_margin_mismatch_rate": (
            high_margin_mismatch / high_margin if high_margin else 0.0
        ),
        "records": comparisons,
    }


def _compare_canary_records(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    records = []
    severe_count = 0
    for base_record, cand_record in zip(baseline, candidate, strict=False):
        base_ids = base_record["output_token_ids"]
        cand_ids = cand_record["output_token_ids"]
        prefix = 0
        for base_token, cand_token in zip(base_ids, cand_ids, strict=False):
            if base_token != cand_token:
                break
            prefix += 1
        base_unique_ratio = len(set(base_ids)) / len(base_ids) if base_ids else 0.0
        cand_unique_ratio = len(set(cand_ids)) / len(cand_ids) if cand_ids else 0.0
        cand_longest_run = _longest_run(cand_ids)
        low_relative_unique = (
            base_unique_ratio == 0.0
            or cand_unique_ratio
            < base_unique_ratio * args.min_canary_relative_unique_ratio
        )
        severe = (
            len(cand_ids) == 0
            or (
                len(cand_ids) >= 32
                and cand_unique_ratio < args.min_canary_unique_ratio
                and low_relative_unique
            )
            or cand_longest_run >= args.max_canary_repeated_run
        )
        severe_count += int(severe)
        records.append(
            {
                "meta": base_record["meta"],
                "same_prompt_hash": (
                    base_record["prompt_token_hash"] == cand_record["prompt_token_hash"]
                ),
                "baseline_token_count": len(base_ids),
                "candidate_token_count": len(cand_ids),
                "exact_prefix_tokens": prefix,
                "baseline_unique_ratio": base_unique_ratio,
                "candidate_unique_ratio": cand_unique_ratio,
                "candidate_longest_repeated_run": cand_longest_run,
                "candidate_finish_reason": cand_record["finish_reason"],
                "severe_candidate_degradation": severe,
            }
        )
    count = len(records)
    return {
        "count": count,
        "severe_candidate_degradation_count": severe_count,
        "severe_candidate_degradation_rate": severe_count / count if count else None,
        "records": records,
    }


def _gate(summary: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    failures: list[str] = []
    probe = summary["probe"]
    canary = summary["canary"]

    def fail_if_metric_gt(name: str, value: float | None, limit: float) -> None:
        if value is None:
            failures.append(f"{name} missing")
        elif value > limit:
            failures.append(f"{name}={value:.6g} > {limit:.6g}")

    if not summary["candidate"]["kv_cache_dtype"].startswith("turboquant_"):
        failures.append("candidate kv_cache_dtype is not turboquant")
    fail_if_metric_gt(
        "mean_top_logprob_jsd", probe["mean_top_logprob_jsd"], args.max_mean_jsd
    )
    fail_if_metric_gt(
        "p95_top_logprob_jsd", probe["p95_top_logprob_jsd"], args.max_p95_jsd
    )
    fail_if_metric_gt(
        "mean_baseline_top1_nll_delta",
        probe["mean_baseline_top1_nll_delta"],
        args.max_mean_baseline_top1_nll_delta,
    )
    fail_if_metric_gt(
        "baseline_top1_missing_rate",
        probe["baseline_top1_missing_rate"],
        args.max_baseline_top1_missing_rate,
    )
    fail_if_metric_gt(
        "high_margin_mismatch_rate",
        probe["high_margin_mismatch_rate"],
        args.max_high_margin_mismatch_rate,
    )
    fail_if_metric_gt(
        "severe_candidate_degradation_rate",
        canary["severe_candidate_degradation_rate"],
        0.0,
    )

    return {
        "passed": not failures,
        "failures": failures,
        "thresholds": {
            "max_mean_jsd": args.max_mean_jsd,
            "max_p95_jsd": args.max_p95_jsd,
            "max_mean_baseline_top1_nll_delta": (args.max_mean_baseline_top1_nll_delta),
            "max_baseline_top1_missing_rate": (args.max_baseline_top1_missing_rate),
            "max_high_margin_mismatch_rate": (args.max_high_margin_mismatch_rate),
            "high_margin_threshold": args.high_margin_threshold,
            "min_canary_unique_ratio": args.min_canary_unique_ratio,
            "min_canary_relative_unique_ratio": (args.min_canary_relative_unique_ratio),
            "max_canary_repeated_run": args.max_canary_repeated_run,
        },
    }


def _build_payload(args: argparse.Namespace) -> dict[str, Any]:
    probe_prompts = _load_probe_prompts(args)
    canary_prompts = _load_canary_prompts(args)

    baseline_reused_from = None
    if args.baseline_json is not None:
        baseline_payload = json.loads(args.baseline_json.read_text())
        baseline = baseline_payload["baseline"]["run"]
        baseline_reused_from = str(args.baseline_json)
    else:
        baseline = _run_case(
            args,
            "baseline",
            args.baseline_kv_cache_dtype,
            probe_prompts,
            canary_prompts,
        )
    candidate = _run_case(
        args,
        "candidate",
        args.candidate_kv_cache_dtype,
        probe_prompts,
        canary_prompts,
    )

    payload = {
        "model": str(args.model),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "env": _tracked_env(),
        "sampling": {
            "seed": args.seed,
            "top_logprobs": args.top_logprobs,
            "canary_output_len": args.canary_output_len,
            "ignore_eos": args.ignore_eos,
        },
        "baseline_reused_from": baseline_reused_from,
        "baseline": {
            "kv_cache_dtype": args.baseline_kv_cache_dtype,
            "run": baseline,
        },
        "candidate": {
            "kv_cache_dtype": args.candidate_kv_cache_dtype,
            "run": candidate,
        },
        "probe": _compare_probe_records(
            baseline["probe"]["records"], candidate["probe"]["records"], args
        ),
        "canary": _compare_canary_records(
            baseline["canary"]["records"], candidate["canary"]["records"], args
        ),
    }
    payload["gate"] = _gate(payload, args)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate SM70 TurboQuant KV-cache output quality."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--baseline-json", type=Path)
    parser.add_argument("--baseline-kv-cache-dtype", default="auto")
    parser.add_argument("--candidate-kv-cache-dtype", default="turboquant_k8v4")
    parser.add_argument("--dtype", default="half")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--engine-arg", action="append", default=[])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--top-logprobs", type=int, default=20)
    parser.add_argument("--prompt-base", action="append", default=[])
    parser.add_argument("--canary-prompt", action="append", default=[])
    parser.add_argument("--context-len", type=int, action="append", default=[])
    parser.add_argument("--max-probe-prompts", type=int, default=8)
    parser.add_argument("--max-canary-prompts", type=int, default=2)
    parser.add_argument("--canary-output-len", type=int, default=128)
    parser.add_argument("--missing-logprob-floor", type=float, default=-80.0)
    parser.add_argument("--max-mean-jsd", type=float, default=0.03)
    parser.add_argument("--max-p95-jsd", type=float, default=0.10)
    parser.add_argument(
        "--max-mean-baseline-top1-nll-delta",
        type=float,
        default=0.20,
    )
    parser.add_argument("--max-baseline-top1-missing-rate", type=float, default=0.10)
    parser.add_argument("--high-margin-threshold", type=float, default=1.0)
    parser.add_argument("--max-high-margin-mismatch-rate", type=float, default=0.0)
    parser.add_argument("--min-canary-unique-ratio", type=float, default=0.05)
    parser.add_argument("--min-canary-relative-unique-ratio", type=float, default=0.5)
    parser.add_argument("--max-canary-repeated-run", type=int, default=32)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_canary_prompts > args.max_probe_prompts:
        raise ValueError("--max-canary-prompts cannot exceed --max-probe-prompts")
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    payload = _build_payload(args)
    args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    summary = {
        "model": payload["model"],
        "candidate_kv_cache_dtype": payload["candidate"]["kv_cache_dtype"],
        "probe": {
            key: value for key, value in payload["probe"].items() if key != "records"
        },
        "canary": {
            key: value for key, value in payload["canary"].items() if key != "records"
        },
        "gate": payload["gate"],
        "json_out": str(args.json_out),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if payload["gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
