# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""End-to-end 9B AWQ long-context prefill comparison for Flash-V100."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
from typing import Any

import torch


def _make_prompt_token_ids(length: int, *, seed: int) -> list[int]:
    # Keep ids in a conservative non-special range for Qwen tokenizers.
    state = seed & 0x7FFFFFFF
    ids: list[int] = []
    for _ in range(length):
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        ids.append(1000 + state % 30000)
    return ids


def _hash_ids(token_ids: list[int]) -> str:
    payload = ",".join(str(token_id) for token_id in token_ids).encode()
    return hashlib.sha256(payload).hexdigest()


def _request_metrics(output: Any, elapsed_s: float, input_len: int) -> dict[str, Any]:
    metrics = getattr(output, "metrics", None)
    if metrics is None:
        return {
            "elapsed_s": elapsed_s,
            "ttft_s": elapsed_s,
            "prefill_tps": input_len / elapsed_s if elapsed_s > 0 else None,
        }

    first_token_latency = getattr(metrics, "first_token_latency", None)
    scheduled_ts = getattr(metrics, "scheduled_ts", 0.0)
    first_token_ts = getattr(metrics, "first_token_ts", 0.0)
    prefill_time = (
        first_token_ts - scheduled_ts
        if first_token_ts and scheduled_ts and first_token_ts >= scheduled_ts
        else first_token_latency
    )
    ttft_s = first_token_latency or prefill_time or elapsed_s
    return {
        "elapsed_s": elapsed_s,
        "ttft_s": ttft_s,
        "prefill_time_s": prefill_time,
        "prefill_tps": input_len / ttft_s if ttft_s and ttft_s > 0 else None,
        "raw": {
            "arrival_time": getattr(metrics, "arrival_time", None),
            "queued_ts": getattr(metrics, "queued_ts", None),
            "scheduled_ts": scheduled_ts,
            "first_token_ts": first_token_ts,
            "last_token_ts": getattr(metrics, "last_token_ts", None),
            "num_generation_tokens": getattr(metrics, "num_generation_tokens", None),
        },
    }


def _run_once(
    llm: Any,
    sampling_params: Any,
    *,
    variant: str,
    input_len: int,
    seed: int,
) -> dict[str, Any]:
    token_ids = _make_prompt_token_ids(input_len, seed=seed)
    prompt = {"prompt_token_ids": token_ids}

    torch.cuda.synchronize()
    start = time.perf_counter()
    outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
    torch.cuda.synchronize()
    elapsed_s = time.perf_counter() - start

    request_output = outputs[0]
    output_ids = list(request_output.outputs[0].token_ids)
    metrics = _request_metrics(request_output, elapsed_s, input_len)
    return {
        "variant": variant,
        "input_len": input_len,
        "prompt_hash": _hash_ids(token_ids),
        "output_tokens": len(output_ids),
        "output_hash": _hash_ids(output_ids),
        "finish_reason": request_output.outputs[0].finish_reason,
        "metrics": metrics,
    }


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in ("elapsed_s", "ttft_s", "prefill_time_s", "prefill_tps"):
        values = [
            float(record["metrics"][key])
            for record in records
            if record.get("metrics", {}).get(key) is not None
        ]
        if not values:
            continue
        summary[f"{key}_values"] = values
        summary[f"{key}_median"] = statistics.median(values)
        summary[f"{key}_mean"] = statistics.mean(values)
        summary[f"{key}_min"] = min(values)
        summary[f"{key}_max"] = max(values)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("/home/ymzx/models/Qwen3.5-9B-AWQ"))
    parser.add_argument("--lengths", type=int, nargs="+", default=[8192, 32768, 65536, 131072, 261120])
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--long-repeat", type=int, default=1)
    parser.add_argument("--long-threshold", type=int, default=200000)
    parser.add_argument("--warmup-len", type=int, default=4096)
    parser.add_argument("--max-model-len", type=int, default=262144)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260618)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--variant", choices=("baseline", "low_smem"), required=True)
    return parser.parse_args()


def _configure_variant(variant: str) -> None:
    if variant == "low_smem":
        os.environ["VLLM_FLASH_V100_PREFILL_D256_LOW_SMEM"] = "1"
    elif variant == "baseline":
        os.environ.pop("VLLM_FLASH_V100_PREFILL_D256_LOW_SMEM", None)
    else:
        raise ValueError(f"unknown variant: {variant}")


def main() -> None:
    args = _parse_args()
    _configure_variant(args.variant)

    from vllm import LLM, SamplingParams

    sampling_params = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        ignore_eos=True,
        skip_special_tokens=False,
    )

    load_start = time.perf_counter()
    llm = LLM(
        model=str(args.model),
        trust_remote_code=True,
        tensor_parallel_size=1,
        dtype="half",
        quantization="awq",
        kv_cache_dtype="auto",
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        attention_backend="FLASH_ATTN_V100",
        enforce_eager=args.enforce_eager,
        disable_log_stats=True,
        disable_custom_all_reduce=True,
        seed=args.seed,
    )
    load_s = time.perf_counter() - load_start

    records: list[dict[str, Any]] = []
    records.append(
        _run_once(
            llm,
            sampling_params,
            variant=args.variant,
            input_len=args.warmup_len,
            seed=args.seed,
        )
        | {"warmup": True}
    )

    for input_len in args.lengths:
        repeats = args.long_repeat if input_len >= args.long_threshold else args.repeat
        for rep in range(repeats):
            seed = args.seed + input_len * 17 + rep * 1009
            record = _run_once(
                llm,
                sampling_params,
                variant=args.variant,
                input_len=input_len,
                seed=seed,
            )
            record["repeat"] = rep
            record["warmup"] = False
            records.append(record)
            ttft = record["metrics"].get("ttft_s")
            tps = record["metrics"].get("prefill_tps")
            print(
                f"{args.variant} len={input_len} rep={rep} "
                f"ttft={ttft:.6f}s prefill_tps={tps:.1f}"
                if ttft and tps
                else f"{args.variant} len={input_len} rep={rep} metrics={record['metrics']}",
                flush=True,
            )

    measured = [record for record in records if not record.get("warmup")]
    by_case: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for record in measured:
        key = str(record["input_len"])
        by_case.setdefault(key, {}).setdefault(record["variant"], []).append(record)

    summaries: dict[str, Any] = {}
    for input_len, variants in by_case.items():
        summaries[input_len] = {
            variant: _summarize(items) for variant, items in variants.items()
        }
        base = summaries[input_len].get("baseline", {})
        opt = summaries[input_len].get("low_smem", {})
        base_ttft = base.get("ttft_s_median")
        opt_ttft = opt.get("ttft_s_median")
        if base_ttft and opt_ttft:
            summaries[input_len]["speedup"] = base_ttft / opt_ttft - 1.0
            summaries[input_len]["time_reduction"] = 1.0 - opt_ttft / base_ttft

    result = {
        "model": str(args.model),
        "load_s": load_s,
        "settings": {
            "variant": args.variant,
            "max_model_len": args.max_model_len,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "kv_cache_dtype": "auto",
            "attention_backend": "FLASH_ATTN_V100",
            "enforce_eager": args.enforce_eager,
        },
        "records": records,
        "summaries": summaries,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
