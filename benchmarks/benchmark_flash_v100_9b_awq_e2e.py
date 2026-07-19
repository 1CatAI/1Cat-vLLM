# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""End-to-end 9B AWQ long-context prefill comparison for Flash-V100."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import torch

_FLASH_V100_ENV_KEYS = (
    "VLLM_FLASH_V100_KERNEL_BLOCK_SIZE16",
    "VLLM_FLASH_V100_DENSE_D256_LOW_SMEM",
    "VLLM_FLASH_V100_DENSE_D256_WMMA_QK",
    "VLLM_FLASH_V100_PREFILL_D256_LOW_SMEM",
    "VLLM_FLASH_V100_PREFILL_D256_SCALAR_QK",
    "VLLM_FLASH_V100_PREFILL_D256_BM32",
    "VLLM_FLASH_V100_PREFILL_CONTIG_DENSE",
    "VLLM_FLASH_V100_PREFILL_CONTIG_DENSE_ALLOW_COPY",
    "VLLM_FLASH_V100_PREFILL_CONTIG_DENSE_MIN_Q",
    "VLLM_FLASH_V100_PREFILL_CONTIG_DENSE_MIN_KV",
    "VLLM_FLASH_V100_PREFILL_USE_PAGED_CACHE",
    "VLLM_FLASH_V100_PREFILL_CONTIG_FAST",
    "VLLM_FLASH_V100_PREFILL_SPLIT_KV",
    "VLLM_FLASH_V100_PREFILL_SPLIT_KV_TOKENS",
    "VLLM_FLASH_V100_BFLA_PREFILL",
    "VLLM_FLASH_V100_BFLA_MIN_Q",
    "VLLM_FLASH_V100_BFLA_MIN_KV",
    "VLLM_FLASH_V100_BFLA_MASK_BLOCK_N",
    "VLLM_FLASH_V100_BFLA_KEEP_MASS",
    "VLLM_FLASH_V100_BFLA_KEEP_RATIO",
    "VLLM_FLASH_V100_BFLA_MIN_KEEP_BLOCKS",
    "VLLM_FLASH_V100_BFLA_THRESHOLD",
    "VLLM_FLASH_V100_BFLA_LOCAL_BLOCKS",
    "VLLM_FLASH_V100_BFLA_POOL",
    "VLLM_FLASH_V100_BFLA_SPEC_STRIDE",
    "VLLM_FLASH_V100_BFLA_SPEC_PROB",
    "VLLM_FLASH_V100_BFLA_SPEC_SEED",
    "VLLM_FLASH_V100_ROUTE_SUMMARY",
)


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


def _serialize_logprobs(logprobs: Any) -> Any:
    if logprobs is None:
        return None
    serialized: list[dict[str, Any] | None] = []
    for step in logprobs:
        if step is None:
            serialized.append(None)
            continue
        items = []
        for token_id, value in step.items():
            items.append(
                {
                    "token_id": int(token_id),
                    "logprob": float(getattr(value, "logprob", value)),
                    "rank": getattr(value, "rank", None),
                    "decoded_token": getattr(value, "decoded_token", None),
                }
            )
        items.sort(key=lambda item: item["logprob"], reverse=True)
        serialized.append({"top": items})
    return serialized


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

    torch.accelerator.synchronize()
    start = time.perf_counter()
    outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
    torch.accelerator.synchronize()
    elapsed_s = time.perf_counter() - start

    request_output = outputs[0]
    output_ids = list(request_output.outputs[0].token_ids)
    output_logprobs = _serialize_logprobs(
        getattr(request_output.outputs[0], "logprobs", None)
    )
    metrics = _request_metrics(request_output, elapsed_s, input_len)
    return {
        "variant": variant,
        "input_len": input_len,
        "prompt_hash": _hash_ids(token_ids),
        "output_tokens": len(output_ids),
        "output_ids": output_ids,
        "output_hash": _hash_ids(output_ids),
        "output_logprobs": output_logprobs,
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


def _env_snapshot() -> dict[str, str | None]:
    return {key: os.environ.get(key) for key in _FLASH_V100_ENV_KEYS}


def _write_result(
    args: argparse.Namespace,
    load_s: float,
    records: list[dict[str, Any]],
) -> None:
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
            "tensor_parallel_size": args.tensor_parallel_size,
            "max_model_len": args.max_model_len,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "quantization": args.quantization,
            "kv_cache_dtype": args.kv_cache_dtype,
            "attention_backend": "FLASH_ATTN_V100",
            "enforce_eager": args.enforce_eager,
            "logprobs": args.logprobs,
            "max_tokens": args.max_tokens,
            "flash_v100_env": _env_snapshot(),
        },
        "records": records,
        "summaries": summaries,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, default=Path("/home/ymzx/models/Qwen3.5-9B-AWQ")
    )
    parser.add_argument(
        "--lengths", type=int, nargs="+", default=[8192, 32768, 65536, 131072, 261120]
    )
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--long-repeat", type=int, default=1)
    parser.add_argument("--long-threshold", type=int, default=200000)
    parser.add_argument("--warmup-len", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=262144)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260618)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--quantization", default="awq")
    parser.add_argument("--kv-cache-dtype", default="auto")
    parser.add_argument("--safetensors-load-strategy", type=str, default=None)
    parser.add_argument("--logprobs", type=int, default=0)
    parser.add_argument("--bfla-min-q", type=int, default=None)
    parser.add_argument("--bfla-min-kv", type=int, default=None)
    parser.add_argument("--bfla-mask-block-n", type=int, default=None)
    parser.add_argument("--bfla-keep-mass", type=float, default=None)
    parser.add_argument("--bfla-keep-ratio", type=float, default=None)
    parser.add_argument("--bfla-min-keep-blocks", type=int, default=None)
    parser.add_argument("--bfla-threshold", type=float, default=None)
    parser.add_argument("--bfla-local-blocks", type=int, default=None)
    parser.add_argument("--bfla-pool", type=str, default=None)
    parser.add_argument("--bfla-spec-stride", type=int, default=None)
    parser.add_argument("--bfla-spec-prob", type=float, default=None)
    parser.add_argument("--bfla-spec-seed", type=int, default=None)
    parser.add_argument(
        "--variant",
        choices=(
            "baseline",
            "low_smem",
            "dense_low_smem",
            "all_low_smem",
            "dense_wmma",
            "all_wmma",
            "kbs16",
            "bfla_allkeep",
            "bfla_sparse",
            "bfla_ratio05",
            "bfla_ratio10",
            "bfla_local32",
        ),
        required=True,
    )
    return parser.parse_args()


def _configure_variant(variant: str) -> None:
    os.environ.pop("VLLM_FLASH_V100_KERNEL_BLOCK_SIZE16", None)
    os.environ.pop("VLLM_FLASH_V100_PREFILL_CONTIG_FAST", None)
    os.environ.pop("VLLM_FLASH_V100_DENSE_D256_LOW_SMEM", None)
    os.environ.pop("VLLM_FLASH_V100_DENSE_D256_WMMA_QK", None)
    os.environ.pop("VLLM_FLASH_V100_PREFILL_D256_SCALAR_QK", None)
    for key in _FLASH_V100_ENV_KEYS:
        if key.startswith("VLLM_FLASH_V100_BFLA_"):
            os.environ.pop(key, None)
    if variant == "low_smem":
        os.environ["VLLM_FLASH_V100_DENSE_D256_WMMA_QK"] = "0"
        os.environ["VLLM_FLASH_V100_PREFILL_D256_LOW_SMEM"] = "1"
    elif variant == "dense_low_smem":
        os.environ["VLLM_FLASH_V100_DENSE_D256_WMMA_QK"] = "0"
        os.environ.pop("VLLM_FLASH_V100_PREFILL_D256_LOW_SMEM", None)
        os.environ["VLLM_FLASH_V100_DENSE_D256_LOW_SMEM"] = "1"
    elif variant == "all_low_smem":
        os.environ["VLLM_FLASH_V100_DENSE_D256_WMMA_QK"] = "0"
        os.environ["VLLM_FLASH_V100_PREFILL_D256_LOW_SMEM"] = "1"
        os.environ["VLLM_FLASH_V100_DENSE_D256_LOW_SMEM"] = "1"
    elif variant == "dense_wmma":
        os.environ.pop("VLLM_FLASH_V100_PREFILL_D256_LOW_SMEM", None)
        os.environ["VLLM_FLASH_V100_DENSE_D256_WMMA_QK"] = "1"
    elif variant == "all_wmma":
        os.environ["VLLM_FLASH_V100_PREFILL_D256_LOW_SMEM"] = "1"
        os.environ["VLLM_FLASH_V100_DENSE_D256_WMMA_QK"] = "1"
    elif variant == "kbs16":
        os.environ["VLLM_FLASH_V100_DENSE_D256_WMMA_QK"] = "0"
        os.environ["VLLM_FLASH_V100_PREFILL_D256_LOW_SMEM"] = "1"
        os.environ["VLLM_FLASH_V100_KERNEL_BLOCK_SIZE16"] = "1"
    elif variant == "bfla_allkeep":
        os.environ["VLLM_FLASH_V100_DENSE_D256_WMMA_QK"] = "0"
        os.environ["VLLM_FLASH_V100_PREFILL_D256_LOW_SMEM"] = "1"
        os.environ["VLLM_FLASH_V100_BFLA_PREFILL"] = "1"
        os.environ["VLLM_FLASH_V100_BFLA_KEEP_MASS"] = "1.0"
        os.environ["VLLM_FLASH_V100_BFLA_THRESHOLD"] = "0.0"
    elif variant == "bfla_sparse":
        os.environ["VLLM_FLASH_V100_DENSE_D256_WMMA_QK"] = "0"
        os.environ["VLLM_FLASH_V100_PREFILL_D256_LOW_SMEM"] = "1"
        os.environ["VLLM_FLASH_V100_BFLA_PREFILL"] = "1"
    elif variant == "bfla_ratio05":
        os.environ["VLLM_FLASH_V100_DENSE_D256_WMMA_QK"] = "0"
        os.environ["VLLM_FLASH_V100_PREFILL_D256_LOW_SMEM"] = "1"
        os.environ["VLLM_FLASH_V100_BFLA_PREFILL"] = "1"
        os.environ["VLLM_FLASH_V100_BFLA_KEEP_RATIO"] = "0.05"
    elif variant == "bfla_ratio10":
        os.environ["VLLM_FLASH_V100_DENSE_D256_WMMA_QK"] = "0"
        os.environ["VLLM_FLASH_V100_PREFILL_D256_LOW_SMEM"] = "1"
        os.environ["VLLM_FLASH_V100_BFLA_PREFILL"] = "1"
        os.environ["VLLM_FLASH_V100_BFLA_KEEP_RATIO"] = "0.10"
    elif variant == "bfla_local32":
        os.environ["VLLM_FLASH_V100_DENSE_D256_WMMA_QK"] = "0"
        os.environ["VLLM_FLASH_V100_PREFILL_D256_LOW_SMEM"] = "1"
        os.environ["VLLM_FLASH_V100_BFLA_PREFILL"] = "1"
        os.environ["VLLM_FLASH_V100_BFLA_LOCAL_BLOCKS"] = "32"
    elif variant == "baseline":
        os.environ["VLLM_FLASH_V100_DENSE_D256_WMMA_QK"] = "0"
        os.environ.pop("VLLM_FLASH_V100_PREFILL_D256_LOW_SMEM", None)
    else:
        raise ValueError(f"unknown variant: {variant}")


def main() -> None:
    args = _parse_args()
    _configure_variant(args.variant)
    if args.bfla_min_q is not None:
        os.environ["VLLM_FLASH_V100_BFLA_MIN_Q"] = str(args.bfla_min_q)
    if args.bfla_min_kv is not None:
        os.environ["VLLM_FLASH_V100_BFLA_MIN_KV"] = str(args.bfla_min_kv)
    if args.bfla_mask_block_n is not None:
        os.environ["VLLM_FLASH_V100_BFLA_MASK_BLOCK_N"] = str(args.bfla_mask_block_n)
    if args.bfla_keep_mass is not None:
        os.environ["VLLM_FLASH_V100_BFLA_KEEP_MASS"] = str(args.bfla_keep_mass)
    if args.bfla_keep_ratio is not None:
        os.environ["VLLM_FLASH_V100_BFLA_KEEP_RATIO"] = str(args.bfla_keep_ratio)
    if args.bfla_min_keep_blocks is not None:
        os.environ["VLLM_FLASH_V100_BFLA_MIN_KEEP_BLOCKS"] = str(
            args.bfla_min_keep_blocks
        )
    if args.bfla_threshold is not None:
        os.environ["VLLM_FLASH_V100_BFLA_THRESHOLD"] = str(args.bfla_threshold)
    if args.bfla_local_blocks is not None:
        os.environ["VLLM_FLASH_V100_BFLA_LOCAL_BLOCKS"] = str(args.bfla_local_blocks)
    if args.bfla_pool is not None:
        os.environ["VLLM_FLASH_V100_BFLA_POOL"] = str(args.bfla_pool)
    if args.bfla_spec_stride is not None:
        os.environ["VLLM_FLASH_V100_BFLA_SPEC_STRIDE"] = str(args.bfla_spec_stride)
    if args.bfla_spec_prob is not None:
        os.environ["VLLM_FLASH_V100_BFLA_SPEC_PROB"] = str(args.bfla_spec_prob)
    if args.bfla_spec_seed is not None:
        os.environ["VLLM_FLASH_V100_BFLA_SPEC_SEED"] = str(args.bfla_spec_seed)

    from vllm import LLM, SamplingParams

    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        ignore_eos=True,
        skip_special_tokens=False,
        logprobs=args.logprobs or None,
    )

    load_start = time.perf_counter()
    llm = LLM(
        model=str(args.model),
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="half",
        quantization=args.quantization,
        kv_cache_dtype=args.kv_cache_dtype,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        attention_backend="FLASH_ATTN_V100",
        enforce_eager=args.enforce_eager,
        safetensors_load_strategy=args.safetensors_load_strategy,
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
    _write_result(args, load_s, records)

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
            if ttft and tps:
                message = (
                    f"{args.variant} len={input_len} rep={rep} "
                    f"ttft={ttft:.6f}s prefill_tps={tps:.1f}"
                )
            else:
                message = (
                    f"{args.variant} len={input_len} rep={rep} "
                    f"metrics={record['metrics']}"
                )
            print(message, flush=True)
            _write_result(args, load_s, records)

    _write_result(args, load_s, records)


if __name__ == "__main__":
    main()
