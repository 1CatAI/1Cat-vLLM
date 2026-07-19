# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""LongBench quality runner for Flash-V100 BFLA experiments.

This runner uses the official LongBench prompt templates and metrics, but
performs generation through vLLM so dense Flash-V100 and BFLA can be compared
under identical prompts, seeds, model, and decoding settings.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

LONG_BENCH_ROOT = Path("third_party/LongBench/LongBench")
NO_CHAT_DATASETS = {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}


def _configure_flash_variant(args: argparse.Namespace) -> None:
    for key in tuple(os.environ):
        if key.startswith("VLLM_FLASH_V100_BFLA_"):
            os.environ.pop(key, None)
    os.environ["VLLM_FLASH_V100_DENSE_D256_WMMA_QK"] = "0"
    os.environ["VLLM_FLASH_V100_PREFILL_D256_LOW_SMEM"] = "1"

    if args.variant == "low_smem":
        return
    if args.variant != "bfla_sparse":
        raise ValueError(f"unknown variant: {args.variant}")

    os.environ["VLLM_FLASH_V100_BFLA_PREFILL"] = "1"
    os.environ["VLLM_FLASH_V100_BFLA_MIN_Q"] = str(args.bfla_min_q)
    os.environ["VLLM_FLASH_V100_BFLA_MIN_KV"] = str(args.bfla_min_kv)
    os.environ["VLLM_FLASH_V100_BFLA_MASK_BLOCK_N"] = str(args.bfla_mask_block_n)
    os.environ["VLLM_FLASH_V100_BFLA_KEEP_MASS"] = str(args.bfla_keep_mass)
    if args.bfla_keep_ratio is not None:
        os.environ["VLLM_FLASH_V100_BFLA_KEEP_RATIO"] = str(args.bfla_keep_ratio)
    os.environ["VLLM_FLASH_V100_BFLA_MIN_KEEP_BLOCKS"] = str(args.bfla_min_keep_blocks)
    os.environ["VLLM_FLASH_V100_BFLA_THRESHOLD"] = str(args.bfla_threshold)
    os.environ["VLLM_FLASH_V100_BFLA_LOCAL_BLOCKS"] = str(args.bfla_local_blocks)
    os.environ["VLLM_FLASH_V100_BFLA_POOL"] = args.bfla_pool
    os.environ["VLLM_FLASH_V100_BFLA_SPEC_STRIDE"] = str(args.bfla_spec_stride)
    os.environ["VLLM_FLASH_V100_BFLA_SPEC_PROB"] = str(args.bfla_spec_prob)
    os.environ["VLLM_FLASH_V100_BFLA_SPEC_SEED"] = str(args.bfla_spec_seed)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_dataset(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for idx, line in enumerate(f):
            row = json.loads(line)
            row["_source_index"] = idx
            rows.append(row)
    return rows


def _select_rows(
    rows: list[dict[str, Any]],
    *,
    min_length: int,
    limit: int | None,
    strategy: str,
    source_indexes: set[int] | None = None,
) -> list[dict[str, Any]]:
    selected = [row for row in rows if int(row.get("length", 0)) >= min_length]
    if source_indexes is not None:
        return [row for row in selected if int(row["_source_index"]) in source_indexes]
    if strategy == "longest":
        selected.sort(key=lambda row: int(row.get("length", 0)), reverse=True)
    elif strategy == "first":
        pass
    elif strategy == "strided":
        if limit is not None and 0 < limit < len(selected):
            if limit == 1:
                selected = [selected[len(selected) // 2]]
            else:
                denom = limit - 1
                last = len(selected) - 1
                selected = [selected[round(i * last / denom)] for i in range(limit)]
    else:
        raise ValueError(f"unknown sample strategy: {strategy}")
    if limit is not None and strategy != "strided":
        selected = selected[:limit]
    return selected


def _truncate_middle(tokenizer: Any, prompt: str, max_input_tokens: int) -> str:
    token_ids = tokenizer(prompt, truncation=False, add_special_tokens=False).input_ids
    if len(token_ids) <= max_input_tokens:
        return prompt
    half = max_input_tokens // 2
    left = tokenizer.decode(token_ids[:half], skip_special_tokens=True)
    right = tokenizer.decode(token_ids[-half:], skip_special_tokens=True)
    return left + right


def _build_prompt(
    tokenizer: Any,
    *,
    dataset: str,
    row: dict[str, Any],
    prompt_format: str,
    max_input_tokens: int,
    chat_template: str,
) -> tuple[str, int]:
    raw_prompt = prompt_format.format(**row)
    target_tokens = max_input_tokens
    for _ in range(8):
        prompt = _truncate_middle(tokenizer, raw_prompt, target_tokens)
        if chat_template == "always" or (
            chat_template == "official" and dataset not in NO_CHAT_DATASETS
        ):
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        final_ids = tokenizer(
            prompt,
            truncation=False,
            add_special_tokens=False,
        ).input_ids
        if len(final_ids) <= max_input_tokens:
            return prompt, len(final_ids)
        overage = len(final_ids) - max_input_tokens
        target_tokens = max(128, target_tokens - overage - 256)
    raise ValueError(
        f"failed to fit prompt for dataset={dataset} "
        f"source_index={row.get('_source_index')} under {max_input_tokens} tokens"
    )


def _score_one(dataset: str, prediction: str, row: dict[str, Any]) -> float:
    sys.path.insert(0, str(LONG_BENCH_ROOT.resolve()))
    from eval import dataset2metric  # type: ignore

    score = 0.0
    pred = prediction
    if dataset in {"trec", "triviaqa", "samsum", "lsht"}:
        pred = pred.lstrip("\n").split("\n")[0]
    for answer in row["answers"]:
        score = max(
            score,
            dataset2metric[dataset](
                pred,
                answer,
                all_classes=row.get("all_classes", []),
            ),
        )
    return float(score)


def _request_metrics(
    output: Any, elapsed_s: float, prompt_tokens: int
) -> dict[str, Any]:
    metrics = getattr(output, "metrics", None)
    first_token_latency = (
        getattr(metrics, "first_token_latency", None) if metrics else None
    )
    ttft_s = first_token_latency or elapsed_s
    return {
        "elapsed_s": elapsed_s,
        "ttft_s": ttft_s,
        "prefill_tps": prompt_tokens / ttft_s if ttft_s else None,
    }


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [float(record["score"]) for record in records]
    ttfts = [float(record["metrics"]["ttft_s"]) for record in records]
    prompt_tokens = [int(record["prompt_tokens"]) for record in records]
    route_hits = sum(1 for record in records if record["prompt_tokens"] > 8192)
    return {
        "samples": len(records),
        "score": round(100.0 * statistics.mean(scores), 4) if scores else None,
        "score_values": scores,
        "prompt_tokens_min": min(prompt_tokens) if prompt_tokens else None,
        "prompt_tokens_median": statistics.median(prompt_tokens)
        if prompt_tokens
        else None,
        "prompt_tokens_max": max(prompt_tokens) if prompt_tokens else None,
        "samples_over_8k_tokens": route_hits,
        "ttft_s_median": statistics.median(ttfts) if ttfts else None,
        "ttft_s_mean": statistics.mean(ttfts) if ttfts else None,
    }


def _write_outputs(
    out_dir: Path,
    args: argparse.Namespace,
    records: list[dict[str, Any]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_dataset.setdefault(record["dataset"], []).append(record)

    settings = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    summary = {
        "settings": settings,
        "env": {
            key: value
            for key, value in os.environ.items()
            if key.startswith("VLLM_FLASH_V100")
        },
        "datasets": {
            dataset: _summarize(items) for dataset, items in sorted(by_dataset.items())
        },
    }
    all_scores = [
        dataset_summary["score"]
        for dataset_summary in summary["datasets"].values()
        if dataset_summary.get("score") is not None
    ]
    summary["average_score"] = (
        round(statistics.mean(all_scores), 4) if all_scores else None
    )
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for dataset, items in by_dataset.items():
        with (out_dir / f"{dataset}.jsonl").open("w", encoding="utf-8") as f:
            for record in items:
                f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _read_existing_records(out_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not out_dir.exists():
        return records
    for path in sorted(out_dir.glob("*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, default=Path("/home/ymzx/models/Qwen3.5-9B-AWQ")
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("benchmark-data/longbench/data")
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-length", type=int, default=0)
    parser.add_argument(
        "--source-indexes",
        type=str,
        default="",
        help=(
            "Comma-separated dataset source indexes to run after min-length filtering."
        ),
    )
    parser.add_argument(
        "--sample-strategy",
        choices=("first", "longest", "strided"),
        default="longest",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--max-input-tokens", type=int, default=65536)
    parser.add_argument("--max-model-len", type=int, default=65536)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.86)
    parser.add_argument("--quantization", default="awq")
    parser.add_argument("--kv-cache-dtype", default="auto")
    parser.add_argument("--safetensors-load-strategy", default=None)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--variant", choices=("low_smem", "bfla_sparse"), required=True)
    parser.add_argument(
        "--chat-template", choices=("official", "always", "none"), default="official"
    )
    parser.add_argument("--seed", type=int, default=20260619)
    parser.add_argument("--bfla-min-q", type=int, default=256)
    parser.add_argument("--bfla-min-kv", type=int, default=256)
    parser.add_argument("--bfla-mask-block-n", type=int, default=256)
    parser.add_argument("--bfla-keep-mass", type=float, default=0.99)
    parser.add_argument("--bfla-keep-ratio", type=float, default=None)
    parser.add_argument("--bfla-min-keep-blocks", type=int, default=0)
    parser.add_argument("--bfla-threshold", type=float, default=999.0)
    parser.add_argument("--bfla-local-blocks", type=int, default=8)
    parser.add_argument("--bfla-pool", default="flat64")
    parser.add_argument("--bfla-spec-stride", type=int, default=0)
    parser.add_argument("--bfla-spec-prob", type=float, default=0.0)
    parser.add_argument("--bfla-spec-seed", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _configure_flash_variant(args)
    source_indexes = (
        {int(value) for value in args.source_indexes.split(",") if value.strip()}
        if args.source_indexes
        else None
    )

    from vllm import LLM, SamplingParams

    dataset2prompt = _load_json(LONG_BENCH_ROOT / "config/dataset2prompt.json")
    dataset2maxlen = _load_json(LONG_BENCH_ROOT / "config/dataset2maxlen.json")
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model),
        trust_remote_code=True,
        use_fast=True,
    )

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
        disable_log_stats=True,
        disable_custom_all_reduce=True,
        seed=args.seed,
        enforce_eager=args.enforce_eager,
        safetensors_load_strategy=args.safetensors_load_strategy,
    )

    records: list[dict[str, Any]] = (
        _read_existing_records(args.out_dir) if args.skip_existing else []
    )
    completed = {
        (str(record["dataset"]), int(record["source_index"])) for record in records
    }
    for dataset in args.datasets:
        rows = _load_dataset(args.data_dir / f"{dataset}.jsonl")
        selected = _select_rows(
            rows,
            min_length=args.min_length,
            limit=args.limit,
            strategy=args.sample_strategy,
            source_indexes=source_indexes,
        )
        sampling_params = SamplingParams(
            max_tokens=int(dataset2maxlen[dataset]),
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            skip_special_tokens=True,
        )
        for ordinal, row in enumerate(selected):
            record_key = (dataset, int(row["_source_index"]))
            if record_key in completed:
                print(
                    f"{args.variant} {dataset} {ordinal + 1}/{len(selected)} "
                    f"source_index={row['_source_index']} skipped",
                    flush=True,
                )
                continue
            prompt, prompt_tokens = _build_prompt(
                tokenizer,
                dataset=dataset,
                row=row,
                prompt_format=dataset2prompt[dataset],
                max_input_tokens=args.max_input_tokens,
                chat_template=args.chat_template,
            )
            torch.accelerator.synchronize()
            start = time.perf_counter()
            outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
            torch.accelerator.synchronize()
            elapsed_s = time.perf_counter() - start

            output = outputs[0]
            completion = output.outputs[0]
            prediction = completion.text
            score = _score_one(dataset, prediction, row)
            record = {
                "variant": args.variant,
                "dataset": dataset,
                "ordinal": ordinal,
                "source_index": row["_source_index"],
                "id": row.get("_id"),
                "length": row.get("length"),
                "prompt_tokens": prompt_tokens,
                "pred": prediction,
                "answers": row["answers"],
                "all_classes": row.get("all_classes", []),
                "score": score,
                "output_tokens": len(completion.token_ids),
                "finish_reason": completion.finish_reason,
                "metrics": _request_metrics(output, elapsed_s, prompt_tokens),
            }
            records.append(record)
            completed.add(record_key)
            _write_outputs(args.out_dir, args, records)
            print(
                f"{args.variant} {dataset} {ordinal + 1}/{len(selected)} "
                f"len={row.get('length')} tokens={prompt_tokens} "
                f"score={100 * score:.2f} ttft={record['metrics']['ttft_s']:.3f}s",
                flush=True,
            )

    _write_outputs(args.out_dir, args, records)


if __name__ == "__main__":
    main()
