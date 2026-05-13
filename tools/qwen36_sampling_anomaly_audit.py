#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Sampling anomaly audit for Qwen3.6 27B on V100.

This is a regression harness, not a benchmark. It keeps prompts and sampling
fixed while switching attention backend / LM-head fast path in separate child
processes, then records repetition and punctuation anomaly metrics.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def make_base_text() -> str:
    return (
        "1Cat vLLM V100 regression prompt. "
        "请严格比较注意力后端在长上下文、中文、英文和代码混合输入下的行为。 "
        "The model should continue deterministically and preserve factual style. "
        "def attention_backend_check(x): return x * 2\n"
    )


def make_prompt_by_tokens(tokenizer: Any, target_len: int) -> str:
    unit = make_base_text()
    unit_tokens = max(1, len(tokenizer.encode(unit, add_special_tokens=False)))
    text = unit * max(4, target_len // unit_tokens + 4)
    lo = 0
    hi = len(text)
    best = ""
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = text[:mid]
        length = len(tokenizer.encode(candidate, add_special_tokens=False))
        if length <= target_len:
            best = candidate
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def apply_chat(tokenizer: Any, prompt: str, enable_thinking: bool) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def build_prompts(tokenizer: Any, enable_thinking: bool,
                  long_tokens: int) -> list[dict[str, str]]:
    raw = [
        {
            "id": "minimal_exact_ok_zh",
            "prompt": "只输出OK",
        },
        {
            "id": "minimal_exact_ok_en",
            "prompt": "Say exactly OK",
        },
        {
            "id": "zh_no_exclaim",
            "prompt": (
                "请用自然、克制的中文回答：为什么长上下文推理需要同时关注 "
                "prefill 和 decode？请不要使用感叹号。"
            ),
        },
        {
            "id": "en_no_exclaim",
            "prompt": (
                "Write a calm technical explanation of why deterministic "
                "decoding is useful for backend regression. Do not use "
                "exclamation marks."
            ),
        },
        {
            "id": "code_summary",
            "prompt": (
                "写一个 Python 函数 summarize_latency(samples)，返回 p50、p90 "
                "和 tokens_per_second，并解释边界情况。"
            ),
        },
        {
            "id": "math_reasoning",
            "prompt": (
                "Solve this carefully and give the final integer at the end: "
                "(137 * 29 + 418) - (56 * 7)."
            ),
        },
        {
            "id": "long_repeat_resistance",
            "prompt": (
                make_prompt_by_tokens(tokenizer, long_tokens)
                + "\n\n最后只用三句话总结上面的测试主题，不要继续复制原文。"
            ),
        },
        {
            "id": "punctuation_stability",
            "prompt": (
                "Generate a 400-token incident report about a model output "
                "quality regression. Use varied sentence structure and avoid "
                "repeating punctuation."
            ),
        },
    ]
    return [{
        "id": item["id"],
        "prompt": apply_chat(tokenizer, item["prompt"], enable_thinking),
    } for item in raw]


def max_run(values: list[Any]) -> int:
    best = 0
    current = object()
    run = 0
    for value in values:
        if value == current:
            run += 1
        else:
            current = value
            run = 1
        best = max(best, run)
    return best


def max_ngram_count(tokens: list[int], n: int) -> int:
    if len(tokens) < n:
        return 0
    grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return max(collections.Counter(grams).values(), default=0)


def anomaly_metrics(text: str, token_ids: list[int]) -> dict[str, Any]:
    chars = [ch for ch in text if not ch.isspace()]
    exclaims = sum(1 for ch in chars if ch in {"!", "！"})
    punct = sum(1 for ch in chars if ch in set("!！?？。.,，;；:：、"))
    unique_ratio = len(set(token_ids)) / len(token_ids) if token_ids else 1.0
    metrics = {
        "char_count": len(chars),
        "output_tokens": len(token_ids),
        "exclamation_count": exclaims,
        "exclamation_ratio": exclaims / max(1, len(chars)),
        "punctuation_ratio": punct / max(1, len(chars)),
        "max_char_run": max_run(chars),
        "max_token_run": max_run(token_ids),
        "max_bigram_count": max_ngram_count(token_ids, 2),
        "max_4gram_count": max_ngram_count(token_ids, 4),
        "unique_token_ratio": unique_ratio,
    }
    metrics["flagged"] = bool(
        metrics["exclamation_ratio"] >= 0.30
        or metrics["max_char_run"] >= 20
        or metrics["max_token_run"] >= 20
        or metrics["max_4gram_count"] >= 24
        or (len(token_ids) >= 128 and unique_ratio <= 0.08)
    )
    return metrics


def child_main(args: argparse.Namespace) -> int:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=True,
    )
    prompts = build_prompts(tokenizer, args.enable_thinking, args.long_tokens)
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
        trust_remote_code=True,
        seed=args.seed,
        enable_prefix_caching=False,
        attention_config={"backend": args.backend},
        compilation_config={
            "cudagraph_mode": "full_and_piecewise",
            "cudagraph_capture_sizes": [1, 2, 4, 8],
        },
        disable_log_stats=True,
        limit_mm_per_prompt={"image": 0, "video": 0},
        mm_processor_cache_gb=0,
        skip_mm_profiling=True,
    )

    rows = []
    warmup = SamplingParams(temperature=0, max_tokens=1, ignore_eos=True)
    llm.generate(["sampling anomaly warmup"], warmup, use_tqdm=False)
    for seed in parse_csv_ints(args.seeds):
        sampling = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_tokens=args.max_tokens,
            seed=seed,
        )
        start = time.perf_counter()
        outputs = llm.generate(
            [item["prompt"] for item in prompts],
            sampling,
            use_tqdm=False,
        )
        batch_latency = time.perf_counter() - start
        for item, output in zip(prompts, outputs, strict=True):
            completion = output.outputs[0]
            token_ids = [int(token_id) for token_id in completion.token_ids]
            text = completion.text
            metrics = anomaly_metrics(text, token_ids)
            rows.append({
                "seed": seed,
                "case": item["id"],
                "prompt_tokens": len(output.prompt_token_ids),
                "finish_reason": completion.finish_reason,
                "stop_reason": completion.stop_reason,
                "batch_latency_sec": batch_latency,
                "metrics": metrics,
                "output_prefix": normalize(text)[:600],
                "output_text": text,
                "output_token_ids": token_ids,
            })
            print(json.dumps({
                "backend": args.backend,
                "seed": seed,
                "case": item["id"],
                "flagged": metrics["flagged"],
                "out_tokens": metrics["output_tokens"],
                "exclamation_ratio": metrics["exclamation_ratio"],
                "max_4gram_count": metrics["max_4gram_count"],
                "unique_token_ratio": metrics["unique_token_ratio"],
                "prefix": normalize(text)[:120],
            }, ensure_ascii=False), flush=True)

    flagged = [row for row in rows if row["metrics"]["flagged"]]
    payload = {
        "backend": args.backend,
        "model": args.model,
        "kv_cache_dtype": args.kv_cache_dtype,
        "lm_head_fastpath": args.lm_head_fastpath,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens": args.max_tokens,
        "seeds": parse_csv_ints(args.seeds),
        "summary": {
            "total": len(rows),
            "flagged": len(flagged),
            "flagged_cases": [
                {"seed": row["seed"], "case": row["case"]}
                for row in flagged
            ],
            "median_output_tokens": statistics.median(
                row["metrics"]["output_tokens"] for row in rows),
        },
        "rows": rows,
    }
    args.child_output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


def run_child(args: argparse.Namespace, backend: str,
              lm_head_fastpath: str, output_dir: Path) -> dict[str, Any]:
    label = f"{backend}_lmhead{lm_head_fastpath}"
    child_output = output_dir / f"{label}.json"
    child_log = output_dir / f"{label}.log"
    env = os.environ.copy()
    env.update({
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": args.cuda_visible_devices,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "VLLM_USE_V1": "1",
        "VLLM_ATTENTION_BACKEND": backend,
        "VLLM_SM70_ENABLE_LM_HEAD_FASTPATH": lm_head_fastpath,
    })
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--backend",
        backend,
        "--lm-head-fastpath",
        lm_head_fastpath,
        "--model",
        args.model,
        "--dtype",
        args.dtype,
        "--kv-cache-dtype",
        args.kv_cache_dtype,
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--seeds",
        args.seeds,
        "--seed",
        str(args.seed),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--top-k",
        str(args.top_k),
        "--max-tokens",
        str(args.max_tokens),
        "--long-tokens",
        str(args.long_tokens),
        "--child-output",
        str(child_output),
    ]
    if args.disable_custom_all_reduce:
        cmd.append("--disable-custom-all-reduce")
    if not args.enable_thinking:
        cmd.append("--disable-thinking")
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
    if proc.returncode != 0:
        raise RuntimeError(
            f"{label} failed with exit={proc.returncode}; see {child_log}")
    data = json.loads(child_output.read_text(encoding="utf-8"))
    data["log_path"] = str(child_log)
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--backend", default="FLASH_ATTN_V100")
    parser.add_argument("--backends", default="FLASH_ATTN_V100,TRITON_ATTN")
    parser.add_argument("--lm-head-fastpath", default="1")
    parser.add_argument("--lm-head-fastpaths", default="1")
    parser.add_argument("--model", default="/home/ymzx/models/Qwen3.6-27B-AWQ")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--kv-cache-dtype", default="auto")
    parser.add_argument("--cuda-visible-devices", default="1,2,3,4")
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--disable-custom-all-reduce", action="store_true")
    parser.add_argument("--disable-thinking", action="store_false",
                        dest="enable_thinking")
    parser.add_argument("--enable-thinking", action="store_true",
                        dest="enable_thinking")
    parser.set_defaults(enable_thinking=False)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--seed", type=int, default=20260508)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--long-tokens", type=int, default=8192)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--child-output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.child:
        if args.child_output is None:
            raise SystemExit("--child requires --child-output")
        return child_main(args)

    output_dir = args.output_dir or Path(
        f"/tmp/qwen36_sampling_anomaly_{time.strftime('%Y%m%d_%H%M%S')}")
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for backend in [item.strip() for item in args.backends.split(",") if item.strip()]:
        for fastpath in [
            item.strip() for item in args.lm_head_fastpaths.split(",")
            if item.strip()
        ]:
            print(
                f"running anomaly audit backend={backend} "
                f"lm_head_fastpath={fastpath}",
                flush=True,
            )
            results[f"{backend}_lmhead{fastpath}"] = run_child(
                args, backend, fastpath, output_dir)

    combined = {"results": results}
    combined_json = output_dir / "combined.json"
    combined_json.write_text(
        json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8")
    for label, data in results.items():
        summary = data["summary"]
        print(
            f"{label}: flagged={summary['flagged']}/{summary['total']} "
            f"cases={summary['flagged_cases']}")
    print(f"wrote {combined_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
