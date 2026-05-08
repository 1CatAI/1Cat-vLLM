#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Real vLLM backend regression for FLASH_ATTN_V100 vs TRITON_ATTN.

Run outside the source checkout when possible:

  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
  python /path/to/tools/vllm_v100_backend_regression.py \
    --model /home/ymzx/models/Qwen3.5-27B-AWQ \
    --prompt-style qwen35-chat
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


BACKENDS = ("FLASH_ATTN_V100", "TRITON_ATTN")


def make_base_text() -> str:
    return (
        "1Cat vLLM V100 regression prompt. "
        "请严格比较注意力后端在长上下文、中文、英文和代码混合输入下的行为。 "
        "The model should continue deterministically and preserve factual style. "
        "def attention_backend_check(x): return x * 2\n"
    )


def make_prompt_by_tokens(tokenizer: Any, target_len: int, suffix: str = "") -> str:
    unit = make_base_text() + suffix + "\n"
    unit_tokens = max(1, len(tokenizer.encode(unit, add_special_tokens=False)))
    text = unit * max(4, target_len // unit_tokens + 4)
    while len(tokenizer.encode(text, add_special_tokens=False)) < target_len:
        text += text

    # Tokenize final text prefixes instead of decoding arbitrary token slices.
    # Some tokenizers do not round-trip `decode(encode(text)[:N])` to N tokens.
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


def maybe_apply_chat_template(tokenizer: Any, user_content: str,
                              args: argparse.Namespace) -> str:
    if args.prompt_style == "raw":
        return user_content

    messages = [{"role": "user", "content": user_content}]
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


def build_quality_prompts(tokenizer: Any,
                          args: argparse.Namespace) -> list[dict[str, str]]:
    prompts = [
        {
            "name": "short_zh_reasoning",
            "prompt": "请用三句话分析 V100 上长上下文推理为什么要同时关注预填充速度和解码速度。",
        },
        {
            "name": "short_en_reasoning",
            "prompt": (
                "Explain why deterministic inference matters when comparing "
                "two attention backends on the same model."
            ),
        },
        {
            "name": "code",
            "prompt": (
                "写一个 Python 函数 summarize_latency(samples)，返回 p50、p90 "
                "和 tokens_per_second，并解释每一步。"
            ),
        },
        {
            "name": "medium_mixed",
            "prompt": make_prompt_by_tokens(tokenizer, 512, "medium"),
        },
        {
            "name": "long_prefill",
            "prompt": make_prompt_by_tokens(tokenizer, args.long_prompt_tokens,
                                            "long-prefill"),
        },
    ]
    return [
        {
            "name": item["name"],
            "prompt": maybe_apply_chat_template(tokenizer, item["prompt"], args),
        }
        for item in prompts
    ]


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
    items.sort(key=lambda x: (
        x["rank"] if x["rank"] is not None else 1_000_000,
        -x["logprob"],
    ))
    return items[:limit]


def selected_logprob(logprob_map: Any, token_id: int) -> float | None:
    if not logprob_map or token_id not in logprob_map:
        return None
    return float(getattr(logprob_map[token_id], "logprob"))


def serialize_request_output(output: Any,
                             include_prompt_token_ids: bool = True) -> dict[str, Any]:
    completion = output.outputs[0]
    prompt_token_ids = [int(x) for x in output.prompt_token_ids]
    output_token_ids = [int(x) for x in completion.token_ids]
    gen_logprobs = []
    for idx, token_id in enumerate(output_token_ids):
        logprob_map = None
        if completion.logprobs is not None and idx < len(completion.logprobs):
            logprob_map = completion.logprobs[idx]
        gen_logprobs.append({
            "token_id": token_id,
            "selected_logprob": selected_logprob(logprob_map, token_id),
            "top": serialize_logprob_map(logprob_map),
        })

    prompt_samples = []
    prompt_logprobs = getattr(output, "prompt_logprobs", None)
    if prompt_logprobs:
        sample_idxs = sorted({
            1,
            2,
            3,
            max(1, len(prompt_token_ids) // 2),
            max(1, len(prompt_token_ids) - 1),
        })
        for idx in sample_idxs:
            if idx < len(prompt_logprobs):
                prompt_samples.append({
                    "index": idx,
                    "token_id": prompt_token_ids[idx],
                    "selected_logprob": selected_logprob(
                        prompt_logprobs[idx], prompt_token_ids[idx]),
                    "top": serialize_logprob_map(prompt_logprobs[idx]),
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
        "generated_logprobs": gen_logprobs,
        "prompt_logprob_samples": prompt_samples,
    }


def child_main(args: argparse.Namespace) -> int:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    quality_prompts = []
    if not args.skip_quality:
        quality_prompts = build_quality_prompts(tokenizer, args)
        if args.long_only_quality:
            quality_prompts = [
                item for item in quality_prompts if item["name"] == "long_prefill"
            ]

    compilation_config = {
        "cudagraph_mode": "full_and_piecewise",
        "cudagraph_capture_sizes": [1, 2, 4, 8],
    }
    llm_kwargs = {}
    if args.disable_mm:
        llm_kwargs.update(
            limit_mm_per_prompt={
                "image": 0,
                "video": 0,
            },
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
        seed=0,
        enable_prefix_caching=False,
        enforce_eager=args.enforce_eager,
        attention_config={"backend": args.backend},
        compilation_config=compilation_config,
        disable_log_stats=True,
        **llm_kwargs,
    )

    quality_sampling = SamplingParams(
        temperature=0,
        top_p=1.0,
        max_tokens=args.quality_max_tokens,
        logprobs=args.logprobs if args.logprobs > 0 else None,
        prompt_logprobs=(
            args.prompt_logprobs if args.prompt_logprobs > 0 else None),
        seed=0,
    )
    warmup_sampling = SamplingParams(
        temperature=0,
        max_tokens=4,
        ignore_eos=True,
        seed=0,
    )
    llm.generate(["warmup prompt for backend regression"], warmup_sampling,
                 use_tqdm=False)

    quality = {}
    if quality_prompts:
        quality_outputs = llm.generate(
            [item["prompt"] for item in quality_prompts],
            quality_sampling,
            use_tqdm=False,
        )
        for item, output in zip(quality_prompts, quality_outputs, strict=True):
            quality[item["name"]] = serialize_request_output(
                output,
                include_prompt_token_ids=not args.omit_prompt_token_ids,
            )

    speed = {}
    if not args.skip_speed:
        if args.long_only_speed:
            scenarios = [
                {
                    "name": "batch1_prefill_long_decode1",
                    "lengths": [args.long_prompt_tokens],
                    "max_tokens": 1,
                    "iters": args.speed_iters,
                },
                {
                    "name": f"batch1_prefill_long_decode{args.long_decode_tokens}",
                    "lengths": [args.long_prompt_tokens],
                    "max_tokens": args.long_decode_tokens,
                    "iters": args.speed_iters,
                },
            ]
        else:
            scenarios = [
            {
                "name": "batch1_prefill512_decode1",
                "lengths": [512],
                "max_tokens": 1,
                "iters": args.speed_iters,
            },
            {
                "name": "batch1_prefill512_decode64",
                "lengths": [512],
                "max_tokens": 64,
                "iters": args.speed_iters,
            },
            {
                "name": "batch1_prefill_long_decode1",
                "lengths": [args.long_prompt_tokens],
                "max_tokens": 1,
                "iters": args.speed_iters,
            },
            {
                "name": "batch1_prefill_long_decode64",
                "lengths": [args.long_prompt_tokens],
                "max_tokens": 64,
                "iters": args.speed_iters,
            },
            {
                "name": "batch4_mixed_decode32",
                "lengths": [64, 128, 256, 512],
                "max_tokens": 32,
                "iters": args.speed_iters,
            },
            {
                "name": "batch4_decode_heavy128",
                "lengths": [64, 64, 64, 64],
                "max_tokens": 128,
                "iters": max(2, args.speed_iters // 2),
            },
            ]
        for scenario in scenarios:
            prompts = [
                maybe_apply_chat_template(
                    tokenizer,
                    make_prompt_by_tokens(tokenizer, length,
                                          f"{scenario['name']}-{idx}"),
                    args,
                )
                for idx, length in enumerate(scenario["lengths"])
            ]
            sampling = SamplingParams(
                temperature=0,
                top_p=1.0,
                max_tokens=scenario["max_tokens"],
                ignore_eos=True,
                seed=0,
            )
            for _ in range(args.speed_warmup):
                llm.generate(prompts, sampling, use_tqdm=False)
            latencies = []
            output_tokens = []
            for _ in range(scenario["iters"]):
                start = time.perf_counter()
                outputs = llm.generate(prompts, sampling, use_tqdm=False)
                latencies.append(time.perf_counter() - start)
                output_tokens.append(
                    sum(len(out.outputs[0].token_ids) for out in outputs))
            input_tokens = [
                len(tokenizer.encode(prompt, add_special_tokens=False))
                for prompt in prompts
            ]
            median = statistics.median(latencies)
            speed[scenario["name"]] = {
                "input_tokens": input_tokens,
                "max_tokens": scenario["max_tokens"],
                "iters": scenario["iters"],
                "latencies_sec": latencies,
                "median_sec": median,
                "mean_sec": statistics.mean(latencies),
                "min_sec": min(latencies),
                "max_sec": max(latencies),
                "median_output_tokens": statistics.median(output_tokens),
                "output_tokens_per_sec": statistics.median(output_tokens) / median,
                "total_tokens_per_sec": (
                    (sum(input_tokens) + statistics.median(output_tokens)) / median
                ),
            }

    result = {
        "backend": args.backend,
        "model": args.model,
        "dtype": args.dtype,
        "kv_cache_dtype": args.kv_cache_dtype,
        "tensor_parallel_size": args.tensor_parallel_size,
        "prompt_style": args.prompt_style,
        "enable_thinking": args.enable_thinking,
        "max_model_len": args.max_model_len,
        "quality": quality,
        "speed": speed,
    }
    args.child_output.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                                 encoding="utf-8")
    return 0


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
        "--quality-max-tokens",
        str(args.quality_max_tokens),
        "--long-prompt-tokens",
        str(args.long_prompt_tokens),
        "--logprobs",
        str(args.logprobs),
        "--prompt-logprobs",
        str(args.prompt_logprobs),
        "--speed-warmup",
        str(args.speed_warmup),
        "--speed-iters",
        str(args.speed_iters),
        "--long-decode-tokens",
        str(args.long_decode_tokens),
        "--child-output",
        str(child_json),
    ]
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")
    if args.enforce_eager:
        cmd.append("--enforce-eager")
    if args.skip_speed:
        cmd.append("--skip-speed")
    if args.skip_quality:
        cmd.append("--skip-quality")
    if not args.enable_thinking:
        cmd.append("--disable-thinking")
    if args.disable_mm:
        cmd.append("--disable-mm")
    if args.disable_custom_all_reduce:
        cmd.append("--disable-custom-all-reduce")
    if args.long_only_quality:
        cmd.append("--long-only-quality")
    if args.long_only_speed:
        cmd.append("--long-only-speed")
    if args.omit_prompt_token_ids:
        cmd.append("--omit-prompt-token-ids")
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
            f"{backend} child failed with exit code {proc.returncode}; "
            f"see {child_log}")
    data = json.loads(child_json.read_text(encoding="utf-8"))
    data["log_path"] = str(child_log)
    return data


def compare_top_tokens(a_top: list[dict[str, Any]],
                       b_top: list[dict[str, Any]]) -> dict[str, Any]:
    a_ids = [x["token_id"] for x in a_top]
    b_ids = [x["token_id"] for x in b_top]
    common = sorted(set(a_ids) & set(b_ids))
    max_common_delta = 0.0
    for token_id in common:
        av = next(x["logprob"] for x in a_top if x["token_id"] == token_id)
        bv = next(x["logprob"] for x in b_top if x["token_id"] == token_id)
        max_common_delta = max(max_common_delta, abs(av - bv))
    return {
        "top1_match": bool(a_ids and b_ids and a_ids[0] == b_ids[0]),
        "common_count": len(common),
        "max_common_logprob_delta": max_common_delta,
    }


def compare_quality(flash: dict[str, Any],
                    triton: dict[str, Any]) -> dict[str, Any]:
    comparisons = {}
    failures = []

    def valid_delta(a: float | None, b: float | None) -> tuple[float | None, bool]:
        if a is None or b is None:
            return None, True
        if not math.isfinite(a) or not math.isfinite(b):
            return None, False
        return abs(a - b), True

    for name, fa in flash["quality"].items():
        tr = triton["quality"][name]
        token_exact = fa["output_token_ids"] == tr["output_token_ids"]
        text_exact = fa["output_text"] == tr["output_text"]
        cum_a = fa["cumulative_logprob"]
        cum_b = tr["cumulative_logprob"]
        cum_delta, cum_finite = valid_delta(cum_a, cum_b)

        gen_deltas = []
        gen_nonfinite = 0
        top1_matches = []
        for fa_lp, tr_lp in zip(fa["generated_logprobs"],
                                tr["generated_logprobs"],
                                strict=False):
            delta, finite = valid_delta(fa_lp["selected_logprob"],
                                        tr_lp["selected_logprob"])
            if not finite:
                gen_nonfinite += 1
            elif delta is not None:
                gen_deltas.append(delta)
            top_cmp = compare_top_tokens(fa_lp["top"], tr_lp["top"])
            top1_matches.append(top_cmp["top1_match"])

        prompt_top1_matches = []
        prompt_deltas = []
        prompt_nonfinite = 0
        for fa_lp, tr_lp in zip(fa["prompt_logprob_samples"],
                                tr["prompt_logprob_samples"],
                                strict=False):
            delta, finite = valid_delta(fa_lp["selected_logprob"],
                                        tr_lp["selected_logprob"])
            if not finite:
                prompt_nonfinite += 1
            elif delta is not None:
                prompt_deltas.append(delta)
            prompt_top1_matches.append(
                compare_top_tokens(fa_lp["top"], tr_lp["top"])["top1_match"])

        result = {
            "token_exact": token_exact,
            "text_exact": text_exact,
            "cumulative_logprob_delta": cum_delta,
            "max_generated_selected_logprob_delta": (
                max(gen_deltas) if gen_deltas else None),
            "generated_top1_match_rate": (
                sum(top1_matches) / len(top1_matches)
                if top1_matches else None),
            "max_prompt_sample_selected_logprob_delta": (
                max(prompt_deltas) if prompt_deltas else None),
            "prompt_sample_top1_match_rate": (
                sum(prompt_top1_matches) / len(prompt_top1_matches)
                if prompt_top1_matches else None),
            "cumulative_logprob_finite": cum_finite,
            "generated_nonfinite_count": gen_nonfinite,
            "prompt_sample_nonfinite_count": prompt_nonfinite,
        }
        passed = (
            token_exact
            and text_exact
            and cum_finite
            and gen_nonfinite == 0
            and prompt_nonfinite == 0
            and (cum_delta is None or cum_delta <= 0.25)
            and (result["max_generated_selected_logprob_delta"] is None
                 or result["max_generated_selected_logprob_delta"] <= 0.15)
            and (result["generated_top1_match_rate"] is None
                 or result["generated_top1_match_rate"] >= 0.95)
        )
        result["passed"] = passed
        comparisons[name] = result
        if not passed:
            failures.append(name)
    return {"cases": comparisons, "failures": failures, "passed": not failures}


def compare_speed(flash: dict[str, Any],
                  triton: dict[str, Any]) -> dict[str, Any]:
    cases = {}
    failures = []
    for name, fa in flash["speed"].items():
        tr = triton["speed"][name]
        ratio = fa["median_sec"] / tr["median_sec"]
        speedup = tr["median_sec"] / fa["median_sec"]
        passed = ratio <= 1.05
        cases[name] = {
            "flash_median_sec": fa["median_sec"],
            "triton_median_sec": tr["median_sec"],
            "ratio_flash_over_triton": ratio,
            "speedup_vs_triton": speedup,
            "flash_output_tokens_per_sec": fa["output_tokens_per_sec"],
            "triton_output_tokens_per_sec": tr["output_tokens_per_sec"],
            "passed": passed,
        }
        if not passed:
            failures.append(name)
    return {
        "cases": cases,
        "failures": failures,
        "passed": not failures,
        "skipped": not flash["speed"] and not triton["speed"],
    }


def parent_main(args: argparse.Namespace) -> int:
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path(
            f"/tmp/vllm_v100_backend_regression_{time.strftime('%Y%m%d_%H%M%S')}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    backend_results = {}
    for backend in BACKENDS:
        print(f"running backend {backend} ...", flush=True)
        backend_results[backend] = run_backend(args, backend, output_dir)

    quality = compare_quality(backend_results["FLASH_ATTN_V100"],
                              backend_results["TRITON_ATTN"])
    speed = compare_speed(backend_results["FLASH_ATTN_V100"],
                          backend_results["TRITON_ATTN"])
    combined = {
        "model": args.model,
        "cuda_visible_devices": args.cuda_visible_devices,
        "results": backend_results,
        "quality_compare": quality,
        "speed_compare": speed,
    }
    combined_json = output_dir / "combined.json"
    combined_json.write_text(json.dumps(combined, indent=2, ensure_ascii=False),
                             encoding="utf-8")

    print(f"wrote {combined_json}")
    print(f"quality passed={quality['passed']} failures={quality['failures']}")
    for name, case in speed["cases"].items():
        status = "PASS" if case["passed"] else "FAIL"
        print(
            f"{status} {name}: flash={case['flash_median_sec']:.4f}s "
            f"triton={case['triton_median_sec']:.4f}s "
            f"speedup={case['speedup_vs_triton']:.3f}x")
    if quality["failures"] or speed["failures"]:
        return 2
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--backend", choices=BACKENDS)
    parser.add_argument("--model", default="/home/ymzx/models/Qwen3.5-27B-AWQ")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--kv-cache-dtype", default="auto")
    parser.add_argument("--cuda-visible-devices", default="1")
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.45)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--disable-custom-all-reduce", action="store_true")
    parser.add_argument("--prompt-style", choices=("raw", "qwen35-chat"),
                        default="raw")
    parser.add_argument("--disable-thinking", action="store_false",
                        dest="enable_thinking")
    parser.add_argument("--disable-mm", action="store_true")
    parser.add_argument("--quality-max-tokens", type=int, default=32)
    parser.add_argument("--long-prompt-tokens", type=int, default=1536)
    parser.add_argument("--long-decode-tokens", type=int, default=64)
    parser.add_argument("--long-only-quality", action="store_true")
    parser.add_argument("--long-only-speed", action="store_true")
    parser.add_argument("--omit-prompt-token-ids", action="store_true")
    parser.add_argument("--logprobs", type=int, default=8)
    parser.add_argument("--prompt-logprobs", type=int, default=8)
    parser.add_argument("--speed-warmup", type=int, default=1)
    parser.add_argument("--speed-iters", type=int, default=3)
    parser.add_argument("--skip-speed", action="store_true")
    parser.add_argument("--skip-quality", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--no-trust-remote-code", action="store_false",
                        dest="trust_remote_code")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--offline", action="store_true", default=True)
    parser.add_argument("--online", action="store_false", dest="offline")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--child-output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.child:
        if args.child_output is None or args.backend is None:
            raise SystemExit("--child requires --backend and --child-output")
        return child_main(args)
    return parent_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
