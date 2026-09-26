# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Concurrent Qwen3.8 decode measurement using complete engine intervals.

Synthetic greedy throughput, natural text-health checks, and prefill are
reported separately. No persistent API; engine workers shut down in finally.
"""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from benchmarks.benchmark_sm70_model_tokens import (
    _metric_snapshot,
    _request_metrics_dict,
    _spec_decoding_delta,
)
from benchmarks.sm70_qwen38_baseline import FIXED_PROMPT, engine_args
from vllm import LLM, SamplingParams


def summarize(records, width):
    eligible = [
        i
        for i, row in enumerate(records)
        if row["running"] == width
        and row["waiting"] == 0
        and len(row["counts"]) == width
        and all(n > 0 for n in row["counts"])
        and not row["prefill"]
        and not row["finished"]
    ]
    selected = set(eligible[8:-8])
    intervals = []
    tokens = 0
    for i in sorted(selected):
        if i - 1 not in selected:
            continue
        prev, row = records[i - 1], records[i]
        if prev["request_ids"] != row["request_ids"]:
            continue
        duration = row["timestamp"] - prev["timestamp"]
        if duration <= 0:
            continue
        intervals.append(duration)
        tokens += sum(row["counts"])
    if len(intervals) < 16:
        raise RuntimeError(
            f"Insufficient fixed-width C{width} intervals: {len(intervals)}"
        )
    ordered = sorted(intervals)
    return {
        "concurrency": width,
        "intervals": len(intervals),
        "emitted_tokens": tokens,
        "engine_seconds": sum(intervals),
        "aggregate_decode_tps": tokens / sum(intervals),
        "per_stream_decode_tps": tokens / sum(intervals) / width,
        "step_ms_mean": statistics.mean(intervals) * 1000,
        "step_ms_p50": statistics.median(intervals) * 1000,
        "step_ms_p90": ordered[int((len(ordered) - 1) * 0.90)] * 1000,
        "step_ms_p99": ordered[int((len(ordered) - 1) * 0.99)] * 1000,
    }


def compare_tokens(requests, reference):
    """Report the first zero-based difference without aborting a loaded engine."""
    differences = []
    for actual, expected in zip(requests, reference, strict=True):
        a, b = actual["token_ids"], expected["token_ids"]
        first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
        if first is None and len(a) != len(b):
            first = min(len(a), len(b))
        differences.append(first)
    return differences


def finalize_measurements(report):
    """Collect every planned case, but never accept failed token parity."""
    report["measurements_complete"] = True
    checks = [
        matched
        for case in report["cases"]
        for key in ("tokens_match_reference", "tokens_match_first_repeat")
        for matched in case.get(key, [])
    ]
    if "reference_accepted" in report:
        checks.append(report["reference_accepted"])
    report["token_parity_passed"] = all(checks) if checks else None
    if checks and not all(checks):
        raise RuntimeError(
            "Token parity failed; all planned measurements were collected, "
            "but this run is not an accepted quality/speed result"
        )
    report["complete"] = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--mode", choices=("nomtp", "mtp"), default="nomtp")
    parser.add_argument("--widths", default="1,4,8,16")
    parser.add_argument("--input-len", type=int, default=8192)
    parser.add_argument("--output-len", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--baseline-reference", type=Path)
    parser.add_argument(
        "--reference", type=Path, help="Same-contract token parity gate"
    )
    parser.add_argument(
        "--diagnostic-reference",
        action="store_true",
        help=(
            "Allow a fully measured but nonrepeatable reference for diagnosis; "
            "the candidate cannot be marked accepted against that reference"
        ),
    )
    parser.add_argument("--measure-prefill", action="store_true")
    parser.add_argument("--health", action="store_true")
    args = parser.parse_args()
    widths = [int(x) for x in args.widths.split(",")]
    if not widths or any(w <= 0 for w in widths) or len(set(widths)) != len(widths):
        raise ValueError("widths must be distinct positive integers")
    if args.input_len <= 0 or args.output_len < 40 or args.repeats <= 0:
        raise ValueError("input/repeats must be positive and output-len >= 40")
    model = str(args.model)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    config = engine_args(model, use_defaults=True)
    config.pop("worker_extension_cls")
    config["max_num_seqs"] = max(widths)
    if args.mode == "mtp":
        config["speculative_config"] = {
            "method": "mtp",
            "num_speculative_tokens": 4,
            "draft_sample_method": "greedy",
        }
    report = {
        "source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "mode": args.mode,
        "engine": config,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "env": {
            k: v for k, v in os.environ.items() if k.startswith(("VLLM_", "CUDA_"))
        },
        "native_wheel": os.environ.get("QWEN38_NATIVE_WHEEL"),
        "input_len": args.input_len,
        "output_len": args.output_len,
        "sampling": {"temperature": 0, "top_p": 1, "top_k": -1, "ignore_eos": True},
        "measurement": (
            "Tokens divided by consecutive engine timestamp intervals at fixed "
            "live width; trim 8 head/tail steps; exclude prefill/finished requests"
        ),
        "cases": [],
        "prefill_cases": [],
        "traces": [],
        "measurements_complete": False,
        "complete": False,
    }
    import vllm._C as native

    report["native_path"] = native.__file__
    report["native_sha256"] = hashlib.sha256(
        Path(native.__file__).read_bytes()
    ).hexdigest()
    reference_cases = {}
    if args.reference:
        reference = json.loads(args.reference.read_text())
        for key in ("engine", "input_len", "output_len", "mode", "sampling"):
            if reference[key] != report[key]:
                raise ValueError(f"Reference contract differs at {key}")
        if not reference.get("complete") and not (
            args.diagnostic_reference and reference.get("measurements_complete")
        ):
            raise ValueError("Reference run is incomplete")
        report["reference_accepted"] = bool(reference.get("complete"))
        report["reference_path"] = str(args.reference)
        reference_cases = {
            (c["concurrency"], c["repeat"]): c for c in reference["cases"]
        }

    def save():
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    topics = (
        "reliable distributed systems and memory ordering",
        "numerical optimization and stable convergence",
        "parsers, compilers, and software testing",
        "database recovery and transaction isolation",
        "network fault diagnosis and observability",
        "dynamic programming and graph algorithms",
        "scientific inference and experimental design",
        "safe repository refactoring with tool calls",
    )
    prompts = []
    for i in range(max(widths)):
        piece = tokenizer.encode(
            f"Independent decode stream {i}: {topics[i % len(topics)]}. ",
            add_special_tokens=False,
        )
        prompts.append(
            {
                "prompt_token_ids": (piece * (args.input_len // len(piece) + 1))[
                    : args.input_len
                ]
            }
        )
    llm = None
    try:
        started = time.perf_counter()
        llm = LLM(**config)
        report["load_seconds"] = time.perf_counter() - started
        save()
        warm = llm.generate(
            "Explain why testing software is useful.",
            SamplingParams(max_tokens=48, temperature=0),
            use_tqdm=False,
        )[0]
        report["warmup_text"] = warm.outputs[0].text
        if args.health:
            import regex as re

            official = json.loads((Path(model) / "generation_config.json").read_text())
            natural = SamplingParams(
                max_tokens=513,
                seed=0,
                temperature=official["temperature"],
                top_p=official["top_p"],
                top_k=official["top_k"],
                ignore_eos=False,
                skip_special_tokens=False,
            )
            report["health"] = []
            for prompt, pattern in (
                (
                    "请计算 19 × 23。请在最后一行写 RESULT=计算结果。",
                    r"RESULT\s*=\s*437\b",
                ),
                (
                    "项目代号 CEDAR-47，编号 8261。最后一行准确输出 CEDAR-47|8261。",
                    r"CEDAR-47\s*\|\s*8261",
                ),
            ):
                ids = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=True,
                    return_dict=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )
                result = llm.generate(
                    {"prompt_token_ids": ids}, natural, use_tqdm=False
                )[0].outputs[0]
                passed = result.finish_reason == "stop" and bool(
                    re.search(pattern, result.text.rsplit("</think>", 1)[-1])
                )
                report["health"].append(
                    {
                        "prompt": prompt,
                        "text": result.text,
                        "finish_reason": result.finish_reason,
                        "passed": passed,
                    }
                )
                save()
                if not passed:
                    raise RuntimeError("Natural output health check failed")
        if args.baseline_reference:
            previous = json.loads(args.baseline_reference.read_text())
            reference = next(c for c in previous["cases"] if c["name"] == "fixed_8192")
            reference_ids = reference["runs"][0]["token_ids"]
            piece = tokenizer.encode(FIXED_PROMPT, add_special_tokens=False)
            fixed_ids = (piece * (8192 // len(piece) + 1))[:8192]
            report["baseline_reference"] = str(args.baseline_reference)
            report["baseline_runs"] = []
            params = SamplingParams(
                max_tokens=513,
                temperature=0,
                seed=0,
                ignore_eos=True,
                skip_special_tokens=False,
            )
            for repeat in range(-1, args.repeats):
                result = llm.generate(
                    {"prompt_token_ids": fixed_ids}, params, use_tqdm=False
                )[0]
                ids = list(result.outputs[0].token_ids)
                metrics = _request_metrics_dict(result.metrics, len(ids))
                first_diff = next(
                    (i for i, (a, b) in enumerate(zip(ids, reference_ids)) if a != b),
                    None,
                )
                row = {
                    "repeat": repeat,
                    "warmup": repeat == -1,
                    "metrics": metrics,
                    "prefill_tps": 8192 / metrics["prefill_time"],
                    "matches_reference": ids == reference_ids,
                    "first_reference_difference": first_diff,
                    "token_ids": ids,
                    "text": result.outputs[0].text,
                }
                report["baseline_runs"].append(row)
                save()
                print(
                    json.dumps(
                        {k: v for k, v in row.items() if k not in ("token_ids", "text")}
                    ),
                    flush=True,
                )
        for width in widths:
            for repeat in range(args.repeats):
                records = []
                core = llm.llm_engine.engine_core
                original = core.get_output

                def observed(original=original, records=records):
                    output = original()
                    scheduler = output.scheduler_stats
                    items = sorted(output.outputs, key=lambda x: x.request_id)
                    records.append(
                        {
                            "timestamp": output.timestamp,
                            "running": scheduler.num_running_reqs
                            if scheduler
                            else None,
                            "waiting": scheduler.num_waiting_reqs
                            if scheduler
                            else None,
                            "counts": [len(x.new_token_ids) for x in items],
                            "request_ids": [x.request_id for x in items],
                            "prefill": any(x.prefill_stats is not None for x in items),
                            "finished": any(x.finished for x in items),
                        }
                    )
                    return output

                before = _metric_snapshot(llm)
                core.get_output = observed
                start = time.perf_counter()
                try:
                    outputs = llm.generate(
                        prompts[:width],
                        SamplingParams(
                            max_tokens=args.output_len,
                            temperature=0,
                            top_p=1,
                            top_k=-1,
                            seed=0,
                            ignore_eos=True,
                        ),
                        use_tqdm=False,
                    )
                finally:
                    core.get_output = original
                elapsed = time.perf_counter() - start
                requests = []
                for output in outputs:
                    completion = output.outputs[0]
                    if len(completion.token_ids) != args.output_len:
                        raise RuntimeError(f"Incomplete output: {output.request_id}")
                    requests.append(
                        {
                            "request_id": output.request_id,
                            "metrics": _request_metrics_dict(
                                output.metrics, len(completion.token_ids)
                            ),
                            "token_ids": list(completion.token_ids),
                            "text": completion.text,
                            "sha256": hashlib.sha256(
                                bytes(str(list(completion.token_ids)), "utf-8")
                            ).hexdigest(),
                            "finish_reason": completion.finish_reason,
                        }
                    )
                case = {
                    "concurrency": width,
                    "repeat": repeat,
                    "wall_seconds": elapsed,
                    "wall_output_tps": width * args.output_len / elapsed,
                    "spec_decoding": _spec_decoding_delta(
                        before, _metric_snapshot(llm)
                    ),
                    "summary": summarize(records, width),
                    "requests": requests,
                    "raw_steps": records,
                }
                report["cases"].append(case)
                if repeat > 0:
                    first = next(
                        c
                        for c in report["cases"]
                        if c["concurrency"] == width and c["repeat"] == 0
                    )
                    differences = compare_tokens(requests, first["requests"])
                    case["first_repeat_token_differences"] = differences
                    case["tokens_match_first_repeat"] = [i is None for i in differences]
                if args.reference:
                    expected = reference_cases[(width, repeat)]
                    differences = compare_tokens(requests, expected["requests"])
                    case["reference_token_differences"] = differences
                    case["tokens_match_reference"] = [i is None for i in differences]
                if any(
                    not all(case.get(key, [True]))
                    for key in ("tokens_match_reference", "tokens_match_first_repeat")
                ):
                    print(
                        f"C{width} repeat {repeat}: token parity FAILED; "
                        "retaining remaining diagnostic cases before failing the run",
                        flush=True,
                    )
                save()
                print(json.dumps({"repeat": repeat, **case["summary"]}), flush=True)
            if args.measure_prefill:
                # One output token completes prefill without a competing decode
                # phase. Use a union wall interval, never sum overlapping TTFTs.
                for repeat in range(args.repeats):
                    start = time.perf_counter()
                    outputs = llm.generate(
                        prompts[:width],
                        SamplingParams(
                            max_tokens=1, temperature=0, seed=0, ignore_eos=True
                        ),
                        use_tqdm=False,
                    )
                    elapsed = time.perf_counter() - start
                    metrics = [
                        _request_metrics_dict(o.metrics, len(o.outputs[0].token_ids))
                        for o in outputs
                    ]
                    if len(metrics) != width or any(
                        len(o.outputs[0].token_ids) != 1 for o in outputs
                    ):
                        raise RuntimeError("Incomplete prefill-only cohort")
                    span = max(m["raw"]["first_token_ts"] for m in metrics) - min(
                        m["raw"]["scheduled_ts"] for m in metrics
                    )
                    row = {
                        "concurrency": width,
                        "repeat": repeat,
                        "input_tokens": width * args.input_len,
                        "engine_prefill_seconds": span,
                        "aggregate_prefill_tps": width * args.input_len / span,
                        "client_wall_seconds": elapsed,
                        "client_input_tps": width * args.input_len / elapsed,
                        "ttft_mean_seconds": statistics.mean(
                            m["first_token_latency"] for m in metrics
                        ),
                        "ttft_max_seconds": max(
                            m["first_token_latency"] for m in metrics
                        ),
                        "request_metrics": metrics,
                    }
                    report["prefill_cases"].append(row)
                    save()
                    print(
                        json.dumps(
                            {
                                "prefill_only": True,
                                **{
                                    k: v
                                    for k, v in row.items()
                                    if k != "request_metrics"
                                },
                            }
                        ),
                        flush=True,
                    )
        report["measurements_complete"] = True
        if args.health and max(widths) > 1:
            # Exercise actual concurrent natural-EOS traffic after the timed
            # cohorts. This is a text-health check, not token-parity evidence.
            health_prompts, expected = [], []
            for i in range(max(widths)):
                marker = f"CEDAR-{47 + i}|{8261 + i}"
                health_prompts.append(
                    {
                        "prompt_token_ids": tokenizer.apply_chat_template(
                            [
                                {
                                    "role": "user",
                                    "content": (
                                        f"项目代号 CEDAR-{47 + i}，编号 {8261 + i}。"
                                        f"最后一行准确输出 {marker}。"
                                    ),
                                }
                            ],
                            tokenize=True,
                            return_dict=False,
                            add_generation_prompt=True,
                            enable_thinking=True,
                        )
                    }
                )
                expected.append(marker)
            results = llm.generate(health_prompts, natural, use_tqdm=False)
            report["batch_health"] = []
            for result, marker in zip(results, expected, strict=True):
                completion = result.outputs[0]
                passed = completion.finish_reason == "stop" and marker in (
                    completion.text.rsplit("</think>", 1)[-1].replace(" ", "")
                )
                report["batch_health"].append(
                    {
                        "expected": marker,
                        "text": completion.text,
                        "finish_reason": completion.finish_reason,
                        "passed": passed,
                    }
                )
            save()
            if not all(case["passed"] for case in report["batch_health"]):
                raise RuntimeError("Concurrent natural output health check failed")
        finalize_measurements(report)
        save()
    except Exception as error:
        report["error"] = repr(error)
        save()
        raise
    finally:
        if llm is not None:
            llm.llm_engine.engine_core.shutdown(timeout=30.0)


if __name__ == "__main__":
    main()
