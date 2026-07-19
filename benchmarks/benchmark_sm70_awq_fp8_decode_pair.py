# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paired SM70 AWQ-vs-FP8 decode gate.

This wrapper runs benchmark_sm70_decode.py in isolated subprocesses with the
same decode shapes, TP size, CUDA graph/eager mode, and communication policy.
It exists to prevent comparing AWQ and FP8 artifacts that were produced with
different all-reduce, scheduler, or graph settings.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DECODE_BENCH = Path(__file__).with_name("benchmark_sm70_decode.py")


def _parse_case(raw: str) -> dict[str, Any]:
    parts = raw.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--case must be NAME:INPUT_LEN:OUTPUT_LEN")
    name, input_len, output_len = parts
    return {
        "name": name,
        "input_len": int(input_len),
        "output_len": int(output_len),
    }


def _env_set(base: dict[str, str], values: dict[str, str | None]) -> dict[str, str]:
    env = dict(base)
    for key, value in values.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    return env


def _selector_env(kind: str, selector: str) -> dict[str, str | None]:
    if kind == "awq":
        if selector == "fixed":
            return {
                "VLLM_SM70_AWQ_TUNE_SMALL_SHAPES": "0",
                "VLLM_SM70_AWQ_PRESERVE_DEFAULT_SPLITS": "1",
                "VLLM_SM70_AWQ_PRESERVE_DEFAULT_SPLITS_ONLY": "0",
            }
        if selector == "fast_preserve":
            return {
                "VLLM_SM70_AWQ_TUNE_SMALL_SHAPES": "1",
                "VLLM_SM70_AWQ_PRESERVE_DEFAULT_SPLITS": "1",
                "VLLM_SM70_AWQ_PRESERVE_DEFAULT_SPLITS_ONLY": "0",
            }
        if selector == "fast_split_count":
            return {
                "VLLM_SM70_AWQ_TUNE_SMALL_SHAPES": "1",
                "VLLM_SM70_AWQ_PRESERVE_DEFAULT_SPLITS": "0",
                "VLLM_SM70_AWQ_PRESERVE_DEFAULT_SPLITS_ONLY": "1",
            }
        if selector == "fast_nopreserve":
            return {
                "VLLM_SM70_AWQ_TUNE_SMALL_SHAPES": "1",
                "VLLM_SM70_AWQ_PRESERVE_DEFAULT_SPLITS": "0",
                "VLLM_SM70_AWQ_PRESERVE_DEFAULT_SPLITS_ONLY": "0",
            }
    elif kind == "fp8":
        if selector == "fixed":
            return {
                "VLLM_SM70_FP8_TUNE_SMALL_SHAPES": "0",
                "VLLM_SM70_FP8_SAFE_FAST_SELECTOR": "0",
                "VLLM_SM70_FP8_PRESERVE_DEFAULT_SPLITS": "1",
                "VLLM_SM70_FP8_PRESERVE_DEFAULT_SPLITS_ONLY": "0",
                "VLLM_SM70_FP8_0DOT3_DENSE_SELECTOR": None,
            }
        if selector == "safe_splits":
            return {
                "VLLM_SM70_FP8_TUNE_SMALL_SHAPES": "1",
                "VLLM_SM70_FP8_SAFE_FAST_SELECTOR": "1",
                "VLLM_SM70_FP8_PRESERVE_DEFAULT_SPLITS": "1",
                "VLLM_SM70_FP8_PRESERVE_DEFAULT_SPLITS_ONLY": "0",
                "VLLM_SM70_FP8_0DOT3_DENSE_SELECTOR": None,
            }
        if selector == "safe_split_count":
            return {
                "VLLM_SM70_FP8_TUNE_SMALL_SHAPES": "1",
                "VLLM_SM70_FP8_SAFE_FAST_SELECTOR": "1",
                "VLLM_SM70_FP8_PRESERVE_DEFAULT_SPLITS": "0",
                "VLLM_SM70_FP8_PRESERVE_DEFAULT_SPLITS_ONLY": "1",
                "VLLM_SM70_FP8_0DOT3_DENSE_SELECTOR": None,
            }
        if selector == "dynamic":
            return {
                "VLLM_SM70_FP8_TUNE_SMALL_SHAPES": "1",
                "VLLM_SM70_FP8_SAFE_FAST_SELECTOR": "0",
                "VLLM_SM70_FP8_PRESERVE_DEFAULT_SPLITS": "0",
                "VLLM_SM70_FP8_PRESERVE_DEFAULT_SPLITS_ONLY": "0",
                "VLLM_SM70_FP8_0DOT3_DENSE_SELECTOR": None,
            }
        if selector == "0dot3":
            return {
                "VLLM_SM70_FP8_TUNE_SMALL_SHAPES": None,
                "VLLM_SM70_FP8_SAFE_FAST_SELECTOR": "0",
                "VLLM_SM70_FP8_PRESERVE_DEFAULT_SPLITS": "0",
                "VLLM_SM70_FP8_PRESERVE_DEFAULT_SPLITS_ONLY": "0",
                "VLLM_SM70_FP8_0DOT3_DENSE_SELECTOR": "1",
            }
    raise ValueError(f"unsupported {kind} selector: {selector}")


def _tracked_env(env: dict[str, str]) -> dict[str, str]:
    prefixes = ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "VLLM_SM70_")
    return {key: env[key] for key in sorted(env) if key.startswith(prefixes)}


def _run_decode(
    *,
    args: argparse.Namespace,
    env: dict[str, str],
    model: Path,
    quantization: str,
    dtype: str,
    out: Path,
    cases_json: Path,
) -> None:
    cmd = [
        args.python,
        str(DECODE_BENCH),
        "--model",
        str(model),
        "--out",
        str(out),
        "--case-json",
        str(cases_json),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--dtype",
        dtype,
        "--quantization",
        quantization,
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--warmup",
        str(args.warmup),
        "--repeat",
        str(args.repeat),
        "--seed",
        str(args.seed),
        "--ignore-eos",
    ]
    if args.max_num_batched_tokens is not None:
        cmd.extend(["--max-num-batched-tokens", str(args.max_num_batched_tokens)])
    if args.max_num_seqs is not None:
        cmd.extend(["--max-num-seqs", str(args.max_num_seqs)])
    if args.attention_backend:
        cmd.extend(["--attention-backend", args.attention_backend])
    if args.disable_custom_all_reduce:
        cmd.append("--disable-custom-all-reduce")
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")
    if args.enforce_eager:
        cmd.append("--enforce-eager")

    log_path = out.with_suffix(".log")
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("$ " + " ".join(cmd) + "\n")
        log_file.flush()
        subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            check=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _case_rows(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = {}
    for case in payload.get("cases", []):
        summary = case.get("summary", {})
        repeats = case.get("repeats", [])
        token_hash = repeats[0].get("token_hash") if repeats else None
        rows[str(case.get("name"))] = {
            "input_len": case.get("prompt", {}).get("input_len"),
            "output_len": case.get("sampling_params", {}).get("max_tokens"),
            "steady_decode_tps_mean": summary.get("steady_decode_tps_mean"),
            "tpot_seconds_mean": summary.get("tpot_seconds_mean"),
            "token_hash": token_hash,
        }
    return rows


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def _build_summary(
    awq: dict[str, Any],
    fp8_payloads: dict[str, dict[str, Any]],
    threshold_pct: float,
) -> dict[str, Any]:
    awq_rows = _case_rows(awq)
    fp8_rows = {
        selector: _case_rows(payload) for selector, payload in fp8_payloads.items()
    }
    fixed_rows = fp8_rows.get("fixed", {})

    comparisons = []
    for selector, rows in fp8_rows.items():
        for case_name, fp8_row in rows.items():
            awq_row = awq_rows.get(case_name, {})
            tpot_ratio = _ratio(
                fp8_row.get("tpot_seconds_mean"),
                awq_row.get("tpot_seconds_mean"),
            )
            tps_ratio = _ratio(
                fp8_row.get("steady_decode_tps_mean"),
                awq_row.get("steady_decode_tps_mean"),
            )
            fixed_hash = fixed_rows.get(case_name, {}).get("token_hash")
            token_hash = fp8_row.get("token_hash")
            hash_equal_to_fixed = (
                None
                if fixed_hash is None or selector == "fixed"
                else token_hash == fixed_hash
            )
            tpot_over_awq_pct = (
                None if tpot_ratio is None else (tpot_ratio - 1.0) * 100.0
            )
            comparisons.append(
                {
                    "case": case_name,
                    "fp8_selector": selector,
                    "awq_tpot_seconds_mean": awq_row.get("tpot_seconds_mean"),
                    "fp8_tpot_seconds_mean": fp8_row.get("tpot_seconds_mean"),
                    "tpot_over_awq_pct": tpot_over_awq_pct,
                    "awq_steady_decode_tps_mean": awq_row.get("steady_decode_tps_mean"),
                    "fp8_steady_decode_tps_mean": fp8_row.get("steady_decode_tps_mean"),
                    "decode_tps_ratio_fp8_over_awq": tps_ratio,
                    "fp8_token_hash": token_hash,
                    "fp8_hash_equal_to_fixed": hash_equal_to_fixed,
                    "passes_tpot_threshold": (
                        tpot_over_awq_pct is not None
                        and tpot_over_awq_pct <= threshold_pct
                    ),
                }
            )

    return {
        "awq": awq_rows,
        "fp8": fp8_rows,
        "comparisons": comparisons,
        "threshold": {
            "tpot_over_awq_pct": threshold_pct,
            "note": (
                "FP8 passes only when TPOT is within the threshold versus the "
                "paired AWQ run. Safe-fast selectors must also preserve the "
                "FP8 fixed-selector token hash before they can be accepted."
            ),
        },
    }


def _write_markdown(summary: dict[str, Any], out: Path) -> None:
    lines = [
        "# SM70 AWQ vs FP8 Decode Pair",
        "",
        "| Case | FP8 selector | AWQ tok/s | FP8 tok/s | FP8 TPOT over AWQ | "
        "FP8 hash == fixed | Pass |",
        "| --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for row in summary["comparisons"]:
        over = row["tpot_over_awq_pct"]
        over_text = "n/a" if over is None else f"{over:.2f}%"
        awq_tps = row["awq_steady_decode_tps_mean"]
        fp8_tps = row["fp8_steady_decode_tps_mean"]
        hash_equal = row["fp8_hash_equal_to_fixed"]
        lines.append(
            (
                "| {case} | {selector} | {awq_tps} | {fp8_tps} | {over} | "
                "{hash_equal} | {passed} |"
            ).format(
                case=row["case"],
                selector=row["fp8_selector"],
                awq_tps="n/a" if awq_tps is None else f"{awq_tps:.3f}",
                fp8_tps="n/a" if fp8_tps is None else f"{fp8_tps:.3f}",
                over=over_text,
                hash_equal=hash_equal,
                passed=row["passes_tpot_threshold"],
            )
        )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--awq-model",
        type=Path,
        default=Path("/home/ymzx/models/Qwen3.6-27B-AWQ"),
    )
    parser.add_argument(
        "--fp8-model",
        type=Path,
        default=Path("/home/ymzx/models/Qwen3.6-27B-FP8"),
    )
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--cuda-visible-devices")
    parser.add_argument("--dtype-awq", default="half")
    parser.add_argument("--dtype-fp8", default="auto")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attention-backend")
    parser.add_argument("--disable-custom-all-reduce", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--awq-selector",
        choices=("fixed", "fast_preserve", "fast_split_count", "fast_nopreserve"),
        default="fast_nopreserve",
    )
    parser.add_argument(
        "--fp8-selector",
        action="append",
        choices=("fixed", "safe_splits", "safe_split_count", "dynamic", "0dot3"),
        default=None,
    )
    parser.add_argument(
        "--case",
        type=_parse_case,
        action="append",
        default=None,
        help="Decode case as NAME:INPUT_LEN:OUTPUT_LEN.",
    )
    parser.add_argument("--tpot-threshold-pct", type=float, default=5.0)
    parser.add_argument("--fail-on-threshold", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or REPO_ROOT / "bench_results" / (
        f"awq_fp8_decode_pair_{timestamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = args.case or [
        {"name": "i512_o128", "input_len": 512, "output_len": 128},
        {"name": "i128_o512", "input_len": 128, "output_len": 512},
    ]
    cases_json = out_dir / "cases.json"
    cases_json.write_text(json.dumps(cases, indent=2) + "\n", encoding="utf-8")

    base_env = dict(os.environ)
    base_env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    if args.cuda_visible_devices:
        base_env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    base_env.update(
        {
            "VLLM_SM70_AWQ_TURBOMIND": "1",
            "VLLM_SM70_FP8_TURBOMIND": "1",
            "VLLM_SM70_FLASH_ATTN_V100": "1",
        }
    )

    fp8_selectors = args.fp8_selector or [
        "fixed",
        "safe_splits",
        "safe_split_count",
        "dynamic",
    ]

    awq_out = out_dir / f"awq_{args.awq_selector}.json"
    awq_env = _env_set(base_env, _selector_env("awq", args.awq_selector))
    _run_decode(
        args=args,
        env=awq_env,
        model=args.awq_model,
        quantization="awq",
        dtype=args.dtype_awq,
        out=awq_out,
        cases_json=cases_json,
    )

    fp8_payloads: dict[str, dict[str, Any]] = {}
    for selector in fp8_selectors:
        fp8_out = out_dir / f"fp8_{selector}.json"
        fp8_env = _env_set(base_env, _selector_env("fp8", selector))
        _run_decode(
            args=args,
            env=fp8_env,
            model=args.fp8_model,
            quantization="fp8",
            dtype=args.dtype_fp8,
            out=fp8_out,
            cases_json=cases_json,
        )
        fp8_payloads[selector] = _load_json(fp8_out)

    payload = {
        "config": {
            "awq_model": str(args.awq_model),
            "fp8_model": str(args.fp8_model),
            "awq_selector": args.awq_selector,
            "fp8_selectors": fp8_selectors,
            "tensor_parallel_size": args.tensor_parallel_size,
            "max_model_len": args.max_model_len,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_num_seqs": args.max_num_seqs,
            "disable_custom_all_reduce": args.disable_custom_all_reduce,
            "enforce_eager": args.enforce_eager,
            "tracked_env": _tracked_env(base_env),
            "cases": cases,
        },
        "summary": _build_summary(
            _load_json(awq_out),
            fp8_payloads,
            args.tpot_threshold_pct,
        ),
    }
    summary_json = out_dir / "summary.json"
    summary_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_markdown(payload["summary"], out_dir / "summary.md")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))

    if args.fail_on_threshold:
        failed = [
            row
            for row in payload["summary"]["comparisons"]
            if not row["passes_tpot_threshold"]
        ]
        return 1 if failed else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
