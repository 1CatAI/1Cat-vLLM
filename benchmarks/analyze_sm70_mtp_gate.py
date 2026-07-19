# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize SM70 MTP candidate artifacts into a cheap promotion gate.

This script is intentionally read-only.  It consumes JSON files produced by
``benchmarks/benchmark_sm70_model_tokens.py`` and emits a compact table with the
fields we repeatedly need before deciding whether an MTP/AWQ candidate deserves
expensive quality or Nsight validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

OFFICIAL_TEMPERATURE = 1.0
OFFICIAL_TOP_P = 0.95
OFFICIAL_TOP_K = 20


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain one benchmark result object")
    return data


def _first_record(data: dict[str, Any]) -> dict[str, Any]:
    records = data.get("records")
    if not isinstance(records, list) or not records:
        return {}
    record = records[0]
    return record if isinstance(record, dict) else {}


def _first_output(data: dict[str, Any]) -> dict[str, Any]:
    record = _first_record(data)
    outputs = record.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        return {}
    output = outputs[0]
    return output if isinstance(output, dict) else {}


def _token_ids(data: dict[str, Any]) -> list[int]:
    raw = _first_output(data).get("token_ids")
    if not isinstance(raw, list):
        return []
    return [int(token_id) for token_id in raw]


def _hash_ids(token_ids: list[int]) -> str | None:
    if not token_ids:
        return None
    payload = ",".join(str(token_id) for token_id in token_ids).encode()
    return hashlib.sha256(payload).hexdigest()


def _first_mismatch(left: list[int], right: list[int]) -> int | None:
    for index, (left_id, right_id) in enumerate(zip(left, right)):
        if left_id != right_id:
            return index
    return None


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _official_sampling(data: dict[str, Any]) -> bool:
    sampling = data.get("sampling_params") or {}
    if not isinstance(sampling, dict):
        return False
    return (
        sampling.get("temperature") == OFFICIAL_TEMPERATURE
        and sampling.get("top_p") == OFFICIAL_TOP_P
        and sampling.get("top_k") == OFFICIAL_TOP_K
        and sampling.get("seed") is not None
    )


def _format_float(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _format_pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.1f}%"


def _candidate_summary(
    path: Path,
    *,
    control_tokens: list[int] | None,
    speed_baseline_tps: float | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    data = _load_json(path)
    record = _first_record(data)
    request_metrics = record.get("request_metrics") or {}
    engine_kwargs = data.get("engine_kwargs") or {}
    sampling = data.get("sampling_params") or {}
    spec_metrics = data.get("spec_decoding_metrics") or {}
    env = data.get("env") or {}

    tokens = _token_ids(data)
    token_hash = _hash_ids(tokens)
    tps = _float_or_none(request_metrics.get("steady_decode_tps"))
    tpot_s = _float_or_none(request_metrics.get("tpot_seconds"))
    mean_acceptance = _float_or_none(spec_metrics.get("mean_acceptance_length"))
    draft_acceptance = _float_or_none(spec_metrics.get("draft_acceptance_rate"))
    first_mismatch = (
        _first_mismatch(control_tokens, tokens) if control_tokens is not None else None
    )
    exact_equal = (
        control_tokens is not None
        and first_mismatch is None
        and len(control_tokens) == len(tokens)
    )
    prefix_equal = (
        control_tokens is not None
        and first_mismatch is None
        and len(control_tokens) != len(tokens)
    )
    speedup = tps / speed_baseline_tps if tps and speed_baseline_tps else None

    tp = _int_or_none(engine_kwargs.get("tensor_parallel_size"))
    output_tokens = _int_or_none(data.get("total_output_tokens")) or len(tokens)
    max_tokens = _int_or_none(sampling.get("max_tokens"))
    official = _official_sampling(data)
    eager = bool(engine_kwargs.get("enforce_eager"))
    spec_config = engine_kwargs.get("speculative_config")
    spec_tokens = None
    spec_method = None
    if isinstance(spec_config, dict):
        spec_tokens = _int_or_none(spec_config.get("num_speculative_tokens"))
        spec_method = spec_config.get("method")

    reasons = []
    if args.require_official_sampling and not official:
        reasons.append("sampling")
    if args.require_tp is not None and tp != args.require_tp:
        reasons.append(f"tp{tp}")
    if args.min_output_tokens is not None and output_tokens < args.min_output_tokens:
        reasons.append(f"output<{args.min_output_tokens}")
    if eager:
        reasons.append("eager")
    if args.target_tps is not None and (tps is None or tps < args.target_tps):
        reasons.append(f"tps<{args.target_tps:g}")
    if args.min_mean_acceptance is not None and (
        mean_acceptance is None or mean_acceptance < args.min_mean_acceptance
    ):
        reasons.append(f"accept<{args.min_mean_acceptance:g}")

    status = "screen-only" if reasons else "target-speed-candidate"
    if exact_equal:
        status = f"{status}+exact"
    elif prefix_equal:
        status = f"{status}+prefix"
    elif control_tokens is not None:
        status = f"{status}+drift"

    return {
        "path": str(path),
        "name": path.name,
        "model": data.get("model"),
        "quantization": engine_kwargs.get("quantization"),
        "dtype": engine_kwargs.get("dtype"),
        "tp": tp,
        "max_model_len": engine_kwargs.get("max_model_len"),
        "max_num_batched_tokens": engine_kwargs.get("max_num_batched_tokens"),
        "gpu_memory_utilization": engine_kwargs.get("gpu_memory_utilization"),
        "official_sampling": official,
        "sampling": {
            "temperature": sampling.get("temperature"),
            "top_p": sampling.get("top_p"),
            "top_k": sampling.get("top_k"),
            "seed": sampling.get("seed"),
            "ignore_eos": sampling.get("ignore_eos"),
            "max_tokens": max_tokens,
        },
        "spec_method": spec_method,
        "spec_tokens": spec_tokens,
        "output_tokens": output_tokens,
        "steady_decode_tps": tps,
        "tpot_ms": tpot_s * 1000.0 if tpot_s is not None else None,
        "speedup_vs_baseline": speedup,
        "mean_acceptance_length": mean_acceptance,
        "draft_acceptance_rate": draft_acceptance,
        "per_position_acceptance_rate": spec_metrics.get(
            "per_position_acceptance_rate"
        ),
        "token_hash": token_hash,
        "exact_equal_to_control": exact_equal if control_tokens is not None else None,
        "prefix_equal_to_control": prefix_equal if control_tokens is not None else None,
        "first_mismatch_vs_control": first_mismatch,
        "common_prefix_len_vs_control": (
            None if control_tokens is None else min(len(control_tokens), len(tokens))
        ),
        "env": {
            "CUDA_VISIBLE_DEVICES": env.get("CUDA_VISIBLE_DEVICES"),
            "VLLM_SM70_QUANT_BACKEND": env.get("VLLM_SM70_QUANT_BACKEND"),
            "VLLM_SM70_AWQ_TURBOMIND": env.get("VLLM_SM70_AWQ_TURBOMIND"),
            "VLLM_SM70_AWQ_TUNE_SMALL_SHAPES": env.get(
                "VLLM_SM70_AWQ_TUNE_SMALL_SHAPES"
            ),
            "VLLM_SM70_AWQ_PRESERVE_DEFAULT_SPLITS": env.get(
                "VLLM_SM70_AWQ_PRESERVE_DEFAULT_SPLITS"
            ),
            "VLLM_SM70_LM_HEAD_TOP1_TC": env.get("VLLM_SM70_LM_HEAD_TOP1_TC"),
        },
        "status": status,
        "screen_reasons": reasons,
    }


def _markdown_table(summaries: list[dict[str, Any]]) -> str:
    lines = [
        "| artifact | status | out | tps | tpot ms | speedup | accept len | "
        "draft acc | mtp | token cmp | hash |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in summaries:
        lines.append(
            "| {name} | {status} | {out} | {tps} | {tpot} | {speedup} | "
            "{accept} | {draft} | {mtp} | {diff} | `{hash}` |".format(
                name=item["name"],
                status=item["status"],
                out=item["output_tokens"],
                tps=_format_float(item["steady_decode_tps"]),
                tpot=_format_float(item["tpot_ms"]),
                speedup=_format_float(item["speedup_vs_baseline"], 2),
                accept=_format_float(item["mean_acceptance_length"]),
                draft=_format_pct(item["draft_acceptance_rate"]),
                mtp=item["spec_tokens"] if item["spec_tokens"] is not None else "-",
                diff=(
                    "exact"
                    if item["exact_equal_to_control"] is True
                    else (
                        f"prefix{item['common_prefix_len_vs_control']}"
                        if item["prefix_equal_to_control"] is True
                        else (
                            "-"
                            if item["first_mismatch_vs_control"] is None
                            else item["first_mismatch_vs_control"]
                        )
                    )
                ),
                hash=(item["token_hash"] or "-")[:16],
            )
        )
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument(
        "--control",
        type=Path,
        help="Artifact whose output tokens are used for cheap exact-drift checks.",
    )
    parser.add_argument(
        "--speed-baseline",
        type=Path,
        help="Artifact whose steady decode TPS is used for speedup ratios.",
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--md-out", type=Path)
    parser.add_argument("--target-tps", type=float, default=100.0)
    parser.add_argument("--min-mean-acceptance", type=float)
    parser.add_argument("--min-output-tokens", type=int, default=512)
    parser.add_argument("--require-tp", type=int, default=2)
    parser.add_argument("--require-official-sampling", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    control_tokens = _token_ids(_load_json(args.control)) if args.control else None
    speed_baseline_tps = None
    if args.speed_baseline is not None:
        baseline = _load_json(args.speed_baseline)
        record = _first_record(baseline)
        metrics = record.get("request_metrics") or {}
        speed_baseline_tps = _float_or_none(metrics.get("steady_decode_tps"))

    summaries = [
        _candidate_summary(
            artifact,
            control_tokens=control_tokens,
            speed_baseline_tps=speed_baseline_tps,
            args=args,
        )
        for artifact in args.artifacts
    ]
    payload = {
        "control": str(args.control) if args.control else None,
        "speed_baseline": str(args.speed_baseline) if args.speed_baseline else None,
        "gate": {
            "target_tps": args.target_tps,
            "min_output_tokens": args.min_output_tokens,
            "require_tp": args.require_tp,
            "require_official_sampling": args.require_official_sampling,
            "min_mean_acceptance": args.min_mean_acceptance,
        },
        "summaries": summaries,
    }
    markdown = _markdown_table(summaries)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if args.md_out is not None:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(markdown)
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
