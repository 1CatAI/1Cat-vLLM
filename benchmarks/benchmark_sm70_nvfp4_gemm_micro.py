# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Microbenchmark SM70 TurboMind NVFP4 dense GEMMs for 27B decode.

The default case set models one Qwen3.5-27B-NVFP4 decode token on one TP rank
with tensor_parallel_size=2. It times only nvfp4_gemm_sm70_out after synthetic
weights have been prepared and dispatch has been warmed, so the weighted total
is comparable to the Nsight "TurboMind NVFP4 GEMM" critical-path bucket.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import torch

DEFAULT_QWEN35_27B_TP2_CASES = (
    # label, per-rank N, per-rank K, per-rank call count per decode token
    ("linear_attn_in_proj_qkvz", 8192, 5120, 48),
    ("out_proj_all", 5120, 3072, 64),
    ("full_attn_qkv_proj", 7168, 5120, 16),
    ("mlp_gate_up_proj", 17408, 5120, 64),
    ("mlp_down_proj", 5120, 8704, 64),
)


@dataclass(frozen=True)
class BenchCase:
    label: str
    n: int
    k: int
    count: int
    m: int


def _import_sm70_ops() -> Any:
    from vllm import _sm70_ops

    return _sm70_ops


def _require_sm70(device: torch.device) -> None:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA device.")
    capability = torch.cuda.get_device_capability(device)
    if capability != (7, 0):
        raise RuntimeError(f"Expected SM70, got sm_{capability[0]}{capability[1]}.")


def _make_input(m: int, k: int, device: torch.device) -> torch.Tensor:
    values = torch.arange(m * k, device=device, dtype=torch.int32)
    values = ((values % 1024).to(torch.float32) / 512.0) - 1.0
    return values.reshape(m, k).to(torch.float16)


def _make_qweight(k: int, n: int, device: torch.device) -> torch.Tensor:
    values = torch.arange(k * n, device=device, dtype=torch.int32)
    return (values.reshape(k, n) & 15).to(torch.uint8).contiguous()


def _pack_qweight_u4(qweight: torch.Tensor) -> torch.Tensor:
    if qweight.size(1) % 8 != 0:
        raise ValueError("N must be divisible by 8 for packed raw GEMV.")
    packed = torch.zeros(
        (qweight.size(0), qweight.size(1) // 8),
        dtype=torch.int32,
        device=qweight.device,
    )
    q_i32 = qweight.to(torch.int32)
    for offset in range(8):
        packed |= q_i32[:, offset::8] << (4 * offset)
    return packed.contiguous()


def _make_scales(k: int, n: int, group_size: int, device: torch.device) -> torch.Tensor:
    groups = k // group_size
    values = torch.arange(groups * n, device=device, dtype=torch.int32)
    values = ((values % 127).to(torch.float32) + 1.0) / 128.0
    return values.reshape(groups, n).to(torch.float16).contiguous()


def _time_cuda_call(
    fn: Any,
    device: torch.device,
    warmup: int,
    iters: int,
    use_cuda_graph: bool,
) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.accelerator.synchronize(device)

    timed_fn = fn
    if use_cuda_graph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        torch.accelerator.synchronize(device)
        timed_fn = graph.replay
        for _ in range(min(warmup, 10)):
            timed_fn()
        torch.accelerator.synchronize(device)

    times_ms: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        timed_fn()
        end.record()
        end.synchronize()
        times_ms.append(float(start.elapsed_time(end)))

    ordered = sorted(times_ms)
    p90_index = min(len(ordered) - 1, int(0.9 * (len(ordered) - 1)))
    p99_index = min(len(ordered) - 1, int(0.99 * (len(ordered) - 1)))
    return {
        "mean_us": statistics.fmean(times_ms) * 1000.0,
        "min_us": min(times_ms) * 1000.0,
        "p50_us": statistics.median(times_ms) * 1000.0,
        "p90_us": ordered[p90_index] * 1000.0,
        "p99_us": ordered[p99_index] * 1000.0,
        "max_us": max(times_ms) * 1000.0,
    }


def _parse_case(raw: str, default_m: int) -> BenchCase:
    parts = raw.split(":")
    if len(parts) not in (4, 5):
        raise ValueError(
            f"--case must be LABEL:N:K:COUNT or LABEL:N:K:COUNT:M, got {raw!r}"
        )
    label, n, k, count = parts[:4]
    m = int(parts[4]) if len(parts) == 5 else default_m
    return BenchCase(label=label, n=int(n), k=int(k), count=int(count), m=m)


def _default_cases(default_m: int) -> list[BenchCase]:
    return [
        BenchCase(label=label, n=n, k=k, count=count, m=default_m)
        for label, n, k, count in DEFAULT_QWEN35_27B_TP2_CASES
    ]


def _run_case(
    ops: Any,
    case: BenchCase,
    group_size: int,
    device: torch.device,
    warmup: int,
    iters: int,
    use_cuda_graph: bool,
    mode: str,
    gemv_split_k: int,
) -> dict[str, Any]:
    if case.k % group_size != 0:
        raise ValueError(f"{case.label}: K={case.k} not divisible by {group_size}.")
    qweight = _make_qweight(case.k, case.n, device)
    scales = _make_scales(case.k, case.n, group_size, device)
    x = _make_input(case.m, case.k, device)
    out = torch.empty((case.m, case.n), dtype=torch.float16, device=device)

    k_ld = 0
    q_ld = 0
    tm_weight = None
    tm_scales = None
    qweight_packed = None
    partials = None
    if mode == "gemm":
        tm_weight, tm_scales, meta = ops.nvfp4_sm70_prepare(
            qweight, scales, group_size, False
        )
        k_ld = int(meta[0].item())
        q_ld = int(meta[1].item())

        run = partial(
            ops.nvfp4_gemm_sm70_out,
            out,
            x,
            tm_weight,
            tm_scales,
            group_size,
            k_ld,
            q_ld,
            False,
        )
    elif mode in ("raw-gemv", "raw-gemv-warp", "raw-gemv-h2"):
        if case.m != 1:
            raise ValueError(f"{mode} mode only supports M=1.")
        qweight_packed = _pack_qweight_u4(qweight)
        if mode == "raw-gemv":
            partials = (
                torch.empty((gemv_split_k, case.n), dtype=torch.float32, device=device)
                if gemv_split_k > 1
                else torch.empty((0,), dtype=torch.float32, device=device)
            )

            run = partial(
                ops.nvfp4_gemv_sm70_raw_out,
                out,
                x,
                qweight_packed,
                scales,
                partials,
                group_size,
                gemv_split_k,
            )
        elif mode == "raw-gemv-h2":
            partials = (
                torch.empty((gemv_split_k, case.n), dtype=torch.float16, device=device)
                if gemv_split_k > 1
                else torch.empty((0,), dtype=torch.float16, device=device)
            )

            run = partial(
                ops.nvfp4_gemv_sm70_h2_out,
                out,
                x,
                qweight_packed,
                scales,
                partials,
                group_size,
                gemv_split_k,
            )
        else:
            partials = torch.empty((0,), dtype=torch.float32, device=device)

            run = partial(
                ops.nvfp4_gemv_sm70_warp_out,
                out,
                x,
                qweight_packed,
                scales,
                group_size,
            )
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    timing = _time_cuda_call(run, device, warmup, iters, use_cuda_graph)
    weighted_mean_ms = float(case.count) * timing["mean_us"] / 1000.0
    row = {
        "mode": mode,
        "label": case.label,
        "count": case.count,
        "m": case.m,
        "n": case.n,
        "k": case.k,
        "group_size": group_size,
        "gemv_split_k": gemv_split_k if mode in ("raw-gemv", "raw-gemv-h2") else 0,
        "k_ld": k_ld,
        "q_ld": q_ld,
        "desc_hint": (
            f"raw_gemv_split{gemv_split_k}_{case.m}x{case.n}x{case.k}"
            if mode == "raw-gemv"
            else (
                f"raw_gemv_warp_{case.m}x{case.n}x{case.k}"
                if mode == "raw-gemv-warp"
                else (
                    f"raw_gemv_h2_split{gemv_split_k}_{case.m}x{case.n}x{case.k}"
                    if mode == "raw-gemv-h2"
                    else f"sm70_f16_nvfp4k{group_size}_f16_tnt_fff_"
                    f"{case.m}x{case.n}x{case.k}_1"
                )
            )
        ),
        **timing,
        "weighted_mean_ms": weighted_mean_ms,
    }
    del qweight, scales, tm_weight, tm_scales, qweight_packed, partials, x, out
    torch.accelerator.empty_cache()
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "mode",
        "label",
        "count",
        "m",
        "n",
        "k",
        "group_size",
        "gemv_split_k",
        "k_ld",
        "q_ld",
        "desc_hint",
        "mean_us",
        "min_us",
        "p50_us",
        "p90_us",
        "p99_us",
        "max_us",
        "weighted_mean_ms",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument("--m", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        help="Time CUDA graph replay of each single-op case after warmup.",
    )
    parser.add_argument(
        "--mode",
        choices=("gemm", "raw-gemv", "raw-gemv-warp", "raw-gemv-h2"),
        default="gemm",
        help="Operator implementation to benchmark.",
    )
    parser.add_argument(
        "--gemv-split-k",
        type=int,
        default=8,
        help="K split count for --mode raw-gemv.",
    )
    parser.add_argument(
        "--case",
        action="append",
        default=None,
        help="LABEL:N:K:COUNT or LABEL:N:K:COUNT:M. Repeatable.",
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--csv-out", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    device = torch.device(args.device)
    _require_sm70(device)
    ops = _import_sm70_ops()
    if not hasattr(torch.ops._C, "nvfp4_sm70_prepare"):
        raise RuntimeError("Missing _C::nvfp4_sm70_prepare.")
    if not hasattr(torch.ops._C, "nvfp4_gemm_sm70_out"):
        raise RuntimeError("Missing _C::nvfp4_gemm_sm70_out.")
    if args.mode == "raw-gemv" and not hasattr(torch.ops._C, "nvfp4_gemv_sm70_raw_out"):
        raise RuntimeError("Missing _C::nvfp4_gemv_sm70_raw_out.")
    if args.mode == "raw-gemv-warp" and not hasattr(
        torch.ops._C, "nvfp4_gemv_sm70_warp_out"
    ):
        raise RuntimeError("Missing _C::nvfp4_gemv_sm70_warp_out.")
    if args.mode == "raw-gemv-h2" and not hasattr(
        torch.ops._C, "nvfp4_gemv_sm70_h2_out"
    ):
        raise RuntimeError("Missing _C::nvfp4_gemv_sm70_h2_out.")

    cases = (
        [_parse_case(raw, args.m) for raw in args.case]
        if args.case
        else _default_cases(args.m)
    )
    rows = [
        _run_case(
            ops=ops,
            case=case,
            group_size=args.group_size,
            device=device,
            warmup=args.warmup,
            iters=args.iters,
            use_cuda_graph=args.cuda_graph,
            mode=args.mode,
            gemv_split_k=args.gemv_split_k,
        )
        for case in cases
    ]
    total_ms = sum(float(row["weighted_mean_ms"]) for row in rows)
    payload = {
        "suite": "qwen3.5-27b-nvfp4-tp2-rank-decode",
        "device": args.device,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device_name": torch.cuda.get_device_name(device),
        "device_capability": list(torch.cuda.get_device_capability(device)),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "group_size": args.group_size,
        "mode": args.mode,
        "gemv_split_k": (
            args.gemv_split_k if args.mode in ("raw-gemv", "raw-gemv-h2") else 0
        ),
        "warmup": args.warmup,
        "iters": args.iters,
        "cuda_graph": args.cuda_graph,
        "total_weighted_mean_ms": total_ms,
        "stage1_target_ms": 10.0,
        "cases": rows,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    if args.csv_out is not None:
        _write_csv(args.csv_out, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
