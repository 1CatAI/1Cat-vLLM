# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark a synthetic TurboMind W4A16 draft LM head on SM70.

This measures only the prepared AWQ GEMM. It does not estimate the proposal
acceptance loss caused by quantizing an FP16 LM head.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable
from pathlib import Path

import torch

from vllm import _sm70_ops as sm70_ops


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=1)
    parser.add_argument("--n", type=int, action="append")
    parser.add_argument("--k", type=int, default=5120)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def _require_sm70(device: torch.device) -> None:
    if not torch.cuda.is_available() or device.type != "cuda":
        raise RuntimeError("This benchmark requires a CUDA device.")
    capability = torch.cuda.get_device_capability(device)
    if capability != (7, 0):
        raise RuntimeError(f"Expected SM70, got sm_{capability[0]}{capability[1]}.")
    for name in ("awq_sm70_prepare", "awq_gemm_sm70_out"):
        if not hasattr(torch.ops._C, name):
            raise RuntimeError(f"Missing torch op _C::{name}.")


def _time_cuda(
    fn: Callable[[], None],
    device: torch.device,
    warmup: int,
    iters: int,
) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)

    events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(iters)
    ]
    for start, end in events:
        start.record()
        fn()
        end.record()
    events[-1][1].synchronize()

    times_ms = [float(start.elapsed_time(end)) for start, end in events]
    ordered = sorted(times_ms)
    return {
        "mean_ms": statistics.fmean(times_ms),
        "p50_ms": statistics.median(times_ms),
        "p90_ms": ordered[int(0.9 * (len(ordered) - 1))],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def _run_case(
    m: int,
    n: int,
    k: int,
    group_size: int,
    device: torch.device,
    warmup: int,
    iters: int,
) -> dict[str, object]:
    if n % 8 != 0 or k % group_size != 0:
        raise ValueError("N must be divisible by 8 and K by group size.")

    qweight = torch.zeros((k, n // 8), dtype=torch.int32, device=device)
    scales = torch.ones((k // group_size, n), dtype=torch.float16, device=device)
    qzeros = torch.zeros((k // group_size, n // 8), dtype=torch.int32, device=device)
    tm_weight, tm_scales, meta = sm70_ops.awq_sm70_prepare(
        qweight, scales, qzeros, group_size
    )
    k_ld = int(meta[0].item())
    q_ld = int(meta[1].item())
    del qweight, scales, qzeros, meta

    x = torch.randn((m, k), dtype=torch.float16, device=device)
    out = torch.empty((m, n), dtype=torch.float16, device=device)

    def run() -> None:
        sm70_ops.awq_gemm_sm70_out(
            out,
            x,
            tm_weight,
            tm_scales,
            group_size,
            k_ld,
            q_ld,
            False,
        )

    timing = _time_cuda(run, device, warmup, iters)
    return {
        "shape": {"m": m, "n": n, "k": k},
        "group_size": group_size,
        "k_ld": k_ld,
        "q_ld": q_ld,
        "prepared_weight_bytes": tm_weight.numel() * tm_weight.element_size(),
        "prepared_scale_bytes": tm_scales.numel() * tm_scales.element_size(),
        **timing,
    }


def main() -> int:
    args = _parse_args()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    _require_sm70(device)
    vocab_sizes = args.n or [124160]
    cases = [
        _run_case(
            args.m,
            n,
            args.k,
            args.group_size,
            device,
            args.warmup,
            args.iters,
        )
        for n in vocab_sizes
    ]
    result = {
        "device": torch.cuda.get_device_name(device),
        "device_capability": list(torch.cuda.get_device_capability(device)),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "warmup": args.warmup,
        "iters": args.iters,
        "cases": cases,
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
