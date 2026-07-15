# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Microbenchmark exact split-K reduction alternatives for TP4 AWQ MLP-down."""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


def _load_extension() -> object:
    source = Path(__file__).resolve().parent / "csrc" / "sm70_awq_splitk_reducer_micro.cu"
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.0")
    return load(
        name="sm70_awq_splitk_reducer_micro_v1",
        sources=[str(source)],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-lineinfo"],
        verbose=os.getenv("VLLM_SM70_REDUCER_EXTENSION_VERBOSE") == "1",
    )


def _time(fn: object, device: torch.device, warmup: int, iters: int, repeats: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)

    values: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            fn()
        end.record()
        end.synchronize()
        values.append(float(start.elapsed_time(end)) * 1000.0 / repeats)

    ordered = sorted(values)
    return {
        "mean_us": statistics.fmean(values),
        "min_us": min(values),
        "p50_us": statistics.median(values),
        "p90_us": ordered[int(0.9 * (len(ordered) - 1))],
        "max_us": max(values),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--splits", type=int, default=7)
    parser.add_argument("--tiles", type=int, default=20)
    parser.add_argument("--elements", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--batch-repeats", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--candidate-first", action="store_true")
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    device = torch.device(args.device)
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(device) != (7, 0):
        raise RuntimeError("This benchmark requires an SM70 CUDA device.")
    if args.splits < 2 or args.tiles < 1 or args.elements < 1 or args.batch_repeats < 1:
        raise ValueError("splits, tiles, elements, and batch-repeats must be positive.")

    extension = _load_extension()
    torch.manual_seed(args.seed)
    partials = torch.randn(
        (args.splits, args.tiles, args.elements), device=device, dtype=torch.float32
    )
    serial_running = torch.empty_like(partials)
    last_staged = torch.empty_like(partials)
    serial_locks = torch.zeros(args.tiles, device=device, dtype=torch.int32)
    last_counters = torch.zeros(args.tiles, device=device, dtype=torch.int32)
    serial_output = torch.empty((args.tiles, args.elements), device=device, dtype=torch.float32)
    last_output = torch.empty_like(serial_output)

    def serial() -> None:
        extension.serial(partials, serial_running, serial_locks, serial_output)

    def last_arrival() -> None:
        extension.last_arrival(partials, last_staged, last_counters, last_output)

    serial()
    last_arrival()
    torch.cuda.synchronize(device)
    exact = bool(torch.equal(serial_output, last_output))
    if not exact:
        raise RuntimeError("Last-arrival reduction changed FP32 output bits.")
    if bool(torch.count_nonzero(serial_locks).item()) or bool(torch.count_nonzero(last_counters).item()):
        raise RuntimeError("Reducer did not restore its per-tile synchronization state.")

    if args.candidate_first:
        last_timing = _time(last_arrival, device, args.warmup, args.iters, args.batch_repeats)
        serial_timing = _time(serial, device, args.warmup, args.iters, args.batch_repeats)
    else:
        serial_timing = _time(serial, device, args.warmup, args.iters, args.batch_repeats)
        last_timing = _time(last_arrival, device, args.warmup, args.iters, args.batch_repeats)
    delta_us = last_timing["mean_us"] - serial_timing["mean_us"]
    payload = {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "device_capability": list(torch.cuda.get_device_capability(device)),
        "shape": {
            "splits": args.splits,
            "tiles": args.tiles,
            "elements": args.elements,
            "note": "TP4 MLP-down has 20 N256 tiles and split 7; only M=1 output rows are modeled.",
        },
        "warmup": args.warmup,
        "iters": args.iters,
        "batch_repeats": args.batch_repeats,
        "seed": args.seed,
        "candidate_first": args.candidate_first,
        "serial": serial_timing,
        "last_arrival": last_timing,
        "delta_us": delta_us,
        "delta_pct": 100.0 * delta_us / serial_timing["mean_us"],
        "output_exact": exact,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
