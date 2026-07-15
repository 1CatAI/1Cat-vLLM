# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure dynamic draft-vocabulary weight packing and M1 LM-head GEMM."""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable
from pathlib import Path

import torch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-vocab-size", type=int, default=124160)
    parser.add_argument("--hidden-size", type=int, default=5120)
    parser.add_argument("--active-size", type=int, action="append")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def _require_sm70(device: torch.device) -> None:
    if not torch.cuda.is_available() or device.type != "cuda":
        raise RuntimeError("This benchmark requires a CUDA device.")
    capability = torch.cuda.get_device_capability(device)
    if capability != (7, 0):
        raise RuntimeError(f"Expected SM70, got sm_{capability[0]}{capability[1]}.")


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


def main() -> int:
    args = _parse_args()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    _require_sm70(device)
    if args.local_vocab_size <= 0 or args.hidden_size <= 0:
        raise ValueError("Vocabulary and hidden sizes must be positive.")

    active_sizes = args.active_size or [1024, 1536, 2048, 4096, 8192, 16384]
    if any(size <= 0 or size > args.local_vocab_size for size in active_sizes):
        raise ValueError("Active sizes must be in [1, local vocab size].")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    weight = torch.zeros(
        (args.local_vocab_size, args.hidden_size),
        dtype=torch.float16,
        device=device,
    )
    hidden = torch.randn((1, args.hidden_size), dtype=torch.float16, device=device)
    cases = []

    for active_size in active_sizes:
        indices = torch.randperm(args.local_vocab_size, device=device)[:active_size]
        packed = torch.empty(
            (active_size, args.hidden_size), dtype=torch.float16, device=device
        )
        logits = torch.empty((1, active_size), dtype=torch.float16, device=device)

        def gather(
            indices: torch.Tensor = indices,
            packed: torch.Tensor = packed,
        ) -> None:
            torch.index_select(weight, 0, indices, out=packed)

        def gemm(
            packed: torch.Tensor = packed,
            logits: torch.Tensor = logits,
        ) -> None:
            torch.mm(hidden, packed.t(), out=logits)

        gather_timing = _time_cuda(gather, device, args.warmup, args.iters)
        gemm_timing = _time_cuda(gemm, device, args.warmup, args.iters)
        bytes_copied = active_size * args.hidden_size * weight.element_size()
        cases.append(
            {
                "active_size": active_size,
                "bytes_copied": bytes_copied,
                "gather_bandwidth_gbps_at_p50": (
                    bytes_copied / gather_timing["p50_ms"] / 1_000_000.0
                ),
                "gather": gather_timing,
                "torch_mm": gemm_timing,
            }
        )

    result = {
        "device": torch.cuda.get_device_name(device),
        "device_capability": list(torch.cuda.get_device_capability(device)),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "local_vocab_size": args.local_vocab_size,
        "hidden_size": args.hidden_size,
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
