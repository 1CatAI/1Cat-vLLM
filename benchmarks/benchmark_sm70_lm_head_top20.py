# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check and time the experimental SM70 TurboMind FP16 top-20 epilogue."""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from vllm import _sm70_ops as sm70_ops


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=32768)
    parser.add_argument("--k", type=int, default=5120)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def _time_cuda(
    operation: Callable[[], Any],
    device: torch.device,
    warmup: int,
    iters: int,
) -> dict[str, float]:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize(device)

    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    ordered = sorted(samples)
    return {
        "mean_ms": statistics.fmean(samples),
        "p50_ms": statistics.median(samples),
        "p90_ms": ordered[int(0.9 * (len(ordered) - 1))],
        "p99_ms": ordered[int(0.99 * (len(ordered) - 1))],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def main() -> int:
    args = _parse_args()
    if args.top_k != 20:
        raise ValueError("The prototype epilogue is fixed to top-k=20.")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    capability = torch.cuda.get_device_capability(device)
    if capability != (7, 0):
        raise RuntimeError(f"Expected SM70, got sm_{capability[0]}{capability[1]}.")
    if not hasattr(torch.ops._C, "sm70_f16_lm_head_top20_tc_out"):
        raise RuntimeError("The experimental top-20 op is not built.")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    hidden = torch.randn((1, args.k), dtype=torch.float16, device=device)
    weight = torch.randn((args.n, args.k), dtype=torch.float16, device=device)
    prepared = sm70_ops.sm70_f16_prepare(weight)
    tm_weight = prepared[0]
    k_ld = int(prepared[1][0].item())

    dense_logits = torch.empty((1, args.n), dtype=torch.float16, device=device)
    fused_values = torch.empty((1, args.top_k), dtype=torch.float32, device=device)
    fused_indices = torch.empty((1, args.top_k), dtype=torch.int64, device=device)

    def dense_gemm() -> None:
        sm70_ops.sm70_f16_gemm_out(
            dense_logits,
            hidden,
            tm_weight,
            k_ld,
            False,
        )

    def dense_gemm_topk() -> tuple[torch.Tensor, torch.Tensor]:
        dense_gemm()
        return torch.topk(dense_logits.float(), k=args.top_k, dim=-1)

    def fused_top20() -> None:
        sm70_ops.sm70_f16_lm_head_top20_tc_out(
            fused_values,
            fused_indices,
            hidden,
            tm_weight,
            k_ld,
            0,
            0,
        )

    reference_values, reference_indices = dense_gemm_topk()
    fused_top20()
    torch.cuda.synchronize(device)
    values_equal = bool(torch.equal(reference_values, fused_values))
    indices_equal = bool(torch.equal(reference_indices, fused_indices))
    reference_order = reference_indices.argsort(dim=-1)
    fused_order = fused_indices.argsort(dim=-1)
    support_indices_equal = bool(
        torch.equal(
            reference_indices.gather(-1, reference_order),
            fused_indices.gather(-1, fused_order),
        )
    )
    support_values_equal = bool(
        torch.equal(
            reference_values.gather(-1, reference_order),
            fused_values.gather(-1, fused_order),
        )
    )
    max_abs_diff = float((reference_values - fused_values).abs().max().item())

    result = {
        "device": torch.cuda.get_device_name(device),
        "device_capability": list(capability),
        "shape": {"m": 1, "n": args.n, "k": args.k, "top_k": args.top_k},
        "correctness": {
            "values_equal": values_equal,
            "indices_equal": indices_equal,
            "support_indices_equal": support_indices_equal,
            "support_values_equal": support_values_equal,
            "values_max_abs_diff": max_abs_diff,
            "reference_indices": reference_indices.cpu().tolist(),
            "fused_indices": fused_indices.cpu().tolist(),
        },
        "timings": {
            "sm70_f16_gemm_only": _time_cuda(
                dense_gemm, device, args.warmup, args.iters
            ),
            "sm70_f16_gemm_then_torch_topk": _time_cuda(
                dense_gemm_topk, device, args.warmup, args.iters
            ),
            "sm70_f16_fused_top20": _time_cuda(
                fused_top20, device, args.warmup, args.iters
            ),
        },
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    return 0 if support_indices_equal and support_values_equal else 1


if __name__ == "__main__":
    raise SystemExit(main())
