# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Strict checks for migrated SM70 LM-head top1 ops.

The fused top1 route is a greedy-only optimization candidate. It is not a
sampling-path substitute. By default this checker fails unless the selected
top1 values are bitwise identical to `torch.mm(...).max(...)` and the indices
match exactly.
"""

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from vllm import _sm70_ops as sm70_ops


def _require_sm70(device: torch.device) -> None:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This check requires a CUDA device.")
    capability = torch.cuda.get_device_capability(device)
    if capability != (7, 0):
        raise RuntimeError(f"Expected SM70, got sm_{capability[0]}{capability[1]}.")


def _require_torch_op(name: str) -> None:
    if not hasattr(torch.ops._C, name):
        raise RuntimeError(
            f"Missing torch op _C::{name}. Rebuild vLLM with CUDA arch 7.0 "
            "and the SM70 TurboMind extension before running this check."
        )


def _bench_cuda_ms(fn: Callable[[], Any], warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / iters)


def _max_abs_diff(left: torch.Tensor, right: torch.Tensor) -> float:
    diff = (left - right).abs()
    return float(diff.max().item()) if diff.numel() else 0.0


def _stats(
    name: str,
    values: torch.Tensor,
    indices: torch.Tensor,
    ref_values: torch.Tensor,
    ref_indices: torch.Tensor,
    elapsed_ms: float,
) -> dict[str, Any]:
    values_f32 = values.float()
    ref_values_f32 = ref_values.float()
    value_max_diff = _max_abs_diff(values_f32, ref_values_f32)
    return {
        "name": name,
        "index_equal": bool(torch.equal(indices, ref_indices)),
        "value_equal": bool(torch.equal(values_f32, ref_values_f32)),
        "value_max_diff": value_max_diff,
        "value_mean_diff": float(
            (values_f32 - ref_values_f32).abs().float().mean().item()
        ),
        "elapsed_ms": elapsed_ms,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=1)
    parser.add_argument("--n", type=int, default=151936)
    parser.add_argument("--k", type=int, default=5120)
    parser.add_argument("--pad", type=int, default=0)
    parser.add_argument("--vocab-start", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--allow-nonzero-diff", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.pad < 0 or args.pad >= args.n:
        raise ValueError("--pad must be in [0, n).")

    device = torch.device(args.device)
    _require_sm70(device)
    if args.m == 1:
        _require_torch_op("sm70_f16_lm_head_top1_out")
    else:
        _require_torch_op("sm70_f16_lm_head_top1_tc_out")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    x = torch.randn((args.m, args.k), device=device, dtype=torch.float16)
    weight = torch.randn((args.n, args.k), device=device, dtype=torch.float16)
    torch_logits_out = torch.empty(
        (args.m, args.n), device=device, dtype=torch.float16
    )
    prepared = sm70_ops.sm70_f16_prepare(weight)
    tm_weight = prepared[0]
    k_ld = int(prepared[1][0].item())

    def torch_mm_max() -> tuple[torch.Tensor, torch.Tensor]:
        logits = torch.mm(x, weight.t())
        if args.pad:
            logits[:, -args.pad:] = -float("inf")
        values, indices = logits.max(dim=-1)
        return values, indices.to(torch.int64) + args.vocab_start

    def torch_mm_only() -> torch.Tensor:
        return torch.mm(x, weight.t(), out=torch_logits_out)

    ref_values, ref_indices = torch_mm_max()
    torch.cuda.synchronize(device)

    results = []

    if args.m == 1:
        values = torch.empty((args.m,), device=device, dtype=torch.float32)
        indices = torch.empty((args.m,), device=device, dtype=torch.int64)

        def scalar_top1() -> None:
            sm70_ops.sm70_f16_lm_head_top1_out(
                values,
                indices,
                x,
                weight,
                int(weight.stride(0)),
                args.vocab_start,
                args.pad,
            )

        scalar_top1()
        torch.cuda.synchronize(device)
        scalar_ms = _bench_cuda_ms(scalar_top1, args.warmup, args.iters)
        results.append(
            _stats("scalar", values, indices, ref_values, ref_indices, scalar_ms)
        )

    if hasattr(torch.ops._C, "sm70_f16_lm_head_top1_tc_out"):
        tc_values = torch.empty((args.m,), device=device, dtype=torch.float32)
        tc_indices = torch.empty((args.m,), device=device, dtype=torch.int64)

        def tensor_core_top1() -> None:
            sm70_ops.sm70_f16_lm_head_top1_tc_out(
                tc_values,
                tc_indices,
                x,
                tm_weight,
                k_ld,
                args.vocab_start,
                args.pad,
            )

        tensor_core_top1()
        torch.cuda.synchronize(device)
        tc_ms = _bench_cuda_ms(tensor_core_top1, args.warmup, args.iters)
        results.append(
            _stats(
                "tensor_core",
                tc_values,
                tc_indices,
                ref_values,
                ref_indices,
                tc_ms,
            )
        )

    logits_out = torch.empty((args.m, args.n), device=device, dtype=torch.float16)

    def sm70_gemm_max() -> tuple[torch.Tensor, torch.Tensor]:
        sm70_ops.sm70_f16_gemm_out(logits_out, x, tm_weight, k_ld, False)
        if args.pad:
            logits_out[:, -args.pad:] = -float("inf")
        values, indices = logits_out.max(dim=-1)
        return values, indices.to(torch.int64) + args.vocab_start

    sm70_values, sm70_indices = sm70_gemm_max()
    sm70_ms = _bench_cuda_ms(lambda: sm70_gemm_max(), args.warmup, args.iters)
    results.append(
        _stats("sm70_gemm_then_max", sm70_values, sm70_indices, ref_values,
               ref_indices, sm70_ms)
    )

    torch_mm_ms = _bench_cuda_ms(lambda: torch_mm_max(), args.warmup, args.iters)
    torch_mm_only_ms = _bench_cuda_ms(torch_mm_only, args.warmup, args.iters)
    strict_pass = all(
        result["index_equal"]
        and result["value_equal"]
        and result["value_max_diff"] == 0.0
        for result in results
    )
    report = {
        "strict": not args.allow_nonzero_diff,
        "strict_pass": strict_pass,
        "shape": {"m": args.m, "n": args.n, "k": args.k},
        "pad": args.pad,
        "vocab_start": args.vocab_start,
        "device_capability": list(torch.cuda.get_device_capability(device)),
        "torch_mm_max_ms": torch_mm_ms,
        "torch_mm_only_ms": torch_mm_only_ms,
        "results": results,
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n")

    if not strict_pass and not args.allow_nonzero_diff:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
