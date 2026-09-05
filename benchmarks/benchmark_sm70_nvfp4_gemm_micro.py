# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Microbenchmark SM70 TurboMind and QPN4 NVFP4 dense GEMMs.

The default case set models one Qwen3.5-27B-NVFP4 decode token on one TP rank
with tensor_parallel_size=2. It times only nvfp4_gemm_sm70_out after synthetic
weights have been prepared and dispatch has been warmed, so the weighted total
is comparable to the Nsight "TurboMind NVFP4 GEMM" critical-path bucket. The
``qpn4-prefill`` mode measures the existing bounded-workspace large-M route,
including weight dequantization on every call.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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


def _make_input(
    m: int,
    k: int,
    device: torch.device,
    *,
    data_pattern: str,
    generator: torch.Generator,
) -> torch.Tensor:
    if data_pattern == "random":
        return torch.randn(
            (m, k),
            dtype=torch.float16,
            device=device,
            generator=generator,
        )
    values = torch.arange(m * k, device=device, dtype=torch.int32)
    values = ((values % 1024).to(torch.float32) / 512.0) - 1.0
    return values.reshape(m, k).to(torch.float16)


def _make_qweight(
    k: int,
    n: int,
    device: torch.device,
    *,
    data_pattern: str,
    generator: torch.Generator,
) -> torch.Tensor:
    if data_pattern == "random":
        return torch.randint(
            0,
            16,
            (k, n),
            dtype=torch.uint8,
            device=device,
            generator=generator,
        )
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


def _make_scales(
    k: int,
    n: int,
    group_size: int,
    device: torch.device,
    *,
    data_pattern: str,
    generator: torch.Generator,
) -> torch.Tensor:
    groups = k // group_size
    if data_pattern == "random":
        return (
            torch.rand(
                (groups, n),
                dtype=torch.float32,
                device=device,
                generator=generator,
            )
            * 0.9921875
            + 0.0078125
        ).to(torch.float16)
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
    gated_silu: bool,
    num_experts: int,
    tensor_out: Path | None,
    data_pattern: str,
    generator: torch.Generator,
    expert_row_counts: list[int] | None,
    prime_slot_groups: bool,
    prime_active_groups: int,
) -> dict[str, Any]:
    if case.k % group_size != 0:
        raise ValueError(f"{case.label}: K={case.k} not divisible by {group_size}.")
    qweight = _make_qweight(
        case.k,
        case.n,
        device,
        data_pattern=data_pattern,
        generator=generator,
    )
    scales = _make_scales(
        case.k,
        case.n,
        group_size,
        device,
        data_pattern=data_pattern,
        generator=generator,
    )
    x = _make_input(
        case.m,
        case.k,
        device,
        data_pattern=data_pattern,
        generator=generator,
    )
    if gated_silu and case.n % 2 != 0:
        raise ValueError(f"{case.label}: gated-SiLU requires even N, got {case.n}.")
    output_size = case.n // 2 if gated_silu else case.n
    out = torch.empty((case.m, output_size), dtype=torch.float16, device=device)

    k_ld = 0
    q_ld = 0
    tm_weight = None
    tm_scales = None
    qpn4_codes = None
    qpn4_scales = None
    dense_weight = None
    qweight_packed = None
    partials = None
    grouped_weights = None
    grouped_scales = None
    ptrs_w = None
    ptrs_s = None
    expert_offsets = None
    expert_ids = None
    if mode == "gemm":
        tm_weight, tm_scales, meta = ops.nvfp4_sm70_prepare(
            qweight, scales, group_size, gated_silu
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
            gated_silu,
        )
    elif mode == "qpn4-prefill":
        qpn4_codes, qpn4_scales = ops.nvfp4_qpn4_prepare_sm70(qweight, scales)
        dense_weight = torch.empty((case.k, case.n), dtype=torch.float16, device=device)
        run = partial(
            ops.nvfp4_qpn4_dispatch_sm70_out,
            out,
            dense_weight.data_ptr(),
            x,
            qpn4_codes,
            qpn4_scales,
            0.0,
            False,
            gated_silu,
        )
    elif mode == "grouped-moe":
        if gated_silu:
            raise ValueError("grouped-moe selector screens require --gated-silu off.")
        if num_experts <= 0 or case.m != num_experts:
            raise ValueError("grouped-moe M must equal --num-experts.")
        tm_weight, tm_scales, meta = ops.nvfp4_sm70_prepare(
            qweight, scales, group_size, False
        )
        k_ld = int(meta[0].item())
        q_ld = int(meta[1].item())
        grouped_weights = tm_weight.unsqueeze(0).repeat(num_experts, 1, 1)
        grouped_scales = tm_scales.unsqueeze(0).repeat(num_experts, 1, 1)
        ptrs_w, ptrs_s = ops.awq_moe_build_strided_ptrs(
            grouped_weights,
            grouped_scales,
            k_ld,
            q_ld,
            num_experts,
        )
        if expert_row_counts is None:
            expert_offsets = torch.arange(
                num_experts + 1, dtype=torch.int32, device=device
            )
        else:
            if any(count <= 0 for count in expert_row_counts):
                raise ValueError("Expert row counts must all be positive.")
            if sum(expert_row_counts) != case.m:
                raise ValueError(
                    "Expert row counts must sum to grouped-moe M: "
                    f"sum={sum(expert_row_counts)}, M={case.m}."
                )
            if len(expert_row_counts) > num_experts:
                raise ValueError(
                    "Expert row count entries cannot exceed --num-experts."
                )
            offsets = [0]
            for count in expert_row_counts:
                offsets.append(offsets[-1] + count)
            offsets.extend([case.m] * (num_experts - len(expert_row_counts)))
            expert_offsets = torch.tensor(offsets, dtype=torch.int32, device=device)
        expert_ids = torch.arange(num_experts, dtype=torch.int32, device=device)
        run = partial(
            ops.nvfp4_moe_dense_stage_sm70_out,
            out,
            x,
            expert_offsets,
            expert_ids,
            ptrs_w,
            ptrs_s,
            num_experts,
            case.k,
            case.n,
            group_size,
        )
        if prime_slot_groups and expert_row_counts is not None:
            slot_offsets = torch.arange(
                num_experts + 1, dtype=torch.int32, device=device
            )
            ops.nvfp4_moe_dense_stage_sm70_out(
                out,
                x,
                slot_offsets,
                expert_ids,
                ptrs_w,
                ptrs_s,
                num_experts,
                case.k,
                case.n,
                group_size,
            )
            torch.cuda.synchronize(device)
        elif prime_active_groups > 0 and expert_row_counts is not None:
            if prime_active_groups > num_experts or case.m < prime_active_groups:
                raise ValueError(
                    "--prime-active-groups must not exceed grouped M or experts."
                )
            base, remainder = divmod(case.m, prime_active_groups)
            prime_counts = [
                base + (group < remainder) for group in range(prime_active_groups)
            ]
            offsets = [0]
            for count in prime_counts:
                offsets.append(offsets[-1] + count)
            offsets.extend([case.m] * (num_experts - prime_active_groups))
            prime_offsets = torch.tensor(offsets, dtype=torch.int32, device=device)
            ops.nvfp4_moe_dense_stage_sm70_out(
                out,
                x,
                prime_offsets,
                expert_ids,
                ptrs_w,
                ptrs_s,
                num_experts,
                case.k,
                case.n,
                group_size,
            )
            torch.cuda.synchronize(device)
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
    output_cpu = out.detach().contiguous().cpu()
    output_sha256 = hashlib.sha256(
        output_cpu.view(torch.uint8).numpy().tobytes()
    ).hexdigest()
    if tensor_out is not None:
        tensor_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(output_cpu, tensor_out)
    weighted_mean_ms = float(case.count) * timing["mean_us"] / 1000.0
    row = {
        "mode": mode,
        "label": case.label,
        "count": case.count,
        "m": case.m,
        "n": case.n,
        "k": case.k,
        "group_size": group_size,
        "gated_silu": gated_silu,
        "num_experts": num_experts if mode == "grouped-moe" else 1,
        "active_expert_groups": (
            len(expert_row_counts)
            if mode == "grouped-moe" and expert_row_counts is not None
            else num_experts
            if mode == "grouped-moe"
            else 1
        ),
        "expert_row_counts": expert_row_counts if mode == "grouped-moe" else None,
        "prime_slot_groups": bool(prime_slot_groups and mode == "grouped-moe"),
        "prime_active_groups": (prime_active_groups if mode == "grouped-moe" else 0),
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
                    else (
                        f"qpn4_prefill_{case.m}x{case.n}x{case.k}"
                        if mode == "qpn4-prefill"
                        else (
                            f"grouped_moe_e{num_experts}_{case.m}x{case.n}x{case.k}"
                            if mode == "grouped-moe"
                            else f"sm70_f16_nvfp4k{group_size}_f16_tnt_fff_"
                            f"{case.m}x{case.n}x{case.k}_1"
                        )
                    )
                )
            )
        ),
        "output_sha256": output_sha256,
        "tensor_out": str(tensor_out) if tensor_out is not None else None,
        **timing,
        "weighted_mean_ms": weighted_mean_ms,
    }
    del (
        qweight,
        scales,
        tm_weight,
        tm_scales,
        qpn4_codes,
        qpn4_scales,
        dense_weight,
        qweight_packed,
        partials,
        grouped_weights,
        grouped_scales,
        ptrs_w,
        ptrs_s,
        expert_offsets,
        expert_ids,
        x,
        out,
    )
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
        "gated_silu",
        "num_experts",
        "active_expert_groups",
        "expert_row_counts",
        "prime_slot_groups",
        "prime_active_groups",
        "gemv_split_k",
        "k_ld",
        "q_ld",
        "desc_hint",
        "output_sha256",
        "tensor_out",
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--data-pattern",
        choices=("periodic", "random"),
        default="periodic",
        help="Synthetic tensor pattern; random is intended for numerical A/B gates.",
    )
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        help="Time CUDA graph replay of each single-op case after warmup.",
    )
    parser.add_argument(
        "--mode",
        choices=(
            "gemm",
            "grouped-moe",
            "qpn4-prefill",
            "raw-gemv",
            "raw-gemv-warp",
            "raw-gemv-h2",
        ),
        default="gemm",
        help="Operator implementation to benchmark.",
    )
    parser.add_argument(
        "--gated-silu",
        action="store_true",
        help="Fuse gate/up SiLU and emit N/2 output columns.",
    )
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument(
        "--expert-row-counts",
        help=(
            "Comma-separated positive row counts for grouped-moe. Counts must "
            "sum to M; unused expert-offset entries become graph-safe empty tails."
        ),
    )
    parser.add_argument(
        "--prime-slot-groups",
        action="store_true",
        help=(
            "Before timing grouped expert rows, tune the same shape with one row "
            "per expert to reproduce the production warmup cache choice."
        ),
    )
    parser.add_argument(
        "--prime-active-groups",
        type=int,
        default=0,
        help=(
            "Tune grouped-MoE on a balanced number of active groups before "
            "timing --expert-row-counts."
        ),
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
    parser.add_argument(
        "--tensor-out",
        type=Path,
        help="Save the exact output tensor; requires exactly one --case.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.prime_active_groups < 0:
        raise ValueError("--prime-active-groups must be non-negative.")
    if args.prime_slot_groups and args.prime_active_groups:
        raise ValueError(
            "--prime-slot-groups and --prime-active-groups are mutually exclusive."
        )
    device = torch.device(args.device)
    _require_sm70(device)
    ops = _import_sm70_ops()
    if not hasattr(torch.ops._C, "nvfp4_sm70_prepare"):
        raise RuntimeError("Missing _C::nvfp4_sm70_prepare.")
    if not hasattr(torch.ops._C, "nvfp4_gemm_sm70_out"):
        raise RuntimeError("Missing _C::nvfp4_gemm_sm70_out.")
    if args.mode == "qpn4-prefill":
        required_qpn4_ops = (
            "nvfp4_qpn4_prepare_sm70",
            "nvfp4_qpn4_dispatch_sm70_out",
        )
        missing_qpn4_ops = [
            name for name in required_qpn4_ops if not hasattr(torch.ops._C, name)
        ]
        if missing_qpn4_ops:
            raise RuntimeError(f"Missing QPN4 operators: {missing_qpn4_ops}.")
    if args.mode == "grouped-moe" and not hasattr(
        torch.ops._C, "nvfp4_moe_dense_stage_sm70_out"
    ):
        raise RuntimeError("Missing _C::nvfp4_moe_dense_stage_sm70_out.")
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
    if args.tensor_out is not None and len(cases) != 1:
        raise ValueError("--tensor-out requires exactly one benchmark case.")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    expert_row_counts = (
        [int(value) for value in args.expert_row_counts.split(",")]
        if args.expert_row_counts
        else None
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
            gated_silu=args.gated_silu,
            num_experts=args.num_experts,
            tensor_out=args.tensor_out,
            data_pattern=args.data_pattern,
            generator=generator,
            expert_row_counts=expert_row_counts,
            prime_slot_groups=args.prime_slot_groups,
            prime_active_groups=args.prime_active_groups,
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
        "num_experts": args.num_experts if args.mode == "grouped-moe" else 1,
        "expert_row_counts": expert_row_counts,
        "prime_slot_groups": args.prime_slot_groups,
        "prime_active_groups": args.prime_active_groups,
        "mode": args.mode,
        "gated_silu": args.gated_silu,
        "gemv_split_k": (
            args.gemv_split_k if args.mode in ("raw-gemv", "raw-gemv-h2") else 0
        ),
        "warmup": args.warmup,
        "iters": args.iters,
        "cuda_graph": args.cuda_graph,
        "data_pattern": args.data_pattern,
        "seed": args.seed,
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
