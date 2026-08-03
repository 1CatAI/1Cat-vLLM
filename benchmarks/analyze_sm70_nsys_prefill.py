#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize one SM70 prefill request from an Nsight Systems SQLite export."""

from __future__ import annotations

import argparse
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Kernel:
    start: int
    end: int
    graph_node: int | None
    name: str
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    registers: int
    dynamic_smem: int


def _category(name: str) -> str:
    lower = name.lower()
    if "turbomind::gemm" in lower and "fp4_e2m1_t" in lower:
        return "TurboMind MXFP4 MoE GEMM"
    if "turbomind::gemm" in lower and "__nv_fp8_e4m3" in lower:
        return "TurboMind FP8 dense GEMM"
    if "turbomind::gemm" in lower:
        return "TurboMind other GEMM"
    if "nccl" in lower:
        return "NCCL collectives"
    if "mhc_" in lower or "hc_prenorm" in lower or "hc_head" in lower:
        return "mHC"
    if any(
        marker in lower
        for marker in (
            "kv_compress",
            "insert_k",
            "gather_k",
            "slot_mapping",
            "qnorm_rope",
            "inverse_rope",
            "save_partial_states",
        )
    ):
        return "KV compression/indexer/rope"
    if "sparse_gathered" in lower or "sparse_attn" in lower:
        return "SM70 sparse MLA/SWA attention"
    if any(
        marker in lower
        for marker in (
            "moerouting",
            "topkgating",
            "experttoken",
            "expertfirsttoken",
            "expandinputrows",
            "radixsort",
        )
    ):
        return "MoE routing"
    if any(marker in lower for marker in ("volta_", "cublas", "cutlass")):
        return "FP16/CUTLASS GEMM"
    if any(marker in lower for marker in ("rms_norm", "rmsnorm", "act_and_mul")):
        return "Norm/activation"
    if any(marker in lower for marker in ("argmax", "softmax")):
        return "LM head/sample"
    return "Other"


def _union_ns(intervals: list[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    total = 0
    current_start, current_end = sorted(intervals)[0]
    for start, end in sorted(intervals)[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    return total + current_end - current_start


def _short_name(name: str) -> str:
    return name if len(name) <= 120 else name[:117] + "..."


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("sqlite", type=Path)
    parser.add_argument(
        "--before-first-graph",
        action="store_true",
        help="Exclude decode beginning at the first CUDA Graph kernel per GPU.",
    )
    parser.add_argument("--device", type=int)
    parser.add_argument("--top-kernels", type=int, default=20)
    args = parser.parse_args()

    connection = sqlite3.connect(args.sqlite)
    rows = list(
        connection.execute(
            """
            SELECT k.deviceId, k.start, k.end, k.graphNodeId, s.value,
                   k.gridX, k.gridY, k.gridZ,
                   k.blockX, k.blockY, k.blockZ,
                   k.registersPerThread, k.dynamicSharedMemory
            FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
            JOIN StringIds AS s ON s.id = k.demangledName
            ORDER BY k.deviceId, k.start
            """
        )
    )
    by_device: dict[int, list[Kernel]] = defaultdict(list)
    for (
        device,
        start,
        end,
        graph_node,
        name,
        grid_x,
        grid_y,
        grid_z,
        block_x,
        block_y,
        block_z,
        registers,
        dynamic_smem,
    ) in rows:
        by_device[device].append(
            Kernel(
                start=start,
                end=end,
                graph_node=graph_node,
                name=name,
                grid=(grid_x, grid_y, grid_z),
                block=(block_x, block_y, block_z),
                registers=registers,
                dynamic_smem=dynamic_smem,
            )
        )

    selected: dict[int, list[Kernel]] = {}
    for device, kernels in by_device.items():
        graph_starts = [
            kernel.start for kernel in kernels if kernel.graph_node is not None
        ]
        cutoff = min(graph_starts) if args.before_first_graph and graph_starts else None
        selected[device] = [
            kernel
            for kernel in kernels
            if (cutoff is None or (kernel.graph_node is None and kernel.end <= cutoff))
        ]

    summaries = {}
    for device, kernels in selected.items():
        if not kernels:
            continue
        intervals = [(kernel.start, kernel.end) for kernel in kernels]
        start = min(item[0] for item in intervals)
        end = max(item[1] for item in intervals)
        summaries[device] = {
            "count": len(kernels),
            "service_ns": sum(item[1] - item[0] for item in intervals),
            "busy_ns": _union_ns(intervals),
            "envelope_ns": end - start,
        }

    print("device  kernels  service_ms  busy_union_ms  envelope_ms  idle_gap_ms")
    for device, summary in sorted(summaries.items()):
        idle_ns = summary["envelope_ns"] - summary["busy_ns"]
        print(
            f"{device:>6}  {summary['count']:>7}  "
            f"{summary['service_ns'] / 1e6:>10.3f}  "
            f"{summary['busy_ns'] / 1e6:>13.3f}  "
            f"{summary['envelope_ns'] / 1e6:>11.3f}  "
            f"{idle_ns / 1e6:>11.3f}"
        )

    if not summaries:
        raise SystemExit("No CUDA kernels found in the selected window")
    device = args.device
    if device is None:
        device = max(summaries, key=lambda item: summaries[item]["envelope_ns"])
    kernels = selected[device]
    total_service_ns = sum(kernel.end - kernel.start for kernel in kernels)

    categories: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    kernel_totals: dict[
        tuple[
            str,
            tuple[int, int, int],
            tuple[int, int, int],
            int,
            int,
        ],
        list[int],
    ] = defaultdict(lambda: [0, 0])
    for kernel in kernels:
        duration = kernel.end - kernel.start
        category = _category(kernel.name)
        categories[category][0] += duration
        categories[category][1] += 1
        signature = (
            kernel.name,
            kernel.grid,
            kernel.block,
            kernel.registers,
            kernel.dynamic_smem,
        )
        kernel_totals[signature][0] += duration
        kernel_totals[signature][1] += 1

    print(f"\nCritical device: {device}")
    print("category                            service_ms  service_%  launches")
    for category, (duration, count) in sorted(
        categories.items(), key=lambda item: item[1][0], reverse=True
    ):
        print(
            f"{category:<35} {duration / 1e6:>10.3f}  "
            f"{duration / total_service_ns * 100:>8.2f}  {count:>8}"
        )

    print("\nTop kernels by summed service time")
    print("service_ms  service_%  launches  launch geometry and kernel")
    for signature, (duration, count) in sorted(
        kernel_totals.items(), key=lambda item: item[1][0], reverse=True
    )[: args.top_kernels]:
        name, grid, block, registers, dynamic_smem = signature
        geometry = (
            f"grid={grid} block={block} regs={registers} dynamic_smem={dynamic_smem}B"
        )
        print(
            f"{duration / 1e6:>10.3f}  {duration / total_service_ns * 100:>8.2f}  "
            f"{count:>8}  {geometry}  {_short_name(name)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
