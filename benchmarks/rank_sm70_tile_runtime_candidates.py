#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rank SM70 TileRT-style GEMM/all-reduce candidates from nsys sqlite.

This diagnostic is intentionally offline-only. It reads an Nsight Systems
sqlite export and estimates which GEMM -> TP all-reduce boundaries are worth
turning into a tile-level runtime experiment first.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def _mean(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _load_strings(cur: sqlite3.Cursor) -> dict[int, str]:
    return {row[0]: row[1] for row in cur.execute("SELECT id, value FROM StringIds")}


def _load_graph_pairs(
    cur: sqlite3.Cursor, strings: dict[int, str]
) -> list[tuple[int, int]]:
    graph_ids = [
        sid for sid, value in strings.items() if value == "cudaGraphLaunch_v10000"
    ]
    if not graph_ids:
        raise RuntimeError("no cudaGraphLaunch_v10000 rows found")
    placeholders = ",".join("?" for _ in graph_ids)
    launches = cur.execute(
        f"""
        SELECT start, end
        FROM CUPTI_ACTIVITY_KIND_RUNTIME
        WHERE nameId IN ({placeholders})
        ORDER BY start
        """,
        graph_ids,
    ).fetchall()
    pairs: list[tuple[int, int]] = []
    for idx in range(0, len(launches) - 1, 2):
        a_start, a_end = launches[idx]
        b_start, b_end = launches[idx + 1]
        pairs.append((min(a_start, b_start), max(a_end, b_end)))
    return pairs


def _load_kernels(cur: sqlite3.Cursor, strings: dict[int, str]) -> list[dict[str, Any]]:
    kernels: list[dict[str, Any]] = []
    for row in cur.execute(
        """
        SELECT start, end, deviceId, streamId, shortName, demangledName,
               gridX, gridY, gridZ, blockX, blockY, blockZ,
               registersPerThread, dynamicSharedMemory
        FROM CUPTI_ACTIVITY_KIND_KERNEL
        ORDER BY start
        """
    ):
        (
            start,
            end,
            device,
            stream,
            short_id,
            demangled_id,
            grid_x,
            grid_y,
            grid_z,
            block_x,
            block_y,
            block_z,
            regs,
            smem,
        ) = row
        kernels.append(
            {
                "start": start,
                "end": end,
                "duration_us": (end - start) / 1000.0,
                "device": device,
                "stream": stream,
                "short": strings.get(short_id, str(short_id)),
                "demangled": strings.get(demangled_id, str(demangled_id)),
                "grid": [grid_x, grid_y, grid_z],
                "block": [block_x, block_y, block_z],
                "regs": regs,
                "smem": smem,
            }
        )
    return kernels


def _grid_key(kernel: dict[str, Any]) -> str:
    grid = kernel["grid"]
    return f"{grid[0]}x{grid[1]}x{grid[2]}"


def _cta_count(kernel: dict[str, Any]) -> int:
    gx, gy, gz = kernel["grid"]
    return int(gx) * int(gy) * int(gz)


def _is_allreduce(kernel: dict[str, Any]) -> bool:
    return kernel["short"] in {
        "cross_device_reduce_1stage",
        "cross_device_reduce_2stage",
        "cross_device_reduce_sum2_1stage",
        "cross_device_reduce_sum2_2stage",
    }


def _is_gemm(kernel: dict[str, Any]) -> bool:
    return kernel["short"] == "gemm_kernel"


def analyze(sqlite_path: Path, skip_pairs: int, sm_count: int) -> dict[str, Any]:
    con = sqlite3.connect(sqlite_path)
    cur = con.cursor()
    strings = _load_strings(cur)
    pairs = _load_graph_pairs(cur, strings)
    kernels = _load_kernels(cur, strings)

    windows: list[tuple[int, int, int]] = []
    for idx in range(skip_pairs, len(pairs) - 1):
        windows.append((idx, pairs[idx][0], pairs[idx + 1][0]))
    if not windows:
        raise RuntimeError("no steady graph windows found")

    token_ms = [
        (windows[i + 1][1] - windows[i][1]) / 1e6 for i in range(len(windows) - 1)
    ]

    stats: dict[tuple[int, str], dict[str, Any]] = defaultdict(
        lambda: {
            "count": 0,
            "ar_us": 0.0,
            "ar_samples": [],
            "prev_gemm_us": 0.0,
            "prev_gemm_samples": [],
            "cta_count": 0,
            "grid": None,
            "regs": None,
            "smem": None,
            "block": None,
            "awq_demangled_seen": False,
        }
    )

    kernel_idx = 0
    for _, start, end in windows:
        while kernel_idx < len(kernels) and kernels[kernel_idx]["end"] <= start:
            kernel_idx += 1
        selected: list[dict[str, Any]] = []
        scan_idx = kernel_idx
        while scan_idx < len(kernels) and kernels[scan_idx]["start"] < end:
            if kernels[scan_idx]["start"] >= start:
                selected.append(kernels[scan_idx])
            scan_idx += 1

        per_device: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for kernel in selected:
            per_device[int(kernel["device"])].append(kernel)

        for device, seq in per_device.items():
            for idx, kernel in enumerate(seq):
                if not _is_allreduce(kernel):
                    continue
                prev = seq[idx - 1] if idx > 0 else None
                if prev is None or not _is_gemm(prev):
                    continue
                grid = _grid_key(prev)
                key = (device, grid)
                item = stats[key]
                item["count"] += 1
                item["ar_us"] += kernel["duration_us"]
                item["ar_samples"].append(kernel["duration_us"])
                item["prev_gemm_us"] += prev["duration_us"]
                item["prev_gemm_samples"].append(prev["duration_us"])
                item["cta_count"] = _cta_count(prev)
                item["grid"] = prev["grid"]
                item["regs"] = prev["regs"]
                item["smem"] = prev["smem"]
                item["block"] = prev["block"]
                item["awq_demangled_seen"] = item["awq_demangled_seen"] or (
                    "uint4_t" in prev["demangled"]
                    or "Operand_B_Pack<turbomind::uint4" in prev["demangled"]
                )

    candidates = []
    for (device, grid), item in sorted(stats.items()):
        calls_per_token = item["count"] / len(windows)
        ar_us_per_token = item["ar_us"] / len(windows)
        prev_gemm_us_per_token = item["prev_gemm_us"] / len(windows)
        cta_count = int(item["cta_count"])
        # This is an optimistic upper bound. Real overlap can be lower if the
        # producer publishes tiles too late, flags are expensive, or occupancy
        # drops after fusion.
        overlap_ceiling_us = min(ar_us_per_token, prev_gemm_us_per_token)
        candidates.append(
            {
                "device": device,
                "prev_gemm_grid": grid,
                "prev_gemm_ctas": cta_count,
                "eligible_ctas_ge_sm_count": cta_count >= sm_count,
                "awq_demangled_seen": bool(item["awq_demangled_seen"]),
                "calls_per_token": calls_per_token,
                "allreduce_us_per_token": ar_us_per_token,
                "prev_gemm_us_per_token": prev_gemm_us_per_token,
                "optimistic_overlap_ceiling_us_per_token": overlap_ceiling_us,
                "mean_allreduce_us": _mean(item["ar_samples"]),
                "median_allreduce_us": _median(item["ar_samples"]),
                "mean_prev_gemm_us": _mean(item["prev_gemm_samples"]),
                "median_prev_gemm_us": _median(item["prev_gemm_samples"]),
                "block": item["block"],
                "regs": item["regs"],
                "dynamic_smem": item["smem"],
            }
        )

    candidates.sort(
        key=lambda row: (
            row["optimistic_overlap_ceiling_us_per_token"],
            row["allreduce_us_per_token"],
        ),
        reverse=True,
    )
    return {
        "sqlite": str(sqlite_path),
        "steady_windows": len(windows),
        "graph_pairs": len(pairs),
        "skip_pairs": skip_pairs,
        "sm_count": sm_count,
        "token_start_gap_ms": {
            "mean": _mean(token_ms),
            "median": _median(token_ms),
            "min": min(token_ms) if token_ms else 0.0,
            "max": max(token_ms) if token_ms else 0.0,
        },
        "candidates": candidates,
    }


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# Tile Runtime Candidate Ranking",
        "",
        f"- sqlite: `{payload['sqlite']}`",
        f"- steady windows: `{payload['steady_windows']}`",
        f"- mean token start gap: `{payload['token_start_gap_ms']['mean']:.3f} ms`",
        f"- SM count threshold: `{payload['sm_count']}`",
        "",
        "| rank | device | prev GEMM grid | CTAs | calls/token | AR us/token | "
        "GEMM us/token | overlap ceiling us/token | mean AR us | mean GEMM us | AWQ |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for idx, row in enumerate(payload["candidates"], 1):
        lines.append(
            "| {rank} | {device} | `{grid}` | {ctas} | {calls:.3f} | "
            "{ar:.3f} | {gemm:.3f} | {ceil:.3f} | {mean_ar:.3f} | "
            "{mean_gemm:.3f} | {awq} |".format(
                rank=idx,
                device=row["device"],
                grid=row["prev_gemm_grid"],
                ctas=row["prev_gemm_ctas"],
                calls=row["calls_per_token"],
                ar=row["allreduce_us_per_token"],
                gemm=row["prev_gemm_us_per_token"],
                ceil=row["optimistic_overlap_ceiling_us_per_token"],
                mean_ar=row["mean_allreduce_us"],
                mean_gemm=row["mean_prev_gemm_us"],
                awq="yes" if row["awq_demangled_seen"] else "unknown",
            )
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("--skip-pairs", type=int, default=1)
    parser.add_argument("--sm-count", type=int, default=80)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--md-out", type=Path)
    args = parser.parse_args()

    payload = analyze(args.sqlite, args.skip_pairs, args.sm_count)
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        args.json_out.write_text(text + "\n")
    if args.md_out:
        write_markdown(payload, args.md_out)


if __name__ == "__main__":
    main()
