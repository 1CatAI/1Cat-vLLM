#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure GPU busy intervals and idle gaps inside Nsight replay ranges."""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

GLOBAL_PID_MASK = -16777216


def _family(name: str) -> str:
    lower = name.lower()
    if name == "[replay start]" or name == "[next replay]":
        return name
    if "sm70_tp8_hierarchical_reduce_push" in lower:
        return "TP8 push all-reduce"
    if "sm70_tp8_hierarchical" in lower:
        return "TP8 pull all-reduce"
    if "nccl" in lower:
        return "NCCL"
    if "gemm_kernel<turbomind" in lower:
        return "TurboMind NVFP4 GEMM"
    if "sm70_glm_mhc_pre_norm" in lower:
        return "mHC pre/norm"
    if "sm70_mhc_dot" in lower:
        return "mHC dot"
    if "sm70_mhc_post" in lower:
        return "mHC post"
    if "cutlass_70_wmma" in lower:
        return "CUTLASS WMMA GEMM"
    if "volta_fp16" in lower:
        return "cuBLAS Volta GEMM"
    if "splitkreduce" in lower:
        return "cuBLAS split-K reduce"
    if "grouped_topk" in lower or "moerouting" in lower or "radixsort" in lower:
        return "MoE routing"
    if "gated_delta_rule" in lower or "causal_conv1d" in lower:
        return "GDN"
    if "flash" in lower or "attention" in lower:
        return "attention"
    if "fp8_kv" in lower:
        return "FP8 KV"
    if "copy" in lower or "elementwise" in lower:
        return "elementwise/copy"
    if name.startswith("[CUDA memcpy"):
        return "CUDA memcpy"
    if name.startswith("[CUDA memset"):
        return "CUDA memset"
    return name[:96]


def _stats(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)

    def percentile(q: float) -> float:
        if not ordered:
            return 0.0
        index = (len(ordered) - 1) * q
        low = int(index)
        high = min(low + 1, len(ordered) - 1)
        weight = index - low
        return ordered[low] * (1.0 - weight) + ordered[high] * weight

    return {
        "mean": statistics.fmean(values) if values else 0.0,
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p99": percentile(0.99),
        "min": ordered[0] if ordered else 0.0,
        "max": ordered[-1] if ordered else 0.0,
    }


def _load_ranges(
    con: sqlite3.Connection, label: str
) -> dict[int, list[tuple[int, int | None]]]:
    ranges: dict[int, list[tuple[int, int | None]]] = defaultdict(list)
    query = (
        "select n.start,n.end,n.globalTid from NVTX_EVENTS n "
        "left join StringIds s on s.id=n.textId "
        "where coalesce(n.text,s.value)=? order by n.globalTid,n.start"
    )
    for start, end, tid in con.execute(query, (label,)):
        ranges[tid].append((start, end))
    return ranges


def _create_intervals(
    con: sqlite3.Connection,
    ranges: dict[int, list[tuple[int, int | None]]],
    edge_drop: int,
) -> list[dict[str, int]]:
    counts = {len(rows) for rows in ranges.values()}
    if not ranges or len(counts) != 1:
        raise RuntimeError({tid: len(rows) for tid, rows in ranges.items()})
    count = next(iter(counts))
    if count <= 2 * edge_drop + 1:
        raise RuntimeError("not enough replay ranges after dropping edges")

    con.execute(
        "create temp table replay_intervals("
        "globalPid integer,step integer,rank integer,start integer,end integer)"
    )
    records = []
    rows = []
    for rank, tid in enumerate(sorted(ranges)):
        starts = [item[0] for item in ranges[tid]]
        for step in range(edge_drop, count - edge_drop):
            record = {
                "global_pid": tid & GLOBAL_PID_MASK,
                "step": step - edge_drop,
                "rank": rank,
                "start": starts[step],
                "end": starts[step + 1],
            }
            records.append(record)
            rows.append(tuple(record.values()))
    con.executemany("insert into replay_intervals values(?,?,?,?,?)", rows)
    con.execute(
        "create index replay_intervals_pid_start_end "
        "on replay_intervals(globalPid,start,end)"
    )
    return records


def _load_activity(
    con: sqlite3.Connection, strings: dict[int, str]
) -> dict[tuple[int, int], list[tuple[int, int, str]]]:
    activity: dict[tuple[int, int], list[tuple[int, int, str]]] = defaultdict(list)
    kernel_query = """
        select i.step,i.rank,k.start,k.end,
               coalesce(k.demangledName,k.shortName)
        from CUPTI_ACTIVITY_KIND_KERNEL k
        join replay_intervals i
          on k.globalPid=i.globalPid and k.start>=i.start and k.start<i.end
        order by i.step,i.rank,k.start,k.end
    """
    for step, rank, start, end, name_id in con.execute(kernel_query):
        activity[(step, rank)].append((start, end, strings.get(name_id, "")))

    for table, name in (
        ("CUPTI_ACTIVITY_KIND_MEMCPY", "[CUDA memcpy]"),
        ("CUPTI_ACTIVITY_KIND_MEMSET", "[CUDA memset]"),
    ):
        exists = con.execute(
            "select 1 from sqlite_master where type='table' and name=?", (table,)
        ).fetchone()
        if not exists:
            continue
        query = f"""
            select i.step,i.rank,a.start,a.end
            from {table} a
            join replay_intervals i
              on a.globalPid=i.globalPid and a.start>=i.start and a.start<i.end
        """
        for step, rank, start, end in con.execute(query):
            activity[(step, rank)].append((start, end, name))

    for items in activity.values():
        items.sort(key=lambda item: (item[0], item[1]))
    return activity


def _analyze_interval(
    interval: dict[str, int],
    activity: list[tuple[int, int, str]],
    min_gap_ns: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    interval_start = interval["start"]
    interval_end = interval["end"]
    gaps: list[dict[str, Any]] = []
    raw_service_ns = sum(end - start for start, end, _name in activity)
    if not activity:
        gaps.append(
            {
                "kind": "empty",
                "start": interval_start,
                "end": interval_end,
                "duration_us": (interval_end - interval_start) / 1_000,
                "before": "[replay start]",
                "after": "[next replay]",
            }
        )
        union_service_ns = 0
    else:
        current_start, current_end, current_name = activity[0]
        union_service_ns = 0
        if current_start - interval_start >= min_gap_ns:
            gaps.append(
                {
                    "kind": "front",
                    "start": interval_start,
                    "end": current_start,
                    "duration_us": (current_start - interval_start) / 1_000,
                    "before": "[replay start]",
                    "after": current_name,
                }
            )
        for start, end, name in activity[1:]:
            if start <= current_end:
                if end > current_end:
                    current_end = end
                    current_name = name
                continue
            union_service_ns += current_end - current_start
            if start - current_end >= min_gap_ns:
                gaps.append(
                    {
                        "kind": "internal",
                        "start": current_end,
                        "end": start,
                        "duration_us": (start - current_end) / 1_000,
                        "before": current_name,
                        "after": name,
                    }
                )
            current_start, current_end, current_name = start, end, name
        union_service_ns += current_end - current_start
        if interval_end - current_end >= min_gap_ns:
            gaps.append(
                {
                    "kind": "tail",
                    "start": current_end,
                    "end": interval_end,
                    "duration_us": (interval_end - current_end) / 1_000,
                    "before": current_name,
                    "after": "[next replay]",
                }
            )

    for gap in gaps:
        gap.update(step=interval["step"], rank=interval["rank"])
        gap["before_family"] = _family(gap["before"])
        gap["after_family"] = _family(gap["after"])

    interval_ns = interval_end - interval_start
    result = {
        "step": interval["step"],
        "rank": interval["rank"],
        "interval_ms": interval_ns / 1_000_000,
        "union_gpu_busy_ms": union_service_ns / 1_000_000,
        "raw_gpu_service_ms": raw_service_ns / 1_000_000,
        "overlap_ms": (raw_service_ns - union_service_ns) / 1_000_000,
        "idle_ms": (interval_ns - union_service_ns) / 1_000_000,
        "front_idle_ms": sum(
            gap["duration_us"] for gap in gaps if gap["kind"] == "front"
        )
        / 1_000,
        "internal_idle_ms": sum(
            gap["duration_us"] for gap in gaps if gap["kind"] == "internal"
        )
        / 1_000,
        "tail_idle_ms": sum(gap["duration_us"] for gap in gaps if gap["kind"] == "tail")
        / 1_000,
        "gap_count": len(gaps),
    }
    return result, gaps


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    con = sqlite3.connect(f"file:{args.sqlite}?mode=ro", uri=True)
    con.execute("pragma temp_store=file")
    strings = dict(con.execute("select id,value from StringIds"))
    ranges = _load_ranges(con, args.replay_nvtx)
    if len(ranges) != args.ranks:
        raise RuntimeError(f"found {len(ranges)} replay ranks, expected {args.ranks}")
    intervals = _create_intervals(con, ranges, args.edge_drop)
    activity = _load_activity(con, strings)

    rows = []
    gaps = []
    for interval in intervals:
        row, interval_gaps = _analyze_interval(
            interval,
            activity[(interval["step"], interval["rank"])],
            round(args.min_gap_us * 1_000),
        )
        rows.append(row)
        gaps.extend(interval_gaps)

    by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_rank: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_step[row["step"]].append(row)
        by_rank[row["rank"]].append(row)
    critical_rows = [
        max(step_rows, key=lambda row: row["interval_ms"])
        for step_rows in by_step.values()
    ]

    pair_rows: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for gap in gaps:
        key = (gap["kind"], gap["before_family"], gap["after_family"])
        pair_rows[key].append(gap["duration_us"])
    top_pairs = []
    for (kind, before, after), values in sorted(
        pair_rows.items(), key=lambda item: -sum(item[1])
    )[: args.top]:
        top_pairs.append(
            {
                "kind": kind,
                "before": before,
                "after": after,
                "count": len(values),
                "total_ms": sum(values) / 1_000,
                "mean_us": statistics.fmean(values),
                "max_us": max(values),
            }
        )

    keys = (
        "interval_ms",
        "union_gpu_busy_ms",
        "raw_gpu_service_ms",
        "overlap_ms",
        "idle_ms",
        "front_idle_ms",
        "internal_idle_ms",
        "tail_idle_ms",
    )
    return {
        "sqlite": str(args.sqlite),
        "replay_nvtx": args.replay_nvtx,
        "ranks": args.ranks,
        "edge_drop": args.edge_drop,
        "min_gap_us": args.min_gap_us,
        "interval_count": len(rows),
        "step_count": len(by_step),
        "all_rank_stats": {key: _stats([row[key] for row in rows]) for key in keys},
        "critical_interval_stats": {
            key: _stats([row[key] for row in critical_rows]) for key in keys
        },
        "rank_means": {
            str(rank): {
                key: statistics.fmean(row[key] for row in rank_rows) for key in keys
            }
            for rank, rank_rows in sorted(by_rank.items())
        },
        "top_gap_pairs": top_pairs,
        "largest_gaps": sorted(gaps, key=lambda gap: -gap["duration_us"])[: args.top],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("--replay-nvtx", default="execute_context_0(0)_generation_1(8)")
    parser.add_argument("--ranks", type=int, default=8)
    parser.add_argument("--edge-drop", type=int, default=1)
    parser.add_argument("--min-gap-us", type=float, default=1.0)
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = analyze(args)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
