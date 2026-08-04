#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Analyze rank skew around a graph-captured SM70 TP8 collective.

CUDA graph kernels can cross the next host replay range boundary. This tool
therefore aligns collective nodes by their global per-rank ordinal and only
then divides them into graph replays. It does not use replay timestamps to
assign individual collective calls to tokens.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import sqlite3
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

GLOBAL_PID_MASK = -16777216
DEFAULT_REPLAY_NVTX = "breakable_cudagraph.replay"
DEFAULT_KERNEL_MARKER = "sm70_tp8_hierarchical_reduce"


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values) if values else 0.0,
        "p50": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
        "p99": _percentile(values, 0.99),
        "min": min(values, default=0.0),
        "max": max(values, default=0.0),
    }


def _kernel_category(name: str, short_name: str) -> str:
    lowered = f"{name} {short_name}".lower()
    if DEFAULT_KERNEL_MARKER in lowered:
        return "TP all-reduce"
    if "turbomind::gemm::gemm_kernel" in lowered:
        if "fp4_e2m1" in lowered or "operand_b_pack<turbomind::fp4" in lowered:
            return "MXFP4 MoE"
        if "__nv_fp8" in lowered or "fp8_e4m3" in lowered:
            return "FP8 dense GEMM"
    if "_sm70_sparse_paged_fp8" in lowered or "_sm70_sparse_gathered" in lowered:
        return "Sparse MLA"
    if "mhc_" in lowered or "hc_prenorm" in lowered or "hc_head_fuse" in lowered:
        return "mHC"
    if any(
        marker in lowered
        for marker in (
            "topkgating",
            "moe_",
            "expert",
            "act_and_mul",
            "silu_and_mul",
            "finalizemoerouting",
        )
    ):
        return "MoE routing/activation"
    if any(
        marker in lowered
        for marker in ("quantize_and_insert_k", "kv_", "qnorm", "rope")
    ):
        return "Q/KV preparation"
    if "norm" in lowered or "rsqrt" in lowered:
        return "Norm/residual"
    if "elementwise" in lowered:
        return "Elementwise"
    if "copy" in lowered or "cast" in lowered:
        return "Copy/cast"
    if "gemm" in lowered or "gemv" in lowered:
        return "Other GEMM/GEMV"
    if "index" in lowered or "topk" in lowered:
        return "Indexer/top-k"
    return "Other"


def _union_duration_us(
    intervals: list[tuple[int, int]], start_ns: int, end_ns: int
) -> float:
    clipped = sorted(
        (max(start, start_ns), min(end, end_ns))
        for start, end in intervals
        if end > start_ns and start < end_ns
    )
    if not clipped:
        return 0.0
    merged_start, merged_end = clipped[0]
    total_ns = 0
    for start, end in clipped[1:]:
        if start <= merged_end:
            merged_end = max(merged_end, end)
        else:
            total_ns += merged_end - merged_start
            merged_start, merged_end = start, end
    return (total_ns + merged_end - merged_start) / 1000.0


def _load_replay_tids(
    connection: sqlite3.Connection, label: str
) -> tuple[list[int], int]:
    rows = list(
        connection.execute(
            "select n.globalTid,count(*) from NVTX_EVENTS n "
            "left join StringIds s on s.id=n.textId "
            "where coalesce(n.text,s.value)=? "
            "group by n.globalTid order by n.globalTid",
            (label,),
        )
    )
    if not rows:
        raise RuntimeError(f"No NVTX replay ranges named {label!r}")
    counts = {int(count) for _tid, count in rows}
    if len(counts) != 1:
        raise RuntimeError(f"Replay counts differ by worker: {rows}")
    return [int(tid) for tid, _count in rows], counts.pop()


def _load_events(
    connection: sqlite3.Connection,
    rank_pids: dict[int, int],
    collective_marker: str,
) -> tuple[
    dict[int, list[tuple[int, int, int, str, str]]],
    dict[int, list[tuple[int, int, int, str]]],
    dict[int, int],
]:
    strings = {
        row[0]: row[1] or ""
        for row in connection.execute("select id,value from StringIds")
    }
    connection.execute("create temp table rank_pids(globalPid integer,rank integer)")
    connection.executemany(
        "insert into rank_pids values(?,?)",
        [(global_pid, rank) for rank, global_pid in rank_pids.items()],
    )
    query = """
        select p.rank,k.start,k.end,k.deviceId,k.streamId,
               k.demangledName,k.shortName
        from CUPTI_ACTIVITY_KIND_KERNEL k
        join rank_pids p on p.globalPid=k.globalPid
        order by p.rank,k.start
    """
    events: dict[int, list[tuple[int, int, int, str, str]]] = defaultdict(list)
    collectives: dict[int, list[tuple[int, int, int, str]]] = defaultdict(list)
    devices: dict[int, int] = {}
    marker = collective_marker.lower()
    for rank, start, end, device, stream, demangled_id, short_id in connection.execute(
        query
    ):
        name = strings.get(demangled_id, "")
        short_name = strings.get(short_id, "")
        category = _kernel_category(name, short_name)
        events[rank].append((start, end, stream, category, short_name))
        if marker in name.lower() or marker in short_name.lower():
            collectives[rank].append((start, end, stream, category))
            devices[rank] = device
    return events, collectives, devices


def _segment_kind(ordinal: int, calls_per_replay: int) -> str | None:
    if ordinal == 0:
        return "token boundary"
    if ordinal % 2 == 1:
        return "attention"
    if ordinal < calls_per_replay:
        return "moe"
    return None


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    connection = sqlite3.connect(f"file:{args.sqlite}?mode=ro", uri=True)
    tids, replay_count = _load_replay_tids(connection, args.replay_nvtx)
    if len(tids) != args.ranks:
        raise RuntimeError(f"Expected {args.ranks} ranks, found {len(tids)}")
    rank_pids = {rank: tid & GLOBAL_PID_MASK for rank, tid in enumerate(tids)}
    events, collectives, devices = _load_events(
        connection, rank_pids, args.collective_marker
    )
    connection.close()

    counts = {rank: len(rows) for rank, rows in collectives.items()}
    if len(counts) != args.ranks or len(set(counts.values())) != 1:
        raise RuntimeError(f"Collective counts differ by rank: {counts}")
    total_per_rank = next(iter(counts.values()))
    if total_per_rank % replay_count:
        raise RuntimeError(
            f"{total_per_rank} collectives are not divisible by "
            f"{replay_count} graph replays"
        )
    calls_per_replay = total_per_rank // replay_count
    if args.expected_calls and calls_per_replay != args.expected_calls:
        raise RuntimeError(
            f"Expected {args.expected_calls} calls/replay, got {calls_per_replay}"
        )

    event_starts = {
        rank: [event[0] for event in rank_events]
        for rank, rank_events in events.items()
    }
    edge_drop = min(args.edge_drop, max((replay_count - 1) // 2, 0))
    steady_replays = range(edge_drop, replay_count - edge_drop)

    collective_metrics: dict[str, list[float]] = defaultdict(list)
    last_arrival_counts: Counter[int] = Counter()
    per_rank_lateness: dict[int, list[float]] = defaultdict(list)
    predecessor_counts: Counter[str] = Counter()
    segment_metrics: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    segment_category_service: dict[str, dict[str, dict[int, list[float]]]] = (
        defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    )

    for replay in steady_replays:
        base = replay * calls_per_replay
        for ordinal in range(calls_per_replay):
            index = base + ordinal
            aligned = [collectives[rank][index] for rank in range(args.ranks)]
            starts = [row[0] for row in aligned]
            ends = [row[1] for row in aligned]
            durations = [(row[1] - row[0]) / 1000.0 for row in aligned]
            first_start = min(starts)
            last_rank = max(range(args.ranks), key=lambda rank: starts[rank])
            last_arrival_counts[last_rank] += 1
            for rank, start in enumerate(starts):
                per_rank_lateness[rank].append((start - first_start) / 1000.0)
            collective_metrics["arrival_skew_us"].append(
                (max(starts) - first_start) / 1000.0
            )
            collective_metrics["completion_skew_us"].append(
                (max(ends) - min(ends)) / 1000.0
            )
            collective_metrics["tail_after_last_arrival_us"].append(
                (max(ends) - max(starts)) / 1000.0
            )
            collective_metrics["envelope_us"].append((max(ends) - first_start) / 1000.0)
            collective_metrics["rank_mean_service_us"].append(
                statistics.mean(durations)
            )

            kind = _segment_kind(ordinal, calls_per_replay)
            if kind not in {"attention", "moe"}:
                continue
            rank_spans: list[float] = []
            rank_busy: list[float] = []
            predecessor_categories: list[str] = []
            for rank in range(args.ranks):
                previous_end = collectives[rank][index - 1][1]
                current_start = collectives[rank][index][0]
                current_stream = collectives[rank][index][2]
                rank_spans.append((current_start - previous_end) / 1000.0)
                rank_events = events[rank]
                lower = bisect.bisect_left(event_starts[rank], previous_end)
                upper = bisect.bisect_left(event_starts[rank], current_start)
                selected = [
                    event
                    for event in rank_events[lower:upper]
                    if event[1] <= current_start and event[3] != "TP all-reduce"
                ]
                rank_busy.append(
                    _union_duration_us(
                        [(event[0], event[1]) for event in selected],
                        previous_end,
                        current_start,
                    )
                )
                category_sums: Counter[str] = Counter()
                for start, end, _stream, category, _short_name in selected:
                    category_sums[category] += (end - start) / 1000.0
                same_stream = [
                    event for event in selected if event[2] == current_stream
                ]
                predecessor_categories.append(
                    same_stream[-1][3] if same_stream else "Unknown"
                )
                for category, duration_us in category_sums.items():
                    segment_category_service[kind][category][rank].append(duration_us)

            segment_metrics[kind]["rank_span_mean_us"].append(
                statistics.mean(rank_spans)
            )
            segment_metrics[kind]["rank_span_max_us"].append(max(rank_spans))
            segment_metrics[kind]["rank_span_spread_us"].append(
                max(rank_spans) - min(rank_spans)
            )
            segment_metrics[kind]["arrival_skew_us"].append(
                (max(starts) - min(starts)) / 1000.0
            )
            segment_metrics[kind]["busy_union_mean_us"].append(
                statistics.mean(rank_busy)
            )
            segment_metrics[kind]["idle_union_mean_us"].append(
                statistics.mean(
                    span - busy for span, busy in zip(rank_spans, rank_busy)
                )
            )

            predecessor = Counter(predecessor_categories).most_common(1)[0][0]
            predecessor_counts[predecessor] += 1

    steady_count = replay_count - 2 * edge_drop
    segment_rows: dict[str, Any] = {}
    for kind, metrics in segment_metrics.items():
        categories = []
        for category, rank_values in segment_category_service[kind].items():
            per_rank_means = [
                statistics.mean(rank_values[rank]) if rank_values[rank] else 0.0
                for rank in range(args.ranks)
            ]
            categories.append(
                {
                    "category": category,
                    "rank_average_us_per_segment": statistics.mean(per_rank_means),
                    "per_rank_mean_us": per_rank_means,
                    "per_rank_mean_spread_us": max(per_rank_means)
                    - min(per_rank_means),
                }
            )
        categories.sort(key=lambda row: -row["rank_average_us_per_segment"])
        segment_rows[kind] = {
            "segments_per_replay": len(metrics["rank_span_mean_us"]) / steady_count,
            "metrics": {name: _stats(values) for name, values in metrics.items()},
            "category_service": categories,
        }

    return {
        "sqlite": str(args.sqlite),
        "replay_nvtx": args.replay_nvtx,
        "rank_pids": rank_pids,
        "rank_devices": devices,
        "graph_replays": replay_count,
        "edge_drop": edge_drop,
        "steady_replays": steady_count,
        "collectives_per_replay": calls_per_replay,
        "aligned_collectives": steady_count * calls_per_replay,
        "collective_metrics": {
            name: _stats(values) for name, values in collective_metrics.items()
        },
        "arrival_skew_sum_ms_per_replay": sum(collective_metrics["arrival_skew_us"])
        / steady_count
        / 1000.0,
        "last_arrival_rank_counts": dict(last_arrival_counts),
        "per_rank_start_lateness_us": {
            rank: _stats(values) for rank, values in per_rank_lateness.items()
        },
        "immediate_predecessor_counts": dict(predecessor_counts),
        "segments": segment_rows,
    }


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    collective = payload["collective_metrics"]
    lines = [
        "# SM70 TP8 Collective Skew",
        "",
        f"- Trace: `{payload['sqlite']}`",
        f"- Graph replays: {payload['graph_replays']}",
        f"- Steady replays: {payload['steady_replays']}",
        f"- Collectives/replay: {payload['collectives_per_replay']}",
        "",
        "| metric | mean us | p50 | p90 | p99 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in (
        "arrival_skew_us",
        "tail_after_last_arrival_us",
        "envelope_us",
        "rank_mean_service_us",
    ):
        row = collective[name]
        lines.append(
            f"| {name} | {row['mean']:.3f} | {row['p50']:.3f} | "
            f"{row['p90']:.3f} | {row['p99']:.3f} |"
        )
    for kind, segment in payload["segments"].items():
        lines.extend(
            [
                "",
                f"## {kind.title()} Segments",
                "",
                "| metric | mean us | p50 | p90 | p99 |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for name, row in segment["metrics"].items():
            lines.append(
                f"| {name} | {row['mean']:.3f} | {row['p50']:.3f} | "
                f"{row['p90']:.3f} | {row['p99']:.3f} |"
            )
        lines.extend(
            [
                "",
                "| category | rank-average us/segment | rank-mean spread us |",
                "|---|---:|---:|",
            ]
        )
        for row in segment["category_service"]:
            lines.append(
                f"| {row['category']} | {row['rank_average_us_per_segment']:.3f} "
                f"| {row['per_rank_mean_spread_us']:.3f} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument("--out-prefix", type=Path, required=True)
    parser.add_argument("--ranks", type=int, default=8)
    parser.add_argument("--edge-drop", type=int, default=1)
    parser.add_argument("--expected-calls", type=int, default=87)
    parser.add_argument("--replay-nvtx", default=DEFAULT_REPLAY_NVTX)
    parser.add_argument("--collective-marker", default=DEFAULT_KERNEL_MARKER)
    args = parser.parse_args()

    payload = analyze(args)
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = args.out_prefix.with_suffix(".json")
    markdown_path = args.out_prefix.with_suffix(".md")
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _write_markdown(payload, markdown_path)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
