# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reclassify Nsight Systems MTP decode kernels into SM70 work buckets."""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument(
        "--windows-json",
        type=Path,
        required=True,
        help="JSON from parse_nsys_decode_tokens.py.",
    )
    parser.add_argument("--out-prefix", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=30)
    return parser.parse_args()


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": sum(values) / len(values) if values else 0.0,
        "p50": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
        "p99": _percentile(values, 0.99),
        "min": min(values) if values else 0.0,
        "max": max(values) if values else 0.0,
    }


def _category(name: str, short_name: str) -> str:
    name_lower = name.lower()
    short_lower = short_name.lower()

    if "turbomind::gemm::gemm_kernel" in name_lower:
        if "top20epilogue" in name_lower:
            return "Draft dynamic LM-head stage 1"
        if "operand_b_pack<turbomind::uint4_t>" in name_lower:
            return "TurboMind AWQ GEMM"
        if "fp4" in name_lower or "nvfp4" in name_lower:
            return "TurboMind NVFP4 GEMM"
        if "operand_b_pack<__half>" in name_lower:
            return "TurboMind FP16 GEMM"
        return "Other TurboMind GEMM"

    if any(
        marker in short_lower
        for marker in (
            "sm70_f16_lm_head_top20_stage2",
            "sm70_merge_tail_top20",
            "sm70_sample_packed_top20",
        )
    ):
        return "Draft dynamic LM-head merge/sample"

    if any(
        marker in name_lower
        for marker in (
            "fused_recurrent_gated_delta_rule",
            "causal_conv1d",
            "fused_gdn_gating",
            "gdn_decode",
            "qwen_gdn",
            "linear_attention",
            "flashqla",
        )
    ):
        return "Target GDN / causal conv"
    if "kernel_unified_attention" in name_lower:
        return "MTP draft attention"
    if "flash_attention_decode" in name_lower or "pagedattention" in name_lower:
        return "Target FlashAttn V100"
    if any(
        marker in name_lower
        for marker in (
            "cross_device_reduce",
            "allreduce",
            "all_reduce",
            "nccldevkernel",
        )
    ):
        return "TP communication"
    if "reshape_and_cache" in name_lower or "kv_cache" in name_lower:
        return "KV-cache update"

    if any(
        marker in short_lower
        for marker in (
            "_topk_topp_kernel",
            "sample_recovered_tokens",
            "rejection_random_sample",
            "cunn_softmaxforward",
            "deviceselectsweepkernel",
            "devicereducesingletilekernel",
            "devicecompactinitkernel",
            "devicescankernel",
            "devicescaninitkernel",
        )
    ):
        return "Target rejection/sample"
    if any(
        marker in short_lower
        for marker in (
            "eagle_step_slot_mapping_metadata",
            "eagle_prepare_next_token",
            "eagle_prepare_inputs",
            "compute_slot_mapping",
        )
    ):
        return "MTP metadata kernels"

    if (
        "cutlass::kernel" in name_lower
        or "cublas" in name_lower
        or "gemv2t_kernel" in name_lower
        or "splitkreduce" in name_lower
    ):
        return "CUTLASS/cuBLAS FP16 GEMM/GEMV"
    if any(
        marker in name_lower
        for marker in (
            "pow_rsqrt",
            "rmsnorm",
            "layernorm",
            "layer_norm",
        )
    ):
        return "RMSNorm/residual"
    if "silu" in name_lower or "sigmoid" in name_lower or "gating" in name_lower:
        return "SiLU/gating"
    if "copy" in name_lower or "catarraybatchedcopy" in name_lower:
        return "Copy/cast"
    if any(
        marker in name_lower
        for marker in (
            "scatter",
            "gather",
            "index",
            "reduce_kernel",
            "reduce_segments",
            "expand_kernel",
        )
    ):
        return "Index/reduce/scatter"
    if "elementwise" in name_lower or "triton_" in name_lower:
        return "Other elementwise/Triton"
    return "Other kernels"


def _load_kernels(
    connection: sqlite3.Connection,
) -> list[tuple[int, int, int, int | None, int | None, str, str]]:
    strings = {
        row[0]: row[1] or ""
        for row in connection.execute("SELECT id, value FROM StringIds")
    }
    return [
        (
            start,
            end,
            device,
            correlation_id,
            global_pid,
            strings[demangled],
            strings[short],
        )
        for (
            start,
            end,
            device,
            correlation_id,
            global_pid,
            demangled,
            short,
        ) in connection.execute(
            "SELECT start, end, deviceId, correlationId, globalPid, "
            "demangledName, shortName "
            "FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start"
        )
    ]


def _memcpy_category(copy_kind: int) -> str:
    return {
        1: "CUDA memcpy HtoD",
        2: "CUDA memcpy DtoH",
        8: "CUDA memcpy DtoD",
        10: "CUDA memcpy P2P",
    }.get(copy_kind, "CUDA memcpy other")


def _load_memcpys(
    connection: sqlite3.Connection,
) -> list[tuple[int, int, int, int | None, int | None, int]]:
    table_exists = connection.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' "
        "AND name='CUPTI_ACTIVITY_KIND_MEMCPY'"
    ).fetchone()[0]
    if not table_exists:
        return []
    return [
        (start, end, device, correlation_id, global_pid, copy_kind)
        for start, end, device, correlation_id, global_pid, copy_kind in connection.execute(
            "SELECT start, end, deviceId, correlationId, globalPid, copyKind "
            "FROM CUPTI_ACTIVITY_KIND_MEMCPY ORDER BY start"
        )
    ]


def _load_graph_launches(
    connection: sqlite3.Connection,
) -> tuple[list[tuple[int, int, int]], dict[str, list[tuple[int, int, int]]]]:
    strings = {
        row[0]: row[1] or ""
        for row in connection.execute("SELECT id, value FROM StringIds")
    }
    launches = [
        (start, correlation_id, global_tid)
        for start, correlation_id, global_tid, name_id in connection.execute(
            "SELECT start, correlationId, globalTid, nameId "
            "FROM CUPTI_ACTIVITY_KIND_RUNTIME ORDER BY start"
        )
        if strings[name_id].startswith("cudaGraphLaunch")
        and correlation_id is not None
        and global_tid is not None
    ]
    replays: dict[str, list[tuple[int, int, int]]] = {}
    for label in ("cudagraph.FULL.replay", "cudagraph.PIECEWISE.replay"):
        replays[label] = [
            (start, end, global_tid)
            for start, end, global_tid in connection.execute(
                "SELECT start, end, globalTid FROM NVTX_EVENTS "
                "WHERE text=? ORDER BY start",
                (label,),
            )
            if end is not None and global_tid is not None
        ]
    return launches, replays


def _scope_correlations(
    *,
    start_ns: int,
    end_ns: int,
    label: str,
    launches: list[tuple[int, int, int]],
    replays: dict[str, list[tuple[int, int, int]]],
) -> set[tuple[int, int]]:
    correlations: set[tuple[int, int]] = set()
    for replay_start, replay_end, replay_tid in replays[label]:
        if not start_ns <= replay_start < end_ns:
            continue
        process_key = replay_tid >> 24
        for launch_start, correlation_id, launch_tid in launches:
            if launch_start < replay_start:
                continue
            if launch_start > replay_end:
                break
            if launch_tid >> 24 == process_key:
                correlations.add((correlation_id, process_key))
    return correlations


def _category_stats(
    rows: list[dict[str, Any]],
    categories: set[str],
) -> list[dict[str, Any]]:
    result = []
    for category in categories:
        critical_values = []
        gpu_sum_values = []
        count_values = []
        for row in rows:
            per_device = [
                values.get(category, 0.0)
                for values in row["device_category_ms"].values()
            ]
            critical_values.append(max(per_device, default=0.0))
            gpu_sum_values.append(sum(per_device))
            count_values.append(float(row["category_counts"].get(category, 0)))
        result.append(
            {
                "category": category,
                "critical_ms": _stats(critical_values),
                "tp_gpu_sum_ms": _stats(gpu_sum_values),
                "kernel_count": _stats(count_values),
            }
        )
    return sorted(result, key=lambda row: -row["critical_ms"]["mean"])


def _analyze(args: argparse.Namespace) -> dict[str, Any]:
    payload = json.loads(args.windows_json.read_text(encoding="utf-8"))
    all_windows = payload["tokens"]
    edge_drop = int(payload.get("edge_drop", 0))
    windows = all_windows[edge_drop:-edge_drop] if edge_drop else all_windows
    connection = sqlite3.connect(args.sqlite)
    kernels = _load_kernels(connection)
    memcpys = _load_memcpys(connection)
    launches, replays = _load_graph_launches(connection)
    connection.close()

    window_rows: list[dict[str, Any]] = []
    scope_rows: dict[str, list[dict[str, Any]]] = {
        "target_full_graph": [],
        "draft_piecewise_graph": [],
    }
    kernel_ms: Counter[tuple[str, str]] = Counter()
    kernel_counts: Counter[tuple[str, str]] = Counter()
    for window in windows:
        start_ns = int(window["start_ns"])
        end_ns = int(window["end_ns"])
        scope_correlation_sets = {
            "target_full_graph": _scope_correlations(
                start_ns=start_ns,
                end_ns=end_ns,
                label="cudagraph.FULL.replay",
                launches=launches,
                replays=replays,
            ),
            "draft_piecewise_graph": _scope_correlations(
                start_ns=start_ns,
                end_ns=end_ns,
                label="cudagraph.PIECEWISE.replay",
                launches=launches,
                replays=replays,
            ),
        }
        device_categories: dict[int, dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        category_counts: Counter[str] = Counter()
        scope_device_categories: dict[str, dict[int, dict[str, float]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(float))
        )
        scope_device_category_counts: dict[str, dict[int, Counter[str]]] = defaultdict(
            lambda: defaultdict(Counter)
        )
        scope_device_extents: dict[str, dict[int, list[int]]] = defaultdict(dict)
        scope_category_counts: dict[str, Counter[str]] = defaultdict(Counter)
        for (
            start,
            end,
            device,
            correlation_id,
            global_pid,
            name,
            short_name,
        ) in kernels:
            if start < start_ns:
                continue
            if start >= end_ns:
                break
            duration_ms = (end - start) / 1_000_000.0
            category = _category(name, short_name)
            device_categories[device][category] += duration_ms
            category_counts[category] += 1
            display_name = short_name or name
            kernel_key = (display_name, category)
            kernel_ms[kernel_key] += duration_ms
            kernel_counts[kernel_key] += 1
            if correlation_id is not None and global_pid is not None:
                correlation_key = (correlation_id, global_pid >> 24)
                for scope, correlation_set in scope_correlation_sets.items():
                    if correlation_key in correlation_set:
                        scope_device_categories[scope][device][category] += duration_ms
                        scope_device_category_counts[scope][device][category] += 1
                        scope_category_counts[scope][category] += 1
                        extent = scope_device_extents[scope].setdefault(
                            device, [start, end]
                        )
                        extent[0] = min(extent[0], start)
                        extent[1] = max(extent[1], end)
        for start, end, device, correlation_id, global_pid, copy_kind in memcpys:
            if start < start_ns:
                continue
            if start >= end_ns:
                break
            duration_ms = (end - start) / 1_000_000.0
            category = _memcpy_category(copy_kind)
            device_categories[device][category] += duration_ms
            category_counts[category] += 1
            if correlation_id is not None and global_pid is not None:
                correlation_key = (correlation_id, global_pid >> 24)
                for scope, correlation_set in scope_correlation_sets.items():
                    if correlation_key in correlation_set:
                        scope_device_categories[scope][device][category] += duration_ms
                        scope_device_category_counts[scope][device][category] += 1
                        scope_category_counts[scope][category] += 1
                        extent = scope_device_extents[scope].setdefault(
                            device, [start, end]
                        )
                        extent[0] = min(extent[0], start)
                        extent[1] = max(extent[1], end)
        window_rows.append(
            {
                "window": int(window["token"]),
                "interval_ms": float(window["interval_ms"]),
                "gpu_wall_ms": float(window["gpu_wall_ms"]),
                "device_category_ms": {
                    str(device): dict(values)
                    for device, values in device_categories.items()
                },
                "category_counts": dict(category_counts),
            }
        )
        for scope in scope_rows:
            device_wall_ms = {
                str(device): (extent[1] - extent[0]) / 1_000_000.0
                for device, extent in scope_device_extents[scope].items()
            }
            critical_device = (
                max(
                    scope_device_extents[scope],
                    key=lambda device: scope_device_extents[scope][device][1]
                    - scope_device_extents[scope][device][0],
                )
                if scope_device_extents[scope]
                else None
            )
            scope_rows[scope].append(
                {
                    "device_category_ms": {
                        str(device): dict(values)
                        for device, values in scope_device_categories[scope].items()
                    },
                    "category_counts": dict(scope_category_counts[scope]),
                    "correlation_count": len(scope_correlation_sets[scope]),
                    "device_wall_ms": device_wall_ms,
                    "scope_wall_ms": max(device_wall_ms.values(), default=0.0),
                    "critical_device": critical_device,
                    "critical_device_category_ms": (
                        dict(scope_device_categories[scope][critical_device])
                        if critical_device is not None
                        else {}
                    ),
                    "critical_device_category_counts": (
                        dict(scope_device_category_counts[scope][critical_device])
                        if critical_device is not None
                        else {}
                    ),
                }
            )

    categories = sorted(
        {
            category
            for row in window_rows
            for values in row["device_category_ms"].values()
            for category in values
        },
        key=lambda category: -sum(
            max(
                (
                    values.get(category, 0.0)
                    for values in row["device_category_ms"].values()
                ),
                default=0.0,
            )
            for row in window_rows
        ),
    )
    category_rows = _category_stats(window_rows, set(categories))
    graph_scopes = {}
    for scope, rows in scope_rows.items():
        scope_categories = {
            category
            for row in rows
            for values in row["device_category_ms"].values()
            for category in values
        }
        critical_rank_rows = [
            {
                "device_category_ms": {"critical": row["critical_device_category_ms"]},
                "category_counts": row["critical_device_category_counts"],
            }
            for row in rows
        ]
        graph_scopes[scope] = {
            "category_rows": _category_stats(rows, scope_categories),
            "critical_rank_category_rows": _category_stats(
                critical_rank_rows, scope_categories
            ),
            "correlation_count": _stats(
                [float(row["correlation_count"]) for row in rows]
            ),
            "activity_wall_ms": _stats(
                [float(row["scope_wall_ms"]) for row in rows]
            ),
        }

    gpu_wall_ms = _stats([row["gpu_wall_ms"] for row in window_rows])
    return {
        "sqlite": str(args.sqlite),
        "windows_json": str(args.windows_json),
        "window_semantics": "one TP verifier/MTP round, not one emitted token",
        "steady_windows": len(window_rows),
        "interval_ms": _stats([row["interval_ms"] for row in window_rows]),
        "gpu_wall_ms": gpu_wall_ms,
        "category_rows": category_rows,
        "graph_scopes": graph_scopes,
        "top_kernels": [
            {
                "kernel": key[0],
                "category": key[1],
                "total_ms": total_ms,
                "count": kernel_counts[key],
            }
            for key, total_ms in kernel_ms.most_common(args.top_n)
        ],
        "windows": window_rows,
    }


def _write_markdown(result: dict[str, Any], path: Path) -> None:
    gpu_wall = result["gpu_wall_ms"]["mean"]
    lines = [
        "# SM70 MTP Nsight Kernel Breakdown",
        "",
        f"Source: `{result['sqlite']}`",
        "",
        "Each window is one TP verifier/MTP round, not one emitted token. "
        "Graph-node tracing is used for composition only because it adds overhead.",
        "",
        f"Steady middle windows: {result['steady_windows']}",
        f"Mean replay interval: {result['interval_ms']['mean']:.3f} ms",
        f"Mean GPU wall: {gpu_wall:.3f} ms",
        "",
        "| category | critical mean ms/round | p50 | p90 | "
        "TP GPU-sum ms/round | CUDA activities/round | traced wall share |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["category_rows"]:
        critical = row["critical_ms"]
        if critical["mean"] < 0.001:
            continue
        share = critical["mean"] / gpu_wall * 100.0 if gpu_wall else 0.0
        lines.append(
            f"| {row['category']} | {critical['mean']:.3f} | "
            f"{critical['p50']:.3f} | {critical['p90']:.3f} | "
            f"{row['tp_gpu_sum_ms']['mean']:.3f} | "
            f"{row['kernel_count']['mean']:.1f} | {share:.1f}% |"
        )
    for scope, title in (
        ("target_full_graph", "Target Verifier FULL Graph"),
        ("draft_piecewise_graph", "MTP Draft PIECEWISE Graphs"),
    ):
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "Graph-correlated kernel and memcpy activities only; work "
                "submitted outside the graph is listed only in the whole-round table.",
                f"Graph activity wall (max rank): "
                f"{result['graph_scopes'][scope]['activity_wall_ms']['mean']:.3f} ms "
                "mean. Per-category max-rank rows below are non-additive.",
                "",
                "| category | per-category max-rank mean ms/round | p50 | p90 | "
                "TP GPU-sum ms/round | CUDA activities/round |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in result["graph_scopes"][scope]["category_rows"]:
            critical = row["critical_ms"]
            if critical["mean"] < 0.001:
                continue
            lines.append(
                f"| {row['category']} | {critical['mean']:.3f} | "
                f"{critical['p50']:.3f} | {critical['p90']:.3f} | "
                f"{row['tp_gpu_sum_ms']['mean']:.3f} | "
                f"{row['kernel_count']['mean']:.1f} |"
            )
        lines.extend(
            [
                "",
                "### Critical-Rank Composition",
                "",
                "For each round, categories are taken from the one rank with "
                "the longest graph activity span. These rows are composable on "
                "that selected rank, but still sum CUDA activity durations rather "
                "than proving a strict DAG critical path.",
                "",
                "| category | selected-rank mean ms/round | p50 | p90 | "
                "activities/round |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in result["graph_scopes"][scope]["critical_rank_category_rows"]:
            critical = row["critical_ms"]
            if critical["mean"] < 0.001:
                continue
            lines.append(
                f"| {row['category']} | {critical['mean']:.3f} | "
                f"{critical['p50']:.3f} | {critical['p90']:.3f} | "
                f"{row['kernel_count']['mean']:.1f} |"
            )
    lines.extend(
        [
            "",
            "## Top Kernels",
            "",
            "| kernel | category | total ms | launches |",
            "|---|---|---:|---:|",
        ]
    )
    for row in result["top_kernels"]:
        lines.append(
            f"| `{row['kernel']}` | {row['category']} | "
            f"{row['total_ms']:.3f} | {row['count']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    result = _analyze(args)
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = args.out_prefix.with_suffix(".json")
    markdown_path = args.out_prefix.with_suffix(".md")
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    _write_markdown(result, markdown_path)
    print(markdown_path)
    print(json_path)


if __name__ == "__main__":
    main()
