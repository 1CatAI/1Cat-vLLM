# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode ledger for profile_both.py captures, ending at the prefill pause."""

import argparse
import json
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path

import regex as re

BUCKETS = (
    ("moe_gemm", r"skinny_nvfp4_moe|moe_dec_kernel|moe_w13_swiglu|moe_w2_route"),
    ("moe_route", r"_swiglu_prob|_route_sum|route_plan|_route_i32|_plan_kernel"),
    ("dense_fp8", r"qpn8"),
    ("head", r"h16_gemv.*"),
    (
        "lm_head",
        r"s161616gemm|hgemv|_markov|[Aa]rg[Mm]ax|argmax|_global_top1|_local_pair",
    ),
    (
        "attn",
        r"_attn_prep|_attn_post|_gather_attention|_gather_draft|_sink_softmax|attn_dec|s884gemm|_insert_slots|_rotate",
    ),
    ("indexer", r"index|_gather_index|topk"),
    (
        "compress",
        r"compress|_pool|_insert|ring_insert|engram|_wait|_publish|_decode_native|_commit",
    ),
    ("allreduce", r"cross_device|[Aa]ll[Rr]educe|custom_ar|ll_allreduce"),
    ("allgather", r"ll_allgather"),
    ("mhc", r"_transition|_hc_|hc_|collapse|_expand|mixes|sinkhorn"),
    ("norm", r"rms|_short_rms|norm"),
    ("copy/elem", r"elementwise|copy|CatArray|fill|where|masked"),
)


def bucket(name):
    for category, pattern in BUCKETS:
        if re.search(pattern, name):
            return category
    return "other"


def union_us(rows):
    end = total = 0
    for row in sorted(rows, key=lambda row: row["start"]):
        total += max(0, row["end"] - max(end, row["start"]))
        end = max(end, row["end"])
    return total / 1000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    with sqlite3.connect(args.report) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT k.start, k.end, s.value AS name, k.gridX AS grid_x,
                   k.gridY AS grid_y, k.blockX AS block_x, k.graphNodeId AS graph_node
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN StringIds s ON k.demangledName = s.id
            WHERE k.deviceId = 0
            ORDER BY k.start
            """
        ).fetchall()
    cuts = [
        after["start"]
        for before, after in zip(rows, rows[1:])
        if after["start"] - before["end"] > 500_000_000
    ]
    if not cuts:
        raise ValueError(
            "Expected the pause between decode and prefill in profile_both.py"
        )
    rows = [row for row in rows if row["start"] < cuts[0]]
    boundaries = [index for index, row in enumerate(rows) if "Broadcast" in row["name"]]
    periods, active, counts = [], [], []
    totals = defaultdict(float)
    kernels = defaultdict(list)
    for first, last in zip(boundaries[5:-2], boundaries[6:-1]):
        work = [
            row
            for row in rows[first + 1 : last]
            if row["start"] >= rows[first]["end"] and "nccl" not in row["name"].lower()
        ]
        if not work:
            continue
        periods.append((rows[last]["end"] - rows[first]["end"]) / 1000)
        active.append(union_us(work))
        counts.append(len(work))
        for row in work:
            name = row["name"]
            category = bucket(name)
            if "h16_gemv" in name:
                if row["grid_x"] > 1000:
                    category = "head"
                elif row["grid_x"] in (4, 12):
                    category = "gate"
                else:
                    category = "other_h16"
            duration = (row["end"] - row["start"]) / 1000
            totals[category] += duration
            key = (
                name,
                row["grid_x"],
                row["grid_y"],
                row["block_x"],
                bool(row["graph_node"]),
            )
            kernels[key].append(duration)
    rounds = len(periods)
    if not rounds:
        raise ValueError(
            "No complete decode rounds after excluding warmup and final steps"
        )
    print(
        json.dumps(
            {
                "report": str(args.report),
                "rounds": rounds,
                "profiled_period_us": round(statistics.median(periods), 2),
                "gpu_active_union_us": round(statistics.median(active), 2),
                "kernels_per_round": statistics.median(counts),
                "bucket_us_per_round": {
                    name: round(total / rounds, 2)
                    for name, total in sorted(totals.items(), key=lambda item: -item[1])
                },
            }
        )
    )
    for key, durations in sorted(kernels.items(), key=lambda item: -sum(item[1]))[:24]:
        name, grid_x, grid_y, block_x, graph = key
        print(
            json.dumps(
                {
                    "name": name,
                    "grid": [grid_x, grid_y],
                    "block": block_x,
                    "graph": graph,
                    "median_us": round(statistics.median(durations), 2),
                    "calls_per_round": round(len(durations) / rounds, 2),
                }
            )
        )


if __name__ == "__main__":
    main()
