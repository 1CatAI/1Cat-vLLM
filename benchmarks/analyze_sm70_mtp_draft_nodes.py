# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rank draft nodes with single/small grids or at most one warp per CTA.

Input is a correlated node-service report. These profiled service sums are
diagnostic and must not be substituted for unprofiled complete-round latency.
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def summarize_draft_nodes(report, rank=None):
    draft = [r for r in report["kernels"] if "draft_step/" in r["phase"]]
    if not draft:
        raise ValueError("No correlated draft nodes")
    rank_service = defaultdict(float)
    for row in draft:
        rank_service[row["rank"]] += row["service_ms_per_round"]
    selected_automatically = rank is None
    if selected_automatically:
        rank = max(rank_service, key=rank_service.get)
    if rank not in rank_service:
        raise ValueError("Requested rank has no draft nodes")
    nodes = []
    phases = defaultdict(lambda: {"calls": 0.0, "service_ms": 0.0})
    for row in draft:
        if row["rank"] != rank:
            continue
        phase = phases[row["phase"]]
        phase["calls"] += row["calls_per_round"]
        phase["service_ms"] += row["service_ms_per_round"]
        ctas = math.prod(row["grid"])
        threads = math.prod(row["block"])
        if ctas < 1 or threads < 1:
            raise ValueError("Invalid launch geometry")
        flags = {
            "single_cta": ctas == 1,
            "grid_at_most_four": ctas <= 4,
            "at_most_one_warp": threads <= 32,
        }
        if any(flags.values()):
            nodes.append({**row, **flags, "ctas": ctas, "threads_per_cta": threads})
    nodes.sort(key=lambda r: r["service_ms_per_round"], reverse=True)
    return {
        "measurement": "profiled node service, not unprofiled round decomposition",
        "selected_rank": rank,
        "rank_selection": "largest summed draft service"
        if selected_automatically
        else "explicit",
        "phases": dict(phases),
        "flagged_calls": sum(r["calls_per_round"] for r in nodes),
        "flagged_service_ms": sum(r["service_ms_per_round"] for r in nodes),
        "nodes": nodes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rank", type=int)
    args = parser.parse_args()
    result = summarize_draft_nodes(json.loads(args.input.read_text()), args.rank)
    result["source_report"] = str(args.input)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "nodes"}))


if __name__ == "__main__":
    main()
