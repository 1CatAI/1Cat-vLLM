# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attribute paired layer graph intervals without inventing an HBM curve.

V100 has no Nsight Systems GPU Metrics support. This records kernel intervals
and weight-read opportunities; per-kernel NCU counters remain separate. The
cold-L2 eviction node and each arm's first/last replay are excluded.
"""

import argparse
import json
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path


def phase(name, ordinal):
    if "resident_out_norm" in name:
        return "out + communication/norm + next weights"
    if "resident_down_norm" in name:
        return "down + communication/norm"
    if "paired_gated" in name:
        return "gate/up weights"
    if "nvfp4_qpn2_sm70_kernel" in name:
        return "down weights"
    if "fp8_qpn8_sm70_kernel" in name:
        return "qkvz/a/b weights" if ordinal < 5 else "out weights"
    if "push_allreduce" in name:
        return "communication/norm"
    if "gemma_fused_add" in name:
        return "entry norm"
    if "conv_gate" in name:
        return "conv/gating"
    if "delta_rule" in name:
        return "GDN state update"
    if "layer_norm" in name:
        return "gated norm"
    return name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(f"file:{args.sqlite}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    pid_device = dict(
        conn.execute(
            "SELECT DISTINCT globalPid,deviceId FROM CUPTI_ACTIVITY_KIND_KERNEL"
        )
    )
    frames = defaultdict(list)
    ranges = conn.execute(
        "SELECT start,end,globalTid,text FROM NVTX_EVENTS "
        "WHERE text IN ('layer/control','layer/candidate') ORDER BY start"
    )
    for nvtx in ranges:
        rank = pid_device[nvtx["globalTid"] & ~((1 << 24) - 1)]
        rows = conn.execute(
            "SELECT k.*,s.value AS name FROM CUPTI_ACTIVITY_KIND_KERNEL k "
            "JOIN StringIds s ON k.shortName=s.id "
            "WHERE k.deviceId=? AND k.start>=? AND k.end<=? ORDER BY k.start",
            (rank, nvtx["start"], nvtx["end"]),
        ).fetchall()
        rows = [r for r in rows if r["name"] != "vectorized_elementwise_kernel"]
        assert rows and all(r["graphNodeId"] for r in rows)
        kernels = []
        for i, row in enumerate(rows):
            kernels.append(
                {
                    "name": row["name"],
                    "phase": phase(row["name"], i),
                    "start_ns": row["start"],
                    "end_ns": row["end"],
                    "duration_us": (row["end"] - row["start"]) / 1000,
                    "gap_before_us": 0
                    if not i
                    else (row["start"] - rows[i - 1]["end"]) / 1000,
                    "grid": [row[k] for k in ("gridX", "gridY", "gridZ")],
                    "block": [row[k] for k in ("blockX", "blockY", "blockZ")],
                    "registers": row["registersPerThread"],
                    "shared_bytes": row["staticSharedMemory"]
                    + row["dynamicSharedMemory"],
                }
            )
        frames[(nvtx["text"], rank)].append(kernels)
    summary = []
    raw = []
    for (arm, rank), replays in sorted(frames.items()):
        assert len(replays) >= 5
        counts = {len(k) for k in replays}
        assert len(counts) == 1, (arm, rank, counts)
        retained = replays[1:-1]
        service = defaultdict(list)
        for kernels in retained:
            grouped = defaultdict(float)
            for k in kernels:
                grouped[k["phase"]] += k["duration_us"]
            for label, value in grouped.items():
                service[label].append(value)
        summary.append(
            {
                "arm": arm,
                "rank": rank,
                "replays": len(retained),
                "kernels": len(retained[0]),
                "gpu_envelope_mean_us": statistics.mean(
                    (k[-1]["end_ns"] - k[0]["start_ns"]) / 1000 for k in retained
                ),
                "gap_sum_mean_us": statistics.mean(
                    sum(n["gap_before_us"] for n in k) for k in retained
                ),
                "phases_mean_us": {k: statistics.mean(v) for k, v in service.items()},
            }
        )
        raw.append({"arm": arm, "rank": rank, "replays": replays})
    result = {
        "scope": "Profiled layer graph, not whole-model C1 or instantaneous HBM",
        "limitations": [
            "V100 does not support continuous Nsight Systems GPU Metrics.",
            "State/KV and peer traffic also consume memory bandwidth.",
            "Node tracing perturbs launch skew and collective waiting.",
            "Do not sum maxima of individual rank categories into a wall time.",
        ],
        "summary": summary,
        "frames": raw,
    }
    (args.out / "weight-windows.json").write_text(json.dumps(result, indent=2))
    text = [
        "# Layer weight-read windows",
        "",
        "These are CUDA graph kernel intervals. They are not instantaneous HBM "
        "counters or unprofiled C1 results. GDN and collectives consume state and "
        "peer bytes even when they do not read projection weights.",
        "",
        "| Arm | Rank | Kernel nodes | GPU envelope us | Gaps us |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        text.append(
            f"| {row['arm']} | {row['rank']} | {row['kernels']} | "
            f"{row['gpu_envelope_mean_us']:.3f} | {row['gap_sum_mean_us']:.3f} |"
        )
    text += [
        "",
        "The first and last replay of each arm/rank and the eviction "
        "node are excluded. Each CUDA graph node is retained in the JSON.",
    ]
    (args.out / "weight-windows.md").write_text("\n".join(text) + "\n")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    colors = {
        "weight": "#236a91",
        "other": "#a0a7ad",
        "comm": "#d6862e",
        "mixed": "#4a9162",
    }
    fig, ax = plt.subplots(figsize=(13, 5))
    for arm_index, arm in enumerate(("layer/control", "layer/candidate")):
        group = [frames[(arm, rank)][5] for rank in range(4)]
        origin = min(k[0]["start_ns"] for k in group)
        for rank, kernels in enumerate(group):
            y = arm_index * 5 + rank
            for k in kernels:
                label = k["phase"]
                category = (
                    "mixed"
                    if "+" in label
                    else "weight"
                    if "weights" in label
                    else "comm"
                    if "communication" in label
                    else "other"
                )
                ax.broken_barh(
                    [((k["start_ns"] - origin) / 1000, k["duration_us"])],
                    (y - 0.35, 0.7),
                    facecolors=colors[category],
                )
    ax.set_yticks(
        [*range(4), *range(5, 9)],
        [f"Control rank {i}" for i in range(4)]
        + [f"Candidate rank {i}" for i in range(4)],
    )
    ax.invert_yaxis()
    ax.set_xlabel("Microseconds from the first compute node of the paired replay")
    ax.set_title(
        "Kernel intervals and weight-read opportunity; no instantaneous HBM measurement"
    )
    ax.legend(
        handles=[
            Patch(color=color, label=label)
            for label, color in zip(
                (
                    "Projection",
                    "Other / state",
                    "Communication + norm",
                    "Fused projection + communication + prefetch",
                ),
                colors.values(),
            )
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
    )
    fig.tight_layout()
    fig.savefig(args.out / "weight-windows.png", dpi=180, bbox_inches="tight")
    fig.savefig(args.out / "weight-windows.svg", bbox_inches="tight")


if __name__ == "__main__":
    main()
