# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Account for retained Flash-Next graph nodes on critical stream 364.

Weight metadata comes from the source-verified step0 prepared layouts.
No DRAM traffic is inferred from Nsight Systems timing or parameter sizes.
"""

import argparse
import collections
import hashlib
import json
import sqlite3
import statistics
from pathlib import Path

import regex as re

p = argparse.ArgumentParser()
p.add_argument("--trace", type=Path, required=True)
p.add_argument("--weights", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
args = p.parse_args()
c = sqlite3.connect(f"file:{args.trace}?mode=ro", uri=True)
c.row_factory = sqlite3.Row
names = dict(c.execute("select id,value from StringIds"))
all_nodes = [
    dict(n)
    for n in c.execute("select * from CUPTI_ACTIVITY_KIND_KERNEL order by start")
]
for n in all_nodes:
    n["name"] = names[n["demangledName"]]
weights = json.loads(args.weights.read_text())
lookup = {(r["name"], tuple(r["grid"])): r for r in weights["kernels"]}
groups = collections.defaultdict(list)
for n in all_nodes:
    if n["graphNodeId"]:
        groups[(n["globalPid"], n["correlationId"])].append(n)


def union(intervals):
    result = []
    for a, b in sorted(intervals):
        if result and a <= result[-1][1]:
            result[-1] = (result[-1][0], max(b, result[-1][1]))
        else:
            result.append((a, b))
    return result


def total(intervals):
    return sum(b - a for a, b in union(intervals)) / 1000


def intersection(a, b):
    return total(
        [
            (max(x, u), min(y, v))
            for x, y in union(a)
            for u, v in union(b)
            if max(x, u) < min(y, v)
        ]
    )


rows = []
families = collections.defaultdict(list)
ranks = []
for pid in sorted({n["globalPid"] for n in all_nodes}):
    gs = sorted(
        [n for (p, _), n in groups.items() if p == pid and len(n) == 1349],
        key=lambda n: n[0]["start"],
    )
    assert len(gs) == 8
    for step in range(2, 7):
        ns = gs[step]
        norms = [
            i
            for i, n in enumerate(ns)
            if n["name"] in ("_hc_combine_norm_kernel", "_grouped_gemma_rmsnorm_kernel")
        ]
        assert len(norms) == 97
        starts = norms[:96:2]
        ple = next(
            i
            for i, n in enumerate(ns)
            if n["name"] == "_dequantize_ple_fp8_bytes_kernel"
        )
        starts[1] = ple - 1
        ranges = (
            [("embedding", 0, starts[0])]
            + [
                (str(i), s, e)
                for i, (s, e) in enumerate(zip(starts, starts[1:] + [norms[-1]]))
            ]
            + [("final_mixer", norms[-1], len(ns))]
        )
        ranks.append(
            {
                "device": ns[0]["deviceId"],
                "step": step,
                "interval_us": (gs[step + 1][0]["start"] - ns[0]["start"]) / 1000,
            }
        )
        for layer, a, b in ranges:
            part = ns[a:b]
            main = [n for n in part if n["streamId"] == 364]
            aux = [n for n in part if n["streamId"] != 364]
            begin = part[0]["start"]
            end = ns[b]["start"] if b < len(ns) else max(n["end"] for n in part)
            mi = [(n["start"], n["end"]) for n in main]
            ai = [(n["start"], n["end"]) for n in aux]
            overlap = intersection(mi, ai)
            wb = 0
            for i, n in enumerate(part, a):
                r = lookup[(n["name"], (n["gridX"], n["gridY"], n["gridZ"]))]
                byte = r["weight_bytes_per_call"]
                if "gemv2T" in n["name"] and (n["gridX"], n["gridY"], n["gridZ"]) == (
                    40,
                    1,
                    10,
                ):
                    byte = 320 * (10240 if layer == "final_mixer" else 2560) * 2
                wb += byte
                if n["streamId"] == 364:
                    k = (n["name"], n["gridX"], n["gridY"], n["gridZ"])
                    families[k].append(
                        {
                            "us": (n["end"] - n["start"]) / 1000,
                            "bytes": byte,
                            "group": r["group"],
                        }
                    )
            main_us = total(mi)
            aux_us = total(ai)
            span = (end - begin) / 1000
            rows.append(
                {
                    "device": part[0]["deviceId"],
                    "step": step,
                    "layer": layer,
                    "kind": ("QSA" if int(layer) % 4 == 3 else "GDN")
                    if layer.isdigit()
                    else layer,
                    "kernel_count": len(part),
                    "main_count": len(main),
                    "aux_count": len(aux),
                    "main_service_us": main_us,
                    "aux_service_us": aux_us,
                    "aux_overlap_main_us": overlap,
                    "aux_in_main_gaps_us": aux_us - overlap,
                    "span_us": span,
                    "main_gaps_us": span - main_us,
                    "uncovered_us": span - total(mi + ai),
                    "weight_bytes": wb,
                    "weight_floor_us": wb / 750000,
                    "main_sync_kernels": sum(
                        bool(
                            re.search(
                                "cross_device_reduce|hc_.*(?:push|gather)|nccl",
                                n["name"],
                                re.I,
                            )
                        )
                        for n in main
                    ),
                }
            )
layerrows = []
for layer in dict.fromkeys(r["layer"] for r in rows):
    rr = [r for r in rows if r["layer"] == layer]
    result = {"layer": layer, "kind": rr[0]["kind"]}
    for key in rr[0]:
        if key not in ("device", "step", "layer", "kind"):
            result[key] = statistics.mean(r[key] for r in rr)
    result["excess_us"] = result["span_us"] - result["weight_floor_us"]
    result["estimated_unprofiled_us"] = 0.88 * result["span_us"]
    layerrows.append(result)
kernelrows = []
for (name, x, y, z), rr in families.items():
    count = len(rr) / 20
    us = statistics.mean(r["us"] for r in rr)
    byte = statistics.mean(r["bytes"] for r in rr)
    kernelrows.append(
        {
            "name": name,
            "grid": [x, y, z],
            "calls_per_step": count,
            "mean_us": us,
            "weight_bytes_per_call": byte,
            "weight_floor_us_per_call": byte / 750000,
            "excess_us_per_step": count * (us - byte / 750000),
            "service_us_per_step": count * us,
            "group": rr[0]["group"],
            "actual_dram_bytes": None,
        }
    )
report = {
    "trace_sha256": hashlib.sha256(args.trace.read_bytes()).hexdigest(),
    "stream": 364,
    "bandwidth_GBps": 750,
    "nsys_correction": 0.88,
    "complete_graphs": 20,
    "weight_definition": weights["weight_byte_definition"],
    "step_mean_us": statistics.mean(r["interval_us"] for r in ranks),
    "rank_step_means": {
        str(d): statistics.mean(r["interval_us"] for r in ranks if r["device"] == d)
        for d in range(4)
    },
    "layers": layerrows,
    "kernels": sorted(kernelrows, key=lambda r: r["excess_us_per_step"], reverse=True),
    "raw_layers": rows,
    "limitations": [
        (
            "Auxiliary service is clipped and intersected with stream 364 intervals; "
            "exposed service in gaps is an upper bound, not proof of a wait dependency."
        ),
        "Weight bytes are addressed prepared parameters, not DRAM counters.",
        (
            "Kernel completion/spinning time is not decomposed without flag "
            "instrumentation."
        ),
    ],
}
args.output.write_text(json.dumps(report, indent=2) + "\n")
for kind in ("GDN", "QSA"):
    rr = [r for r in layerrows if r["kind"] == kind and r["layer"] != "1"]
    print(
        kind,
        {
            k: round(statistics.mean(r[k] for r in rr), 3)
            for k in (
                "span_us",
                "main_service_us",
                "aux_service_us",
                "aux_overlap_main_us",
                "weight_floor_us",
                "main_gaps_us",
            )
        },
    )
for group in (
    "HC",
    "shared + router",
    "QSA attention chain",
    "GDN core",
    "routed experts",
):
    rr = [r for r in kernelrows if r["group"] == group]
    print(
        group,
        "main_us",
        round(sum(r["service_us_per_step"] for r in rr), 3),
        "calls",
        sum(r["calls_per_step"] for r in rr),
    )
print(
    "TOP",
    [
        (
            r["name"].split("(")[0][:60],
            round(r["excess_us_per_step"], 3),
            r["calls_per_step"],
            r["grid"],
        )
        for r in report["kernels"][:12]
    ],
)
