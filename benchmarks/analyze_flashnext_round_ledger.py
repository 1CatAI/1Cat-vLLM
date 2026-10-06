# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Close four-rank MTP round windows without summing concurrent GPU service."""

import argparse
import collections
import json
import sqlite3
import statistics
from pathlib import Path

from benchmarks.analyze_flashnext_graph_nodes import family


def union_ns(intervals):
    end = total = 0
    for start, stop in sorted(intervals):
        total += max(0, stop - max(start, end))
        end = max(end, stop)
    return total


def clipped(intervals, start, end):
    return [(max(a, start), min(b, end)) for a, b in intervals if a < end and b > start]


def exclusive_activity_ns(scopes, start, end):
    """Partition the window, prioritizing graph work over overlapping copies."""
    points = collections.defaultdict(collections.Counter)
    points[start]
    points[end]
    for label, spans in scopes.items():
        for a, b in clipped(spans, start, end):
            points[a][label] += 1
            points[b][label] -= 1
    active = collections.Counter()
    totals = collections.Counter()
    previous = start
    priority = ("target", "draft", "outside graphs", "copies")
    for at, changes in sorted(points.items()):
        label = next((key for key in priority if active[key]), "no activity")
        totals[label] += at - previous
        active.update(changes)
        previous = at
    assert sum(totals.values()) == end - start
    return dict(totals)


def stats(values):
    ordered = sorted(values)
    return {
        "mean": statistics.mean(ordered),
        "p50": statistics.median(ordered),
        "p90": ordered[round(0.9 * (len(ordered) - 1))],
        "max": ordered[-1],
    }


def classify(name):
    for marker, label in (
        ("hcx", "HC complete boundary"),
        ("down3", "HC down"),
        ("up3", "HC up"),
        ("dense_mv", "GGUF dense MMA"),
        ("swiglu_mv", "Shared expert gate/up"),
        ("quantize_q8", "Activation quantization"),
        ("dmvq", "GGUF dense dp4a"),
        ("dmv", "GGUF dense MMA"),
        ("dp4a", "GGUF dp4a"),
        ("gate_up", "Expert gate/up"),
        ("gdn", "GDN"),
        ("delta_rule", "GDN"),
        ("conv1d", "Convolution"),
        ("rmsnorm_gated", "Gated normalization"),
        ("top1", "Head/argmax"),
        ("lm_head", "Head/argmax"),
    ):
        if marker in name.lower():
            return label
    return family(name)


def analyze(sqlite_path, benchmark_path, trim=8):
    report = json.loads(benchmark_path.read_text())
    workers = report["node_trace"]["workers"]
    ranks = {w["pid"]: w["rank"] for w in workers}
    assert sorted(ranks.values()) == [0, 1, 2, 3]
    with sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True) as db:
        names = dict(db.execute("select id,value from StringIds"))
        ranges = collections.defaultdict(list)
        for start, end, tid, label in db.execute(
            "select n.start,n.end,n.globalTid,coalesce(n.text,s.value) "
            "from NVTX_EVENTS n left join StringIds s on s.id=n.textId "
            "where coalesce(n.text,s.value) in "
            "('graph_parity.target.replay','graph_parity.draft.propose') "
            "order by n.start"
        ):
            if (tid >> 24) & 0xFFFFFF in ranks:
                ranges[tid].append((start, end, label))
        scope = {}
        target_ids = collections.defaultdict(list)
        for start, end, tid, cid in db.execute(
            "select a.start,a.end,a.globalTid,a.correlationId "
            "from CUPTI_ACTIVITY_KIND_RUNTIME a join StringIds s on s.id=a.nameId "
            "where s.value like 'cudaGraphLaunch%' order by a.start"
        ):
            pid = (tid >> 24) & 0xFFFFFF
            if pid not in ranks:
                continue
            labels = [label for a, b, label in ranges[tid] if a <= start and end <= b]
            if "graph_parity.target.replay" in labels:
                target_ids[pid].append(cid)
                scope[pid, cid] = "target"
            elif "graph_parity.draft.propose" in labels:
                scope[pid, cid] = "draft"
        assert set(target_ids) == set(ranks), "Missing target replay NVTX ranges"
        assert len({len(v) for v in target_ids.values()}) == 1
        kernels = collections.defaultdict(list)
        envelopes = {}
        for pid, start, end, cid, name in db.execute(
            "select (globalPid >> 24)&16777215,start,end,correlationId,demangledName "
            "from CUPTI_ACTIVITY_KIND_KERNEL order by start"
        ):
            if pid not in ranks:
                continue
            kernels[pid].append(
                (start, end, scope.get((pid, cid), "outside graphs"), names[name])
            )
            if scope.get((pid, cid)) == "target":
                a, b = envelopes.get((pid, cid), (start, end))
                envelopes[pid, cid] = min(a, start), max(b, end)
        copies = collections.defaultdict(list)
        tables = {r[0] for r in db.execute("select name from sqlite_master")}
        for table in ("CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"):
            if table not in tables:
                continue
            for pid, start, end in db.execute(
                f"select (globalPid >> 24)&16777215,start,end from {table}"
            ):
                if pid in ranks:
                    copies[pid].append((start, end))
    count = len(next(iter(target_ids.values())))
    assert count > 2 * trim + 2
    rows = []
    for ordinal in range(trim, count - trim - 1):
        starts = {
            pid: envelopes[pid, ids[ordinal]][0] for pid, ids in target_ids.items()
        }
        ends = {pid: envelopes[pid, ids[ordinal]][1] for pid, ids in target_ids.items()}
        # Common windows align TP by replay ordinal, never by temporal clustering.
        start = min(starts.values())
        end = min(
            envelopes[pid, ids[ordinal + 1]][0] for pid, ids in target_ids.items()
        )
        assert end > start
        rank_rows = {}
        for pid, rank in ranks.items():
            active = [
                (a, b, s, name)
                for a, b, s, name in kernels[pid]
                if a < end and b > start
            ]

            by_scope = collections.defaultdict(list)
            by_family = collections.defaultdict(list)
            for a, b, s, name in active:
                by_scope[s].append((a, b))
                by_family[s + ": " + classify(name)].append((a, b))
            kernel_intervals = clipped([(a, b) for a, b, _, _ in active], start, end)
            copy_intervals = clipped(copies[pid], start, end)
            busy = union_ns(kernel_intervals + copy_intervals)
            kernel_busy = union_ns(kernel_intervals)
            service = sum(b - a for a, b in kernel_intervals)
            rank_rows[rank] = {
                "busy_ms": busy / 1e6,
                "no_recorded_activity_ms": (end - start - busy) / 1e6,
                "concurrent_kernel_overlap_ms": (service - kernel_busy) / 1e6,
                "scope_union_ms": {
                    k: union_ns(clipped(v, start, end)) / 1e6
                    for k, v in by_scope.items()
                },
                "family_union_ms": {
                    k: union_ns(clipped(v, start, end)) / 1e6
                    for k, v in by_family.items()
                },
                "kernel_count": len(active),
                "exclusive_activity_ms": {
                    k: v / 1e6
                    for k, v in exclusive_activity_ns(
                        {**by_scope, "copies": copy_intervals}, start, end
                    ).items()
                },
            }
        rows.append(
            {
                "ordinal": ordinal,
                "window_ms": (end - start) / 1e6,
                "target_envelope_ms": (max(ends.values()) - start) / 1e6,
                "target_entry_skew_ms": (max(starts.values()) - start) / 1e6,
                "ranks": rank_rows,
            }
        )
    summaries = {}
    for rank in range(4):
        cells = [r["ranks"][rank] for r in rows]
        fields = (
            "busy_ms",
            "no_recorded_activity_ms",
            "concurrent_kernel_overlap_ms",
            "kernel_count",
        )
        summaries[rank] = {k: stats([c[k] for c in cells]) for k in fields}
        for field in ("scope_union_ms", "family_union_ms", "exclusive_activity_ms"):
            keys = set().union(*(c[field] for c in cells))
            summaries[rank][field] = {
                k: stats([c[field].get(k, 0) for c in cells]) for k in sorted(keys)
            }
    return {
        "scope": "Profiled common TP windows; not unprofiled acceptance latency.",
        "warning": (
            "Activity gaps exclude recorded kernels/copies; spinning inside a "
            "kernel is not identified as idle. Family unions overlap and must "
            "not be summed."
        ),
        "rounds": len(rows),
        "window_ms": stats([r["window_ms"] for r in rows]),
        "target_envelope_ms": stats([r["target_envelope_ms"] for r in rows]),
        "target_entry_skew_ms": stats([r["target_entry_skew_ms"] for r in rows]),
        "per_rank": summaries,
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = analyze(args.sqlite, args.benchmark)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("rows", "per_rank")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
