# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Count target graph nodes selected by the benchmark's actual replay ranges."""

import argparse
import collections
import json
import sqlite3
import statistics
from pathlib import Path


def family(name):
    low = name.lower()
    for marker, label in (
        ("raw_grouped_gate_up", "GGUF original gate/up"),
        ("small_grouped_vec", "GGUF canonical down"),
        ("_route_and_gather", "Expert route/gather"),
        ("_unroute", "Expert unroute"),
        ("_ngram", "PLE n-gram"),
        ("_gather_packed_rows", "PLE packed row gather"),
        ("dequantize", "Dequantization"),
        ("row_gemv", "FP16 row GEMV"),
        ("_hc_", "HC"),
        ("qwen38_hc", "HC"),
        ("router", "Router"),
        ("shared_gate", "Shared gate"),
        ("_qsa_", "QSA"),
        ("qsa", "QSA"),
        ("cube_allreduce", "TP ring"),
        ("nccl", "TP NCCL"),
        ("transform_hmma_sm70_lattice", "GGUF lattice GEMM"),
        ("transform_hmma_sm70_lut4", "GGUF LUT GEMM"),
        ("transform_hmma_sm70_bitplane", "GGUF bitplane GEMM"),
        ("turbomind::gemm", "TurboMind GEMM"),
        ("cutlass::", "FP16 CUTLASS"),
        ("catarray", "Tensor concatenation"),
        ("cat_kernel", "Tensor concatenation"),
        ("copy", "Tensor copy"),
        ("scatter", "Tensor scatter"),
        ("indexselect", "Tensor index select"),
        ("arange", "Tensor arange"),
        ("sort", "Tensor sort"),
        ("searchsorted", "Tensor searchsorted"),
    ):
        if marker in low:
            return label
    return "Other"


def analyze(sqlite_path, benchmark_path):
    report = json.loads(benchmark_path.read_text())
    workers = report["node_trace"]["workers"]
    ranks = {w["pid"]: w["rank"] for w in workers}
    assert sorted(ranks.values()) == [0, 1, 2, 3]
    with sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True) as db:
        names = dict(db.execute("select id,value from StringIds"))
        ranges = collections.defaultdict(list)
        tables = {r[0] for r in db.execute("select name from sqlite_master")}
        if "NVTX_EVENTS" in tables:
            for start, end, tid in db.execute(
                "select n.start,n.end,n.globalTid from NVTX_EVENTS n "
                "left join StringIds s on s.id=n.textId "
                "where coalesce(n.text,s.value)='graph_parity.target.replay' "
                "order by n.start"
            ):
                if (tid >> 24) & 0xFFFFFF in ranks:
                    ranges[tid].append((start, end))
        selected = collections.defaultdict(list)
        for start, end, tid, correlation in db.execute(
            "select a.start,a.end,a.globalTid,a.correlationId "
            "from CUPTI_ACTIVITY_KIND_RUNTIME a join StringIds s "
            "on s.id=a.nameId where s.value like 'cudaGraphLaunch%' "
            "order by a.start"
        ):
            if any(a <= start and end <= b for a, b in ranges.get(tid, ())):
                selected[(tid >> 24) & 0xFFFFFF].append(correlation)
        selection = "actual replay NVTX ranges"
        if not ranges:
            # CUDA-only capture avoids old NCCL NVTX extension incompatibility.
            # Select the repeated graph with most nodes, then verify its launch
            # count against the independent M5 target CPU records.
            candidates = collections.defaultdict(list)
            for pid, cid, graph, count in db.execute(
                "select (globalPid >> 24) & 16777215,correlationId,"
                "graphNodeId >> 32,count(*) from CUPTI_ACTIVITY_KIND_KERNEL "
                "where graphNodeId != 0 group by 1,2,3 order by min(start)"
            ):
                if pid in ranks:
                    candidates[(pid, graph)].append((cid, count))
            for worker in workers:
                pid = worker["pid"]
                replay_count = sum(
                    e["label"] == "target.replay"
                    and e.get("tokens") == 5
                    and e.get("requests") == 1
                    for e in worker["events"]
                )
                eligible = [
                    rows
                    for (p, _), rows in candidates.items()
                    if p == pid and abs(len(rows) - replay_count) <= 2
                ]
                assert eligible, (pid, replay_count)
                rows = max(eligible, key=lambda r: statistics.median(n for _, n in r))
                assert statistics.median(n for _, n in rows) > 500
                selected[pid] = [cid for cid, _ in rows]
            selection = "largest repeated graph, checked against M5 CPU replay counts"
        assert set(selected) == set(ranks)
        # Trim transitions independently by rank ordinal, never by a time window.
        lengths = {len(v) for v in selected.values()}
        assert len(lengths) == 1, lengths
        selected = {pid: ids[8:-8] for pid, ids in selected.items()}
        assert min(map(len, selected.values())) >= 8
        db.execute("create temp table launches(pid integer,cid integer)")
        db.executemany(
            "insert into launches values (?,?)",
            [(pid, cid) for pid, ids in selected.items() for cid in ids],
        )
        samples = collections.defaultdict(collections.Counter)
        service = collections.defaultdict(float)
        graphs = collections.Counter()
        for pid, cid, graph, name_id, duration, count in db.execute(
            "select (k.globalPid >> 24) & 16777215,k.correlationId,"
            "k.graphNodeId >> 32,"
            "k.demangledName,sum(k.end-k.start),count(*) "
            "from CUPTI_ACTIVITY_KIND_KERNEL k join launches l "
            "on l.pid=((k.globalPid >> 24) & 16777215) and l.cid=k.correlationId "
            "group by 1,2,3,4"
        ):
            assert graph, (pid, cid, graph)
            samples[(pid, cid)][names[name_id]] += count
            service[names[name_id]] += duration / 1e6
            graphs[(ranks[pid], graph)] += count
        expected = sum(map(len, selected.values()))
        assert len(samples) == expected, (len(samples), expected)
        per_rank = {}
        for pid, ids in selected.items():
            counts = [sum(samples[(pid, cid)].values()) for cid in ids]
            assert len(set(counts)) == 1, (ranks[pid], counts)
            per_rank[ranks[pid]] = dict(rounds=len(ids), kernels=counts[0])
        totals = sum(samples.values(), collections.Counter())
        kernels = [
            dict(
                name=name,
                family=family(name),
                calls_per_rank_round=count / expected,
                rank_mean_service_ms=service[name] / expected,
            )
            for name, count in totals.most_common()
        ]
        families = collections.defaultdict(lambda: [0.0, 0.0])
        for row in kernels:
            cell = families[row["family"]]
            cell[0] += row["calls_per_rank_round"]
            cell[1] += row["rank_mean_service_ms"]
        return dict(
            per_rank=per_rank,
            selection=selection,
            rank_mean_kernel_count=statistics.mean(
                r["kernels"] for r in per_rank.values()
            ),
            families={
                k: dict(calls=v[0], service_ms=v[1]) for k, v in families.items()
            },
            kernels=kernels,
            graph_ids=[dict(rank=r, graph=g, nodes=n) for (r, g), n in graphs.items()],
            interpretation=(
                "Graph-node service is profiled composition, not round latency."
            ),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = analyze(args.sqlite, args.benchmark)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "kernels"}, indent=2))


if __name__ == "__main__":
    main()
