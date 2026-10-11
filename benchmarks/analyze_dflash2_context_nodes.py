# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attribute graph nodes through launch correlations and recorded round ranges.

GPU service sums from an instrumented trace are diagnostic, not unprofiled
latency. In particular, resident TP kernels can include waits for other ranks.
"""

import argparse
import collections
import json
import sqlite3
import statistics
from pathlib import Path

import regex as re


def family(name: str) -> str:
    for marker, label in (
        ("allreduce", "TP reduce/wait"),
        ("cross_device_reduce", "TP reduce/wait"),
        ("nccl", "TP NCCL"),
        ("grouped_verify", "Grouped attention"),
        ("flash_attention", "Attention"),
        ("nvfp4_", "NVFP4 linear"),
        ("fp8_qpn", "FP8 linear"),
        ("sigmoid_gating_delta_rule", "GDN"),
        ("conv", "Convolution"),
        ("norm", "Normalization"),
        ("rope", "RoPE/cache"),
        ("cache", "Cache"),
        ("sample", "Sampling"),
        ("rejection", "Sampling"),
        ("copy", "Copy"),
    ):
        if marker in name.lower():
            return label
    return "Other"


def analyze(sqlite: Path, rounds_dir: Path, trim: int) -> dict:
    with sqlite3.connect(f"file:{sqlite}?mode=ro", uri=True) as db:
        names = dict(db.execute("select id,value from StringIds"))
        spans = collections.defaultdict(list)
        rounds = collections.defaultdict(dict)
        expression = re.compile(r"^quasar/rank(\d+)/round(\d+)/(.*)$")
        for start, end, tid, label in db.execute(
            "select n.start,n.end,n.globalTid,coalesce(n.text,s.value) "
            "from NVTX_EVENTS n left join StringIds s on n.textId=s.id "
            "where n.end is not null order by n.start"
        ):
            match = expression.match(label or "")
            if match is None:
                continue
            rank, step = map(int, match.group(1, 2))
            method = match.group(3)
            if method.endswith(".propose"):
                stage, priority = "draft", 4
            elif method.endswith(".run_fullgraph"):
                stage, priority = "target graph", 3
            elif method.endswith(".sample_tokens"):
                stage, priority = "sample/state", 2
            else:
                stage, priority = "target setup/state", 1
            spans[tid].append((start, end, priority, rank, step, stage))
            if method.endswith(".execute_model"):
                if step in rounds[rank]:
                    raise ValueError(f"Duplicate execute boundary: rank {rank}, {step}")
                rounds[rank][step] = start
        if set(rounds) != {0, 1, 2, 3}:
            raise ValueError(f"Expected four TP ranks, got {sorted(rounds)}")
        steps = sorted(rounds[0])
        if any(sorted(rows) != steps for rows in rounds.values()):
            raise ValueError("Rank round ordinals differ")
        if trim < 1 or len(steps) <= 2 * trim:
            raise ValueError("Trim must discard both transition rounds")
        selected = set(steps[trim:-trim])
        observations = {}
        for rank in rounds:
            rows = json.loads((rounds_dir / f"rank{rank}-rounds.json").read_text())
            observations[rank] = {row["step"]: row for row in rows}
        emits = {}
        for step in selected:
            values = []
            for rank, rows in observations.items():
                current, following = rows[step], rows[step + 1]
                if current["request_ids"] != following["request_ids"]:
                    raise ValueError("Request changed within the selected round")
                if current["scheduled"] != 8 or len(current["output_tokens"]) != 1:
                    raise ValueError("Expected the captured single-request q8 verifier")
                values.append(
                    following["output_tokens"][0] - current["output_tokens"][0]
                )
            if len(set(values)) != 1 or values[0] <= 0:
                raise ValueError(f"Invalid cross-rank output delta: {step}, {values}")
            emits[step] = values[0]
        service = collections.defaultdict(collections.Counter)
        kernels = collections.Counter()
        counts = collections.Counter()
        coverage = collections.Counter()
        query = (
            "select k.start,k.end,k.globalPid,k.graphNodeId,k.demangledName,"
            "r.start,r.globalTid from CUPTI_ACTIVITY_KIND_KERNEL k "
            "left join CUPTI_ACTIVITY_KIND_RUNTIME r "
            "on k.correlationId=r.correlationId "
            "and k.globalPid=(r.globalTid & -16777216)"
        )
        total = db.execute(
            "select count(*) from CUPTI_ACTIVITY_KIND_KERNEL"
        ).fetchone()[0]
        for start, end, _pid, node, name_id, launch, tid in db.execute(query):
            coverage["rows"] += 1
            if launch is None:
                coverage["missing_launch"] += 1
                continue
            coverage["correlated"] += 1
            active = [span for span in spans[tid] if span[0] <= launch < span[1]]
            if not active:
                coverage["outside_recorded_stages"] += 1
                continue
            _, _, _, rank, step, stage = max(active, key=lambda span: span[2])
            if step not in selected:
                continue
            name = names[name_id]
            duration = (end - start) / 1e6
            service[(rank, step)][(stage, family(name))] += duration
            kernels[(stage, name)] += duration
            counts[(stage, name)] += 1
            coverage["selected"] += 1
            coverage["selected_graph_nodes"] += bool(node)
        if coverage["rows"] != total or coverage["missing_launch"]:
            raise ValueError(
                f"Ambiguous or missing launch correlation: {dict(coverage)}"
            )
        if len(service) != len(selected) * len(rounds):
            raise ValueError("Missing selected rank/round GPU activity")
        records = []
        for step in sorted(selected):
            ranks = []
            for rank in sorted(rounds):
                wall = (rounds[rank][step + 1] - rounds[rank][step]) / 1e6
                cells = service[(rank, step)]
                ranks.append(
                    dict(
                        rank=rank,
                        profiled_round_wall_ms=wall,
                        service_ms=sum(cells.values()),
                        stages=[
                            dict(stage=stage, family=label, service_ms=value)
                            for (stage, label), value in cells.items()
                        ],
                    )
                )
            records.append(dict(step=step, emitted_tokens=emits[step], ranks=ranks))
        denominator = len(service)
        return dict(
            scope="Profiled service composition; not end-to-end performance acceptance",
            selection="CPU NVTX launch containment plus CUDA correlation/global PID",
            sqlite=str(sqlite),
            rounds_dir=str(rounds_dir),
            selected_steps=sorted(selected),
            coverage=dict(coverage),
            emitted_tokens=sum(emits.values()),
            round_wall_rank_max_median_ms=statistics.median(
                max(row["profiled_round_wall_ms"] for row in record["ranks"])
                for record in records
            ),
            kernels=[
                dict(
                    stage=stage,
                    name=name,
                    family=family(name),
                    calls_per_rank_round=counts[(stage, name)] / denominator,
                    rank_mean_service_ms=value / denominator,
                )
                for (stage, name), value in kernels.most_common()
            ],
            rounds=records,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument("--rounds-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trim", type=int, default=1)
    args = parser.parse_args()
    report = analyze(args.sqlite, args.rounds_dir, args.trim)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key not in ("kernels", "rounds")
            }
        )
    )


if __name__ == "__main__":
    main()
