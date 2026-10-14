# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Diagnostic MTP4 verifier phases, aligned by step ordinal across all TP4 ranks."""

import argparse
import json
import statistics
from pathlib import Path

import regex as re

p = argparse.ArgumentParser()
p.add_argument("log", type=Path)
p.add_argument("requests", type=Path)
p.add_argument("output", type=Path)
a = p.parse_args()


def stats(v):
    x = sorted(v)
    return dict(
        n=len(x),
        mean=statistics.mean(x),
        p50=statistics.median(x),
        p90=x[round(0.9 * (len(x) - 1))],
        p99=x[round(0.99 * (len(x) - 1))],
        max=x[-1],
    )


raw = a.log.read_bytes()
result = []
for request in json.loads(a.requests.read_text()):
    c = request["concurrency"]
    groups = {}
    for line in (
        raw[request["profile_byte_start"] : request["profile_byte_end"]]
        .decode(errors="replace")
        .splitlines()
    ):
        match = re.search(
            r"Worker_TP(\d+).*SM70 V2 MTP profile interval_avg_ms "
            r"calls=(\d+) interval_calls=1 tokens=(\d+) drafts=(\d+) (.*)",
            line,
        )
        if not match:
            continue
        rank, ordinal, tokens, drafts = map(int, match.groups()[:4])
        if (tokens, drafts) != (5 * c, 4 * c):
            continue
        values = {k: float(v) for k, v in re.findall(r"([a-z_]+)=([0-9.]+)", match[5])}
        if rank in groups.setdefault(ordinal, {}):
            raise ValueError("Duplicate rank ordinal")
        groups[ordinal][rank] = values
    complete = [(n, g) for n, g in sorted(groups.items()) if set(g) == {0, 1, 2, 3}]
    if request["repetition"] < 0:
        continue
    if len(complete) < 16:
        raise ValueError(("Too few full-concurrency verifier rounds", c, len(complete)))
    records = []
    for ordinal, ranks in complete:
        critical = max(ranks, key=lambda r: ranks[r]["target_verifier_gpu"])
        rec = dict(
            ordinal=ordinal, critical_rank=critical, ranks=ranks, **ranks[critical]
        )
        rec["rank_max_verifier_wall_ms"] = max(
            r["target_verifier_wall_cpu"] for r in ranks.values()
        )
        records.append(rec)
    fields = (
        "target_verifier_gpu",
        "rank_max_verifier_wall_ms",
        "target_forward",
        "target_sample",
        "target_state_update",
        "draft_total",
        "total_gpu",
    )
    row = dict(
        concurrency=c,
        repetition=request["repetition"],
        complete_rounds=len(complete),
        incomplete_rounds=len(groups) - len(complete),
        phases={k: stats([r[k] for r in records]) for k in fields},
        per_round=records,
    )
    result.append(row)
    print(c, request["repetition"], json.dumps(row["phases"]), flush=True)
a.output.write_text(
    json.dumps(
        dict(
            definition=(
                "Per-ordinal TP4 maximum GPU event sum: target forward + sampling "
                "+ state commit. Diagnostic events/fences are enabled. Draft is "
                "separate; stage statistics are taken on the critical verifier rank."
            ),
            cases=result,
        ),
        indent=2,
    )
)
