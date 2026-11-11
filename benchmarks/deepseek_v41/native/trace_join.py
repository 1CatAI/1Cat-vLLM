# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Join v41 traces of 4 nodes into per-round critical-path segments.

Usage: trace_join.py OFFSETS_JSON LOG... (offset_us per host: remote - Dev-1)
Prints median per segment over decode rounds (6 scheduled tokens).
"""

import collections
import json
import statistics as st
import sys

import regex as re

offsets = json.load(open(sys.argv[1]))  # {"C4140-Dev-2": -8320.0, ...} in us
ev = collections.defaultdict(list)  # (role, event) -> [(t_us, fields)]
for path in sys.argv[2:]:
    host = re.search(r"v41trace-(C4140-Dev-\d)", path).group(1)
    off = offsets.get(host, 0.0)
    for line in open(path):
        p = line.split()
        if len(p) < 3:
            continue
        role, event, t = p[0], p[1], int(p[2]) / 1e3 - off
        fields = dict(x.split("=", 1) for x in p[3:] if "=" in x)
        ev[(role, event)].append((t, fields))


def seq(role, event, n=None):
    xs = [t for t, f in ev[(role, event)] if n is None or f.get("n") == str(n)]
    return sorted(xs)


roles = sorted({r for r, _ in ev if r.startswith("w")}, key=lambda r: int(r[1:]))
print(
    "roles", roles, {k: len(v) for k, v in ev.items() if k[1] in ("launch", "e_sched")}
)
sched = sorted(t for t, f in ev[("engine", "e_sched")] if f.get("n") == "6")
upd = seq("engine", "e_update")
res = seq("engine", "e_result")
L = {r: seq(r, "launch", 6) for r in roles}
F = {r: seq(r, "fwd_done", 6) for r in roles}
R = {r: seq(r, "pp_ready") for r in roles}
S = {r: seq(r, "sent") for r in roles}
D = {r: seq(r, "deq") for r in roles}
last = roles[-1]
SE, SX, OS = seq(last, "s_enter"), seq(last, "s_exit"), seq(last, "out_sent")


def after(xs, t, limit=60000):
    for x in xs:
        if x >= t and x - t < limit:
            return x
    return None


segs = collections.defaultdict(list)
for t0 in sched[5:-5]:
    row = {"sched": t0}
    for r in roles:
        row[r + ".launch"] = after(L[r], t0)
        row[r + ".fwd_done"] = after(F[r], t0)
        if r != roles[0]:
            row[r + ".pp_ready"] = after(R[r], t0)
        if r != last:
            row[r + ".sent"] = after(S[r], t0)
    row["s_enter"] = after(SE, row[last + ".fwd_done"] or t0)
    row["s_exit"] = after(SX, row["s_enter"] or t0)
    row["out_sent"] = after(OS, row["s_exit"] or t0)
    row["e_result"] = after(res, row["out_sent"] or t0)
    row["e_update"] = after(upd, row["e_result"] or t0)
    nxt = after(sched, t0 + 1)
    row["next_sched"] = nxt
    if any(v is None for v in row.values()):
        continue
    prev = ("sched", t0)
    order = []
    for r in roles:
        if r != roles[0]:
            order.append(r + ".pp_ready")
        order.append(r + ".launch") if r == roles[0] else None
        order.append(r + ".fwd_done")
        if r != last:
            order.append(r + ".sent")
    order += ["s_enter", "s_exit", "out_sent", "e_result", "e_update", "next_sched"]
    for k in order:
        segs[f"{prev[0]} -> {k}"].append(row[k] - prev[1])
        prev = (k, row[k])
    segs["TOTAL round"].append(row["next_sched"] - t0)
    for r in roles[1:]:
        segs[f"(info) {r}.launch - sched"].append(row[r + ".launch"] - t0)
for k, v in segs.items():
    print(f"{st.median(v) / 1e3:8.3f} ms  {k}  (n={len(v)})")
