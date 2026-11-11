# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Print one steady round of a stage's GPU timeline, collapsing runs of
non-NCCL kernels: name groups with start offset, busy and gaps."""

import sqlite3
import sys

db, which = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 20
con = sqlite3.connect(db)
rows = con.execute("""SELECT k.start, k.end, s.value, k.streamId FROM CUPTI_ACTIVITY_KIND_KERNEL k
    JOIN StringIds s ON k.demangledName = s.id WHERE k.deviceId = 0 ORDER BY k.start""").fetchall()
names = {}
for r in rows:
    if "nccl" in r[2].lower():
        names[r[2][:60]] = names.get(r[2][:60], 0) + 1
print("nccl kernels:", names)
# rounds: split at idle > 1 ms
bursts, cur = [], [rows[0]]
for r in rows[1:]:
    if (
        r[0] - max(x[1] for x in cur[-3:]) > 1e6
        and "nccl" not in cur[-1][2].lower()
        or (r[0] - cur[-1][1] > 1e6)
    ):
        bursts.append(cur)
        cur = [r]
    else:
        cur.append(r)
bursts.append(cur)
print("bursts", len(bursts), [len(b) for b in bursts[which : which + 8]])
t0 = bursts[which][0][0]
for b in bursts[which : which + 4]:
    print(
        f"--- burst at {(b[0][0] - t0) / 1e6:8.3f} ms, {len(b)} kernels, span {(b[-1][1] - b[0][0]) / 1e6:.3f}"
    )
    grp = []
    for r in b:
        nm = "NCCL:" + r[2][14:40] if "nccl" in r[2].lower() else "k"
        if grp and grp[-1][0] == nm and nm == "k" and r[0] - grp[-1][2] < 50e3:
            grp[-1][2] = r[1]
            grp[-1][3] += 1
            grp[-1][4] = r[2][:40]
        else:
            grp.append([nm, r[0], r[1], 1, r[2][:40]])
    prev = None
    for g in grp:
        gap = (g[1] - prev) / 1e6 if prev else 0
        print(
            f"  +{(g[1] - t0) / 1e6:8.3f} gap {gap:6.3f} {g[0]:30s} n={g[3]:4d} dur {(g[2] - g[1]) / 1e6:7.3f} last={g[4]}"
        )
        prev = g[2]
