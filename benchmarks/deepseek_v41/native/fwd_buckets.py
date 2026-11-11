# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bucket kernels of forward bursts (>300 kernels) of a stage sqlite; per step."""

import collections
import sqlite3
import statistics as st
import sys

sys.path.insert(0, "/home/ymzx/v41")
from nsys_buckets import bucket

con = sqlite3.connect(sys.argv[1])
rows = con.execute(
    "SELECT k.start,k.end,s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id WHERE k.deviceId=0 ORDER BY k.start"
).fetchall()
bursts, cur = [], [rows[0]]
for r in rows[1:]:
    if r[0] - cur[-1][1] > 0.5e6:
        bursts.append(cur)
        cur = [r]
    else:
        cur.append(r)
big = [b for b in bursts if len(b) > 300]
med = sorted(len(b) for b in big)[len(big) // 2]
big = [b for b in big if abs(len(b) - med) <= 0.1 * med]
agg, cnt = collections.Counter(), collections.Counter()
span, busy = [], []
for b in big:
    span.append((b[-1][1] - b[0][0]) / 1e6)
    busy.append(sum(e - s for s, e, _ in b) / 1e6)
    for s, e, n in b:
        k = bucket(n)
        agg[k] += (e - s) / 1e3
        cnt[k] += 1
n = len(big)
print(
    f"{sys.argv[1]}: {n} fwd steps, span {st.median(span):.2f} ms, kernel-busy {st.median(busy):.2f} ms, {sum(cnt.values()) / n:.0f} kernels"
)
for k, v in agg.most_common():
    print(f"   {k:14s} {v / n:7.0f} us  {cnt[k] / n:6.1f} kernels")
