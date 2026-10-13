# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Excess main-stream kernel time while the cold-expert transcode side stream runs.

Usage: python nsys_side_stream_excess.py trace.sqlite [deviceId]
Compares each (kernel, grid) family's duration with vs without overlap.
"""

import bisect
import collections
import sqlite3
import statistics as S
import sys

c = sqlite3.connect(sys.argv[1])
dev = int(sys.argv[2]) if len(sys.argv) > 2 else 0
names = dict(c.execute("select id,value from StringIds").fetchall())
ks = c.execute(
    "select start,end,streamId,demangledName,gridX,gridY,gridZ,blockX "
    "from CUPTI_ACTIVITY_KIND_KERNEL where deviceId=? order by start",
    (dev,),
).fetchall()
main = collections.Counter(r[2] for r in ks).most_common(1)[0][0]
mer = []
for s, e, sid, n, *_ in ks:
    if sid != main and "transcode" in names[n] and _[0] <= 8:
        if mer and s <= mer[-1][1]:
            mer[-1][1] = max(mer[-1][1], e)
        else:
            mer.append([s, e])
starts = [m[0] for m in mer]


def ov(s, e):
    i = bisect.bisect_right(starts, e)
    t = 0
    for a, b in mer[max(0, i - 3) : i]:
        t += max(0, min(b, e) - max(a, s))
    return t / max(1, e - s)


fam = collections.defaultdict(lambda: [[], []])
for s, e, sid, n, gx, gy, gz, bx in ks:
    if sid != main:
        continue
    k = (f"{names[n][:70]} g={gx},{gy},{gz} b={bx}", gx, gy, gz, bx)
    o = ov(s, e)
    (fam[k][0] if o > 0.5 else fam[k][1]).append(e - s)
tot = 0
rows = []
for k, (a, b) in fam.items():
    if len(b) >= 3 and a:
        base = S.median(b)
        ex = sum(max(0, x - base) for x in a)
        tot += ex
        rows.append((ex / 1e6, len(a), base / 1e3, S.median(a) / 1e3, k[0][-40:]))
print(f"excess on main stream while cold transcode runs: {tot / 1e6:.0f} ms")
for r in sorted(rows, reverse=True)[:12]:
    ms, count, base, slow, name = r
    print(f"{ms:6.0f} ms n={count:4d} base {base:6.0f} us -> {slow:6.0f} us  {name}")
