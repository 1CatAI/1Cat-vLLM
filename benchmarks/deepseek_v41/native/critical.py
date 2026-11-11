# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-step critical-path pieces of one PP stage from its nsys sqlite (GPU 0).

recv_done -> send_start = this stage's compute on the chain; also reports
the recv wait and, on the last stage, the time after its last kernel."""

import sqlite3
import statistics as st
import sys

db, stage = sys.argv[1], int(sys.argv[2])
con = sqlite3.connect(db)
rows = con.execute("""SELECT k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k
    JOIN StringIds s ON k.demangledName = s.id WHERE k.deviceId = 0 ORDER BY k.start""").fetchall()
bursts, cur = [], [rows[0]]
for r in rows[1:]:
    if r[0] - cur[-1][1] > 2e6:
        bursts.append(cur)
        cur = [r]
    else:
        cur.append(r)
bursts.append(cur)
typical = sorted(len(b) for b in bursts)[len(bursts) // 2]
steps = [b for b in bursts if abs(len(b) - typical) <= 0.1 * typical]
comp, wait, total, starts = [], [], [], []
for b in steps:
    sr = [r for r in b if "SendRecv" in r[2]]
    first, last = b[0][0], b[-1][1]
    starts.append(first)
    if stage == 0:
        send = sr[-1][0] if sr else last
        comp.append((send - first) / 1e6)
        wait.append(0)
    else:
        recv = sr[0]
        end = sr[-1][0] if stage < 3 and len(sr) > 1 else last
        wait.append((recv[1] - recv[0]) / 1e6)
        comp.append((end - recv[1]) / 1e6)
    total.append((last - first) / 1e6)
gaps = [(b - a) / 1e6 for a, b in zip(starts, starts[1:])]
print(
    f"stage {stage}: steps {len(steps)}  compute(chain) {st.median(comp):6.2f} ms  recv-wait {st.median(wait):6.2f}  burst {st.median(total):6.2f}  step-period {st.median(gaps):6.2f} ms"
)
