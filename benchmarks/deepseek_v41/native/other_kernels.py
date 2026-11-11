# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib.util
import sqlite3
import sys
from collections import defaultdict

spec = importlib.util.spec_from_file_location(
    "nb", __file__.replace("other_kernels.py", "nsys_buckets.py")
)
nb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nb)
con = sqlite3.connect(sys.argv[1])
rows = con.execute("""SELECT k.start, k.end, s.value, k.gridX*k.gridY*k.gridZ, k.blockX FROM CUPTI_ACTIVITY_KIND_KERNEL k
    JOIN StringIds s ON k.demangledName = s.id WHERE k.deviceId = 0 ORDER BY k.start""").fetchall()
b = [i for i, r in enumerate(rows) if "Broadcast" in r[2]]
rows = rows[b[0] + 1 : b[-1] + 1]
rounds = len(b) - 1
agg = defaultdict(lambda: [0.0, 0])
for s, e, n, g, bx in rows:
    if "nccl" in n.lower():
        continue
    if nb.bucket(n) not in ("other",) and sys.argv[2:3] != ["all"]:
        continue
    key = n[:110]
    agg[key][0] += (e - s) / 1e6
    agg[key][1] += 1
for k in sorted(agg, key=lambda k: -agg[k][0])[:30]:
    t, c = agg[k]
    print(f"{t / rounds:7.3f} ms {c / rounds:6.1f}x {1000 * t / c:6.1f}us  {k}")
