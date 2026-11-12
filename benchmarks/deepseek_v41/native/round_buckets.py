# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-round GPU kernel time by budget bucket, between the first and last
token broadcast kernel of a stage (async PP NCCL mode). NCCL kernels are
reported separately (their time includes spinning on data)."""

import importlib.util
import sqlite3
import sys
from collections import defaultdict

spec = importlib.util.spec_from_file_location(
    "nb", __file__.replace("round_buckets.py", "nsys_buckets.py")
)
nb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nb)
db = sys.argv[1]
con = sqlite3.connect(db)
rows = con.execute("""SELECT k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k
    JOIN StringIds s ON k.demangledName = s.id WHERE k.deviceId = 0 ORDER BY k.start""").fetchall()
b = [i for i, r in enumerate(rows) if "Broadcast" in r[2]]
rows = rows[b[0] + 1 : b[-1] + 1]
rounds = len(b) - 1
tot = defaultdict(float)
cnt = defaultdict(int)
names = defaultdict(float)
for s, e, n in rows:
    k = "nccl" if "nccl" in n.lower() else nb.bucket(n)
    tot[k] += (e - s) / 1e6
    cnt[k] += 1
    names[n[:80]] += (e - s) / 1e6
comp = sum(v for k, v in tot.items() if k != "nccl")
print(
    f"{db}: rounds {rounds} compute {comp / rounds:.3f} ms/round, kernels {sum(cnt.values()) / rounds:.0f}"
)
for k in sorted(tot, key=lambda k: -tot[k]):
    print(f"  {k:14s} {tot[k] / rounds:7.3f} ms {cnt[k] / rounds:6.1f} k")
print(" top:")
for n in sorted(names, key=lambda k: -names[k])[:18]:
    if "nccl" in n.lower():
        continue
    print(f"  {names[n] / rounds:7.3f}  {n}")
