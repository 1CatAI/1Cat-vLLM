# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefill kernel ledger of one stage: the window after the largest idle gap
(the pause between the profiled decode and the prefill), by bucket."""

import sqlite3
import sys
from collections import defaultdict

import regex as re

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from decode_ledger import bucket  # noqa

con = sqlite3.connect(sys.argv[1])
rows = con.execute("""SELECT k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k
    JOIN StringIds s ON k.demangledName = s.id WHERE k.deviceId = 0 ORDER BY k.start""").fetchall()
gaps = [
    (rows[i + 1][0] - max(r[1] for r in rows[max(0, i - 50) : i + 1]), i)
    for i in range(len(rows) - 1)
]
g, i0 = max(gaps)
big = [r for r in rows[i0 + 1 :] if "nccl" not in r[2].lower() or True]
tot = defaultdict(float)
cnt = defaultdict(int)
top = defaultdict(float)
for s_, e, n in big:
    b = "nccl" if "nccl" in n.lower() else bucket(n)
    tot[b] += (e - s_) / 1e6
    cnt[b] += 1
    top[re.sub(r"[(<].*", "", n)[:70]] += (e - s_) / 1e6
span = (big[-1][1] - big[0][0]) / 1e6
iv = sorted((r[0], r[1]) for r in big if "nccl" not in r[2].lower())
busy = 0
cs, ce = iv[0]
for s, e in iv[1:]:
    if s > ce:
        busy += ce - cs
        cs, ce = s, e
    else:
        ce = max(ce, e)
busy += ce - cs
print(
    f"{sys.argv[1]}: prefill window after {g / 1e6:.0f} ms gap: span {span:.0f} ms, busy(non-nccl) {busy / 1e6:.0f} ms, kernels {len(big)}"
)
for b in sorted(tot, key=lambda k: -tot[k]):
    print(f"  {b:10s} {tot[b]:9.1f} ms  {cnt[b]:7d} k")
print("  top kernels:")
for k, v in sorted(top.items(), key=lambda kv: -kv[1])[:24]:
    print(f"    {v:8.1f} ms  {k}")
