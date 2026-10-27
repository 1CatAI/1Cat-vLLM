# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import sys
from collections import defaultdict

import regex as re

ev = json.load(open(sys.argv[1]))["traceEvents"]
kern = sorted(
    (e for e in ev if e.get("ph") == "X" and e.get("cat") == "kernel"),
    key=lambda e: e["ts"],
)
ts = [k["ts"] for k in kern]
import bisect

r = sorted(
    e["ts"]
    for e in ev
    if e.get("ph") == "X"
    and e.get("cat") == "user_annotation"
    and e["name"] == "V41::target_forward"
)
tot = defaultdict(float)
cnt = defaultdict(int)
n = 0
spans = []
for a, b in zip(r, r[1:]):
    seg = kern[bisect.bisect_left(ts, a) : bisect.bisect_left(ts, b)]
    if (
        not seg
        or any(re.search("transcode|prefill|_partition_tile", k["name"]) for k in seg)
        or b - a > 60000
    ):
        continue
    n += 1
    spans.append(b - a)
    for k in seg:
        nm = re.sub(r"<.*", "<>", re.sub(r"\(.*", "", k["name"]))[:90]
        tot[nm] += k["dur"]
        cnt[nm] += 1
spans.sort()
print(
    "decode rounds",
    n,
    "median span us",
    spans[len(spans) // 2] if spans else None,
    "busy ms",
    round(sum(tot.values()) / n / 1e3, 2),
    "kernels",
    sum(cnt.values()) / n,
)
for k, v in sorted(tot.items(), key=lambda kv: -kv[1])[
    : int(sys.argv[2]) if len(sys.argv) > 2 else 40
]:
    print(f"{v / n / 1e3:7.3f} ms {cnt[k] / n:7.1f}x  {k}")
