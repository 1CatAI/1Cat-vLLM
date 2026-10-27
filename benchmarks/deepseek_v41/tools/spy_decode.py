# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""py-spy raw: decode-only CPU (drop prefill stacks and GPU-wait leaves)."""

import collections
import sys

import regex as re

inc = collections.Counter()
total = 0
waits = 0
pre = 0
WAIT = (
    "parse_output",
    "synchronize",
    "_to_list",
    "item",
    "tolist",
    "acquire_read",
    "sched_yield",
    "wait",
)
for line in open(sys.argv[1]):
    stack, _, n = line.rstrip().rpartition(" ")
    if not stack:
        continue
    n = int(n)
    frames = [
        re.sub(r" \(.*?/([^/]+\.py):(\d+)\)", r" (\1:\2)", f) for f in stack.split(";")
    ]
    if any("prefill_moe" in f or "_ring_pump" in f for f in frames):
        pre += n
        continue
    leaf = frames[-1].split(" ")[0]
    if leaf in WAIT:
        waits += n
        continue
    total += n
    seen = set()
    for f in frames:
        k = re.sub(r":\d+\)", ")", f)
        if k not in seen:
            inc[k] += n
            seen.add(k)
print("decode cpu samples", total, "waits", waits, "prefill", pre)
for f, n in inc.most_common(45):
    print("%6d %s" % (n, f[:140]))
