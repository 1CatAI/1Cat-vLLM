# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize py-spy raw (collapsed) output: inclusive and self share per function."""

import collections
import sys

import regex as re

inc, own = collections.Counter(), collections.Counter()
total = 0
for line in open(sys.argv[1]):
    line = line.rstrip()
    if not line:
        continue
    stack, _, n = line.rpartition(" ")
    n = int(n)
    total += n
    frames = [
        re.sub(r" \(.*?/([^/]+\.py):\d+\)", r" (\1)", f) for f in stack.split(";")
    ]
    seen = set()
    for f in frames:
        if f not in seen:
            inc[f] += n
            seen.add(f)
    own[frames[-1]] += n
print("total samples", total)
print("-- self")
for f, n in own.most_common(int(sys.argv[2]) if len(sys.argv) > 2 else 15):
    print("%5.1f%% %s" % (100 * n / total, f[:150]))
print("-- inclusive")
for f, n in inc.most_common(int(sys.argv[3]) if len(sys.argv) > 3 else 30):
    print("%5.1f%% %s" % (100 * n / total, f[:150]))
