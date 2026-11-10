# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attribute main-stream gaps to recorded cross-stream event dependencies.

Nsight synchronization activity timestamps describe CPU API calls, not GPU
wait duration. Reconstruct a lower bound for event readiness from preceding
GPU work and recursively recorded dependencies on the source stream. Missing
recorded events remain unattributed. Queue/resource contention is not an
explicit dependency and must not be counted as waiting for cold data.

All joins are process-local. Memsets participate in main-stream coverage.
The interval assumes one profiled request; timings remain profiled evidence.
"""

import argparse
import bisect
import collections
import functools
import itertools
import json
import sqlite3
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("trace", type=Path)
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
c = sqlite3.connect(args.trace)
PID, MAIN = c.execute(
    "select globalPid,streamId from CUPTI_ACTIVITY_KIND_KERNEL where deviceId=? "
    "group by globalPid,streamId order by count(*) desc limit 1",
    (args.device,),
).fetchone()
names = dict(c.execute("select id,value from StringIds"))
api = {
    cid: (s, e, names[n])
    for s, e, tid, cid, n in c.execute(
        "select start,end,globalTid,correlationId,nameId "
        "from CUPTI_ACTIVITY_KIND_RUNTIME"
    )
    if tid >> 24 == PID >> 24
}
activity = collections.defaultdict(list)
for table in ["KERNEL", "MEMCPY", "MEMSET"]:
    extra = "shortName" if table == "KERNEL" else "'" + table + "'"
    for s, e, stream, cid, name in c.execute(
        f"select start,end,streamId,correlationId,{extra} "
        f"from CUPTI_ACTIVITY_KIND_{table} where globalPid=?",
        (PID,),
    ):
        if cid not in api:
            continue
        name = names[name] if table == "KERNEL" else name
        activity[stream].append((api[cid][0], api[cid][1], s, e, name))
events = {}
for stream, cid, eid, sid in c.execute(
    "select streamId,correlationId,eventId,eventSyncId "
    "from CUPTI_ACTIVITY_KIND_CUDA_EVENT where globalPid=?",
    (PID,),
):
    if cid in api:
        events[eid, sid] = (stream, api[cid][0], api[cid][1])
waits = collections.defaultdict(list)
for stream, cid, eid, sid in c.execute(
    "select streamId,correlationId,eventId,eventSyncId "
    "from CUPTI_ACTIVITY_KIND_SYNCHRONIZATION where globalPid=? and syncType=2",
    (PID,),
):
    if cid in api:
        waits[stream].append((api[cid][0], (eid, sid)))
for items in activity.values():
    items.sort()
for items in waits.values():
    items.sort()
activity_times = {s: [x[0] for x in items] for s, items in activity.items()}
wait_times = {s: [x[0] for x in items] for s, items in waits.items()}
prefix_end = {
    s: list(itertools.accumulate((x[3] for x in items), max))
    for s, items in activity.items()
}


@functools.lru_cache(None)
def ready(key):
    if key not in events:
        return 0, "unrecorded"
    stream, record_time, record_end = events[key]
    i = bisect.bisect_left(activity_times.get(stream, []), record_time)
    end = prefix_end[stream][i - 1] if i else 0
    source = f"stream{stream}"
    # A record after a cross-stream wait may have no new kernel of its own.
    j = bisect.bisect_left(wait_times.get(stream, []), record_time)
    if j:
        for _, predecessor in waits[stream][:j]:
            if predecessor == key:
                continue
            pe, ps = ready(predecessor)
            if pe > end:
                end, source = pe, ps
    return max(end, record_time), source


main = sorted(activity[MAIN], key=lambda x: x[2])
totals = collections.Counter()
by_source = collections.Counter()
by_next = collections.Counter()
largest = []
previous = main[0]
for current in main[1:]:
    launch, launch_end, s, e, name = current
    prev_end = previous[3]
    if s > prev_end:
        gap = s - prev_end
        totals["gap_ns"] += gap
        totals[
            "fully_queued_gap_ns"
            if launch_end <= prev_end
            else "not_fully_queued_gap_ns"
        ] += gap
        a = bisect.bisect_left(wait_times.get(MAIN, []), previous[0])
        b = bisect.bisect_left(wait_times.get(MAIN, []), launch)
        dependencies = [
            (ready(key), events.get(key), key) for _, key in waits[MAIN][a:b]
        ]
        latest = max(dependencies, default=((0, "none"), None, None))
        dep_end, source = latest[0]
        attributed = max(0, min(s, dep_end) - max(prev_end, launch_end))
        totals["event_dependency_ns"] += attributed
        by_source[source] += attributed
        by_next[name] += attributed
        if gap > 100000:
            largest.append(
                dict(
                    gap_ms=gap / 1e6,
                    dependency_ms=attributed / 1e6,
                    source=source,
                    next=name,
                    previous=previous[4],
                    start_ms=prev_end / 1e6,
                    queued=launch_end <= prev_end,
                    waits=dependencies,
                )
            )
    if e > previous[3]:
        previous = current
result = dict(
    main_span_ms=(main[-1][3] - main[0][2]) / 1e6,
    totals_ms={k.removesuffix("_ns"): v / 1e6 for k, v in totals.items()},
    event_source_ms={k: v / 1e6 for k, v in by_source.most_common()},
    next_kernel_event_ms={k: v / 1e6 for k, v in by_next.most_common(20)},
    largest_gaps=sorted(largest, key=lambda x: -x["gap_ms"])[:30],
)
with args.output.open("w") as f:
    json.dump(result, f, indent=2)
result["largest_gaps"] = result["largest_gaps"][:5]
print(json.dumps(result, indent=2))
