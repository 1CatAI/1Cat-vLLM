# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Critical-path breakdown of one GPU's main stream from an nsys sqlite export.

Usage: python nsys_breakdown.py trace.sqlite <deviceId>
Gaps whose next kernel was launched before the gap began count as GPU wait
(events/side streams); the rest as host gap. Segment names are Flash-Next specific.
"""

import collections
import sqlite3
import sys

c = sqlite3.connect(sys.argv[1])
dev = int(sys.argv[2])
names = dict(c.execute("select id, value from StringIds").fetchall())
api = dict(
    c.execute("select correlationId, end from CUPTI_ACTIVITY_KIND_RUNTIME").fetchall()
)
ks = c.execute(
    "select start,end,streamId,correlationId,demangledName,gridX,gridY "
    "from CUPTI_ACTIVITY_KIND_KERNEL where deviceId=?",
    (dev,),
).fetchall()
ms = c.execute(
    "select start,end,streamId,correlationId,NULL,0,0 "
    "from CUPTI_ACTIVITY_KIND_MEMCPY where deviceId=?",
    (dev,),
).fetchall()
st = collections.Counter(r[2] for r in ks)
main = st.most_common(1)[0][0]
m = sorted(r for r in ks + ms if r[2] == main)
phase = "gu"


def seg(n, gx, gy):
    global phase
    if n is None:
        return "memcpy"
    if "_sub_bounds" in n or "_sub_fill" in n:
        phase = "dn"
    if "_route_rows" in n:
        phase = "gu_next"
    if "unroute" in n:
        phase = "gu"
        return "MoE: unroute (top-k 加权合并)"
    if (
        "gemm_kernel" in n
        and "turbomind" in n.lower()
        or n.startswith("void turbomind")
    ):
        if phase == "dn":
            return "MoE: down GEMM"
        if phase == "gu_next":
            phase = "gu"
            return "MoE: gate|up GEMM"
        return "TurboMind dense GEMM(注意力/共享专家等投影)"
    if any(
        k in n
        for k in (
            "Sort",
            "searchsorted",
            "topkGating",
            "route_rows",
            "_locate",
            "_sub_bounds",
            "_sub_fill",
            "Scan",
            "fill_reverse",
        )
    ):
        return "MoE: 路由/排序"
    if any(k in n for k in ("moe_q8", "quantize_q8", "mtp")):
        return "MTP draft"
    if "partial_fp8_lite" in n or "rows_fp8" in n:
        return "QSA: 稀疏注意力 partial"
    if "merge" in n and "device_qsa" in n:
        return "QSA: merge"
    if any(
        k in n
        for k in (
            "device_qsa",
            "qsa",
            "relu_headsum",
            "expand_qsa",
            "_stage_pages",
            "_write",
            "zero_kv",
            "indexer",
        )
    ):
        return "QSA: indexer/选 token/写 KV/stage"
    if any(
        k in n
        for k in (
            "tilelang",
            "conv1d",
            "post_conv",
            "chunk_local",
            "kkt",
            "gdn",
            "rmsnorm_gated",
            "layer_norm_fwd",
        )
    ):
        return "GDN 线性注意力"
    if "combine_norm" in n:
        return "HC: combine+norm"
    if "AllGather" in n:
        return "通信: AllGather (HC)"
    if any(k in n for k in ("AllReduce", "cross_device")):
        return "通信: AllReduce"
    if "_hc_" in n or "hcx" in n:
        return "HC: 其他小 kernel"
    if "volta" in n or "cutlass" in n or "splitK" in n or "gemv" in n:
        return (
            "cuBLAS: HC down 投影 (N=88)"
            if (gx, gy) == (2, 64)
            else "cuBLAS: 其他 dense GEMM"
        )
    if "segment_native_restore" in n or "gguf" in n.lower():
        return "dense GGUF/共享专家"
    return "elementwise/其他小 kernel"


busy = collections.defaultdict(float)
n = collections.Counter()
wait = collections.defaultdict(float)
host = 0
end = m[0][0]
for s, e, _, cid, nm, gx, gy in m:
    name = names.get(nm) if nm is not None else None
    g = seg(name, gx, gy)
    if s > end:
        if api.get(cid) is not None and api[cid] <= end:
            wait[g] += s - end
        else:
            host += s - end
    if e > end:
        busy[g] += e - max(s, end)
        end = e
    n[g] += 1
span = (m[-1][1] - m[0][0]) / 1e6
print(
    f"GPU{dev} span {span:.0f} ms; host-gap {host / 1e6:.0f}; "
    f"gpu-wait {sum(wait.values()) / 1e6:.0f}"
)
for g in sorted(busy, key=lambda g: -busy[g]):
    print(
        f"{g:36s} {busy[g] / 1e6:7.0f} ms {100 * busy[g] / 1e6 / span:5.1f}%  "
        f"wait-before {wait[g] / 1e6:5.0f}  n={n[g]}"
    )
side = collections.defaultdict(float)
for s, e, sid, cid, nm, gx, gy in ks:
    if sid != main:
        side[(sid, (names.get(nm) or "")[:40])] += e - s
print(
    "side streams:",
    [(k, round(v / 1e6)) for k, v in sorted(side.items(), key=lambda x: -x[1])[:5]],
)
