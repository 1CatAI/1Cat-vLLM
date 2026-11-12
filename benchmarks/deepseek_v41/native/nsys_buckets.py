# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bucket an nsys CUDA kernel trace (sqlite) into the V4.1 budget rows.

Usage: nsys_buckets.py trace.sqlite ROUNDS
Prints per-bucket ms/round, kernel count/round, busy and idle ms/round.
"""

import sqlite3
import sys
from collections import defaultdict

import regex as re

BUCKETS = [
    ("dense_fp8_q8", r"qpn8"),
    ("moe_gemm", r"skinny|moe_qpn"),
    (
        "moe_route",
        r"_plan_kernel|_swiglu_prob|_route_sum|route_topk|_route|topk_gate|sqrt_softplus",
    ),
    (
        "dense_fp8",
        r"fp8_gemv|fp8_reduce|_linear_split_k|_reduce_split_k|_linear\b|hgemv",
    ),
    ("attention", r"attend|sparse|attn|insert_latents|window|latent|softmax|qdq|rope"),
    ("index", r"index|select_compressed|topk|gather_index"),
    ("allreduce", r"[Aa]ll[Rr]educe|cross_device|custom_ar|nccl.*Reduce|push|sum2"),
    ("pp_send_recv", r"nccl.*(Send|Recv|SendRecv)|ncclDevKernel_SendRecv"),
    ("mhc", r"hc_|_hc|mhc|sinkhorn|collapse|expand|transition|mixes"),
    (
        "engram",
        r"engram|_wait|_publish|_copy_request|_consume|_can_publish|_decode_native",
    ),
    ("norm", r"rms|norm"),
    ("gemm_other", r"gemm|sgemm|hgemm|cutlass|volta|turing|ampere"),
]


def bucket(name):
    for b, pat in BUCKETS:
        if re.search(pat, name):
            return b
    return "other"


def main():
    db, rounds = sys.argv[1], float(sys.argv[2])
    con = sqlite3.connect(db)
    device = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    q = """SELECT k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k
           JOIN StringIds s ON k.demangledName = s.id WHERE k.deviceId = ? ORDER BY k.start"""
    rows = con.execute(q, (device,)).fetchall()
    span = (rows[-1][1] - rows[0][0]) / 1e6 if rows else 0
    print(f"device {device}: {len(rows)} kernels over {span:.1f} ms")
    # Steps: kernel bursts separated by > 2 ms idle; keep full decode steps.
    bursts, cur = [], [rows[0]]
    for r in rows[1:]:
        if r[0] - cur[-1][1] > 2e6:
            bursts.append(cur)
            cur = [r]
        else:
            cur.append(r)
    bursts.append(cur)
    sizes = sorted(len(b) for b in bursts)
    typical = sizes[len(sizes) // 2]
    steps = [b for b in bursts if abs(len(b) - typical) <= 0.1 * typical]
    rows = [r for b in steps for r in b]
    rounds = len(steps)
    print(
        f"{len(bursts)} bursts, {rounds} decode steps of ~{typical} kernels; span/step "
        f"{sum((b[-1][1] - b[0][0]) for b in steps) / 1e6 / max(rounds, 1):.2f} ms"
    )
    total = defaultdict(float)
    count = defaultdict(int)
    names = defaultdict(float)
    busy = 0.0
    last_end = None
    idle = 0.0
    for start, end, name in rows:
        d = (end - start) / 1e6
        b = bucket(name)
        total[b] += d
        count[b] += 1
        names[name[:70]] += d
        busy += d
        if last_end is not None and start > last_end:
            gap = (start - last_end) / 1e6
            if gap < 5.0:  # ignore inter-request gaps
                idle += gap
        last_end = max(last_end or 0, end)
    print(
        f"kernels/round {len(rows) / rounds:.0f}  busy {busy / rounds:.3f} ms  idle(<5ms gaps) {idle / rounds:.3f} ms"
    )
    for b in sorted(total, key=lambda k: -total[k]):
        print(
            f"  {b:14s} {total[b] / rounds:7.3f} ms  {count[b] / rounds:6.1f} kernels"
        )
    print("top kernels:")
    for n in sorted(names, key=lambda k: -names[k])[:15]:
        print(f"  {names[n] / rounds:7.3f} ms  {n}")


if __name__ == "__main__":
    main()
