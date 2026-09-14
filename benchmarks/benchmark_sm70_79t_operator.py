# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Operator gate for the historical (79T) SM70 D256 GQA architecture path.

* finite check
* CUDA-event latency + useful causal TFLOP/s (same flop convention as the
  historical stage logs: 4*Hq*D*(Q*(KV-Q) + Q*(Q+1)/2))
* FP32-accumulated reference on sampled query rows (relative L2 / max abs)
"""

import argparse
import json
import statistics

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--library", required=True)
ap.add_argument("--op", default="sm70_d256_gqa_architecture_fwd")
ap.add_argument("--q", type=int, default=8192)
ap.add_argument("--kv", type=int, nargs="+", default=[128000, 256000])
ap.add_argument("--warmup", type=int, default=20)
ap.add_argument("--samples", type=int, default=100)
ap.add_argument("--tag", default="")
args = ap.parse_args()

torch.ops.load_library(args.library)
op = getattr(torch.ops._vllm_fa2_C, args.op)
dense_op = torch.ops._vllm_fa2_C.sm70_d256_splitd_n32_dense_fwd
from vllm.v1.attention.backends.flash_attn_v100 import (  # noqa: E402
    _run_sm70_d256_gqa_79t_dispatch,
    _run_sm70_d256_gqa_79t_q8192_dispatch,
)

q8192_op = getattr(
    torch.ops._vllm_fa2_C,
    "sm70_d256_gqa_architecture_q8192_fwd",
    None,
)

SCALE = 0.0625
HQ, HKV, D = 6, 1, 256

row_candidates = {
    0,
    1,
    63,
    64,
    65,
    191,
    192,
    511,
    999,
    1999,
    3999,
    5999,
    7935,
    7936,
    7997,
    7998,
    7999,
    8000,
    8063,
    8127,
    args.q - 3,
    args.q - 2,
    args.q - 1,
}
rows = torch.tensor(
    sorted(row for row in row_candidates if 0 <= row < args.q),
    device="cuda",
)


def run(candidate_q, candidate_k, candidate_v, candidate_out):
    if int(candidate_q.shape[1]) > 8000 and q8192_op is not None:
        return _run_sm70_d256_gqa_79t_q8192_dispatch(
            candidate_q,
            candidate_k,
            candidate_v,
            candidate_out,
            softmax_scale=SCALE,
            architecture_q8192_op=q8192_op,
        )
    return _run_sm70_d256_gqa_79t_dispatch(
        candidate_q,
        candidate_k,
        candidate_v,
        candidate_out,
        softmax_scale=SCALE,
        architecture_op=op,
        dense_op=dense_op,
    )


selected_op = (
    "sm70_d256_gqa_architecture_q8192_fwd"
    if args.q > 8000 and q8192_op is not None
    else args.op
)
report = {"library": args.library, "op": selected_op, "cases": []}
for kv in args.kv:
    torch.manual_seed(173)
    q = torch.randn(1, args.q, HQ, D, device="cuda", dtype=torch.float16)
    k = torch.randn(1, kv, HKV, D, device="cuda", dtype=torch.float16)
    v = torch.randn(1, kv, HKV, D, device="cuda", dtype=torch.float16)
    out = torch.empty(1, args.q, HQ, D, device="cuda", dtype=torch.float16)

    run(q, k, v, out)
    torch.cuda.synchronize()
    finite = bool(torch.isfinite(out).all())

    # FP32-accumulated reference on sampled rows, attending all keys.
    idx = torch.arange(kv, device="cuda")
    limit = kv - args.q + rows  # inclusive last visible key
    ref = torch.empty(len(rows), HQ, D, device="cuda", dtype=torch.float32)
    qs = q[0, rows].permute(1, 0, 2).float()  # (HQ, R, D)
    kf = k[0, :, 0].float()
    vf = v[0, :, 0].float()
    for r in range(len(rows)):
        s = qs[:, r] @ kf.T * SCALE  # (HQ, kv)
        s = s.masked_fill(idx[None, :] > limit[r], float("-inf"))
        p = torch.softmax(s, dim=-1)
        ref[r] = p @ vf
    got = out[0, rows].float()

    delta = got - ref
    rel_l2 = float(delta.norm() / ref.norm())
    max_abs = float(delta.abs().max())
    row_rel = float((delta.norm(dim=-1) / ref.norm(dim=-1).clamp_min(1e-30)).max())

    if not finite:
        report["cases"].append({"kv": kv, "finite": False})
        print(json.dumps({"kv": kv, "finite": False}), flush=True)
        del q, k, v, out
        torch.cuda.empty_cache()
        continue

    for _ in range(args.warmup):
        run(q, k, v, out)
    torch.cuda.synchronize()
    times = []
    for _ in range(args.samples):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        run(q, k, v, out)
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b))
    med = statistics.median(times)
    ordered = sorted(times)
    flops = 4 * HQ * D * (args.q * (kv - args.q) + args.q * (args.q + 1) / 2)
    case = {
        "kv": kv,
        "finite": True,
        "median_ms": round(med, 4),
        "p10_ms": round(ordered[len(ordered) // 10], 4),
        "p90_ms": round(ordered[9 * len(ordered) // 10], 4),
        "useful_causal_tflops": round(flops / (med * 1e9), 4),
        "relative_l2": rel_l2,
        "max_abs": max_abs,
        "worst_row_relative_l2": row_rel,
    }
    report["cases"].append(case)
    print(json.dumps(case), flush=True)
    del q, k, v, out
    torch.cuda.empty_cache()

print(json.dumps({"summary": report}))
