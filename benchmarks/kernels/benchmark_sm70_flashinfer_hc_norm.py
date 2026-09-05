# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Complete HC combine + Gemma-norm micro, not projection or model timing."""

import argparse
import json
import statistics

import torch

from benchmarks.kernels.benchmark_sm70_flashinfer_gdn_conv import (
    capture,
    check_exclusive,
    error,
)
from benchmarks.kernels.flashinfer_sm70_hc_norm import HCNorm, build


@torch.inference_mode()
def screen(rows, dtype, args):
    from vllm.models.qwen4_exp.nvidia.ops.hc import hc_combine_norm

    r = torch.randn(rows, 4 * 2560, device="cuda", dtype=dtype)
    b = torch.randn(rows, 2560, device="cuda", dtype=torch.float16)
    inj = torch.randn(rows, 4, device="cuda", dtype=torch.float16)
    weight = torch.randn(4 * 2560, device="cuda", dtype=torch.float16) * 0.05
    calls = [lambda: hc_combine_norm(r, b, inj, weight, 1e-6, 4)]
    candidates = [HCNorm(r, warps) for warps in (1, 2, 4, 8)]
    calls += [lambda candidate=c: candidate(r, b, inj, weight) for c in candidates]
    graphs = [capture(fn, args.calls) for fn in calls]
    checks = []
    for cycle, scale in enumerate((0.25, 1.0, 3.0, 1.0)):
        r.normal_().mul_(scale)
        b.normal_().mul_(scale)
        inj.normal_()
        expected = calls[0]()
        for index, candidate in enumerate(candidates):
            eager = [x.clone() for x in calls[index + 1]()]
            candidate.combined.fill_(float("nan"))
            candidate.normalized.fill_(float("nan"))
            graphs[index + 1].replay()
            torch.cuda.synchronize()
            for x, y in zip((candidate.combined, candidate.normalized), eager):
                torch.testing.assert_close(x, y, atol=0, rtol=0)
            diffs = [error(x, y) for x, y in zip(eager, expected)]
            checks.append({"cycle": cycle, "warps": candidate.warps, "errors": diffs})
    gate = all(
        d["finite"] and d["relative_l2"] < 1e-3 for c in checks for d in c["errors"]
    )
    result = {
        "rows": rows,
        "residual_dtype": str(dtype),
        "operator_gate": gate,
        "checks": checks,
    }
    if gate:
        for _ in range(20):
            for graph in graphs:
                graph.replay()
        torch.cuda.synchronize()
        samples = [[] for _ in calls]
        for repeat in range(args.samples):
            order = (
                range(len(calls)) if repeat % 2 == 0 else reversed(range(len(calls)))
            )
            for index in order:
                s, e = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                s.record()
                graphs[index].replay()
                e.record()
                e.synchronize()
                samples[index].append(s.elapsed_time(e) * 1000 / args.calls)
        check_exclusive()
        result.update(
            median_us=[statistics.median(s) for s in samples], samples_us=samples
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", nargs="+", type=int, default=[1, 4, 8, 16])
    parser.add_argument("--calls", type=int, default=100)
    parser.add_argument("--samples", type=int, default=9)
    args = parser.parse_args()
    check_exclusive()
    torch.manual_seed(20260906)
    build()
    for dtype in (torch.float16, torch.float32):
        for rows in args.rows:
            print(json.dumps(screen(rows, dtype, args)), flush=True)


if __name__ == "__main__":
    main()
