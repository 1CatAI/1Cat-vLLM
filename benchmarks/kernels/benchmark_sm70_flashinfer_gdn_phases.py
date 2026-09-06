# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Same-arithmetic GDN cooperative versus stream-ordered split phases.

Includes BA, conv, gating and recurrence. Excludes QKVZ/output projections.
The baseline library must be the frozen integration binary, not a new rebuild.
"""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median

import torch

from benchmarks.kernels.benchmark_sm70_flashinfer_gdn_conv import (
    capture,
    check_exclusive,
    load_weights,
)
from benchmarks.kernels.benchmark_sm70_flashinfer_mqa import measure
from benchmarks.kernels.flashinfer_sm70_gdn_conv import FusedGDN, build
from benchmarks.kernels.sm70_paired_stats import paired_latency_interval


def screen(rows, weights, steps, repeats, candidate="two_phase"):
    hidden, hq, hv, wqkv, ba, cw, A, dt = weights
    torch.manual_seed(20260906)
    x = torch.randn(rows, hidden, device="cuda", dtype=torch.float16)
    raw = (x @ wqkv.T).contiguous()
    pool = rows + 3
    c0 = torch.randn(pool, raw.shape[1], 3, device="cuda", dtype=torch.float16)
    s0 = torch.randn(pool, hv, 128, 128, device="cuda") * 0.01
    convs, states = [c0.clone() for _ in range(2)], [s0.clone() for _ in range(2)]
    indices = torch.arange(rows, device="cuda", dtype=torch.int32)
    packed = ba.T.contiguous()
    bias = x.new_empty(0)
    ops = [
        FusedGDN(rows, hq, hv, hidden=hidden, **{candidate: p}) for p in (False, True)
    ]
    calls = [
        lambda i=i: ops[i](
            x, packed, raw, cw, bias, convs[i], A, dt, states[i], indices
        )
        for i in range(2)
    ]
    graphs = [capture(call) for call in calls]
    for c, s in zip(convs, states):
        c.copy_(c0)
        s.copy_(s0)
    for step in range(steps):
        x.normal_().mul_((0.25, 1.0, 3.0)[step % 3])
        raw.copy_(x @ wqkv.T)
        indices.copy_(torch.randperm(pool, device="cuda")[:rows])
        if step % 8 == 7:
            indices[-1] = -1
        for op, graph in zip(ops, graphs):
            op.output.fill_(torch.nan)
            op.partial.fill_(torch.nan)
            graph.replay()
        torch.cuda.synchronize()
        for a, b in (
            (convs[0], convs[1]),
            (states[0], states[1]),
            (ops[0].output, ops[1].output),
            (ops[0].partial, ops[1].partial),
        ):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
            assert torch.isfinite(a).all()
    # History ends on a padding case: restore all live rows before timing.
    # Otherwise B1 measures only padding and B16 really measures 15 requests.
    indices.copy_(torch.arange(rows, device="cuda", dtype=torch.int32))
    times = [[], []]
    for cycle in range(5):
        for i in [0, 1] if cycle % 2 == 0 else [1, 0]:
            graphs[i].replay()
            times[i].append(measure(graphs[i].replay, repeats))
    for a, b in (
        (convs[0], convs[1]),
        (states[0], states[1]),
        (ops[0].output, ops[1].output),
    ):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        assert torch.isfinite(a).all()
    return dict(
        rows=rows,
        timed_live_rows=rows,
        independent_history_steps=steps,
        output_state_conv_partials_exact=True,
        samples_us=times,
        median_us=list(map(median, times)),
        reduction_pct=100 * (1 - median(times[1]) / median(times[0])),
        paired_interval=paired_latency_interval(*times),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--baseline-library", type=Path, required=True)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 4, 8, 16, 32, 64])
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument(
        "--candidate", choices=("two_phase", "shared_parameters"), default="two_phase"
    )
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or min(args.steps, args.repeats) < 1:
        parser.error("Require fresh output and positive steps/repeats")
    if not all(1 <= rows <= 64 for rows in args.rows):
        parser.error("Rows must be in 1..64")
    check_exclusive()
    torch.ops.load_library(str(args.baseline_library.resolve()))
    weights = load_weights(args.model)
    build(
        hidden=weights[0],
        q_heads=weights[1],
        v_heads=weights[2],
        **{args.candidate: True},
    )
    results = []
    for rows in args.rows:
        result = screen(rows, weights, args.steps, args.repeats, args.candidate)
        print(json.dumps(result), flush=True)
        results.append(result)
        check_exclusive()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            dict(
                candidate=args.candidate,
                scope=(
                    "native GDN BA+conv+recurrent only; not full model quality or speed"
                ),
                baseline_sha256=hashlib.sha256(
                    args.baseline_library.read_bytes()
                ).hexdigest(),
                results=results,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
