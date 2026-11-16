# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU check for VLLM_V41_PLAN_FUSE (one GPU, synthetic inputs).

Today:    q8_aligned  -> gate_up7  (gate_up7 launches moe_plan, then gate/up)
Fused:    q8_plan     -> gate_up7p (q8_plan builds the plan in grid row 0)

1. Bit equality, M = 1..6 (6 routes per row): the whole 512-int plan buffer
   (both prefilled with the same garbage, so untouched entries count too),
   xq/xd, and the gate/up outputs hq/hd. Route mixes: random experts from
   pools of 6..384 (many repeats across rows), dropped routes (id -1, as the
   expert prune or a truncated verify position writes them), one expert in
   every row, and all routes dropped.
2. Time per layer inside a CUDA graph (``--reps`` layers per replay) for
   q8 + gate/up both ways, and the q8 launch alone both ways.

Synthetic data only. Needs a v41k.so built from this tree (q8_plan and
gate_up7p); pass it with --lib or VLLM_V41K_LIB. Exit status 1 on any
difference.

    python benchmarks/deepseek_v41/plan_fuse_check.py --lib /path/v41k.so
"""

import argparse
import json
import os
import sys

import torch

E, ROWS, K_DIM, TOP_K = 384, 576, 5120, 6
# bytes per 256-weight block: IQ2_XXS 66, IQ2_XS 74, IQ1_M 56
BLOCK_BYTES = {16: 66, 17: 74, 29: 56}


def _library(path):
    if not path:
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, "v41k", "build", "v41k.so")
    torch.ops.load_library(path)
    for op in ("q8_aligned", "gate_up7", "q8_plan", "gate_up7p", "table"):
        if not hasattr(torch.ops.v41k, op):
            sys.exit(f"{path} has no v41k.{op}; rebuild it from this tree")
    return path


def _ids(m, trial, g, dev):
    routes = m * TOP_K
    kind = trial % 6
    if kind == 4:  # one expert in every row
        ids = torch.arange(TOP_K, device=dev).repeat(m) + 7
    elif kind == 5 and trial % 12 == 11:  # everything dropped
        return torch.full((routes,), -1, dtype=torch.int32, device=dev)
    else:
        pool = (6, 8, 12, 24, 96, E)[(trial // 6) % 6]
        perm = torch.randperm(E, device=dev, generator=g)[:pool]
        ids = torch.stack(
            [
                perm[torch.randperm(pool, device=dev, generator=g)[:TOP_K]]
                for _ in range(m)
            ]
        ).flatten()
    ids = ids.to(torch.int32)
    if kind in (1, 3, 5):  # drop the lowest-weight routes of some rows
        view = ids.view(m, TOP_K)
        for r in range(m):
            keep = int(torch.randint(1, TOP_K + 1, (1,), device=dev, generator=g))
            view[r, keep:] = -1
    return ids.contiguous()


def _weights(gtype, g, dev):
    stride = K_DIM // 256 * BLOCK_BYTES[gtype]

    # Bytes 0..59: every fp16 scale these bytes can form is finite and below 1.
    def one():
        return torch.randint(
            0, 60, (E, ROWS, stride), dtype=torch.uint8, device=dev, generator=g
        )

    return one(), one()


class Layer:
    def __init__(self, m, ids, x, prob, gate, up, gtype, book, garbage):
        dev = x.device
        routes = m * TOP_K
        self.args = (ids, prob, gate, up, gtype, book)
        self.x = x
        self.xq = torch.zeros((m, 160, 32), dtype=torch.int8, device=dev)
        self.xd = torch.zeros((m, 160), dtype=torch.float32, device=dev)
        self.hq = torch.zeros((routes, 18, 32), dtype=torch.int8, device=dev)
        self.hd = torch.zeros((routes, 18), dtype=torch.float32, device=dev)
        self.plan = garbage.clone()

    def today(self):
        ids, prob, gate, up, gtype, book = self.args
        torch.ops.v41k.q8_aligned(self.xq, self.xd, self.x)
        torch.ops.v41k.gate_up7(
            self.xq,
            self.xd,
            self.hq,
            self.hd,
            ids,
            prob,
            gate,
            up,
            gtype,
            book,
            self.plan,
        )

    def fused(self):
        ids, prob, gate, up, gtype, book = self.args
        torch.ops.v41k.q8_plan(self.xq, self.xd, self.x, ids, self.plan)
        torch.ops.v41k.gate_up7p(
            self.xq,
            self.xd,
            self.hq,
            self.hd,
            ids,
            prob,
            gate,
            up,
            gtype,
            book,
            self.plan,
        )

    def q8_today(self):
        torch.ops.v41k.q8_aligned(self.xq, self.xd, self.x)

    def q8_fused(self):
        ids = self.args[0]
        torch.ops.v41k.q8_plan(self.xq, self.xd, self.x, ids, self.plan)


def _same(a, b):
    if a.dtype == torch.float32:
        a, b = a.view(torch.int32), b.view(torch.int32)
    return torch.equal(a, b)


def _graph_us(fn, reps):
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(reps):
            fn()
    graph.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    best = float("inf")
    for _ in range(5):
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        best = min(best, start.elapsed_time(end) * 1000.0 / reps)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", default=os.environ.get("VLLM_V41K_LIB", ""))
    ap.add_argument("--gtype", type=int, default=16, choices=sorted(BLOCK_BYTES))
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-time", action="store_true", help="bit checks only")
    args = ap.parse_args()
    lib = _library(args.lib)
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(args.seed)
    gate, up = _weights(args.gtype, g, dev)
    book = torch.ops.v41k.table(args.gtype, dev)
    report = {"lib": os.path.basename(lib), "gtype": args.gtype, "mismatch": {}}
    report["us"] = {}
    bad = 0
    for m in range(1, 7):
        miss = {"plan": 0, "xq_xd": 0, "hq_hd": 0}
        finite = 0
        for trial in range(args.trials):
            ids = _ids(m, trial, g, dev)
            x = torch.randn(m, K_DIM, device=dev, generator=g).half()
            prob = torch.rand(m, TOP_K, device=dev, generator=g)
            garbage = torch.randint(
                -(2**31),
                2**31 - 1,
                (512,),
                dtype=torch.int32,
                device=dev,
                generator=g,
            )
            a = Layer(m, ids, x, prob, gate, up, args.gtype, book, garbage)
            b = Layer(m, ids, x, prob, gate, up, args.gtype, book, garbage)
            a.today()
            b.fused()
            miss["plan"] += int(not _same(a.plan, b.plan))
            miss["xq_xd"] += int(not (_same(a.xq, b.xq) and _same(a.xd, b.xd)))
            miss["hq_hd"] += int(not (_same(a.hq, b.hq) and _same(a.hd, b.hd)))
            finite += int(torch.isfinite(a.hd).all())
        bad += sum(miss.values())
        # Synthetic weights keep every fp16 scale finite, so a non-finite hd is a
        # kernel fault; gate it separately (it is a count of good trials).
        bad += int(finite != args.trials)
        report["mismatch"][f"M{m}"] = dict(miss, hd_all_finite_trials=finite)
        if args.no_time:
            continue
        ids = _ids(m, 0, g, dev)
        x = torch.randn(m, K_DIM, device=dev, generator=g).half()
        prob = torch.rand(m, TOP_K, device=dev, generator=g)
        lay = Layer(
            m,
            ids,
            x,
            prob,
            gate,
            up,
            args.gtype,
            book,
            torch.zeros(512, dtype=torch.int32, device=dev),
        )
        us = {
            "q8_aligned+gate_up7": _graph_us(lay.today, args.reps),
            "q8_plan+gate_up7p": _graph_us(lay.fused, args.reps),
            "q8_aligned": _graph_us(lay.q8_today, args.reps),
            "q8_plan": _graph_us(lay.q8_fused, args.reps),
        }
        us["saved_per_layer"] = us["q8_aligned+gate_up7"] - us["q8_plan+gate_up7p"]
        report["us"][f"M{m}"] = {k: round(v, 2) for k, v in us.items()}
    report["trials_per_m"] = args.trials
    report["ok"] = bad == 0
    print(json.dumps(report, indent=1))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
