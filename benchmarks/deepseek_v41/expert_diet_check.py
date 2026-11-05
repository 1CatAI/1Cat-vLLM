# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU check for VLLM_V41_EXPERT_DIET (one GPU, synthetic inputs).

Compares the EXPERT_DIET kernels (v41k_diet.cuh) against the EXPERT_FAST
ones they replace, in the same process:
  gate_up: gate_up_fast  vs gate_up_diet   (build_plan True and False)
  down:    down_rows_fast vs down_rows_diet(sa=0) and, where it fits
           (IQ1_M M <= 4, IQ2 M <= 2), down_rows_diet(sa=1)

Bit equality uses the inputs, poisoning and whole-buffer comparisons of
expert_fast_check.py (its helpers are imported): gtypes 16 / 17 / 29,
M = 1..6, its route mixes (pools of 6..384 experts, dropped routes, one
expert in every row, everything dropped), garbage-filled plan buffers, NaN
hd scales on dropped routes, both accumulation-class counts for down and a
random rank. Any difference, or a NaN in a live output, is a failure.

Timing: per kernel and M = 1..6, microseconds per layer inside a CUDA graph
cycling over 8 layers with disjoint expert slices (cold weights), as in
expert_fast_check.py; fast and diet are timed alternately twice and the best
of each kept. Weight GB/s as there.

Run with the server down, on one GPU, with VLLM_V41_EXPERT_DIET unset (the
script refuses otherwise, since gate_up_fast / down_rows_fast would then be
the diet kernels themselves):

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \\
      python benchmarks/deepseek_v41/expert_diet_check.py --lib /path/v41k.so

Needs a v41k.so built from this tree (gate_up_diet, down_rows_diet,
expert_diet). Exit status 1 on any difference.
"""

import argparse
import contextlib
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import expert_fast_check as efc  # noqa: E402

E, N, ROWS_LOCAL, TOP_K = efc.E, efc.N, efc.ROWS_LOCAL, efc.TOP_K
GU_ROWS, GU_BLOCKS, DN_BLOCKS = efc.GU_ROWS, efc.GU_BLOCKS, efc.DN_BLOCKS
BLOCK_BYTES, LAYERS = efc.BLOCK_BYTES, efc.LAYERS


def _library(path):
    path = efc._library(path)
    for op in ("gate_up_diet", "down_rows_diet", "expert_diet"):
        if not hasattr(torch.ops.v41k, op):
            sys.exit(f"{path} has no v41k.{op}; rebuild it from this tree")
    if int(torch.ops.v41k.expert_diet()) != 0:
        sys.exit("VLLM_V41_EXPERT_DIET is set: unset it (fast would be diet)")
    return path


def _sa_fits(gtype, m):
    return m <= (4 if gtype == 29 else 2)


class GuPair(efc.GuCase):
    """GuCase whose 'old' is gate_up_fast and 'new' is gate_up_diet."""

    def _run(self, op, buf, plan):
        op(
            self.xq,
            self.xd,
            self.hq(buf),
            self.hd(buf),
            self.ids,
            self.prob,
            self.gate,
            self.up,
            self.gtype,
            self.fbook,
            plan,
            not self.plan_fuse,
        )

    def old(self):
        self._run(torch.ops.v41k.gate_up_fast, self.buf_old, self.plan_old)

    def new(self):
        self._run(torch.ops.v41k.gate_up_diet, self.buf_new, self.plan_new)


class DnPair(efc.Case):
    """Case whose 'old' is down_rows_fast and 'new' is down_rows_diet(sa)."""

    sa = 0

    def _args(self, out, classes):
        return (
            out,
            self.gathered,
            self.stride,
            self.routes,
            self.down,
            self.gtype,
            self.rank * ROWS_LOCAL,
            self.fbook,
            classes,
            self.plan,
        )

    def old(self, classes):
        torch.ops.v41k.down_rows_fast(*self._args(self.out_old, classes))

    def new(self, classes):
        torch.ops.v41k.down_rows_diet(*self._args(self.out_new, classes), self.sa)


def _alternate(fn_a, fn_b, reps):
    """Best-of graph timings of two kernel lists, measured a, b, a, b."""
    ta = tb = float("inf")
    for _ in range(2):
        ta = min(ta, efc._graph_us(fn_a, reps))
        tb = min(tb, efc._graph_us(fn_b, reps))
    return ta, tb


def check_gate_up(gtype, args, g, dev, report):
    gate = efc._weights(gtype, GU_ROWS, GU_BLOCKS, g, dev)
    up = efc._weights(gtype, GU_ROWS, GU_BLOCKS, g, dev)
    fbook = torch.ops.v41k.fast_table(gtype, dev)
    row_bytes = GU_BLOCKS * BLOCK_BYTES[gtype]
    bad = 0
    res = {"mismatch": {}, "us": {}}
    for m in range(1, 7):
        per_m = {}
        for plan_fuse in (True, False):
            keys = ("hq", "hd", "past_hd", "plan", "nan_old", "nan_new")
            fail = dict.fromkeys(keys, 0)
            for trial in range(args.trials):
                ids = efc._ids(m, trial, g, dev)
                scale = 10 ** (-3 + 2 * float(torch.rand(1, device=dev, generator=g)))
                case = GuPair(m, gtype, gate, up, None, fbook, ids, scale, plan_fuse, g)
                case.poison(g)
                case.old()
                case.new()
                torch.cuda.synchronize()
                for k, v in efc._compare_gu(case).items():
                    fail[k] += v
            bad += sum(fail.values())
            per_m["plan_fuse" if plan_fuse else "build_plan"] = fail
        res["mismatch"][f"M{m}"] = per_m
        if args.no_time:
            continue
        slices = torch.randperm(E, device=dev, generator=g).view(LAYERS, -1)
        cases = []
        for layer in range(LAYERS):
            ids = efc._ids(m, 0, g, dev, pool_experts=slices[layer])
            cases.append(GuPair(m, gtype, gate, up, None, fbook, ids, 0.01, True, g))
        nu = sum(efc._nu(c.plan0) for c in cases) / LAYERS
        mb = nu * GU_ROWS * 2 * row_bytes / 1e6
        t_fast, t_diet = _alternate(
            [lambda c=c: c.old() for c in cases],
            [lambda c=c: c.new() for c in cases],
            args.reps,
        )
        res["us"][f"M{m}"] = {
            "experts_per_layer": round(nu, 2),
            "weight_MB": round(mb, 2),
            "gate_up_fast_us": round(t_fast, 2),
            "gate_up_diet_us": round(t_diet, 2),
            "gate_up_fast_GBps": round(mb * 1e3 / t_fast, 1),
            "gate_up_diet_GBps": round(mb * 1e3 / t_diet, 1),
            "saved_per_layer_us": round(t_fast - t_diet, 2),
            "speedup": round(t_fast / t_diet, 3),
        }
    report["gtype"].setdefault(str(gtype), {})["gate_up"] = res
    del gate, up
    torch.cuda.empty_cache()
    return bad


def check_down(gtype, args, g, dev, report):
    down = efc._weights(gtype, ROWS_LOCAL, DN_BLOCKS, g, dev)
    fbook = torch.ops.v41k.fast_table(gtype, dev)
    row_bytes = DN_BLOCKS * BLOCK_BYTES[gtype]
    bad = 0
    res = {"mismatch": {}, "us": {}}
    for m in range(1, 7):
        per_m = {}
        sas = (0, 1) if _sa_fits(gtype, m) else (0,)
        for sa in sas:
            for classes in (8, 4):
                fail = {"own_rows": 0, "other_rows": 0, "nan_old": 0, "nan_new": 0}
                for trial in range(args.trials):
                    ids = efc._ids(m, trial, g, dev)
                    x = torch.randn(m, N, device=dev, generator=g).half()
                    plan = efc._plan(m, ids, x, g, dev)
                    gathered = efc._gathered(m, ids, g, dev)
                    rank = int(torch.randint(0, 4, (1,), device=dev, generator=g))
                    case = DnPair(m, gtype, down, None, fbook, plan, gathered, rank)
                    case.sa = sa
                    case.out_old.fill_(float("nan"))
                    case.out_new.fill_(float("nan"))
                    case.old(classes)
                    case.new(classes)
                    torch.cuda.synchronize()
                    for k, v in efc._compare(case).items():
                        fail[k] += v
                bad += sum(fail.values())
                per_m[f"sa{sa}_classes{classes}"] = fail
        res["mismatch"][f"M{m}"] = per_m
        if args.no_time:
            continue
        slices = torch.randperm(E, device=dev, generator=g).view(LAYERS, -1)
        cases = []
        for layer in range(LAYERS):
            ids = efc._ids(m, 0, g, dev, pool_experts=slices[layer])
            x = torch.randn(m, N, device=dev, generator=g).half()
            plan = efc._plan(m, ids, x, g, dev)
            gathered = efc._gathered(m, ids, g, dev)
            cases.append(DnPair(m, gtype, down, None, fbook, plan, gathered, layer % 4))
        nu = sum(efc._nu(c.plan) for c in cases) / LAYERS
        mb = nu * ROWS_LOCAL * row_bytes / 1e6
        row = {"experts_per_layer": round(nu, 2), "weight_MB": round(mb, 2)}
        fast = [lambda c=c: c.old(8) for c in cases]
        for sa in sas:

            def diet(c, sa=sa):
                c.sa = sa
                c.new(8)

            t_fast, t_diet = _alternate(
                fast, [lambda c=c: diet(c) for c in cases], args.reps
            )
            row[f"down_rows_fast_us_vs_sa{sa}"] = round(t_fast, 2)
            row[f"down_rows_diet_sa{sa}_us"] = round(t_diet, 2)
            row[f"down_rows_diet_sa{sa}_GBps"] = round(mb * 1e3 / t_diet, 1)
            row[f"saved_per_layer_us_sa{sa}"] = round(t_fast - t_diet, 2)
            row[f"speedup_sa{sa}"] = round(t_fast / t_diet, 3)
        row["down_rows_fast_GBps"] = round(
            mb * 1e3 / row["down_rows_fast_us_vs_sa0"], 1
        )
        res["us"][f"M{m}"] = row
    report["gtype"].setdefault(str(gtype), {})["down"] = res
    del down
    torch.cuda.empty_cache()
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", default=os.environ.get("VLLM_V41K_LIB", ""))
    ap.add_argument("--gtypes", default="29,16,17")
    ap.add_argument("--roles", default="gate_up,down")
    ap.add_argument("--trials", type=int, default=48)
    ap.add_argument("--reps", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-time", action="store_true")
    args = ap.parse_args()
    lib = _library(args.lib)
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(args.seed)
    report = {
        "lib": os.path.basename(lib),
        "device": torch.cuda.get_device_name(dev),
        "trials_per_m_and_mode": args.trials,
        "gtype": {},
    }
    with contextlib.suppress(Exception):
        report["sm_clock_mhz"] = torch.cuda.clock_rate(dev)
    roles = args.roles.split(",")
    for role in roles:
        if role not in ("gate_up", "down"):
            sys.exit(f"unknown role {role!r} (gate_up, down)")
    bad = 0
    for gtype in (int(v) for v in args.gtypes.split(",")):
        if "gate_up" in roles:
            bad += check_gate_up(gtype, args, g, dev, report)
        if "down" in roles:
            bad += check_down(gtype, args, g, dev, report)
    report["failing_trials"] = bad
    report["ok"] = bad == 0
    print(json.dumps(report, indent=1))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
