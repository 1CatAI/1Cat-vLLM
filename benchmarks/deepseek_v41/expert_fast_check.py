# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU check for VLLM_V41_EXPERT_FAST (one GPU, synthetic inputs).

Role down:
  Today:  down_rows(rows=4, dn_warps=8 or 4)     (VLLM_V41_ROWS_TILE '4,8')
  Fast:   down_rows_fast(classes=8 or 4)         (fbook = fast_table(gtype))
Role gate_up:
  Today:  q8_aligned + gate_up7 (moe_plan inside), or q8_plan + gate_up7p
          (VLLM_V41_PLAN_FUSE=1, the deployed stack)
  Fast:   gate_up_fast(build_plan=True / False)  (fbook = fast_table(gtype))

Role gate_up: bit equality per gtype, M = 1..6, both plan modes, the route
mixes below, an activation scale drawn per trial from 1e-3..1e-1 (so the clamp
at 10 is hit in some trials and not in others). The hq / hd buffer is prefilled
with the same random bytes for both kernels and hd with NaN; the whole buffer
(hq bytes, hd bits, the bytes past hd) and the whole 512-entry plan buffer are
compared, and a NaN left in hd of a live route is a failure of its own. Timing
as below for the deployed pair (gate_up7p vs gate_up_fast(build_plan=False)
after q8_plan), GB/s = experts x 576 rows x 2 matrices x row bytes / time.
Weight tensors in both roles have 64 bytes of slack after the last row (the
old and new IQ2 loaders read the aligned word that ends a row).

Role down:
1. Bit equality, per down gtype (16 IQ2_XXS, 17 IQ2_XS, 29 IQ1_M), M = 1..6
   tokens x top-6 routes, both accumulation-class counts (8 = the deployed
   dn_warps, 4 = any other value), a random rank (row_base) per trial. Each
   kernel writes a fresh NaN-poisoned [M, 5120] half output; the whole output
   is compared as int16 (own rows and the other ranks' zeroed rows reported
   separately), and a NaN left anywhere is a failure of its own. Route mixes as
   plan_fuse_check: random experts from pools of 6..384 (repeats across rows),
   dropped routes (id -1) in some rows, one expert in every row (6 members per
   expert) and all routes dropped. The plan buffer is prefilled with garbage,
   so stale entries past nu / cnt are live in both kernels, and the gathered
   hd scales of dropped routes are NaN in every rank slot, so reading one shows
   up as a mismatch. Weights are random bytes over the whole codebook range;
   only the bytes that form the fp16 block scale are bounded (finite, < 4).
2. Time per layer inside a CUDA graph, old vs fast, with cold weights: the
   graph cycles over 8 layers whose experts come from disjoint slices of the
   384, so consecutive launches share no weight rows (V100 L2 is 6 MB). Weight
   GB/s = experts touched x 1280 rows x row bytes / time.

Synthetic data only. Needs a v41k.so built from this tree (fast_table,
gate_up_fast and down_rows_fast); pass it with --lib or VLLM_V41K_LIB. Exit
status 1 on any difference. ``--no-time --trials 6`` is small enough for
compute-sanitizer (with PYTORCH_NO_CUDA_MEMORY_CACHING=1 every tensor is its
own allocation, so memcheck sees exact bounds).

    python benchmarks/deepseek_v41/expert_fast_check.py --lib /path/v41k.so
    python benchmarks/deepseek_v41/expert_fast_check.py --roles gate_up \\
        --gtypes 29
"""

import argparse
import contextlib
import json
import os
import sys

import torch

E, N, ROWS_LOCAL, K_DIM, TOP_K = 384, 5120, 1280, 2304, 6
GU_ROWS, GU_BLOCKS, DN_BLOCKS = 576, 20, 9  # gate/up rows per rank, K blocks
# bytes per 256-weight block: IQ2_XXS 66, IQ2_XS 74, IQ1_M 56
BLOCK_BYTES = {16: 66, 17: 74, 29: 56}
LAYERS = 8  # timing: layers cycled inside one graph, disjoint expert slices


def _library(path):
    if not path:
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, "v41k", "build", "v41k.so")
    torch.ops.load_library(path)
    for op in (
        "q8_aligned",
        "q8_plan",
        "table",
        "gate_up7",
        "gate_up7p",
        "down_rows",
        "fast_table",
        "gate_up_fast",
        "down_rows_fast",
    ):
        if not hasattr(torch.ops.v41k, op):
            sys.exit(f"{path} has no v41k.{op}; rebuild it from this tree")
    return path


def _stride(routes):
    return (routes * 648 + 255) // 256 * 256


def _ids(m, trial, g, dev, pool_experts=None):
    """Flat [m * 6] int32 expert ids, the route mix chosen by trial."""
    routes = m * TOP_K
    kind = trial % 6
    if pool_experts is not None:  # timing: distinct per row from a slice
        n = pool_experts.numel()
        ids = torch.stack(
            [
                pool_experts[torch.randperm(n, device=dev, generator=g)[:TOP_K]]
                for _ in range(m)
            ]
        ).flatten()
        return ids.to(torch.int32).contiguous()
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


def _weights(gtype, rows, blocks, g, dev):
    """[E, rows, blocks] uint8 blocks, random except the fp16 scale bytes,
    a contiguous view with 64 bytes of slack after the last row."""
    bb = BLOCK_BYTES[gtype]
    n = E * rows * blocks * bb
    flat = torch.randint(0, 256, (n + 64,), dtype=torch.uint8, device=dev, generator=g)
    w = flat[:n].view(E, rows, blocks, bb)
    if gtype in (16, 17):  # d = fp16 bytes 0..1; high byte < 60 -> |d| < 1
        w[..., 1] = w[..., 1] % 60
    else:  # IQ1_M d = top nibbles of the 4 scale words; byte 55 < 64 -> < 4
        w[..., 55] = w[..., 55] % 64
    return w.view(E, rows, blocks * bb)


def _garbage_plan(g, dev):
    return torch.randint(
        -(2**31), 2**31 - 1, (512,), dtype=torch.int32, device=dev, generator=g
    )


def _plan(m, ids, x, g, dev):
    """The serving plan for ids (q8_plan), over a garbage-filled buffer."""
    plan = _garbage_plan(g, dev)
    xq = torch.empty((m, 160, 32), dtype=torch.int8, device=dev)
    xd = torch.empty((m, 160), dtype=torch.float32, device=dev)
    torch.ops.v41k.q8_plan(xq, xd, x, ids, plan)
    return plan


def _gathered(m, ids, g, dev):
    """4 rank slots of [routes, 18, 32] int8 hq + [routes, 18] fp32 hd."""
    routes = m * TOP_K
    stride = _stride(routes)
    buf = torch.randint(0, 256, (4, stride), dtype=torch.uint8, device=dev, generator=g)
    hq = torch.randint(
        -127, 128, (4, routes * 576), dtype=torch.int8, device=dev, generator=g
    )
    hd = torch.rand((4, routes, 18), device=dev, generator=g) * 1e-3 + 1e-5
    hd[:, ids.long() < 0, :] = float("nan")  # dropped routes are never read
    buf[:, : routes * 576] = hq.view(torch.uint8)
    buf[:, routes * 576 : routes * 648] = hd.view(4, -1).view(torch.uint8)
    return buf.flatten().contiguous()


class Case:
    def __init__(self, m, gtype, down, dbook, fbook, plan, gathered, rank):
        self.m, self.gtype, self.rank = m, gtype, rank
        self.routes = m * TOP_K
        self.stride = _stride(self.routes)
        self.down, self.dbook, self.fbook = down, dbook, fbook
        self.plan, self.gathered = plan, gathered
        dev = down.device
        self.out_old = torch.empty((m, N), dtype=torch.float16, device=dev)
        self.out_new = torch.empty((m, N), dtype=torch.float16, device=dev)

    def old(self, warps):
        torch.ops.v41k.down_rows(
            self.out_old,
            self.gathered,
            self.stride,
            self.routes,
            self.down,
            self.gtype,
            self.rank * ROWS_LOCAL,
            self.dbook,
            4,
            warps,
            self.plan,
        )

    def new(self, classes):
        torch.ops.v41k.down_rows_fast(
            self.out_new,
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


def _compare(case):
    """Failure counts (0/1 each) for one trial: own rows, other rows, NaN."""
    a = case.out_old.view(torch.int16)
    b = case.out_new.view(torch.int16)
    own = slice(case.rank * ROWS_LOCAL, (case.rank + 1) * ROWS_LOCAL)
    mask = torch.ones(N, dtype=torch.bool, device=a.device)
    mask[own] = False
    return {
        "own_rows": int(not torch.equal(a[:, own], b[:, own])),
        "other_rows": int(not torch.equal(a[:, mask], b[:, mask])),
        "nan_old": int(bool(torch.isnan(case.out_old).any())),
        "nan_new": int(bool(torch.isnan(case.out_new).any())),
    }


def _graph_us(fns, reps):
    for fn in fns:
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for i in range(reps):
            fns[i % len(fns)]()
    graph.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    best = float("inf")
    for _ in range(7):
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        best = min(best, start.elapsed_time(end) * 1000.0 / reps)
    return best


def _nu(plan):
    return int(plan[0].item())


class GuCase:
    """One gate/up layer: inputs shared by both kernels, separate outputs.

    plan_fuse: q8_plan builds the plan before gate/up (old gate_up7p, new
    build_plan=False); otherwise q8_aligned, and gate_up7 / gate_up_fast
    (build_plan=True) rebuild it over the garbage-filled buffer.
    """

    def __init__(self, m, gtype, gate, up, obook, fbook, ids, scale, plan_fuse, g):
        dev = gate.device
        self.m, self.gtype, self.ids, self.plan_fuse = m, gtype, ids, plan_fuse
        self.gate, self.up, self.obook, self.fbook = gate, up, obook, fbook
        self.routes = m * TOP_K
        self.stride = _stride(self.routes)
        x = (torch.randn(m, N, device=dev, generator=g) * scale).half()
        self.xq = torch.empty((m, 160, 32), dtype=torch.int8, device=dev)
        self.xd = torch.empty((m, 160), dtype=torch.float32, device=dev)
        self.plan0 = _garbage_plan(g, dev)
        if plan_fuse:
            torch.ops.v41k.q8_plan(self.xq, self.xd, x, ids, self.plan0)
        else:
            torch.ops.v41k.q8_aligned(self.xq, self.xd, x)
        self.prob = torch.rand(self.routes, device=dev, generator=g)
        self.plan_old, self.plan_new = self.plan0.clone(), self.plan0.clone()
        self.buf_old = torch.empty(self.stride, dtype=torch.uint8, device=dev)
        self.buf_new = torch.empty(self.stride, dtype=torch.uint8, device=dev)

    def hq(self, buf):
        return buf[: self.routes * 576].view(torch.int8).view(self.routes, 18, 32)

    def hd(self, buf):
        r = self.routes
        return buf[r * 576 : r * 648].view(torch.float32).view(r, 18)

    def poison(self, g):
        junk = torch.randint(
            0,
            256,
            (self.stride,),
            dtype=torch.uint8,
            device=self.gate.device,
            generator=g,
        )
        for buf in (self.buf_old, self.buf_new):
            buf.copy_(junk)
            self.hd(buf).fill_(float("nan"))
        self.plan_old.copy_(self.plan0)
        self.plan_new.copy_(self.plan0)

    def old(self):
        op = torch.ops.v41k.gate_up7p if self.plan_fuse else torch.ops.v41k.gate_up7
        op(
            self.xq,
            self.xd,
            self.hq(self.buf_old),
            self.hd(self.buf_old),
            self.ids,
            self.prob,
            self.gate,
            self.up,
            self.gtype,
            self.obook,
            self.plan_old,
        )

    def new(self):
        torch.ops.v41k.gate_up_fast(
            self.xq,
            self.xd,
            self.hq(self.buf_new),
            self.hd(self.buf_new),
            self.ids,
            self.prob,
            self.gate,
            self.up,
            self.gtype,
            self.fbook,
            self.plan_new,
            not self.plan_fuse,
        )


def _compare_gu(c):
    """Failure counts (0/1 each) for one gate/up trial."""
    r = c.routes
    live = c.ids >= 0
    hd_old, hd_new = c.hd(c.buf_old), c.hd(c.buf_new)
    return {
        "hq": int(not torch.equal(c.hq(c.buf_old), c.hq(c.buf_new))),
        "hd": int(not torch.equal(hd_old.view(torch.int32), hd_new.view(torch.int32))),
        "past_hd": int(not torch.equal(c.buf_old[r * 648 :], c.buf_new[r * 648 :])),
        "plan": int(not torch.equal(c.plan_old, c.plan_new)),
        "nan_old": int(bool(torch.isnan(hd_old[live]).any())),
        "nan_new": int(bool(torch.isnan(hd_new[live]).any())),
    }


def check_gate_up(gtype, args, g, dev, report):
    gate = _weights(gtype, GU_ROWS, GU_BLOCKS, g, dev)
    up = _weights(gtype, GU_ROWS, GU_BLOCKS, g, dev)
    obook = torch.ops.v41k.table(gtype, dev)
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
                ids = _ids(m, trial, g, dev)
                scale = 10 ** (-3 + 2 * float(torch.rand(1, device=dev, generator=g)))
                case = GuCase(
                    m, gtype, gate, up, obook, fbook, ids, scale, plan_fuse, g
                )
                case.poison(g)
                case.old()
                case.new()
                torch.cuda.synchronize()
                for k, v in _compare_gu(case).items():
                    fail[k] += v
            # counts of failing trials only; good trials are never added
            bad += sum(fail.values())
            per_m["gate_up7p" if plan_fuse else "gate_up7"] = fail
        res["mismatch"][f"M{m}"] = per_m
        if args.no_time:
            continue
        slices = torch.randperm(E, device=dev, generator=g).view(LAYERS, -1)
        cases = []
        for layer in range(LAYERS):
            ids = _ids(m, 0, g, dev, pool_experts=slices[layer])
            cases.append(GuCase(m, gtype, gate, up, obook, fbook, ids, 0.01, True, g))
        nu = sum(_nu(c.plan0) for c in cases) / LAYERS
        mb = nu * GU_ROWS * 2 * row_bytes / 1e6
        t_old = _graph_us([lambda c=c: c.old() for c in cases], args.reps)
        t_new = _graph_us([lambda c=c: c.new() for c in cases], args.reps)
        res["us"][f"M{m}"] = {
            "experts_per_layer": round(nu, 2),
            "weight_MB": round(mb, 2),
            "gate_up7p_us": round(t_old, 2),
            "gate_up_fast_us": round(t_new, 2),
            "gate_up7p_GBps": round(mb * 1e3 / t_old, 1),
            "gate_up_fast_GBps": round(mb * 1e3 / t_new, 1),
            "saved_per_layer_us": round(t_old - t_new, 2),
            "speedup": round(t_old / t_new, 3),
        }
    report["gtype"].setdefault(str(gtype), {})["gate_up"] = res
    del gate, up
    torch.cuda.empty_cache()
    return bad


def check_down(gtype, args, g, dev, report):
    down = _weights(gtype, ROWS_LOCAL, DN_BLOCKS, g, dev)
    dbook = torch.ops.v41k.table(gtype, dev)
    fbook = torch.ops.v41k.fast_table(gtype, dev)
    row_bytes = DN_BLOCKS * BLOCK_BYTES[gtype]
    bad = 0
    res = {"mismatch": {}, "us": {}}
    for m in range(1, 7):
        per_m = {}
        for classes in (8, 4):
            fail = {"own_rows": 0, "other_rows": 0, "nan_old": 0, "nan_new": 0}
            for trial in range(args.trials):
                ids = _ids(m, trial, g, dev)
                x = torch.randn(m, N, device=dev, generator=g).half()
                plan = _plan(m, ids, x, g, dev)
                gathered = _gathered(m, ids, g, dev)
                rank = int(torch.randint(0, 4, (1,), device=dev, generator=g))
                case = Case(m, gtype, down, dbook, fbook, plan, gathered, rank)
                case.out_old.fill_(float("nan"))
                case.out_new.fill_(float("nan"))
                case.old(classes)
                case.new(classes)
                torch.cuda.synchronize()
                for k, v in _compare(case).items():
                    fail[k] += v
            # every count is a count of failing trials; good trials are never
            # added to the mismatch total
            bad += sum(fail.values())
            per_m[f"classes{classes}"] = fail
        res["mismatch"][f"M{m}"] = per_m
        if args.no_time:
            continue
        slices = torch.randperm(E, device=dev, generator=g).view(LAYERS, -1)
        cases = []
        for layer in range(LAYERS):
            ids = _ids(m, 0, g, dev, pool_experts=slices[layer])
            x = torch.randn(m, N, device=dev, generator=g).half()
            plan = _plan(m, ids, x, g, dev)
            gathered = _gathered(m, ids, g, dev)
            cases.append(Case(m, gtype, down, dbook, fbook, plan, gathered, layer % 4))
        nu = sum(_nu(c.plan) for c in cases) / LAYERS
        mb = nu * ROWS_LOCAL * row_bytes / 1e6
        t_old = _graph_us([lambda c=c: c.old(8) for c in cases], args.reps)
        t_new = _graph_us([lambda c=c: c.new(8) for c in cases], args.reps)
        res["us"][f"M{m}"] = {
            "experts_per_layer": round(nu, 2),
            "weight_MB": round(mb, 2),
            "down_rows_4x8_us": round(t_old, 2),
            "down_rows_fast_us": round(t_new, 2),
            "down_rows_4x8_GBps": round(mb * 1e3 / t_old, 1),
            "down_rows_fast_GBps": round(mb * 1e3 / t_new, 1),
            "saved_per_layer_us": round(t_old - t_new, 2),
            "speedup": round(t_old / t_new, 3),
        }
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
    with contextlib.suppress(Exception):  # needs pynvml; informational only
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
