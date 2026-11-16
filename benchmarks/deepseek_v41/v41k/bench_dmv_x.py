# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DMV_X microbench: dmv13 (vllm._C) vs dmv13_x (v41k), V4.1 dense shapes.

Standalone (no server, one GPU). Per shape, at M verify rows:
  old    dmv13 with dense_tm's config (wo_a: one launch per O group)
  same   dmv13_x, same warps/split, acc 1 (wo_a: one merged launch);
         expected bitwise equal to old
  probe  dmv13_x loads/staging/barriers only (no decode or mma): floor
  sweep  dmv13_x warps x split x acc (only `same` is bit-identical; the
         others change the FP32 summation order -> near-tie gate)
Times are CUDA-graph replays of back-to-back launches over rotating weight
copies (>= --l2-mb per arm, so the 6 MB L2 holds none of them), reported as
us per projection; --kernel-time adds profiler kernel-only times.

  python benchmarks/deepseek_v41/v41k/bench_dmv_x.py --json out.json
  python benchmarks/deepseek_v41/v41k/bench_dmv_x.py --no-time   # bits only
--no-time runs no timing: for M = 1..8 and every role it compares `same`
(dmv13_x at dmv13's warps/split, acc 1) with `old` bit for bit and exits 1
on any difference.
Env: VLLM_V41K_LIB (default benchmarks/deepseek_v41/v41k/build/v41k.so).
"""

import argparse
import glob
import importlib.util
import json
import math
import os
import statistics

import torch

Q5K, Q8 = 1, 4
# role -> (format, K, widths, block-diagonal groups); per TP4 rank.
SHAPES = {
    "qkv": (Q8, 5120, [1280, 512], 1),
    "q_b": (Q5K, 1280, [8192], 1),
    "wo_a": (Q5K, 4096, [1024], 2),
    "wo_b": (Q5K, 2048, [5120], 1),
    "shared": (Q5K, 5120, [576, 576], 1),
    "head": (Q5K, 5120, [32320], 1),
}
LAYER_ROLES = ("qkv", "q_b", "wo_a", "wo_b", "shared")
FLOOR_GBS = 735.0  # MEASURED lm-head rate on repacked bytes (decode profile)


def load_ops():
    torch.ops.load_library(
        os.environ.get("VLLM_V41K_LIB")
        or os.path.join(os.path.dirname(os.path.abspath(__file__)), "build", "v41k.so")
    )
    assert hasattr(torch.ops.v41k, "dmv13_x"), "v41k library predates dmv13_x"
    spec = importlib.util.find_spec("vllm")
    libs = []
    for root in spec.submodule_search_locations if spec else []:
        libs += glob.glob(os.path.join(root, "_C.*so")) + glob.glob(
            os.path.join(root, "_C.abi3.so")
        )
    if libs:
        torch.ops.load_library(libs[0])
    else:
        import vllm._C  # noqa: F401
    assert hasattr(torch.ops._C, "sm70_dmv13_out"), "vllm._C has no dmv13"


def default_config(k, widths, target=80):
    """dense_tm.default_config (VLLM_V41_DMV_CTAS=80): (warps, split)."""
    tiles = sum((w + 31) // 32 for w in widths)
    groups = (k // 32 + 3) // 4
    split = max(1, min(8, round(target / tiles), groups // 4))
    split = max(c for c in (1, 2, 3, 4, 6, 8) if c <= split)
    return (8, split) if split > 1 else (4, 1)


def planes(fmt, n, k, gen):
    tiles, groups = n // 32, k // 128
    cw = 2 if fmt == Q8 else 1

    def rand_bytes(count):
        return torch.randint(
            0, 256, (count,), dtype=torch.uint8, device="cuda", generator=gen
        )

    codes = rand_bytes(tiles * groups * 2048 * cw)
    high = (
        rand_bytes(tiles * groups * 512)
        if fmt == Q5K
        else torch.empty(0, dtype=torch.uint8, device="cuda")
    )
    # one 32-bit stats word per lane and K step: half2 (scale, min); Q8 uses
    # the low half only.
    words = tiles * groups * 128
    u = torch.rand(words, 2, device="cuda", generator=gen)
    stats = torch.stack(
        [0.002 + 0.018 * u[:, 0], -0.1 * u[:, 1] if fmt == Q5K else 0 * u[:, 1]], 1
    )
    return codes, high, stats.half().view(torch.uint8).flatten()


class Arm:
    """One projection on rotating weight copies; call(i) runs copy i."""

    def __init__(self, role, copies, x, kind, warps, split, acc=1, probe=0):
        fmt, k, widths, groups = SHAPES[role]
        self.role, self.kind, self.copies, self.x = role, kind, copies, x
        self.fmt, self.k, self.groups = fmt, k, groups
        self.warps, self.split, self.acc, self.probe = warps, split, acc, probe
        self.widths = widths * groups
        m = x.shape[0]
        self.out = torch.empty(m, sum(self.widths), dtype=torch.half, device="cuda")
        cols, self.views = 0, []
        for w in self.widths:
            self.views.append(self.out[:, cols : cols + w])
            cols += w
        launches = groups if kind == "old" else 1
        tiles = sum(w // 32 for w in self.widths) // launches
        self.ws = [
            torch.zeros(tiles * split * 256, device="cuda") for _ in range(launches)
        ]
        self.cnt = [
            torch.zeros(tiles, dtype=torch.int32, device="cuda")
            for _ in range(launches)
        ]

    def __call__(self, i):
        codes, high, scale = self.copies[i % len(self.copies)]
        if self.kind == "old" and self.groups > 1:
            for g in range(self.groups):
                torch.ops._C.sm70_dmv13_out(
                    self.x[:, g * self.k : (g + 1) * self.k],
                    [codes[g]], [high[g]], [scale[g]], [self.views[g]],
                    [self.fmt], [self.widths[g]], self.k, self.split,
                    self.warps, self.ws[g], self.cnt[g], 1, None, None,
                )  # fmt: skip
        elif self.kind == "old":
            torch.ops._C.sm70_dmv13_out(
                self.x, codes, high, scale, self.views,
                [self.fmt] * len(self.widths), self.widths, self.k,
                self.split, self.warps, self.ws[0], self.cnt[0], 1, None, None,
            )  # fmt: skip
        else:
            n = len(self.widths)
            offs = [g * self.k for g in range(n)] if self.groups > 1 else [0] * n
            torch.ops.v41k.dmv13_x(
                self.x, codes, high, scale, self.views, offs, self.fmt,
                self.k, self.split, self.warps, self.acc, self.ws[0],
                self.cnt[0], self.probe,
            )  # fmt: skip


def time_arm(arm, reps, iters, kernel_time):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for i in range(len(arm.copies)):
            arm(i)
    torch.cuda.current_stream().wait_stream(side)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for r in range(reps):
            arm(r)
    graph.replay()
    torch.accelerator.synchronize()
    times = []
    for _ in range(iters):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        graph.replay()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b) * 1000.0 / reps)
    result = {"us": statistics.median(times)}
    if kernel_time:
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            graph.replay()
            torch.accelerator.synchronize()
        total = 0.0
        for e in prof.key_averages():
            if "dense_mv" in e.key or "dmv_x" in e.key:
                total += getattr(
                    e, "self_device_time_total", getattr(e, "self_cuda_time_total", 0)
                )
        result["kernel_us"] = total / reps
    del graph
    return result


def diff(ref, out):
    a, b = ref.float(), out.float()
    d = (a - b).abs()
    return {
        "bitwise": bool(torch.equal(ref.view(torch.int16), out.view(torch.int16))),
        "n_diff": int((ref.view(torch.int16) != out.view(torch.int16)).sum()),
        "max_abs": float(d.max()),
        "max_rel": float(d.max() / a.abs().max().clamp_min(1e-6)),
        "finite": bool(torch.isfinite(b).all()),
    }


def check_bits(roles, gen):
    """dmv13_x at dmv13's config vs dmv13, bit for bit, M = 1..8 (no timing)."""
    failed = 0
    for role in roles:
        fmt, k, widths, groups = SHAPES[role]
        one = [planes(fmt, w, k, gen) for w in widths * groups]
        copies = [tuple(list(t) for t in zip(*one))]
        warps, split = default_config(k, widths)
        for m in range(1, 9):
            x = torch.randn(m, k * groups, device="cuda", generator=gen).half()
            old = Arm(role, copies, x, "old", warps, split)
            same = Arm(role, copies, x, "x", warps, split)
            old(0)
            same(0)
            torch.accelerator.synchronize()
            d = diff(old.out, same.out)
            ok = d["bitwise"] and d["finite"]
            failed += not ok
            print(
                f"{role:7s} M={m} w{warps} s{split}: bitwise={d['bitwise']}"
                f" n_diff={d['n_diff']} finite={d['finite']}",
                flush=True,
            )
    print("PASS" if not failed else f"FAIL: {failed} cases differ")
    return 1 if failed else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--m", type=int, default=6)
    ap.add_argument("--roles", default=",".join(SHAPES))
    ap.add_argument("--l2-mb", type=float, default=64.0)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--no-sweep", action="store_true")
    ap.add_argument("--kernel-time", action="store_true")
    ap.add_argument("--layers", type=int, default=40)
    ap.add_argument("--head-calls", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1103)
    ap.add_argument("--json")
    ap.add_argument("--no-time", action="store_true")
    args = ap.parse_args()
    load_ops()
    torch.accelerator.set_device_index(0)
    gen = torch.Generator(device="cuda").manual_seed(args.seed)
    if args.no_time:
        raise SystemExit(check_bits(args.roles.split(","), gen))
    print(torch.cuda.get_device_name(), "M =", args.m)
    report = {"m": args.m, "roles": {}}
    for role in args.roles.split(","):
        fmt, k, widths, groups = SHAPES[role]
        one = [planes(fmt, w, k, gen) for w in widths * groups]
        nbytes = sum(t.numel() for p in one for t in p)
        count = max(1, math.ceil(args.l2_mb * 2**20 / nbytes))
        copies = [one] + [
            [planes(fmt, w, k, gen) for w in widths * groups] for _ in range(count - 1)
        ]
        copies = [tuple(list(t) for t in zip(*c)) for c in copies]
        reps = count * max(1, math.ceil(64 / count))
        x = torch.randn(args.m, k * groups, device="cuda", generator=gen).half()
        warps, split = default_config(k, widths)
        floor = nbytes / (FLOOR_GBS * 1e3)
        row = {"bytes": nbytes, "copies": count, "floor_us": floor}
        old = Arm(role, copies, x, "old", warps, split)
        row["old"] = time_arm(old, reps, args.iters, args.kernel_time)
        old(0)
        ref = old.out.clone()
        row["old"]["config"] = [warps, split, 1]
        arms = [("same", warps, split, 1, 0), ("probe", warps, split, 1, 1)]
        if not args.no_sweep:
            arms += [
                (f"w{w}s{s}a{a}", w, s, a, 0)
                for w in (2, 4, 8)
                for s in (1, 2, 3, 4, 6, 8)
                for a in (1, 2)
                if s <= k // 128
            ]
        row["arms"] = {}
        for name, w, s, a, probe in arms:
            arm = Arm(role, copies, x, "x", w, s, a, probe)
            res = time_arm(arm, reps, args.iters, args.kernel_time)
            res["config"] = [w, s, a]
            if not probe:
                arm(0)
                res.update(diff(ref, arm.out))
            row["arms"][name] = res
            del arm
        swept = {n: r for n, r in row["arms"].items() if n != "probe"}
        best = min(swept, key=lambda n: swept[n]["us"])
        row["best"] = best
        same, probe_r, b = row["arms"]["same"], row["arms"]["probe"], swept[best]
        print(
            f"{role:7s} {nbytes / 2**20:7.2f} MB x{count:<3d} floor {floor:6.1f}"
            f" | old {row['old']['us']:6.1f} (w{warps} s{split})"
            f" | same {same['us']:6.1f} bitwise={same['bitwise']}"
            f" | probe {probe_r['us']:6.1f}"
            f" | best {b['us']:6.1f} {best} n_diff={b['n_diff']}"
            f" max_abs={b['max_abs']:.3g} max_rel={b['max_rel']:.2g}",
            flush=True,
        )
        if args.kernel_time:
            print(
                f"{'':7s} kernel-only: old {row['old'].get('kernel_us', 0):6.1f}"
                f" | same {same.get('kernel_us', 0):6.1f}"
                f" | probe {probe_r.get('kernel_us', 0):6.1f}"
                f" | best {b.get('kernel_us', 0):6.1f}"
            )
        report["roles"][role] = row
        del old, copies, one
        torch.accelerator.empty_cache()
    roles = report["roles"]
    layer = [r for r in LAYER_ROLES if r in roles]
    if layer:
        old_l = sum(roles[r]["old"]["us"] for r in layer)
        same_l = sum(roles[r]["arms"]["same"]["us"] for r in layer)
        best_l = sum(roles[r]["arms"][roles[r]["best"]]["us"] for r in layer)
        print(
            f"per layer ({'+'.join(layer)}): old {old_l:.1f} us, same {same_l:.1f}"
            f" us, best {best_l:.1f} us -> per round ({args.layers} layers):"
            f" same {(old_l - same_l) * args.layers / 1e3:+.3f} ms saved,"
            f" best {(old_l - best_l) * args.layers / 1e3:+.3f} ms saved"
        )
        report["per_layer_us"] = {"old": old_l, "same": same_l, "best": best_l}
    if "head" in roles:
        h = roles["head"]
        gain = h["old"]["us"] - h["arms"][h["best"]]["us"]
        print(f"head x{args.head_calls}: best saves {gain * args.head_calls:.1f} us")
    config = ";".join(
        f"{r}={','.join(map(str, row['arms'][row['best']]['config']))}"
        for r, row in roles.items()
        if row["arms"][row["best"]]["us"] < row["arms"]["same"]["us"] * 0.97
    )
    print("suggested VLLM_V41_DMV_X_CONFIG (near-tie gate):", repr(config))
    report["suggested_config"] = config
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1)
        print("wrote", args.json)


if __name__ == "__main__":
    main()
