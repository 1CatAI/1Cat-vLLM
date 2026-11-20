#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ONE V100, no model, no server: the tensor-core index scorer (v41idx3) against the scorer that serves (v41idx).

    python check_v41idx3.py --ref-lib ../v41idx/build/v41idx.so --lib build/v41idx3.so [--device 0]
                            [--out result.json] [--reps 200] [--skip-time]

Part 1, EQUALITY on generated operands (the contract: bitwise). For every case torch.ops.v41idx.index_scores_tiled
and torch.ops.v41idx3.index_scores_tc write the same inputs into buffers pre-filled with a NaN pattern; the two
buffers must be equal BIT FOR BIT, columns past `end` included. Cases: 1 to 8 tokens, eight heads, end 0, 1, 63, 64,
65, 4097, 62,000, the capacity and past the capacity (a stale position: both stop at the capacity), ratio 1 and 4, four classes of operands (v41idx3_cases.py: the engine's own
class with close scales, the same with spread scales, fp16 rows of any mantissa, every bit pattern with every scale
byte), the guard on (variant 0), every cell through the fallback (2), three and four CTAs an SM (4, 8). With each
case the two counters (cells the first look did not pass; heads of those recomputed by the serving chain) against
the guard's rule stated a second time in numpy (guard_rule.py), and against 0 for the close class. This is where
the card's own tensor cores meet the layout and the adder the host test only models.

Part 2, TIME (a reading, not a verdict). Six tokens, live keys N in 512 .. 262,144, the close class: each kernel
in a CUDA graph replayed `reps` times after a warm-up, CUDA events. Kernels: the serving scorer; v41idx3 with the
guard at 2, 3 and 4 CTAs an SM; v41idx3 without the guard (what the guard costs when it never fires). Then the
RARE FALLBACK at 62,000 and 131,072 keys: a planted fraction of keys whose scale bytes are 114, 127, 127, 140 (in
range, far over the limit: all eight heads of each of their cells are recomputed), fractions 1e-5 to 1e-2, each
checked bitwise first; and one head of one token made a row the guard does not take (that token's every cell gets
the second look and one head recomputed). Last, the rule's two sizes: how much faster than the serving scorer.

Any mismatch or wrong count: VERDICT FAIL. No generated case run: VERDICT INCONCLUSIVE. The operand bits are made
by torch's generator from a fixed seed; no model output is involved.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import guard_rule  # noqa: E402
import v41idx3_cases as cases  # noqa: E402

VARIANTS = {
    "guard": 0,
    "forced fallback": 2,
    "guard, 3 CTAs an SM": 4,
    "guard, 4 CTAs an SM": 8,
}
FAR_BYTES = (114, 127, 127, 140)
# The sizes of a run on a card; --cpu-smoke replaces them (the script's own paths, on the CPU test libraries).
SIZES = {
    "capacity": 65536,
    "tokens": tuple(range(1, 9)),
    "long_ends": (4097, 62000),
    "time_capacity": 262144,
    "time_keys": (512, 8192, 32768, 62000, 131072, 262144),
    "rule_keys": (62000, 131072),
}
SMOKE = {
    "capacity": 1024,
    "tokens": (1, 6),
    "long_ends": (257, 700),
    "time_capacity": 2048,
    "time_keys": (512, 1024, 2048),
    "rule_keys": (1024, 2048),
}


def expected_counts(q, keys, tokens: int, end: int, variant: int) -> tuple[int, int]:
    live = min(end, keys.shape[0])
    if variant & 2:
        return tokens * live, 8 * tokens * live
    return guard_rule.expected_counts(q.cpu(), keys.cpu(), tokens, live)


def equality(device) -> dict:
    ref_op, new_op = (
        torch.ops.v41idx.index_scores_tiled,
        torch.ops.v41idx3.index_scores_tc,
    )
    run = mismatches = miscounts = 0
    bad: list[str] = []
    capacity = SIZES["capacity"]
    for k, kind in enumerate(cases.CLASSES):
        for tokens in SIZES["tokens"]:
            q, w, keys = cases.operands(
                kind,
                tokens,
                capacity,
                torch.Generator().manual_seed(1000 + 10 * k + tokens),
            )
            qd, wd, kd = q.to(device), w.to(device), keys.to(device)
            for ratio in (1, 4):
                a, b = SIZES["long_ends"]
                for end in (
                    0,
                    1,
                    63,
                    64,
                    65,
                    a,
                    b // ratio,
                    capacity // ratio,
                    capacity + 70,
                ):
                    for label, variant in VARIANTS.items():
                        differ, cells, heads, problems = cases.both(
                            ref_op, new_op, qd, wd, kd, end, ratio, variant
                        )
                        if any(p.startswith("PREMISE") for p in problems):
                            raise RuntimeError(
                                "the serving scorer wrote past `end`: the test's premise is wrong"
                            )
                        want = expected_counts(q, keys, tokens, end, variant)
                        if kind == "close" and not variant & 2:
                            want = (
                                (0, 0) if want == (0, 0) else (-1, -1)
                            )  # arithmetic says 0; numpy must agree
                        run += 1
                        mismatches += differ != 0
                        wrong = (cells, heads) != want or bool(problems)
                        miscounts += wrong
                        if (differ or wrong) and len(bad) < 20:
                            bad.append(
                                f"{kind} tokens {tokens} ratio {ratio} end {end} {label}: {differ} cells "
                                f"differ; counters {cells}, {heads}, expected {want[0]}, {want[1]}; "
                                + "; ".join(problems)
                            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {
        "cases": run,
        "mismatches": mismatches,
        "wrong_counts": miscounts,
        "first": bad,
    }


def time_graph(fn, reps: int, device) -> float:
    """Microseconds per replay of a CUDA graph holding one call of fn."""
    if device.type != "cuda":  # --cpu-smoke: the call is made once and nothing is timed
        fn()
        return float("nan")
    graph = torch.cuda.CUDAGraph()
    fn()  # warm-up outside the graph (the library sets the function's shared-memory attribute once)
    torch.cuda.synchronize(device)
    with torch.cuda.graph(graph):
        fn()
    for _ in range(20):
        graph.replay()
    torch.cuda.synchronize(device)
    start, stop = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    start.record()
    for _ in range(reps):
        graph.replay()
    stop.record()
    stop.synchronize()
    return start.elapsed_time(stop) * 1000.0 / reps


def timing(device, reps: int) -> dict:
    ref_op, new_op = (
        torch.ops.v41idx.index_scores_tiled,
        torch.ops.v41idx3.index_scores_tc,
    )
    capacity, tokens = SIZES["time_capacity"], 6
    q, w, keys = cases.operands(
        "close", tokens, capacity, torch.Generator().manual_seed(20261118)
    )
    qd, wd, kd = q.to(device), w.to(device), keys.to(device)
    counters = torch.zeros(4, dtype=torch.int64, device=device)
    out = torch.empty((tokens, capacity), dtype=torch.float16, device=device)
    table = []
    kernels = {
        "v41idx (serving)": None,
        "tc guard": 0,
        "tc guard 3 CTAs": 4,
        "tc guard 4 CTAs": 8,
        "tc no guard": 1,
    }
    for n in SIZES["time_keys"]:
        positions = torch.full((tokens,), n - 1, dtype=torch.int64, device=device)
        row = {"keys": n, "us": {}}
        for name, variant in kernels.items():
            if variant is None:
                row["us"][name] = time_graph(
                    lambda: ref_op(out, qd, wd, kd, positions, 8, 1), reps, device
                )
            else:
                row["us"][name] = time_graph(
                    lambda v=variant: new_op(
                        out, qd, wd, kd, positions, 8, 1, v, counters
                    ),
                    reps,
                    device,
                )
        table.append(row)
        print(
            f"TIME keys {n:7d} us per call: "
            + "  ".join(f"{k} {t:8.1f}" for k, t in row["us"].items()),
            flush=True,
        )
    if int(counters[0]):
        raise RuntimeError(f"the close class fell back in timing: {counters.tolist()}")

    planted = []
    far = torch.tensor(FAR_BYTES, dtype=torch.uint8)
    for n in SIZES["rule_keys"]:
        positions = torch.full((tokens,), n - 1, dtype=torch.int64, device=device)
        base = time_graph(
            lambda: new_op(out, qd, wd, kd, positions, 8, 1, 0, counters), reps, device
        )
        serving = time_graph(
            lambda: ref_op(out, qd, wd, kd, positions, 8, 1), reps, device
        )
        for fraction in (1e-5, 1e-4, 1e-3, 1e-2):
            count = max(1, round(n * fraction))
            where = torch.randperm(
                n, generator=torch.Generator().manual_seed(n + count)
            )[:count]
            keys2 = keys.clone()
            keys2[where, 64:] = far
            k2 = keys2.to(device)
            differ, cells, heads, _ = cases.both(ref_op, new_op, qd, wd, k2, n, 1, 0)
            us = time_graph(
                lambda: new_op(out, qd, wd, k2, positions, 8, 1, 0, counters),
                reps,
                device,
            )
            planted.append(
                {
                    "keys": n,
                    "kind": "keys far over the limit",
                    "planted_keys": count,
                    "cells": cells,
                    "heads": heads,
                    "differ": differ,
                    "want_cells": tokens * count,
                    "want_heads": 8 * tokens * count,
                    "us": us,
                    "us_none": base,
                    "us_serving": serving,
                }
            )
            print(
                f"RARE keys {n:7d}: {count} keys far over the limit ({cells} cells, {heads} heads recomputed, "
                f"{differ} cells differ): {us:8.1f} us per call against {base:8.1f} with none and {serving:8.1f} "
                "for the serving scorer",
                flush=True,
            )
        q2 = q.clone()
        q2.view(torch.int16)[2 * 8 + 5, 77] = (
            0x0001  # one subnormal: token 2 is not taken by the first look
        )
        q2d = q2.to(device)
        differ, cells, heads, _ = cases.both(ref_op, new_op, q2d, wd, kd, n, 1, 0)
        us = time_graph(
            lambda: new_op(out, q2d, wd, kd, positions, 8, 1, 0, counters), reps, device
        )
        planted.append(
            {
                "keys": n,
                "kind": "one row the guard does not take",
                "cells": cells,
                "heads": heads,
                "differ": differ,
                "want_cells": n,
                "want_heads": n,
                "us": us,
                "us_none": base,
                "us_serving": serving,
            }
        )
        print(
            f"RARE keys {n:7d}: one row of one token not taken ({cells} cells get the second look, {heads} "
            f"heads recomputed, {differ} cells differ): {us:8.1f} us per call against {base:8.1f} with none and "
            f"{serving:8.1f} for the serving scorer",
            flush=True,
        )
    return {"table": table, "planted": planted}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-lib", required=True)
    ap.add_argument("--lib", required=True)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--out")
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--skip-time", action="store_true")
    ap.add_argument(
        "--cpu-smoke",
        action="store_true",
        help="NO CARD: this script's own paths on small sizes with the CPU test libraries; says nothing "
        "of a card and times nothing",
    )
    args = ap.parse_args()
    if args.cpu_smoke:
        SIZES.update(SMOKE)
        device, name = torch.device("cpu"), "no card (smoke)"
    else:
        device = torch.device("cuda", args.device)
        props = torch.cuda.get_device_properties(device)
        name = props.name
        if (props.major, props.minor) != (7, 0):
            print(
                f"VERDICT INCONCLUSIVE: device {args.device} is compute capability {props.major}.{props.minor}, "
                "not a V100 (7.0)"
            )
            return 3
    torch.ops.load_library(args.ref_lib)
    torch.ops.load_library(args.lib)
    with (
        contextlib.nullcontext() if args.cpu_smoke else torch.cuda.device(device),
        torch.inference_mode(),
    ):
        eq = equality(device)
        print(
            f"EQUALITY generated: cases {eq['cases']}, mismatches {eq['mismatches']}, wrong counts "
            f"{eq['wrong_counts']}",
            flush=True,
        )
        for line in eq["first"]:
            print("  WRONG", line)
        times = {} if args.skip_time else timing(device, args.reps)
    rule = None
    if times:
        by = {row["keys"]: row["us"] for row in times["table"]}
        small, large = SIZES["rule_keys"]
        rule = {
            n: 100.0 * (1.0 - by[n]["tc guard"] / by[n]["v41idx (serving)"])
            for n in (small, large)
        }
        if args.cpu_smoke:  # nothing was timed: the rule has no reading
            print("RULE: inconclusive in --cpu-smoke (no card, nothing timed)")
        else:
            met = all(v >= 17.0 for v in rule.values())
            print(
                f"RULE (a reading): v41idx3 with its guard is {rule[small]:.1f} % faster than the serving scorer at "
                f"{small:,} keys and {rule[large]:.1f} % at {large:,}; the rule asks at least 17 % at both: "
                f"{'MET' if met else 'NOT MET'}"
            )
    if args.out:
        with open(args.out, "w") as f:
            json.dump(
                {"device": name, "equality": eq, "timing": times, "rule": rule},
                f,
                indent=1,
            )
    planted_bad = sum(
        1
        for p in (times.get("planted", []) if times else [])
        if p["differ"] or (p["cells"], p["heads"]) != (p["want_cells"], p["want_heads"])
    )
    # a smoke run's last line never begins with VERDICT: it is no card's verdict
    word = (
        "SMOKE (no card, the CPU test libraries, small sizes)"
        if args.cpu_smoke
        else "VERDICT"
    )
    if eq["cases"] == 0:
        print(f"{word} INCONCLUSIVE: no generated case ran")
        return 3
    if eq["mismatches"] or eq["wrong_counts"] or planted_bad:
        print(
            f"{word} FAIL: generated cases with a difference {eq['mismatches']} of {eq['cases']}, with a wrong "
            f"count {eq['wrong_counts']}; planted cases wrong {planted_bad}"
        )
        return 1
    print(
        f"{word} PASS (equality): {eq['cases']} generated cases bitwise equal to the serving scorer, every count "
        "as the rule says; timing is a reading, not a verdict"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
