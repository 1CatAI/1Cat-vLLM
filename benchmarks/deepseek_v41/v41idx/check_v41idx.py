# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU check of v41idx.index_scores_tiled (run with no serving workload).

Every case runs three kernels on the same inputs and NaN-poisoned outputs:
  v41k   torch.ops.v41k.index_scores_decode from the reference v41k.so (--v41k)
  ref    torch.ops.v41idx.index_scores_ref (the embedded verbatim copy)
  tiled  torch.ops.v41idx.index_scores_tiled (the new kernel)
and requires tiled == v41k and ref == v41k BITWISE over the whole output row
(written columns and the poison past the live end), plus a minimum number of
written / finite values so the comparison is not of poison alone.

Matrix: heads 8 (tokens 1..8) and 4 (tokens 1..16); capacities 1, 63, 64, 65,
1000, 4097, 65536, 131072; live ends 0, 1, 31, 32, 63, 64, 65, 1000, 37500,
75000 and the capacity, plus stale ends 64 keys beyond aligned capacity;
ratio 1, 4, 8 (positions[-1] at
every offset inside the ratio group); realistic keys (scale exponents near
the observed range) and full-range keys (every scale byte, NaN/Inf scales);
normal and extreme q / weights (+-65504, Inf, NaN); a plain output and a
bank-like one (rows of a wider [16, width] buffer, as bank.partial[:tokens]).
Then CUDA graphs: one capture replayed with changed positions (live ends
5 .. capacity + 64) against eager v41k, and, in a fresh process, a capture whose
first-ever tiled call is inside the capture (T 6 and T 8). Then timing in
graphs (TIME lines, us per call, tiled vs reference v41k).

  --quick      reduced matrix (TP4 runs); --sanitize: small grid for
  compute-sanitizer (one capacity past 2 CTAs x SMs so the persistent loop
  turns), no graphs, no timing; --no-time; --json PATH. Exit 0 only when every check
  passes ("ALL OK"); any mismatch or exception -> "FAILED", exit 1.
"""

import argparse
import json
import os
import subprocess
import sys
import time

import torch

POISON_BASE = 0x7E00


def load(lib, v41k):
    torch.ops.load_library(v41k)
    torch.ops.load_library(lib)
    assert hasattr(torch.ops.v41k, "index_scores_decode"), v41k
    assert int(torch.ops.v41idx.version()) == 1, lib


def poison(rows, width, dev):
    p = (
        torch.arange(rows * width, device=dev, dtype=torch.int32) & 0x1FF
    ) | POISON_BASE
    return p.to(torch.int16).view(torch.float16).reshape(rows, width)


def make_keys(cap, mode, gen, dev):
    k = torch.randint(0, 256, (cap, 68), generator=gen, dtype=torch.uint8, device="cpu")
    if mode == "real":
        # biased E8M0 exponents around the observed index-key range (2^-12 .. 2^4)
        k[:, 64:] = torch.randint(115, 132, (cap, 4), generator=gen, dtype=torch.uint8)
    else:
        k[:, 64:] = torch.randint(0, 256, (cap, 4), generator=gen, dtype=torch.uint8)
        if cap >= 4:
            k[0, 64:] = 255  # NaN/Inf scale
            k[1, 64:] = 0  # zero scale
            k[2, 64:] = 254
            k[3, 64:] = 1
    return k.to(dev)


def make_qw(rows, mode, gen, dev):
    q = torch.randn(rows, 128, generator=gen, dtype=torch.float32)
    w = torch.randn(rows, generator=gen, dtype=torch.float32) * 0.5
    if mode == "extreme":
        q[0, 0] = 65504.0
        q[0, 1] = -65504.0
        q[0, 5] = float("inf")
        w[0] = 65504.0
        if rows > 1:
            q[1, 7] = float("nan")
            w[1] = float("-inf")
        if rows > 2:
            q[2, :] = 1000.0
            w[2] = -0.0
        if rows > 3:
            w[3] = float("nan")
    return q.half().to(dev).contiguous(), w.half().to(dev).contiguous()


def positions_for(tokens, end, ratio, off):
    """tokens consecutive positions whose last one gives (p + 1) // ratio == end."""
    last = ratio - 2 - off if end == 0 else end * ratio - 1 + off
    return torch.arange(last - tokens + 1, last + 1, dtype=torch.int64).clamp_min(0)


class Stats:
    def __init__(self):
        self.cases = 0
        self.bad = []
        self.written = 0
        self.finite = 0

    def ok(self):
        return not self.bad


def run_case(st, H, T, keys, q, w, pos, ratio, bank, tag):
    dev = keys.device
    cap = keys.shape[0]
    width = cap + 40 if bank else cap
    outs = {}
    for name in ("v41k", "ref", "tiled"):
        if bank:
            buf = poison(16, width, dev)
            o = buf[:T]
        else:
            o = poison(T, width, dev)
        args = (o, q, w, keys, pos, H, ratio)
        if name == "v41k":
            torch.ops.v41k.index_scores_decode(*args)
        elif name == "ref":
            torch.ops.v41idx.index_scores_ref(*args)
        else:
            torch.ops.v41idx.index_scores_tiled(*args)
        outs[name] = o
    a = outs["v41k"].view(torch.int16)
    st.cases += 1
    for name in ("ref", "tiled"):
        b = outs[name].view(torch.int16)
        if not torch.equal(a, b):
            n = int((a != b).sum().item())
            st.bad.append(f"{tag} {name}!=v41k ({n} values)")
    pz = poison(T, width, dev).view(torch.int16)
    if bank and not torch.equal(outs["tiled"].view(torch.int16)[:, cap:], pz[:, cap:]):
        st.bad.append(f"{tag} tiled wrote guard columns beyond capacity")
    written = a != pz
    st.written += int(written.sum().item())
    st.finite += int((written & torch.isfinite(outs["v41k"])).sum().item())


def matrix(st, dev, quick, sanitize, seed=517):
    gen = torch.Generator().manual_seed(seed)
    if sanitize:
        # 12000 keys = 188 blocks > 2 CTAs x 80 SMs: the persistent loop runs twice
        caps, hts = [65, 12000], [(8, 1), (8, 6), (8, 8), (4, 16)]
    elif quick:
        caps = [64, 65, 4097, 131072]
        hts = [(8, t) for t in (1, 2, 6, 8)] + [(4, t) for t in (1, 6, 16)]
    else:
        caps = [1, 63, 64, 65, 1000, 4097, 65536, 131072]
        hts = [(8, t) for t in range(1, 9)] + [(4, t) for t in range(1, 17)]
    ends_all = [0, 1, 31, 32, 63, 64, 65, 1000, 37500, 75000]
    ratios = [1, 4] if sanitize else [1, 4, 8]
    for cap in caps:
        for kmode in ("real", "full"):
            keys = make_keys(cap, kmode, gen, dev)
            ends = sorted({e for e in ends_all if e <= cap} | {cap})
            if sanitize or quick:
                ends = sorted({e for e in (0, 1, 63, 65, cap // 2, cap) if e <= cap})
            # A full reference CTA ends exactly at aligned capacity; stale live
            # positions must not make the tiled loop exceed that allocation.
            if cap % 32 == 0:
                ends = sorted(set(ends) | {cap + 64})
            for H, T in hts:
                for qmode in ("normal",) if sanitize else ("normal", "extreme"):
                    q, w = make_qw(T * H, qmode, gen, dev)
                    for ratio in ratios:
                        for end in ends:
                            if end == 0 and ratio < 2:
                                continue
                            offs = {0, ratio - 1} if end else {0}
                            for off in sorted(offs):
                                if end == 0 and off > ratio - 2:
                                    continue
                                pos = positions_for(T, end, ratio, off).to(dev)
                                for bank in (True,) if sanitize else (False, True):
                                    tag = (
                                        f"H{H} T{T} cap{cap} {kmode}/{qmode} r{ratio} "
                                        f"end{end}+{off} {'bank' if bank else 'plain'}"
                                    )
                                    run_case(
                                        st, H, T, keys, q, w, pos, ratio, bank, tag
                                    )
    torch.cuda.synchronize()


def graph_replay(st, dev):
    """One capture, replayed with changed positions; compared with eager v41k."""
    gen = torch.Generator().manual_seed(7)
    H, T, cap = 8, 6, 131072
    keys = make_keys(cap, "real", gen, dev)
    q, w = make_qw(T * H, "normal", gen, dev)
    pos = positions_for(T, 1, 4, 0).to(dev)
    out = poison(T, cap + 40, dev)
    torch.ops.v41idx.index_scores_tiled(out, q, w, keys, pos, H, 4)  # warm (eager)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        torch.ops.v41idx.index_scores_tiled(out, q, w, keys, pos, H, 4)
    for end in (5, 64, 65, 1000, 37500, 75000, cap - 1, cap, cap + 64):
        for off in (0, 3):
            pos.copy_(positions_for(T, end, 4, off).to(dev))
            out.copy_(poison(T, cap + 40, dev))
            g.replay()
            ref = poison(T, cap + 40, dev)
            torch.ops.v41k.index_scores_decode(ref, q, w, keys, pos, H, 4)
            torch.cuda.synchronize()
            st.cases += 1
            if not torch.equal(out.view(torch.int16), ref.view(torch.int16)):
                st.bad.append(f"graph replay end{end}+{off}: tiled!=v41k")
            guard = poison(T, 40, dev).view(torch.int16)
            if not torch.equal(out.view(torch.int16)[:, cap:], guard):
                st.bad.append(f"graph replay end{end}+{off}: guard columns changed")


FIRST_IN_CAPTURE = r"""
import sys, torch
sys.path.insert(0, {here!r})
import check_v41idx as c
c.load({lib!r}, {v41k!r})
dev = torch.device("cuda:0")
torch.cuda.init()
gen = torch.Generator().manual_seed(11)
H, T, cap = 8, {T}, 65536
keys = c.make_keys(cap, "real", gen, dev)
q, w = c.make_qw(T * H, "normal", gen, dev)
pos = c.positions_for(T, 30000, 4, 1).to(dev)
out = c.poison(T, cap, dev)
torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    torch.ops.v41idx.index_scores_tiled(out, q, w, keys, pos, H, 4)
g.replay()
ref = c.poison(T, cap, dev)
torch.ops.v41k.index_scores_decode(ref, q, w, keys, pos, H, 4)
torch.cuda.synchronize()
print("FIRSTCAP", "OK" if torch.equal(out.view(torch.int16), ref.view(torch.int16)) else "MISMATCH")
"""


def first_call_in_capture(st, lib, v41k):
    here = os.path.dirname(os.path.abspath(__file__))
    for T in (6, 8):
        r = subprocess.run(
            [
                sys.executable,
                "-I",
                "-c",
                FIRST_IN_CAPTURE.format(here=here, lib=lib, v41k=v41k, T=T),
            ],
            capture_output=True,
            text=True,
        )
        st.cases += 1
        if r.returncode != 0 or "FIRSTCAP OK" not in r.stdout:
            st.bad.append(
                f"first call inside capture T{T}: rc={r.returncode} "
                f"{(r.stdout + r.stderr)[-300:]}"
            )


def timing(dev, lines):
    gen = torch.Generator().manual_seed(3)
    H, T = 8, 6
    for cap in (65536, 131072):
        keys = make_keys(cap, "real", gen, dev)
        q, w = make_qw(T * H, "normal", gen, dev)
        for end in (1000, 10000, 37500, 75000, 131072):
            if end > cap:
                continue
            pos = positions_for(T, end, 4, 0).to(dev)
            res = {}
            for name in ("v41k", "tiled"):
                out = poison(T, cap, dev)
                op = (
                    torch.ops.v41k.index_scores_decode
                    if name == "v41k"
                    else torch.ops.v41idx.index_scores_tiled
                )
                op(out, q, w, keys, pos, H, 4)
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    for _ in range(20):
                        op(out, q, w, keys, pos, H, 4)
                for _ in range(5):
                    g.replay()
                s, e = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                s.record()
                for _ in range(50):
                    g.replay()
                e.record()
                torch.cuda.synchronize()
                res[name] = s.elapsed_time(e) * 1000.0 / (50 * 20)
            line = (
                f"TIME H{H} T{T} cap{cap} keys{end}: v41k {res['v41k']:.1f} us "
                f"tiled {res['tiled']:.1f} us ({res['v41k'] / res['tiled']:.2f}x)"
            )
            print(line, flush=True)
            lines.append(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True, help="v41idx.so under test")
    ap.add_argument(
        "--v41k", required=True, help="the reference v41k.so (VLLM_V41K_LIB)"
    )
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--sanitize", action="store_true")
    ap.add_argument("--no-time", action="store_true")
    ap.add_argument("--json")
    a = ap.parse_args()
    t0 = time.time()
    load(a.lib, a.v41k)
    dev = torch.device("cuda:0")
    print(
        f"DEVICE {torch.cuda.get_device_name(dev)} SMs "
        f"{torch.cuda.get_device_properties(dev).multi_processor_count} "
        f"grid(131072)={int(torch.ops.v41idx.grid_for(131072, 0))}",
        flush=True,
    )
    st = Stats()
    matrix(st, dev, a.quick, a.sanitize)
    print(
        f"MATRIX {st.cases} cases, {st.written} values written by v41k "
        f"({st.finite} finite), {len(st.bad)} mismatched cases",
        flush=True,
    )
    min_written = 2000 if a.sanitize else 100000
    if st.written < min_written or st.finite < min_written // 2:
        st.bad.append(f"coverage too low: written {st.written} finite {st.finite}")
    if not a.sanitize:
        graph_replay(st, dev)
        first_call_in_capture(st, a.lib, a.v41k)
        print(f"GRAPHS done, {len(st.bad)} mismatched cases so far", flush=True)
    lines = []
    if not (a.no_time or a.sanitize):
        timing(dev, lines)
    for b in st.bad[:20]:
        print("FAIL", b)
    ok = st.ok()
    print(
        ("ALL OK" if ok else "FAILED")
        + f" ({st.cases} cases, {time.time() - t0:.0f} s)"
    )
    if a.json:
        with open(a.json, "w") as f:
            json.dump(
                {
                    "ok": ok,
                    "cases": st.cases,
                    "written": st.written,
                    "finite": st.finite,
                    "bad": st.bad[:200],
                    "time": lines,
                },
                f,
                indent=1,
            )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
