# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cold-L2 graph timings of V4.1 dense projections, M=6.

Usage: q8ks_bench.py SRC CHECKPOINT LAYER
"""

import sys

import torch
from safetensors import safe_open

sys.path.insert(0, sys.argv[1])
torch.ops.load_library(sys.argv[1] + "/benchmarks/deepseek_v41/v41k/build/v41q8.so")
f = safe_open(sys.argv[2], "pt", device="cpu")
L = int(sys.argv[3])
dev = torch.device("cuda", 0)


def t(n):
    return f.get_tensor(f"layers.{L}." + n)


r, TP = 0, 4


def rows(w, s, parts):
    n = w.shape[0] // parts
    return w[r * n : (r + 1) * n], s[r * n // 32 : (r + 1) * n // 32]


def cols(w, s, parts):
    k = w.shape[1] // parts
    return w[:, r * k : (r + 1) * k], s[:, r * k // 32 : (r + 1) * k // 32]


def cat(*ws):
    return torch.cat([w for w, _ in ws]), torch.cat([s for _, s in ws])


qa = (t("attn.wq_a.weight"), t("attn.wq_a.scale"))
kv = (t("attn.wkv.weight"), t("attn.wkv.scale"))
g = rows(t("ffn.shared_experts.w1.weight"), t("ffn.shared_experts.w1.scale"), TP)
u = rows(t("ffn.shared_experts.w3.weight"), t("ffn.shared_experts.w3.scale"), TP)
shapes = {
    "wq_a+wkv": cat(qa, kv),
    "wq_b": rows(t("attn.wq_b.weight"), t("attn.wq_b.scale"), TP),
    "wo_a.g0": (lambda w, s: (w[:1024], s[:32]))(
        *rows(t("attn.wo_a.weight"), t("attn.wo_a.scale"), TP)
    ),
    "wo_b": cols(t("attn.wo_b.weight"), t("attn.wo_b.scale"), TP),
    "shared.w1+w3": cat(g, u),
    "shared.w2": cols(
        t("ffn.shared_experts.w2.weight"), t("ffn.shared_experts.w2.scale"), TP
    ),
}
flush_buffer = torch.randn(16 * 1024 * 1024, device=dev)
flush_result = torch.empty((), device=dev)


def timed(fn, n=30):
    # Each call starts after an ordinary 64 MiB read, including instruction
    # and data cache misses that occur between layers in the serving graph.
    for _ in range(3):
        torch.sum(flush_buffer, dim=0, out=flush_result)
        fn()
    torch.accelerator.synchronize()
    start = torch.cuda.Event(enable_timing=True, external=True)
    finish = torch.cuda.Event(enable_timing=True, external=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        torch.sum(flush_buffer, dim=0, out=flush_result)
        start.record()
        fn()
        finish.record()
    samples = []
    for _ in range(n):
        graph.replay()
        torch.accelerator.synchronize()
        samples.append(start.elapsed_time(finish) * 1000)
    return sorted(samples)[len(samples) // 2]


M = 6
tot0 = tot1 = 0
tb = 0
for name, (w, s) in shapes.items():
    w = w.contiguous().to(dev)
    s = s.contiguous().to(dev)
    n, k = w.shape
    x = (torch.randn(M, k, device=dev) * 0.3).half()
    packed = torch.empty(n * k, dtype=torch.uint8, device=dev)
    torch.ops.v41q8.prepack(packed, w.view(torch.uint8))
    sc = (
        (torch.pow(2.0, s.view(torch.uint8).float() - 127) * 256)
        .half()
        .t()
        .contiguous()
    )
    out0 = torch.empty(M, n, dtype=torch.float16, device=dev)
    out1 = torch.empty_like(out0)
    best0 = None
    for split in (4, 8, 16, 32):
        if (k // 16) % split:
            continue
        for ch in (1, 2):
            us = timed(lambda: torch.ops.v41q8.gemm(out0, x, packed, sc, split, ch))
            if best0 is None or us < best0[0]:
                best0 = (us, split, ch)
    torch.ops.v41q8.gemm(out0, x, packed, sc, best0[1], best0[2])
    tickets = torch.zeros(n // 32, dtype=torch.int32, device=dev)
    best1 = None
    for W in (4, 8, 16):
        for parts in (1, 2, 3, 4, 5, 6, 8, 10, 16, 20):
            if (k // 16) % (W * parts) or (k // 16) // (W * parts) < 2:
                continue
            for ch in (1, 2):
                us = timed(
                    lambda: torch.ops.v41q8.gemm_ksplit(
                        out1, x, packed, sc, tickets, W, parts, ch
                    )
                )
                if best1 is None or us < best1[0]:
                    best1 = (us, W, parts, ch)
    torch.ops.v41q8.gemm_ksplit(
        out1, x, packed, sc, tickets, best1[1], best1[2], best1[3]
    )
    err = ((out1.float() - out0.float()).norm() / out0.float().norm()).item()
    nb = n * k + s.numel()
    tot0 += best0[0]
    tot1 += best1[0]
    tb += nb
    print(
        f"{name:13s} N={n:5d} K={k:5d} gemm {best0[0]:6.1f}us {nb / best0[0] / 1e3:4.0f}GB/s (s{best0[1]}c{best0[2]}) | ksplit {best1[0]:6.1f}us {nb / best1[0] / 1e3:4.0f}GB/s (W{best1[1]} p{best1[2]} c{best1[3]}, {n // 32 * best1[2]} CTAs) rel {err:.1e}"
    )
# wo_a grouped: old path = per-group contiguous + gemm + cat; new = one launch
w, s_ = rows(t("attn.wo_a.weight"), t("attn.wo_a.scale"), TP)
w = w.contiguous().to(dev)
s_ = s_.contiguous().to(dev)
n, k = w.shape
G = n // 1024
x3 = (torch.randn(M, G, k, device=dev) * 0.3).half()


def prep(wt, st):
    p = torch.empty(wt.numel(), dtype=torch.uint8, device=dev)
    torch.ops.v41q8.prepack(p, wt.contiguous().view(torch.uint8))
    return p, (
        torch.pow(2.0, st.view(torch.uint8).float() - 127) * 256
    ).half().t().contiguous()


gp = [prep(w[i * 1024 : (i + 1) * 1024], s_[i * 32 : (i + 1) * 32]) for i in range(G)]
full = prep(w, s_)


def old():
    outs = []
    for i in range(G):
        o = torch.empty(M, 1024, dtype=torch.float16, device=dev)
        torch.ops.v41q8.gemm(o, x3[:, i].contiguous(), gp[i][0], gp[i][1], 32, 2)
        outs.append(o)
    return torch.cat(outs, -1)


ref = old()
t_old = timed(old)
tk = torch.zeros(n // 32, dtype=torch.int32, device=dev)
outg = torch.empty(M, n, dtype=torch.float16, device=dev)
bestg = None
for W in (4, 8, 16, 32):
    for parts in (1, 2, 4, 8):
        if (k // 16) % (W * parts) or (k // 16) // (W * parts) < 2:
            continue
        for ch in (1, 2):
            us = timed(
                lambda: torch.ops.v41q8.gemm_grouped(
                    outg, x3, full[0], full[1], tk, W, parts, ch
                )
            )
            if bestg is None or us < bestg[0]:
                bestg = (us, W, parts, ch)
torch.ops.v41q8.gemm_grouped(
    outg, x3, full[0], full[1], tk, bestg[1], bestg[2], bestg[3]
)
nb = n * k + s_.numel()
print(
    f"wo_a grouped G={G} N={n} K={k}: old {t_old:.1f}us ({nb / t_old / 1e3:.0f} GB/s) | one launch {bestg[0]:.1f}us ({nb / bestg[0] / 1e3:.0f} GB/s) W{bestg[1]} p{bestg[2]} c{bestg[3]} rel {((outg.float() - ref.float()).norm() / ref.float().norm()).item():.1e}"
)
print(
    f"TOTAL gemm {tot0:.1f} us ({tb / tot0 / 1e3:.0f} GB/s)  ksplit {tot1:.1f} us ({tb / tot1 / 1e3:.0f} GB/s)  {tb / 1e6:.1f} MB"
)
