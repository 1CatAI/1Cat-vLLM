# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K2: V4.1 dense FP8 32x32 projections at M=6: current path vs v41q8 (QPN8 HMMA)."""

import sys

import torch
from safetensors import safe_open

sys.path.insert(0, sys.argv[1])
from vllm.models.deepseek_v41 import fp8_linear

torch.ops.load_library(sys.argv[1] + "/benchmarks/deepseek_v41/v41k/build/v41k.so")
torch.ops.load_library(sys.argv[1] + "/benchmarks/deepseek_v41/v41k/build/v41q8.so")
f = safe_open(sys.argv[2], "pt", device="cpu")
L = int(sys.argv[3])
dev = torch.device("cuda", 0)


def t(n):
    return f.get_tensor(f"layers.{L}." + n)


TP, r = 4, 0


def rows(w, s, parts):  # column-parallel slice of N
    n = w.shape[0] // parts
    return w[r * n : (r + 1) * n], s[r * n // 32 : (r + 1) * n // 32]


def cols(w, s, parts):  # row-parallel slice of K
    k = w.shape[1] // parts
    return w[:, r * k : (r + 1) * k], s[:, r * k // 32 : (r + 1) * k // 32]


shapes = {
    "wq_a(rep)": (t("attn.wq_a.weight"), t("attn.wq_a.scale")),
    "wkv(rep)": (t("attn.wkv.weight"), t("attn.wkv.scale")),
    "wq_b": rows(t("attn.wq_b.weight"), t("attn.wq_b.scale"), TP),
    "wo_a.g0": (lambda w, s: (w[:1024], s[:32]))(
        *rows(t("attn.wo_a.weight"), t("attn.wo_a.scale"), TP)
    ),
    "wo_b": cols(t("attn.wo_b.weight"), t("attn.wo_b.scale"), TP),
    "shared.w1": rows(
        t("ffn.shared_experts.w1.weight"), t("ffn.shared_experts.w1.scale"), TP
    ),
    "shared.w2": cols(
        t("ffn.shared_experts.w2.weight"), t("ffn.shared_experts.w2.scale"), TP
    ),
}


def timed(fn, n=300):
    for _ in range(5):
        fn()
    torch.accelerator.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    g.replay()
    torch.accelerator.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(n):
        g.replay()
    b.record()
    torch.accelerator.synchronize()
    return a.elapsed_time(b) / n * 1000


M = 6
tot_old = tot_new = 0.0
tot_bytes = 0
for name, (w, s) in shapes.items():
    w = w.contiguous().to(dev)
    s = s.contiguous().to(dev)
    n, k = w.shape
    x = (torch.randn(M, k, device=dev) * 0.3).half()
    ref = (
        x.float()
        @ (
            w.float()
            * torch.pow(2.0, s.view(torch.uint8).float() - 127)
            .repeat_interleave(32, 0)[:n]
            .repeat_interleave(32, 1)[:, :k]
        ).t()
    )
    old = fp8_linear.fp8_block32_linear(x, w, s)
    t_old = timed(lambda: fp8_linear.fp8_block32_linear(x, w, s))
    packed = torch.empty(n * k, dtype=torch.uint8, device=dev)
    torch.ops.v41q8.prepack(packed, w.view(torch.uint8))
    scales = (
        (torch.pow(2.0, s.view(torch.uint8).float() - 127) * 256)
        .half()
        .t()
        .contiguous()
    )  # [K/32, N/32]
    out = torch.empty(M, n, dtype=torch.float16, device=dev)
    best = None
    for split in (4, 8, 16, 32):
        if (k // 16) % split:
            continue
        for chains in (1, 2):
            us = timed(
                lambda: torch.ops.v41q8.gemm(out, x, packed, scales, split, chains)
            )
            if best is None or us < best[0]:
                best = (us, split, chains)
    torch.ops.v41q8.gemm(out, x, packed, scales, best[1], best[2])
    e_old = ((old.float() - ref).norm() / ref.norm()).item()
    e_new = ((out.float() - ref).norm() / ref.norm()).item()
    nbytes = n * k + s.numel()
    tot_old += t_old
    tot_new += best[0]
    tot_bytes += nbytes
    print(
        f"{name:10s} N={n:5d} K={k:5d}  old {t_old:6.1f} us ({nbytes / t_old / 1e3:5.0f} GB/s) err {e_old:.1e} | q8 {best[0]:6.1f} us ({nbytes / best[0] / 1e3:5.0f} GB/s) split {best[1]} chains {best[2]} err {e_new:.1e}"
    )
print(
    f"TOTAL old {tot_old:.1f} us  q8 {tot_new:.1f} us  bytes {tot_bytes / 1e6:.1f} MB"
)
