# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gate projection [M,5120] x [384,5120]^T (BF16 coefficients, FP32 out):
_row_projection (current) vs _col_projection2 configs; graph-timed, 20 calls/graph."""

import sys

import torch

sys.path.insert(0, sys.argv[1])
from safetensors import safe_open

from vllm.model_executor.kernels.mhc import deepseek_v41 as k

f = safe_open(sys.argv[2], "pt", device="cpu")
w = f.get_tensor("layers.3.ffn.gate.weight").cuda().contiguous()  # BF16
torch.manual_seed(0)


def timed(fn, n=30, inner=20):
    for _ in range(3):
        fn()
    torch.accelerator.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(inner):
            fn()
    g.replay()
    torch.accelerator.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(n):
        g.replay()
    b.record()
    torch.accelerator.synchronize()
    return a.elapsed_time(b) / (n * inner) * 1000


for M in (1, 6):
    x = (torch.randn(M, 5120, device="cuda") * 0.5).half()
    out = torch.empty(M, 384, dtype=torch.float32, device="cuda")
    row = lambda: k._row_projection[(M, 384)](
        x,
        w,
        out,
        K=5120,
        N=384,
        NORMALIZE=False,
        EPS=1e-20,
        BLOCK=1024,
        enable_fp_fusion=False,
    )
    row()
    ref = out.clone()
    t = timed(row)
    print(f"M={M} row_projection {t:6.2f} us ({w.numel() * 2 / t / 1e3:.0f} GB/s eff)")
    for colb in ():
        for blk in (256, 512):
            for nw in (2, 4, 8):
                o2 = torch.empty_like(out)
                fn = lambda: k._col_projection2[(384 // colb,)](
                    x,
                    w,
                    o2,
                    M=M,
                    K=5120,
                    N=384,
                    NORMALIZE=False,
                    EPS=1e-20,
                    COLB=colb,
                    BLOCK=blk,
                    num_warps=nw,
                    enable_fp_fusion=False,
                )
                try:
                    fn()
                    t2 = timed(fn)
                except Exception as e:
                    print("  err", colb, blk, nw, str(e)[:60])
                    continue
                d = (o2 - ref).abs().max().item()
                rel = d / ref.abs().max().item()
                print(
                    f"  col COLB={colb} BLOCK={blk} warps={nw}: {t2:6.2f} us  maxdiff {d:.2e} rel {rel:.1e}"
                )
torch.ops.load_library(sys.argv[1] + "/benchmarks/deepseek_v41/v41k/build/v41sk.so")
for M in (1, 6):
    x = (torch.randn(M, 5120, device="cuda") * 0.5).half()
    out = torch.empty(M, 384, dtype=torch.float32, device="cuda")
    o2 = torch.empty_like(out)
    k._row_projection[(M, 384)](
        x,
        w,
        out,
        K=5120,
        N=384,
        NORMALIZE=False,
        EPS=1e-20,
        BLOCK=1024,
        enable_fp_fusion=False,
    )
    fn = lambda: torch.ops.v41sk.gate_gemv(o2, x, w)
    fn()
    t = timed(fn)
    x6 = x if M == 1 else x[:1].contiguous()
    o1 = torch.empty(1, 384, device="cuda")
    torch.ops.v41sk.gate_gemv(o1, x6, w)
    print(
        f"M={M} cuda gate_gemv {t:6.2f} us ({w.numel() * 2 / t / 1e3:.0f} GB/s)  maxdiff vs row {(o2 - out).abs().max().item():.2e}  row0 M-invariant {torch.equal(o1[0], o2[0])}"
    )
