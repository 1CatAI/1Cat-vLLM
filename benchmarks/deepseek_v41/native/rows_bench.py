# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batched-decode dense FP8 (M = 12..64) on layer-10 TP-rank-0 shapes:
v41q8.gemm_rows (tuned warps/parts/row tiles) vs dequant + cuBLAS vs the
Triton 16-row tiles; L2 flushed before every call, error vs FP32 dequant.
Usage (server stopped): CUDA_VISIBLE_DEVICES=0 python rows_bench.py MODEL_DIR
"""

import os
import sys

import torch
from safetensors import safe_open

from vllm.models.deepseek_v41 import fp8_linear

assert fp8_linear.q8_ready()
dev = torch.device("cuda")
f = safe_open(os.path.join(sys.argv[1], "layer-10.safetensors"), "pt", device="cpu")
t = lambda n: f.get_tensor("layers.10." + n)  # noqa: E731
TP = 4


def rows(w, s):
    n = w.shape[0] // TP
    return w[:n].contiguous(), s[: n // 32].contiguous()


def cols(w, s):
    k = w.shape[1] // TP
    return w[:, :k].contiguous(), s[:, : k // 32].contiguous()


qa, qs = t("attn.wq_a.weight"), t("attn.wq_a.scale")
ka, ks = t("attn.wkv.weight"), t("attn.wkv.scale")
shapes = {
    "wq_a+wkv 1792x5120": (torch.cat([qa, ka]), torch.cat([qs, ks])),
    "wq_b 8192x1280": rows(t("attn.wq_b.weight"), t("attn.wq_b.scale")),
    "wo_b 5120x2048": cols(t("attn.wo_b.weight"), t("attn.wo_b.scale")),
    "shared w13 1152x5120": (
        torch.cat(
            [
                rows(
                    t("ffn.shared_experts.w1.weight"), t("ffn.shared_experts.w1.scale")
                )[0],
                rows(
                    t("ffn.shared_experts.w3.weight"), t("ffn.shared_experts.w3.scale")
                )[0],
            ]
        ),
        torch.cat(
            [
                rows(
                    t("ffn.shared_experts.w1.weight"), t("ffn.shared_experts.w1.scale")
                )[1],
                rows(
                    t("ffn.shared_experts.w3.weight"), t("ffn.shared_experts.w3.scale")
                )[1],
            ]
        ),
    ),
    "shared w2 5120x576": cols(
        t("ffn.shared_experts.w2.weight"), t("ffn.shared_experts.w2.scale")
    ),
}
big = torch.randn(16 * 2**20, device=dev)
sink = torch.empty(1, device=dev)


def timed(fn, n=20):
    def run():
        torch.sum(big, 0, out=sink)
        fn()

    for _ in range(3):
        run()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    torch.accelerator.synchronize()
    flush = torch.cuda.CUDAGraph()
    with torch.cuda.graph(flush):
        torch.sum(big, 0, out=sink)
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(n):
        flush.replay()
    b.record()
    torch.accelerator.synchronize()
    base = a.elapsed_time(b) / n
    a.record()
    for _ in range(n):
        graph.replay()
    b.record()
    torch.accelerator.synchronize()
    return (a.elapsed_time(b) / n - base) * 1000


for name, (w, s) in shapes.items():
    w, s = w.cuda(), s.cuda()
    n, k = w.shape
    prepared = fp8_linear.q8_prepare(w, s)
    packed, half_scales, _ = prepared
    tickets = torch.zeros(n // 32, dtype=torch.int32, device=dev)
    ref_w = (
        w.float()
        * (2.0 ** (s.view(torch.uint8).float() - 127))
        .repeat_interleave(32, 0)
        .repeat_interleave(32, 1)[:n, :k]
    )
    print(f"== {name} ({n * k / 1e6:.1f} MB)", flush=True)
    for m in (12, 24, 48, 64):
        x = (torch.randn(m, k, device=dev) * 0.5).half()
        ref = x.float() @ ref_w.T
        out = torch.empty(m, n, dtype=torch.float16, device=dev)
        rt = (m + 7) // 8
        rt = 2 if rt <= 2 else 4 if rt <= 4 else 8
        best = None
        for warps in (4, 8, 16):
            if warps * rt > 32:
                continue
            for parts in (1, 2, 4, 5, 8, 10, 16, 20):
                if (k // 16) % (warps * parts):
                    continue
                fn = lambda: torch.ops.v41q8.gemm_rows(  # noqa: E731
                    out, x, packed, half_scales, tickets, warps, parts, rt
                )
                try:
                    us = timed(fn)
                except Exception as e:
                    print("   ", warps, parts, rt, str(e)[:80])
                    continue
                if best is None or us < best[0]:
                    best = (us, warps, parts)
        fn = lambda: torch.ops.v41q8.gemm_rows(  # noqa: E731
            out, x, packed, half_scales, tickets, best[1], best[2], rt
        )
        fn()
        err = float((out.float() - ref).norm() / ref.norm())
        dq = timed(lambda: fp8_linear._fp8_dequant_mm(x, w, s, None))
        tri = timed(lambda: fp8_linear.fp8_block32_linear(x, w, s))
        print(
            f"  M={m:2d}: rows {best[0]:6.1f} us (W{best[1]} P{best[2]} RT{rt}, "
            f"{n * k / best[0] / 1e3:4.0f} GB/s, rel err {err:.1e}) | "
            f"dequant+cuBLAS {dq:6.1f} us | triton {tri:6.1f} us",
            flush=True,
        )
