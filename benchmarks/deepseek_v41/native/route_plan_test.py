# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bitwise check and timing: v41sk.route_plan vs _route_i32 + _plan_kernel."""

import os
import sys
import time

import torch

sys.path.insert(0, os.path.expanduser("~/v41/src"))
os.environ["VLLM_V41_ROUTE_CUDA"] = "0"
from vllm.models.deepseek_v41 import native as nt

nt.load_v41sk()
torch.manual_seed(0)
dev = "cuda"
bad = 0
cases = 0
for E, K in ((384, 6), (128, 3)):
    bias = (torch.randn(E, device=dev) * 0.05).float()
    for T in range(1, 9):
        if T * K >= 64:
            continue
        for trial in range(300):
            if trial % 3 == 0:
                logits = torch.randn(T, E, device=dev) * 3
            elif trial % 3 == 1:
                logits = (torch.randn(T, E, device=dev) * 4 - 2).half().float()  # ties
            else:
                logits = (
                    torch.randint(-3, 3, (T, E), device=dev).float() * 0.5
                )  # many ties
            if trial == 7:
                logits[0, :5] = 25.0
            nt._ROUTE_CUDA = False
            p0, i0, (q0, e0, o0) = nt.route_and_plan(logits, bias, K, 1.5)
            nt._ROUTE_CUDA = True
            p1, i1, (q1, e1, o1) = nt.route_and_plan(logits, bias, K, 1.5)
            cases += 1
            ok = (
                torch.equal(p0.view(torch.int32), p1.view(torch.int32))
                and torch.equal(i0, i1)
                and torch.equal(q0, q1)
                and torch.equal(e0, e1)
                and torch.equal(o0, o1)
            )
            if not ok:
                bad += 1
                if bad <= 3:
                    print(
                        "MISMATCH",
                        E,
                        K,
                        T,
                        trial,
                        (p0 - p1).abs().max().item(),
                        torch.equal(i0, i1),
                        torch.equal(q0, q1),
                        torch.equal(e0, e1),
                        torch.equal(o0, o1),
                    )
print(f"cases {cases} mismatches {bad}")
logits = torch.randn(6, 384, device=dev) * 3
bias = torch.zeros(384, device=dev)
for flag in (False, True):
    nt._ROUTE_CUDA = flag
    g = torch.cuda.CUDAGraph()
    nt.route_and_plan(logits, bias, 6, 1.5)
    torch.accelerator.synchronize()
    with torch.cuda.graph(g):
        for _ in range(50):
            nt.route_and_plan(logits, bias, 6, 1.5)
    g.replay()
    torch.accelerator.synchronize()
    t = time.perf_counter()
    for _ in range(20):
        g.replay()
    torch.accelerator.synchronize()
    print(
        "cuda" if flag else "triton",
        f"{(time.perf_counter() - t) / 1000 * 1e6:.2f} us/call (graph)",
    )
