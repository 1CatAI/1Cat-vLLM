# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import sys

import torch

sys.path.insert(0, sys.argv[1])
from vllm.model_executor.kernels.mhc.deepseek_v41_router import route_topk
from vllm.models.deepseek_v41 import native

dev = torch.device("cuda", 0)
g = torch.Generator(device="cpu").manual_seed(0)
bad = 0
for trial in range(400):
    M = 1 + trial % 8
    logits = (torch.randn(M, 384, generator=g) * (1 + trial % 5)).to(dev)
    if trial % 7 == 0:
        logits[:, :20] = logits[:, :1]  # ties
    bias = (torch.randn(384, generator=g) * 0.1).to(dev)
    p1, i1 = route_topk(logits, bias, 6, 1.5)
    p2, i2, plan = native.route_and_plan(logits, bias, 6, 1.5)
    ref = native._route_plan_torch(i1.int())
    ok = (
        torch.equal(p1, p2)
        and torch.equal(i1.int(), i2)
        and all(torch.equal(a, b) for a, b in zip(plan, ref))
    )
    bad += not ok
print("route+plan fused mismatches:", bad, "of 400")
x = torch.randn(6, 384, device=dev)


def t(fn, n=500):
    for _ in range(10):
        fn()
    torch.accelerator.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        fn()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(n):
        gr.replay()
    b.record()
    torch.accelerator.synchronize()
    return a.elapsed_time(b) / n * 1000


bias = torch.zeros(384, device=dev)
print(
    "old route+convert+plan us",
    round(t(lambda: native._route_plan(route_topk(x, bias, 6, 1.5)[1].int())), 1),
)
print("fused us", round(t(lambda: native.route_and_plan(x, bias, 6, 1.5)), 1))
