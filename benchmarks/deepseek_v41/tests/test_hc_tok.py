# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Token-blocked HC transition vs the per-token kernels (T > 8)."""

import torch

import vllm.model_executor.kernels.mhc.deepseek_v41_fused as fused

torch.manual_seed(0)
dev = "cuda"
H = 5120
ok = True
for T, mix in ((37, True), (2048, True), (8192, True), (8192, False)):
    res = torch.randn(T, 4, H, device=dev)
    y = torch.randn(T, H, device=dev).half()
    post = torch.rand(T, 4, device=dev) * 2
    comb = torch.rand(T, 4, 4, device=dev)
    cpre = torch.rand(T, 4, device=dev)
    norm = torch.rand(H, device=dev)
    fn = torch.randn(24, 4 * H, device=dev) * 0.01 if mix else None
    scale = torch.rand(3, device=dev) if mix else None
    base = torch.randn(24, device=dev) if mix else None
    outs, times = [], []
    for blocked in (False, True):
        fused._TOKEN_BLOCKED = blocked
        r = res.clone()
        o, m = fused.hc_transition(r, y, post, comb, cpre, norm, fn, scale, base)
        outs.append((r, o, m))
        rr = res.clone()
        torch.accelerator.synchronize()
        a, b = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        a.record()
        for _ in range(5):
            fused.hc_transition(rr, y, post, comb, cpre, norm, fn, scale, base)
        b.record()
        torch.accelerator.synchronize()
        times.append(a.elapsed_time(b) / 5)
    (r0, o0, m0), (r1, o1, m1) = outs
    same_res = torch.equal(r0, r1)
    dout = (o0.float() - o1.float()).abs().max().item()
    dm = max((a - b).abs().max().item() for a, b in zip(m0, m1)) if mix else 0.0
    good = same_res and dout < 2e-2 and dm < 1e-4
    ok &= good
    print(
        "T=%d mix=%s residual_equal=%s out_maxdiff=%.2e mix_maxdiff=%.2e old %.2f ms new %.2f ms"
        % (T, mix, same_res, dout, dm, times[0], times[1])
    )
print("ALL OK" if ok else "FAILED")
