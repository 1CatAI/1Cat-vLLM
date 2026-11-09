# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Micro-test G1 memory changes against the previous implementations."""

import torch

import vllm.model_executor.kernels.mhc.deepseek_v41_fused as fused
from vllm.model_executor.kernels.mhc.deepseek_v41 import expand_hc
from vllm.model_executor.layers.quantization.gguf_expert_cache import _unroute
from vllm.model_executor.layers.quantization.gguf_shared_prefill import (
    _v41_prefill_accumulate,
)
from vllm.triton_utils import triton

torch.manual_seed(0)
dev = "cuda"
H = 5120
ok = True

# 1. expand_hc in place == out of place.
for T in (1, 7, 300, 3000):
    res = torch.randn(T, 4, H, device=dev) * 3
    x = torch.randn(T, H, device=dev).half()
    post = torch.rand(T, 4, device=dev) * 2
    comb = torch.rand(T, 4, 4, device=dev)
    ref = expand_hc(x, res, post, comb)
    out = expand_hc(x, res.clone(), post, comb, inplace=True)
    same = torch.equal(ref, out)
    ok &= same
    print(
        "expand T=%d equal=%s maxdiff=%.3g" % (T, same, (ref - out).abs().max().item())
    )

# 2. windowed hc_transition == single window.
for T, mix in ((3000, True), (5000, False), (4096, True)):
    res = torch.randn(T, 4, H, device=dev)
    y = torch.randn(T, H, device=dev).half()
    post = torch.rand(T, 4, device=dev) * 2
    comb = torch.rand(T, 4, 4, device=dev)
    cpre = torch.rand(T, 4, device=dev)
    norm = torch.rand(H, device=dev)
    fn = torch.randn(24, 4 * H, device=dev) * 0.01 if mix else None
    scale = torch.rand(3, device=dev) if mix else None
    base = torch.randn(24, device=dev) if mix else None
    outs = []
    for window in (1 << 30, 2048, 1000):
        fused._WINDOW = window
        r = res.clone()
        o, m = fused.hc_transition(r, y, post, comb, cpre, norm, fn, scale, base)
        outs.append((r, o, m))
    for r, o, m in outs[1:]:
        same = torch.equal(r, outs[0][0]) and torch.equal(o, outs[0][1])
        if mix:
            same &= all(torch.equal(a, b) for a, b in zip(m, outs[0][2]))
        ok &= same
        print("transition T=%d mix=%s equal=%s" % (T, mix, same))
fused._WINDOW = 2048

# 3. per-tile accumulation vs the old full down + _unroute.
E, TILE = 384, 32
for T in (1, 13, 2048, 4096):
    ids = torch.stack([torch.randperm(E, device=dev)[:6] for _ in range(T)]).int()
    flat = ids.reshape(-1)
    sorted_ids, order = flat.sort(stable=True)
    offsets = torch.searchsorted(
        sorted_ids,
        torch.arange(E + 1, device=dev, dtype=sorted_ids.dtype),
        out_int32=True,
    )
    inverse = torch.empty_like(order, dtype=torch.int32)
    inverse[order] = torch.arange(order.numel(), device=dev, dtype=torch.int32)
    down = (torch.randn(T * 6, H, device=dev) * 0.5).half()  # sorted-order rows
    ref = torch.empty(T, H, device=dev, dtype=torch.float16)
    ones = torch.ones(T, 6, device=dev)
    _unroute[(T, triton.cdiv(H, 1024))](down, ones, inverse, ref, H, TOPK=6, BLOCK=1024)
    acc = torch.zeros(T, H, device=dev)
    for f in range(0, E, TILE):
        bounds = offsets[f : f + TILE + 1]
        scratch = torch.full_like(down, float("nan"))
        lo, hi = bounds[0].item(), bounds[-1].item()
        scratch[lo:hi] = down[lo:hi]  # only this tile's rows are valid
        _v41_prefill_accumulate[(T, triton.cdiv(H, 1024))](
            scratch, inverse, bounds, acc, T, EXPERTS=TILE, TOPK=6, H=H, BLOCK=1024
        )
    new = acc.half()
    diff = (new.float() - ref.float()).abs()
    rel = diff.max().item() / ref.float().abs().max().item()
    mism = (new != ref).float().mean().item()
    good = torch.isfinite(new).all().item() and rel < 1e-3
    ok &= good
    print(
        "accumulate T=%d max|d|=%.3g rel=%.2g mismatch-frac=%.4f"
        % (T, diff.max().item(), rel, mism)
    )

print("ALL OK" if ok else "FAILED")
