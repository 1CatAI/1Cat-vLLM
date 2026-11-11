# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K1/B1: native V4.1 MXFP4 experts on v41sk vs an FP32 dequant reference.

Real layer weights (TP4 rank slice), M = 1..8 verifier rows, realistic overlap.
"""

import sys

import torch
from safetensors import safe_open

sys.path.insert(0, sys.argv[1])  # source tree
from vllm.model_executor.layers.fused_moe.experts.skinny_sm70_moe import (
    qpn_prepack,
    rebase_e8m0_for_fp16,
)
from vllm.models.deepseek_v41 import native

layer_file, layer, rank = (
    sys.argv[2],
    int(sys.argv[3]),
    int(sys.argv[4]) if len(sys.argv) > 4 else 0,
)
dev = torch.device("cuda", 0)
E, H, I, TP = 384, 5120, 2304, 4
Ip = I // TP
f = safe_open(layer_file, "pt", device="cpu")


def t(n):
    return f.get_tensor(n).view(torch.uint8)


w13 = torch.empty(E, 2 * Ip, H // 2, dtype=torch.uint8)
s13 = torch.empty(E, 2 * Ip, H // 32, dtype=torch.uint8)
w2 = torch.empty(E, H, Ip // 2, dtype=torch.uint8)
s2 = torch.empty(E, H, Ip // 32, dtype=torch.uint8)
for e in range(E):
    p = f"layers.{layer}.ffn.experts.{e}."
    rows = slice(rank * Ip, (rank + 1) * Ip)
    w13[e, :Ip] = t(p + "w1.weight")[rows]
    w13[e, Ip:] = t(p + "w3.weight")[rows]
    s13[e, :Ip] = t(p + "w1.scale")[rows]
    s13[e, Ip:] = t(p + "w3.scale")[rows]
    w2[e] = t(p + "w2.weight")[:, rank * Ip // 2 : (rank + 1) * Ip // 2]
    s2[e] = t(p + "w2.scale")[:, rank * Ip // 32 : (rank + 1) * Ip // 32]
w13, s13, w2, s2 = (x.to(dev) for x in (w13, s13, w2, s2))
LUT = torch.tensor(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], device=dev
)


def deq(codes, scales):  # [N, K/2] u8, [N, K/32] -> fp32 [N, K]
    v = torch.stack([LUT[(codes & 15).long()], LUT[(codes >> 4).long()]], -1).flatten(1)
    return v * torch.pow(2.0, scales.float() - 127).repeat_interleave(32, 1)


ref_w = {}


def ref(x, ids, probs):
    out = torch.zeros(x.shape[0], H, device=dev)
    for m in range(x.shape[0]):
        for j in range(ids.shape[1]):
            e = int(ids[m, j])
            if e not in ref_w:
                ref_w[e] = (deq(w13[e], s13[e]), deq(w2[e], s2[e]))
            a, b = ref_w[e]
            gu = a @ x[m].float()
            act = (
                (
                    torch.nn.functional.silu(gu[:Ip].clamp(max=10))
                    * gu[Ip:].clamp(-10, 10)
                    * probs[m, j]
                )
                .half()
                .float()
            )
            out[m] += b @ act
    return out


# keep raw copies for the reference before in-place prepack
w13r, s13r, w2r, s2r = w13.clone(), s13.clone(), w2.clone(), s2.clone()


class L:
    pass


layer_obj = L()
layer_obj.w13_weight, layer_obj.w2_weight = w13, w2
layer_obj.w13_weight_scale, layer_obj.w2_weight_scale = s13, s2
native.load_v41sk()
layer_obj.v41_g13 = rebase_e8m0_for_fp16(s13).to(dev)
layer_obj.v41_g2 = rebase_e8m0_for_fp16(s2).to(dev)
for e in range(E):
    for c, s in ((w13, s13), (w2, s2)):
        qc, qs = qpn_prepack(c[e], s[e], 32)
        c[e].view(-1).copy_(qc)
        s[e].view(-1).copy_(qs)
layer_obj.v41_split13 = native._splitk(H)
layer_obj.v41_split2 = native._splitk(Ip)
w13, s13, w2, s2 = w13r, s13r, w2r, s2r  # reference uses raw layout
method = native.V41NativeMoEMethod.__new__(native.V41NativeMoEMethod)
g = torch.Generator(device="cpu").manual_seed(0)
per_byte = 2 * Ip * (H // 2 + H // 32) + H * (Ip // 2 + Ip // 32)
print(
    f"rank {rank} layer {layer}: split {layer_obj.v41_split13}/{layer_obj.v41_split2}, {per_byte / 1e6:.2f} MB per expert slice"
)
for M in (1, 2, 4, 6, 8):
    # realistic overlap: top-6 per row from a skewed distribution
    pop = torch.distributions.Categorical(
        torch.tensor([1.0 / (i + 1) ** 0.7 for i in range(E)])
    )
    ids = (
        torch.stack(
            [
                torch.randperm(E, generator=g)[:6]
                if M == 1
                else torch.unique(pop.sample((40,)))[
                    torch.randperm(len(torch.unique(pop.sample((40,)))))
                ][:6]
                for _ in range(M)
            ]
        )
        if False
        else None
    )
    rows = []
    for _ in range(M):
        r = []
        while len(r) < 6:
            e = int(pop.sample())
            if e not in r:
                r.append(e)
        rows.append(r)
    ids = torch.tensor(rows, device=dev, dtype=torch.int32)
    probs = torch.rand(M, 6, generator=g).to(dev) * 0.5
    x = (torch.randn(M, H, generator=g) * 0.5).half().to(dev)
    got = method.apply(layer_obj, x, probs, ids)
    want = ref(x, ids, probs)
    err = ((got - want).norm() / want.norm()).item()
    distinct = len(set(ids.flatten().tolist()))
    for _ in range(5):
        method.apply(layer_obj, x, probs, ids)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        method.apply(layer_obj, x, probs, ids)
    for _ in range(10):
        graph.replay()
    torch.accelerator.synchronize()
    n = 200
    a = torch.cuda.Event(True)
    b = torch.cuda.Event(True)
    a.record()
    for _ in range(n):
        graph.replay()
    b.record()
    torch.accelerator.synchronize()
    us = a.elapsed_time(b) / n * 1000
    gbps = distinct * per_byte / (us * 1e-6) / 1e9
    print(
        f"M={M}: distinct {distinct:2d}  rel_err {err:.2e}  {us:7.1f} us/layer (graph)  {gbps:6.1f} GB/s on unique-expert bytes"
    )


# ---- breakdown at M=6 with ~20 distinct experts (real-traffic mean)
def timed(fn, n=200):
    for _ in range(5):
        fn()
    torch.accelerator.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        fn()
    gr.replay()
    torch.accelerator.synchronize()
    a = torch.cuda.Event(True)
    b = torch.cuda.Event(True)
    a.record()
    for _ in range(n):
        gr.replay()
    b.record()
    torch.accelerator.synchronize()
    return a.elapsed_time(b) / n * 1000


experts = torch.randperm(E, generator=g)[:20].tolist()
rows = [[experts[(m * 3 + j * 7) % 20] for j in range(6)] for m in range(6)]
for r in rows:
    assert len(set(r)) == 6
ids = torch.tensor(rows, device=dev, dtype=torch.int32)
distinct = len(set(ids.flatten().tolist()))
x = (torch.randn(6, H, generator=g) * 0.5).half().to(dev)
probs = torch.rand(6, 6, generator=g).to(dev)
perm, gid, goff = native._route_plan(ids)
gu = torch.empty(36, 2 * Ip, dtype=torch.float16, device=dev)
act = torch.empty(36, Ip, dtype=torch.float16, device=dev)
dn = torch.empty(36, H, dtype=torch.float16, device=dev)
w13p, s13p, w2p, s2p = (
    layer_obj.w13_weight,
    layer_obj.w13_weight_scale,
    layer_obj.w2_weight,
    layer_obj.w2_weight_scale,
)
op = torch.ops.v41sk.skinny_moe_qpn_sm70
b13 = distinct * 2 * Ip * (H // 2 + H // 32)
b2 = distinct * H * (Ip // 2 + Ip // 32)
print(
    f"breakdown M=6 distinct {distinct}: bytes w13 {b13 / 1e6:.1f} MB w2 {b2 / 1e6:.1f} MB"
)
for split, nacc in ((16, 1), (16, 2), (8, 1), (8, 2), (10, 1)):
    try:
        us = timed(
            lambda: op(
                x,
                w13p,
                s13p,
                layer_obj.v41_g13,
                perm,
                gid,
                goff,
                6,
                gu,
                False,
                6,
                split,
                nacc,
                1,
            )
        )
        print(
            f"  w13 split {split} nacc {nacc}: {us:6.1f} us  {b13 / us / 1e3:6.1f} GB/s"
        )
    except Exception as ex:
        print("  w13", split, nacc, "ERR", str(ex)[:60])
for split in (4, 6, 9, 12, 18):
    try:
        us = timed(
            lambda: op(
                act,
                w2p,
                s2p,
                layer_obj.v41_g2,
                perm,
                gid,
                goff,
                6,
                dn,
                True,
                6,
                split,
                1,
                1,
            )
        )
        print(f"  w2 split {split}: {us:6.1f} us  {b2 / us / 1e3:6.1f} GB/s")
    except Exception as ex:
        print("  w2", split, "ERR", str(ex)[:60])
print(f"  route plan: {timed(lambda: native._route_plan(ids)):6.1f} us")


def actf():
    gate, up = gu.float().split(Ip, dim=-1)
    act.copy_(
        (
            torch.nn.functional.silu(gate.clamp(max=10))
            * up.clamp(-10, 10)
            * probs.reshape(-1, 1)
        ).half()
    )


print(f"  activation: {timed(actf):6.1f} us")
print(f"  route sum: {timed(lambda: dn.view(6, 6, -1).float().sum(1)):6.1f} us")
print(f"  full apply: {timed(lambda: method.apply(layer_obj, x, probs, ids)):6.1f} us")
# copy bandwidth reference
big = torch.empty(512 << 20, dtype=torch.uint8, device=dev)
big2 = torch.empty_like(big)
us = timed(lambda: big2.copy_(big), 20)
print(f"  D2D copy: {2 * big.numel() / us / 1e3:6.1f} GB/s (read+write)")
# ---- plan kernel equivalence on random routes
for trial in range(200):
    M = 1 + trial % 8
    ids_t = (
        torch.stack([torch.randperm(E, generator=g)[:6] for _ in range(M)])
        .int()
        .to(dev)
    )
    if trial % 3 == 0:
        ids_t = torch.randint(0, 12, (M, 6), generator=g).int().to(dev)
    a1 = native._route_plan(ids_t)
    a2 = native._route_plan_torch(ids_t)
    assert all(torch.equal(u, v) for u, v in zip(a1, a2)), (trial, ids_t, a1, a2)
print("plan kernel == torch plan on 200 random routes")
# ---- plan kernel equivalence on random routes
for trial in range(200):
    M = 1 + trial % 8
    ids_t = (
        torch.stack([torch.randperm(E, generator=g)[:6] for _ in range(M)])
        .int()
        .to(dev)
    )
    if trial % 3 == 0:
        ids_t = torch.randint(0, 12, (M, 6), generator=g).int().to(dev)
    a1 = native._route_plan(ids_t)
    a2 = native._route_plan_torch(ids_t)
    assert all(torch.equal(u, v) for u, v in zip(a1, a2)), (trial, ids_t, a1, a2)
print("plan kernel == torch plan on 200 random routes")
