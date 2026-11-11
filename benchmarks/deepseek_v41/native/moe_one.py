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

g2 = torch.Generator().manual_seed(1)
experts = torch.randperm(E, generator=g2)[:20].tolist()
rows = [[experts[(m * 3 + j * 7) % 20] for j in range(6)] for m in range(6)]
ids = torch.tensor(rows, device=dev, dtype=torch.int32)
x = (torch.randn(6, H) * 0.5).half().to(dev)
perm, gid, goff = native._route_plan(ids)
gu = torch.empty(36, 2 * Ip, dtype=torch.float16, device=dev)
op = torch.ops.v41sk.skinny_moe_qpn_sm70
for _ in range(3):
    op(
        x,
        layer_obj.w13_weight,
        layer_obj.w13_weight_scale,
        layer_obj.v41_g13,
        perm,
        gid,
        goff,
        6,
        gu,
        False,
        6,
        16,
        1,
        1,
    )
torch.accelerator.synchronize()
