# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""G3 micro-test: row-split down (gate_up7 + down_rows) vs the K-split moe7, 4 ranks simulated."""

import glob
import os
import sys

import gguf
import numpy as np
import torch

lib = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "/home/ymzx/ds41-dev/tools/v41k_g3/build/v41k.so"
)
torch.ops.load_library(lib)
dev = torch.device("cuda")
SIZES = {16: (256, 66), 17: (256, 74), 29: (256, 56)}

tensors = {}
for f in sorted(
    glob.glob("/home/ymzx/ds41-models/gguf/DeepSeek-V4.1-Flash-*-0000*-of-00004.gguf")
):
    for t in gguf.GGUFReader(f).tensors:
        if "exps" in t.name:
            tensors[t.name] = t
picked = {}
for name, t in tensors.items():
    if name.endswith("ffn_down_exps.weight"):
        layer = int(name.split(".")[1])
        kind = int(t.tensor_type)
        picked.setdefault(kind, layer)
print("layers by down type", picked)

E = 12
torch.manual_seed(0)
ok = True
for dkind, layer in sorted(picked.items()):
    g = tensors[f"blk.{layer}.ffn_gate_exps.weight"]
    u = tensors[f"blk.{layer}.ffn_up_exps.weight"]
    d = tensors[f"blk.{layer}.ffn_down_exps.weight"]
    gkind = int(g.tensor_type)
    experts = np.random.default_rng(layer).choice(g.data.shape[0], E, replace=False)
    gate = torch.from_numpy(np.ascontiguousarray(g.data[experts])).to(
        dev
    )  # [E, 2304, rb]
    up = torch.from_numpy(np.ascontiguousarray(u.data[experts])).to(dev)
    down = torch.from_numpy(np.ascontiguousarray(d.data[experts])).to(
        dev
    )  # [E, 5120, rb]
    block, bb = SIZES[dkind]
    gbook = torch.ops.v41k.table(gkind, dev)
    dbook = torch.ops.v41k.table(dkind, dev)
    for m in (1, 3, 6):
        x = (torch.randn(m, 5120, device=dev) * 0.5).half()
        # Distinct per token; expert 0 shared by every token (<= 6 routes).
        ids = torch.stack(
            [
                torch.cat(
                    [
                        torch.zeros(1, device=dev, dtype=torch.long),
                        torch.randperm(E - 1, device=dev)[:5] + 1,
                    ]
                )
                for _ in range(m)
            ]
        ).int()
        prob = torch.rand(m, 6, device=dev)
        prob = (prob / prob.sum(1, keepdim=True) * 1.5).contiguous()
        ids = ids.contiguous()
        xq = torch.empty((m, 160, 32), dtype=torch.int8, device=dev)
        xd = torch.empty((m, 160), dtype=torch.float32, device=dev)
        torch.ops.v41k.q8_aligned(xq, xd, x)
        routes = m * 6
        stride = (routes * 648 + 255) // 256 * 256
        gathered = torch.zeros(4, stride, dtype=torch.uint8, device=dev)
        ref = torch.zeros(m, 5120, device=dev)
        new = torch.zeros(m, 5120, device=dev)
        rows, warps = (16, 8) if dkind == 29 else (8, 4)
        nrows, nwarps = [int(v) for v in os.environ.get("ROWS_TILE", "4,8").split(",")]
        for r in range(4):
            gr = gate[:, 576 * r : 576 * (r + 1)].contiguous()
            ur = up[:, 576 * r : 576 * (r + 1)].contiguous()
            first, end = 576 * r // block, (576 * (r + 1) + block - 1) // block
            left = 576 * r - first * block
            dc = torch.zeros(
                E, 5120, (end - first) * bb + 512, dtype=torch.uint8, device=dev
            )
            dc = dc[:, :, : (end - first) * bb]
            dc.copy_(down[:, :, first * bb : end * bb])
            dc = dc.contiguous()
            hq = torch.empty((routes, 18, 32), dtype=torch.int8, device=dev)
            hd = torch.empty((routes, 18), dtype=torch.float32, device=dev)
            out = torch.empty(m, 5120, dtype=torch.float16, device=dev)
            plan = torch.zeros(512, dtype=torch.int32, device=dev)
            torch.ops.v41k.moe7(
                out,
                xq,
                xd,
                hq,
                hd,
                ids,
                prob,
                gr,
                ur,
                dc,
                gkind,
                dkind,
                left,
                gbook,
                dbook,
                rows,
                warps,
                plan,
            )
            ref += out.float()
            # Row mode: gate/up into this rank's packed slot.
            q = gathered[r, : routes * 576].view(torch.int8).view(routes, 18, 32)
            dd = (
                gathered[r, routes * 576 : routes * 648]
                .view(torch.float32)
                .view(routes, 18)
            )
            plan2 = torch.zeros(512, dtype=torch.int32, device=dev)
            torch.ops.v41k.gate_up7(
                xq, xd, q, dd, ids, prob, gr, ur, gkind, gbook, plan2
            )
            same_h = torch.equal(q, hq) and torch.equal(dd, hd)
            ok &= same_h
        for r in range(4):
            dr = torch.zeros(
                E, 1280, down.shape[2] + 64, dtype=torch.uint8, device=dev
            )[:, :, : down.shape[2]]
            dr.copy_(down[:, 1280 * r : 1280 * (r + 1)])
            dr = dr.contiguous()
            out = torch.zeros(m, 5120, dtype=torch.float16, device=dev)
            plan2 = torch.zeros(512, dtype=torch.int32, device=dev)
            torch.ops.v41k.gate_up7(
                xq,
                xd,
                torch.empty((routes, 18, 32), dtype=torch.int8, device=dev),
                torch.empty((routes, 18), dtype=torch.float32, device=dev),
                ids,
                prob,
                gate[:, :576].contiguous(),
                up[:, :576].contiguous(),
                gkind,
                gbook,
                plan2,
            )  # only fills the plan
            torch.ops.v41k.down_rows(
                out,
                gathered,
                stride,
                routes,
                dr,
                dkind,
                1280 * r,
                dbook,
                nrows,
                nwarps,
                plan2,
            )
            others = torch.ones(5120, dtype=torch.bool, device=dev)
            others[1280 * r : 1280 * (r + 1)] = False
            ok &= bool((out[:, others] == 0).all())
            new += out.float()
        diff = (new - ref).abs()
        rel = (diff.norm() / ref.norm()).item()
        good = torch.isfinite(new).all().item() and rel < 2e-3
        ok &= good
        print(
            "down %d gate %d layer %d m=%d same_h=%s rel=%.2e max=%.3g |ref|max=%.3g"
            % (
                dkind,
                gkind,
                layer,
                m,
                same_h,
                rel,
                diff.max().item(),
                ref.abs().max().item(),
            )
        )
print("ALL OK" if ok else "FAILED")
