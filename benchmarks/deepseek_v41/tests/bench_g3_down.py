# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""G3 kernel timing: moe7 (K-split) vs gate_up7 + down_rows (row-split), one rank, M=6."""

import torch

torch.ops.load_library("/home/ymzx/ds41-dev/tools/v41k_g3/build/v41k.so")
dev = torch.device("cuda")
SIZES = {16: 66, 17: 74, 29: 56}
E, m = 128, 6
routes = m * 6
torch.manual_seed(0)


def timed(fn, n=300):
    for _ in range(20):
        fn()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(n):
        fn()
    b.record()
    torch.accelerator.synchronize()
    return a.elapsed_time(b) / n * 1000


def rand_bank(kind, experts, rows, blocks):
    # Random qs bytes; fp16 scales forced small so dots stay finite.
    bank = torch.randint(
        0, 256, (experts, rows, blocks, SIZES[kind]), dtype=torch.uint8, device=dev
    )
    if kind in (16, 17):
        bank[..., 1] = 0x20
    else:
        bank[..., 49] &= 0x0F
        bank[..., 51] &= 0x0F
        bank[..., 53] &= 0x0F
        bank[..., 55] = 0x20
    return bank.view(experts, rows, blocks * SIZES[kind]).contiguous()


for gkind, dkind in ((29, 29), (29, 16), (16, 17)):
    gate = rand_bank(gkind, E, 576, 20)
    up = rand_bank(gkind, E, 576, 20)
    dcol = rand_bank(dkind, E, 5120, 3)
    drow = rand_bank(dkind, E, 1280, 9)
    gbook, dbook = torch.ops.v41k.table(gkind, dev), torch.ops.v41k.table(dkind, dev)
    x = (torch.randn(m, 5120, device=dev) * 0.5).half()
    ids = (
        torch.stack([torch.randperm(E, device=dev)[:6] for _ in range(m)])
        .int()
        .contiguous()
    )
    prob = torch.full((m, 6), 0.25, device=dev)
    xq = torch.empty((m, 160, 32), dtype=torch.int8, device=dev)
    xd = torch.empty((m, 160), dtype=torch.float32, device=dev)
    torch.ops.v41k.q8_aligned(xq, xd, x)
    hq = torch.empty((routes, 18, 32), dtype=torch.int8, device=dev)
    hd = torch.empty((routes, 18), dtype=torch.float32, device=dev)
    out = torch.empty(m, 5120, dtype=torch.float16, device=dev)
    plan = torch.zeros(512, dtype=torch.int32, device=dev)
    rows, warps = (16, 8) if dkind == 29 else (8, 4)
    t_moe7 = timed(
        lambda: torch.ops.v41k.moe7(
            out,
            xq,
            xd,
            hq,
            hd,
            ids,
            prob,
            gate,
            up,
            dcol,
            gkind,
            dkind,
            64,
            gbook,
            dbook,
            rows,
            warps,
            plan,
        )
    )
    t_gu = timed(
        lambda: torch.ops.v41k.gate_up7(
            xq, xd, hq, hd, ids, prob, gate, up, gkind, gbook, plan
        )
    )
    stride = (routes * 648 + 255) // 256 * 256
    gathered = torch.zeros(4 * stride, dtype=torch.uint8, device=dev)
    res = [f"g{gkind}/d{dkind}: moe7 {t_moe7:.1f} us, gate_up7 {t_gu:.1f} us"]
    for r_tile, w in ((4, 8), (2, 8), (2, 4), (8, 8)):
        o = torch.zeros(m, 5120, dtype=torch.float16, device=dev)
        t = timed(
            lambda: torch.ops.v41k.down_rows(
                o, gathered, stride, routes, drow, dkind, 0, dbook, r_tile, w, plan
            )
        )
        res.append(f"down_rows[{r_tile}x{w}] {t:.1f}")
    print(" | ".join(res))
