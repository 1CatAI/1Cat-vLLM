# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Research only: original FP16 Q/K/V, FP32 probability and PV reduction."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def part(
    Q,
    K,
    V,
    T,
    L,
    P,
    MX,
    SM,
    KS0: tl.constexpr,
    KS1: tl.constexpr,
    KS2: tl.constexpr,
    VS0: tl.constexpr,
    VS1: tl.constexpr,
    VS2: tl.constexpr,
    TS: tl.constexpr,
    PAGE: tl.constexpr,
    PARTS: tl.constexpr,
    BK: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    z = tl.program_id(2)
    qr = z // PARTS
    part = z % PARTS
    length = tl.load(L + b)
    qpos = length - 8 + qr
    lo = tl.maximum(0, length - 8 - 2047)
    key = lo + part * BK + tl.arange(0, BK)
    dim = tl.arange(0, 128)
    valid = (key >= qpos - 2047) & (key <= qpos + 2047) & (key < length) & (key >= 0)
    physical = tl.load(T + b * TS + key // PAGE, valid, other=0)
    q = tl.load(Q + ((b * 8 + qr) * 8 + h) * 128 + dim).to(tl.float32)
    k = tl.load(
        K
        + physical[:, None] * KS0
        + (key % PAGE)[:, None] * KS1
        + (h // 4) * KS2
        + dim[None, :],
        valid[:, None],
        other=0,
    ).to(tl.float32)
    score = tl.sum(k * q[None, :], 1) * 0.08838834764831845
    score = tl.where(valid, score, float("-inf"))
    mx = tl.maximum(tl.max(score, 0), -1e30)
    prob = tl.exp(score - mx)
    sm = tl.sum(prob, 0)
    v = tl.load(
        V
        + physical[:, None] * VS0
        + (key % PAGE)[:, None] * VS1
        + (h // 4) * VS2
        + dim[None, :],
        valid[:, None],
        other=0,
    ).to(tl.float32)
    pv = tl.sum(prob[:, None] * v, 0)
    row = ((b * 8 + qr) * 8 + h) * PARTS + part
    tl.store(P + row * 128 + dim, pv)
    tl.store(MX + row, mx)
    tl.store(SM + row, sm)


@triton.jit
def merge(P, MX, SM, OUTPUT, PARTS: tl.constexpr, BP: tl.constexpr):
    row = tl.program_id(0)
    p = tl.arange(0, BP)
    d = tl.arange(0, 128)
    mx = tl.load(MX + row * PARTS + p, p < PARTS, other=-1e30)
    sm = tl.load(SM + row * PARTS + p, p < PARTS, other=0)
    maximum = tl.max(mx, 0)
    scale = tl.where(sm > 0, tl.exp(mx - maximum), 0.0)
    denominator = tl.sum(sm * scale, 0)
    pv = tl.load(
        P + (row * PARTS + p[:, None]) * 128 + d[None, :], p[:, None] < PARTS, other=0
    )
    result = tl.sum(pv * scale[:, None], 0) / tl.maximum(denominator, 1e-24)
    tl.store(OUTPUT + row * 128 + d, result)


def forward(q, k, v, table, lengths, bk=128, out=None, workspace=None):
    parts = triton.cdiv(2055, bk)
    rows = q.shape[0] * 64
    if workspace is None:
        workspace = (
            torch.empty(rows, parts, 128, device=q.device),
            torch.empty(rows, parts, device=q.device),
            torch.empty(rows, parts, device=q.device),
        )
    if out is None:
        out = torch.empty_like(q)
    p, mx, sm = workspace
    part[(q.shape[0], 8, 8 * parts)](
        q,
        k,
        v,
        table,
        lengths,
        p,
        mx,
        sm,
        *k.stride()[:3],
        *v.stride()[:3],
        table.stride(0),
        k.shape[1],
        parts,
        bk,
        num_warps=8,
        enable_fp_fusion=False,
    )
    merge[(rows,)](
        p,
        mx,
        sm,
        out,
        parts,
        triton.next_power_of_2(parts),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out
