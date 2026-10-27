# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed fused indexer vs the gather + mm path: top-512 agreement and timing."""

import os

import torch

os.environ.setdefault(
    "VLLM_V41K_LIB", "/home/ymzx/ds41-dev/tools/v41k_g3/build/v41k.so"
)
import vllm.models.deepseek_v41.attention_ops as ops
from vllm.models.deepseek_v41.kv_pack import pack_rows
from vllm.models.deepseek_v41.quant_rope import attention_qdq

torch.manual_seed(0)
H = 8
ident = lambda x: x
for T, start, ratio, rows in (
    (6, 9000, 2, 8192),
    (6, 15000, 1, 16384),
    (512, 3000, 2, 8192),
    (2048, 12000, 1, 16384),
    (6, 400000, 1, 524288),
):
    raw = (torch.randn(rows, 128, device="cuda") * 0.5).half()
    ref_keys = attention_qdq(raw, fp4=True, group=32)
    packed = pack_rows(raw, False)
    q = attention_qdq(
        (torch.randn(T, H, 128, device="cuda") * 0.5)
        .half()
        .reshape(T * H, 128)
        .contiguous(),
        fp4=True,
        group=32,
    ).view(T, H, 128)
    w = (torch.randn(T, H, device="cuda") * 0.2).half()
    pos = torch.arange(start, start + T, device="cuda")
    valid = torch.ones(T, dtype=torch.bool, device="cuda")
    old = (
        ops.select_compressed_positions(q, w, ref_keys, pos, valid, ratio, ident)
        if rows <= 16384
        else None
    )
    new = ops.select_packed_positions(q, w, packed, pos, valid, ratio, ident)
    if old is not None:
        agree = []
        for r in range(T):
            a, b = set(old[r].tolist()) - {-1}, set(new[r].tolist()) - {-1}
            agree.append(len(a & b) / max(1, len(a)))
        print(
            "T=%d start=%d ratio=%d: top-512 overlap min %.4f mean %.5f exact rows %d/%d"
            % (
                T,
                start,
                ratio,
                min(agree),
                sum(agree) / T,
                sum(1 for x in agree if x == 1.0),
                T,
            )
        )
    for name, fn in (
        (
            "old",
            lambda: ops.select_compressed_positions(
                q, w, ref_keys, pos, valid, ratio, ident
            ),
        )
        if rows <= 16384
        else ("old", None),
        (
            "new",
            lambda: ops.select_packed_positions(q, w, packed, pos, valid, ratio, ident),
        ),
    ):
        if fn is None:
            continue
        fn()
        torch.accelerator.synchronize()
        a, b = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        a.record()
        for _ in range(5):
            fn()
        b.record()
        torch.accelerator.synchronize()
        print("   %s %.3f ms" % (name, a.elapsed_time(b) / 5))
