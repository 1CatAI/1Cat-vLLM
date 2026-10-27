# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed compressed rows (hot ring + staging / selected buffer) vs the FP16 paged cache."""

from types import SimpleNamespace

import torch

import vllm.models.deepseek_v41.attention_ops as ops
from vllm.models.deepseek_v41.kv_pack import pack_rows
from vllm.models.deepseek_v41.kv_store import _fill_selected
from vllm.models.deepseek_v41.quant_rope import attention_qdq

torch.manual_seed(0)
dev = "cuda"
H, BS, R = 16, 64, 2
ok = True
for T, start, hot_rows in (
    (300, 5000, 1024),
    (6, 9000, 1024),
    (6, 3000, 4096),
    (2048, 12000, 2048),
):
    total = start + T
    rows = total // R + 1
    raw = (torch.randn(rows, 512, device=dev) * 0.3).half()
    fp16 = attention_qdq(raw, fp4=True, group=16, e4m3_scale=True)
    packed = pack_rows(raw, True)
    # paged FP16 reference cache: identity block table
    ccache = torch.zeros(
        rows // (BS // R) + 2, BS // R, 512, dtype=torch.float16, device=dev
    )
    ccache.view(-1, 512)[:rows] = fp16
    table = torch.arange(ccache.shape[0], dtype=torch.int32, device=dev)[None]
    ring = (torch.randn(ops.WINDOW_RING, 512, device=dev) * 0.3).half()
    values = (torch.randn(T, 512, device=dev) * 0.3).half()
    q = (torch.randn(T, H, 512, device=dev) * 0.3).half()
    pos = torch.arange(start, total, device=dev)
    valid = torch.ones(T, dtype=torch.bool, device=dev)
    sink = torch.randn(H, device=dev)
    n_c = (pos + 1) // R
    topk = (torch.rand(T, 512, device=dev) * n_c[:, None]).long()
    topk = (
        torch.where(torch.rand(T, 512, device=dev) < 0.05, -1, topk).int().contiguous()
    )
    ref = ops.attend_paged_latents(
        q,
        pos,
        valid,
        None,
        table,
        sink,
        window_block=BS,
        window_ring=ring,
        window_values=values,
        compressed_cache=ccache,
        compressed_table=table,
        compressed_block=BS,
        ratio=R,
        topk=topk,
    )
    newest = int(total) // R - 1
    hot = torch.zeros(hot_rows, 288, dtype=torch.uint8, device=dev)
    idx = torch.arange(max(0, newest - hot_rows + 1), newest + 1, device=dev)
    hot[idx % hot_rows] = packed[idx]
    old = newest - hot_rows + 1
    staging = torch.zeros(max(rows, 1), 288, dtype=torch.uint8, device=dev)
    if old > 0:
        staging[:old] = packed[:old]
    selected = torch.zeros(8, 512, 288, dtype=torch.uint8, device=dev)
    store = SimpleNamespace(
        hot={0: hot}, staging=staging, hot_rows={0: hot_rows}, selected=selected
    )
    if T <= 8:
        _fill_selected[(T, 8)](
            topk, pos, hot, packed, selected, T, hot_rows, RATIO=R, TOPK=512, BLOCK=64
        )
    new = ops.attend_paged_latents(
        q,
        pos,
        valid,
        None,
        table,
        sink,
        window_block=BS,
        window_ring=ring,
        window_values=values,
        ratio=R,
        topk=topk,
        packed=(store, 0),
    )
    same = torch.equal(ref, new)
    ok &= same
    print(
        "T=%d start=%d hot_rows=%d mode=%s bit-exact=%s maxdiff=%.3g"
        % (
            T,
            start,
            hot_rows,
            "selected" if T <= 8 else "hot+staging",
            same,
            (ref.float() - new.float()).abs().max().item(),
        )
    )
print("ALL OK" if ok else "FAILED")
