# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused sparse attention vs the gather + bmm path (ring window + compressed top-k)."""

import torch

import vllm.models.deepseek_v41.attention_ops as ops

torch.manual_seed(0)
dev = "cuda"
H, BS, R = 16, 64, 2
ok = True
for T, start, comp_rows in ((300, 0, 0), (512, 700, 600), (2048, 5000, 2600)):
    total = start + T
    ring = (torch.randn(ops.WINDOW_RING, 512, device=dev) * 0.3).half()
    values = (torch.randn(T, 512, device=dev) * 0.3).half()
    q = (torch.randn(T, H, 512, device=dev) * 0.3).half()
    pos = torch.arange(start, total, device=dev)
    valid = torch.ones(T, dtype=torch.bool, device=dev)
    valid[5] = False
    sink = torch.randn(H, device=dev)
    table = torch.arange(total // BS + 2, dtype=torch.int32, device=dev)[None]
    kwargs = dict(window_block=BS, window_ring=ring, window_values=values)
    if comp_rows:
        ccache = (torch.randn(total // BS + 2, BS // R, 512, device=dev) * 0.3).half()
        n_c = (pos + 1) // R
        topk = (torch.rand(T, 512, device=dev) * n_c[:, None]).long()
        topk = torch.where(torch.rand(T, 512, device=dev) < 0.05, -1, topk).int()
        kwargs.update(
            compressed_cache=ccache,
            compressed_table=table,
            compressed_block=BS,
            ratio=R,
            topk=topk.contiguous(),
        )
    ops._FUSED_SPARSE = False
    ref = ops.attend_paged_latents(q, pos, valid, None, table, sink, **kwargs)
    ops._FUSED_SPARSE = True
    new = ops.attend_paged_latents(q, pos, valid, None, table, sink, **kwargs)
    diff = (new.float() - ref.float()).abs()
    rel = (diff.norm() / ref.float().norm()).item()
    good = torch.isfinite(new).all().item() and rel < 5e-3
    ok &= good
    for fused in (False, True):
        ops._FUSED_SPARSE = fused
        for _ in range(3):
            ops.attend_paged_latents(q, pos, valid, None, table, sink, **kwargs)
        torch.accelerator.synchronize()
        a, b = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        a.record()
        for _ in range(5):
            ops.attend_paged_latents(q, pos, valid, None, table, sink, **kwargs)
        b.record()
        torch.accelerator.synchronize()
        print(
            "  T=%d compressed=%s fused=%s %.2f ms"
            % (T, bool(comp_rows), fused, a.elapsed_time(b) / 5)
        )
    print("T=%d rel=%.2e max=%.3g ok=%s" % (T, rel, diff.max().item(), good))
print("ALL OK" if ok else "FAILED")
