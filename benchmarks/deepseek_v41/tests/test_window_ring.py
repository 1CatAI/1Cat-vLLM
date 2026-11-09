# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""G4 window ring vs paged window cache: identical attention over chunked prefill + decode."""

import torch

from vllm.models.deepseek_v41.attention_ops import (
    WINDOW_RING,
    attend_paged_latents,
    insert_latents,
    insert_window_ring,
)

torch.manual_seed(0)
dev = "cuda"
H, BS = 16, 64
total = 1200
cache = torch.zeros(total // BS + 2, BS, 512, dtype=torch.float16, device=dev)
table = torch.arange(cache.shape[0], dtype=torch.int32, device=dev)[None]
ring = torch.zeros(WINDOW_RING, 512, dtype=torch.float16, device=dev)
sink = torch.randn(H, device=dev)
ok, start = True, 0
for T in (300, 200, 6, 6, 3, 400, 1, 5):
    pos = torch.arange(start, start + T, device=dev)
    valid = torch.ones(T, dtype=torch.bool, device=dev)
    values = (torch.randn(T, 512, device=dev) * 0.3).half()
    q = (torch.randn(T, H, 512, device=dev) * 0.3).half()
    insert_latents(cache, values, pos, valid, table, block_size=BS)
    old = attend_paged_latents(q, pos, valid, cache, table, sink, window_block=BS)
    new = attend_paged_latents(
        q,
        pos,
        valid,
        None,
        table,
        sink,
        window_block=BS,
        window_ring=ring,
        window_values=values,
    )
    insert_window_ring(ring, values, pos, valid)
    same = torch.equal(old, new)
    ok &= same
    print(
        "chunk T=%d start=%d equal=%s maxdiff=%.3g"
        % (T, start, same, (old.float() - new.float()).abs().max().item())
    )
    start += T
print("ALL OK" if ok else "FAILED")
