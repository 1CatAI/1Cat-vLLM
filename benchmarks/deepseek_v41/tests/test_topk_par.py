# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.models.deepseek_v41.index_topk import index_topk, index_topk_parallel

torch.manual_seed(0)


def tm(fn, n=10):
    fn()
    torch.accelerator.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(n):
        fn()
    b.record()
    torch.accelerator.synchronize()
    return a.elapsed_time(b) / n


ok = True
for rows, width, start in (
    (6, 524288, 400000),
    (6, 524288, 300),
    (3, 131072, 100000),
    (6, 65536, 65000),
):
    sc = torch.randn(rows, width, device="cuda").half()
    sc[:, ::7] = 0.5  # ties
    pos = torch.arange(start, start + rows, device="cuda")
    valid = torch.ones(rows, dtype=torch.bool, device="cuda")
    a = torch.empty(rows, 512, dtype=torch.int32, device="cuda")
    b = torch.empty_like(a)
    index_topk(a, sc, pos, valid, 1, 512)
    index_topk_parallel(b, sc, pos, valid, 1, 512)
    same = torch.equal(a, b)
    ok &= same
    print(
        "rows %d width %d start %d equal=%s single %.3f ms parallel %.3f ms"
        % (
            rows,
            width,
            start,
            same,
            tm(lambda: index_topk(a, sc, pos, valid, 1, 512)),
            tm(lambda: index_topk_parallel(b, sc, pos, valid, 1, 512)),
        )
    )
print("ALL OK" if ok else "FAILED")
