# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""In-kernel copy from pinned host memory (UVA) vs DMA, all GPUs at once.

Copies `count` scattered expert-sized blocks (0.47 MiB) per GPU, as a decode
round's misses would be copied in-graph.
"""

import argparse
import threading
import time

import torch

import vllm._custom_ops  # noqa: F401  (registers _C)
from vllm.triton_utils import tl, triton


@triton.jit
def _gather(src, dst, rows, ROW: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.load(rows + tl.program_id(1))
    base = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    for start in range(0, ROW, BLOCK * tl.num_programs(0)):
        offs = start + base
        mask = offs < ROW
        tl.store(
            dst + tl.program_id(1).to(tl.int64) * ROW + offs,
            tl.load(src + row.to(tl.int64) * ROW + offs, mask=mask),
            mask=mask,
        )


def run(gpu, count, ctas, block, warps, repeats, barrier, out):
    torch.accelerator.set_device_index(gpu)
    row = 490240 // 16  # int4-sized elements
    host = torch.empty((64, row, 4), dtype=torch.int32).pin_memory()
    view = torch.ops._C.get_cuda_view_from_cpu_tensor(host)
    dst = torch.empty((count, row, 4), dtype=torch.int32, device=gpu)
    rows = torch.randperm(64, device=gpu)[:count].to(torch.int32)
    src = view.view(torch.int64)  # 8-byte lanes
    d = dst.view(torch.int64)
    ROW = row * 2
    for mode in ("uva", "dma"):

        def once(mode=mode):
            if mode == "uva":
                _gather[(ctas, count)](
                    src, d, rows, ROW=ROW, BLOCK=block, num_warps=warps
                )
            else:
                for i, r in enumerate(rows.tolist()):
                    dst[i].copy_(host[r], non_blocking=True)

        once()
        torch.accelerator.synchronize()
        barrier.wait()
        t = time.perf_counter()
        for _ in range(repeats):
            once()
        torch.accelerator.synchronize()
        dt = (time.perf_counter() - t) / repeats
        out[gpu, mode] = (count * 490240 / dt / 1e9, dt * 1e6)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--count", type=int, default=8)
    p.add_argument("--ctas", type=int, default=8)
    p.add_argument("--block", type=int, default=1024)
    p.add_argument("--warps", type=int, default=8)
    p.add_argument("--gpus", type=int, default=torch.accelerator.device_count())
    a = p.parse_args()
    for gpus in (1, a.gpus):
        out = {}
        barrier = threading.Barrier(gpus)
        ts = [
            threading.Thread(
                target=run,
                args=(g, a.count, a.ctas, a.block, a.warps, 20, barrier, out),
            )
            for g in range(gpus)
        ]
        [t.start() for t in ts]
        [t.join() for t in ts]
        for (g, mode), (gbps, us) in sorted(out.items()):
            print(
                f"gpus={gpus} gpu{g} {mode}: {gbps:5.2f} GB/s {us:7.0f} us "
                f"for {a.count} experts (ctas={a.ctas} block={a.block})",
                flush=True,
            )


if __name__ == "__main__":
    main()
