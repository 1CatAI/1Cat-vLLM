# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""All-reduce / all-gather timings per NCCL protocol at prefill message sizes.

Usage (4 idle GPUs): python nccl_protocols.py default Simple LL LL128
"""

import os
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def run(rank, proto):
    if proto != "default":
        os.environ["NCCL_PROTO"] = proto
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29733")
    torch.accelerator.set_device_index(rank)
    dist.init_process_group("nccl", rank=rank, world_size=4)
    out = []
    for rows, cols, op in [
        (8192, 2560, "ar"),
        (7344, 2560, "ar"),
        (8192, 2560 // 4, "ag"),
        (8192, 88, "ar"),
        (8192, 352, "ag"),
    ]:
        x = torch.randn(rows, cols, device="cuda").half()
        o = torch.empty(4 * rows, cols, device="cuda").half()
        fn = (
            (lambda x=x: dist.all_reduce(x))
            if op == "ar"
            else (lambda o=o, x=x: dist.all_gather_into_tensor(o, x))
        )
        for _ in range(5):
            fn()
        torch.accelerator.synchronize()
        dist.barrier()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(30):
            fn()
        e.record()
        e.synchronize()
        out.append(
            f"{op} {rows}x{cols} ({x.numel() * 2 / 1e6:.1f} MB) "
            f"{s.elapsed_time(e) / 30 * 1e3:7.0f} us"
        )
    if rank == 0:
        print(proto, " | ".join(out), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    for proto in sys.argv[1:]:
        mp.spawn(run, args=(proto,), nprocs=4)
