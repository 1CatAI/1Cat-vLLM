# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ping-pong NCCL send/recv between two nodes (one GPU each) over IB.
Usage: nccl_p2p.py RANK MASTER_IP"""

import sys
import time

import torch
import torch.distributed as dist

rank, master = int(sys.argv[1]), sys.argv[2]
torch.accelerator.set_device_index(0)
dist.init_process_group(
    "nccl", init_method=f"tcp://{master}:29599", rank=rank, world_size=2
)
peer = 1 - rank
for size in (8 << 10, 123 << 10, 492 << 10, 8 << 20, 64 << 20):
    buf = torch.empty(size // 4, dtype=torch.float32, device="cuda")
    for _ in range(10):
        if rank == 0:
            dist.send(buf, peer)
            dist.recv(buf, peer)
        else:
            dist.recv(buf, peer)
            dist.send(buf, peer)
    torch.accelerator.synchronize()
    n = 200 if size < (8 << 20) else 20
    t = time.perf_counter()
    for _ in range(n):
        if rank == 0:
            dist.send(buf, peer)
            dist.recv(buf, peer)
        else:
            dist.recv(buf, peer)
            dist.send(buf, peer)
    torch.accelerator.synchronize()
    dt = (time.perf_counter() - t) / n / 2
    if rank == 0:
        print(
            f"{size / 1024:9.0f} KiB  one-way {dt * 1e6:8.1f} us  {size / dt / 1e9:6.2f} GB/s",
            flush=True,
        )
dist.destroy_process_group()
