# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""4-process test of v41k.peer_gather: correctness over many epochs + latency (eager and graph)."""

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

LIB = "/home/ymzx/ds41-dev/tools/v41k_g3/build/v41k.so"


class View:
    def __init__(self, ptr, n):
        self.__cuda_array_interface__ = {
            "shape": (n,),
            "typestr": "|u1",
            "data": (ptr, False),
            "version": 3,
        }


def worker(rank):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29533")
    torch.accelerator.set_device_index(rank)
    dist.init_process_group("gloo", rank=rank, world_size=4)
    torch.ops.load_library(LIB)
    from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce

    stride_max = (36 * 648 + 255) // 256 * 256
    flags = 4 * stride_max
    nbytes = flags + 256
    ptrs = CustomAllreduce.create_shared_buffer(nbytes, None)
    own = torch.as_tensor(View(ptrs[rank], nbytes), device=f"cuda:{rank}")
    own.zero_()
    torch.accelerator.synchronize()
    dist.barrier()
    epoch = torch.zeros(1, dtype=torch.int32, device=f"cuda:{rank}")
    ok = True
    for it in range(300):
        routes = 6 * (1 + it % 6)
        stride = (routes * 648 + 255) // 256 * 256
        own[rank * stride : (rank + 1) * stride].fill_((it * 7 + rank * 31) % 251)
        torch.ops.v41k.peer_gather(*ptrs, rank, stride, flags, epoch)
        for r in range(4):
            want = (it * 7 + r * 31) % 251
            if not bool((own[r * stride : (r + 1) * stride] == want).all()):
                ok = False
        # Mimic the MoE output all-reduce that bounds rank skew.
        torch.accelerator.synchronize()
        dist.barrier()
    stride = (36 * 648 + 255) // 256 * 256
    for _ in range(50):
        torch.ops.v41k.peer_gather(*ptrs, rank, stride, flags, epoch)
    torch.accelerator.synchronize()
    dist.barrier()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(1000):
        torch.ops.v41k.peer_gather(*ptrs, rank, stride, flags, epoch)
    b.record()
    torch.accelerator.synchronize()
    eager = a.elapsed_time(b)
    dist.barrier()
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s), torch.cuda.graph(g):
        for _ in range(40):
            torch.ops.v41k.peer_gather(*ptrs, rank, stride, flags, epoch)
    torch.accelerator.synchronize()
    dist.barrier()
    a.record()
    for _ in range(25):
        g.replay()
    b.record()
    torch.accelerator.synchronize()
    graph = a.elapsed_time(b) / 1000 * 1000
    print(
        f"rank {rank} ok={ok} eager {eager:.2f} us/call graph {graph:.2f} us/call epoch={int(epoch.item())}",
        flush=True,
    )
    dist.barrier()


if __name__ == "__main__":
    mp.spawn(worker, nprocs=4)
