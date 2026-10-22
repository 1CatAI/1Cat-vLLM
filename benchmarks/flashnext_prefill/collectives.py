# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TP4 prefill-sized collective screen; launch with torchrun on four idle GPUs.

Uses the same PyNccl and custom-AR implementations as serving. Custom AR's larger
scratch allocation is experimental here, not a change to runtime admission.
"""

import json
import os
import statistics
from functools import partial

import torch
import torch.distributed as dist

from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator


def measure(fn):
    samples = []
    for _ in range(5):
        fn()
    torch.accelerator.synchronize()
    dist.barrier()
    for _ in range(5):
        begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        begin.record()
        for _ in range(20):
            fn()
        end.record()
        end.synchronize()
        times = [None] * dist.get_world_size()
        dist.all_gather_object(times, begin.elapsed_time(end) / 20)
        samples.append(max(times))
    return {"median_ms": statistics.median(samples), "samples_ms": samples}


def main():
    rank = int(os.environ["LOCAL_RANK"])
    torch.accelerator.set_device_index(rank)
    dist.init_process_group("gloo")
    assert dist.get_world_size() == 4
    nccl = PyNcclCommunicator(dist.group.WORLD, rank)
    ca = CustomAllreduce(dist.group.WORLD, rank, max_size=48 * 1024**2)
    assert not ca.disabled and not nccl.disabled
    try:
        for rows in (7344, 5347):
            x = torch.randn(rows, 2560, device="cuda", dtype=torch.half) * 0.03
            control, candidate = torch.empty_like(x), torch.empty_like(x)
            run_nccl = partial(nccl.all_reduce, x, control)
            run_custom = partial(ca.all_reduce, x, out=candidate)
            result = {
                "tokens": rows,
                "reduce_bytes": x.numel() * x.element_size(),
                "nccl_all_reduce": measure(run_nccl),
                "custom_all_reduce_eager": measure(run_custom),
            }
            difference = (candidate.float() - control.float()).norm()
            result["relative_l2"] = (difference / control.float().norm()).item()
            result["finite"] = bool(torch.isfinite(candidate).all())
            # Two HC packets per module: N=88 and N=640 per rank.
            for n in (88, 640):
                packet = torch.randn(rows, n, device="cuda", dtype=torch.half)
                gathered = torch.empty(4 * rows, n, device="cuda", dtype=torch.half)
                result[f"nccl_all_gather_n{n}"] = measure(
                    partial(nccl.all_gather, gathered, packet)
                )
            if rank == 0:
                print(json.dumps(result), flush=True)
    finally:
        torch.accelerator.synchronize()
        ca.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
