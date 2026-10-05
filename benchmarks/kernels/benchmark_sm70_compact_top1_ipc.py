# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TP4 compact top1 vs NCCL, four draft decisions in one CUDA graph.

This measures communication only, not an MTP round. Launch with torchrun.
"""

import argparse
import faulthandler
import json
import os
import statistics
from pathlib import Path

import torch
import torch.distributed as dist

from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce


def main():
    faulthandler.dump_traceback_later(90)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    assert torch.cuda.get_device_capability() == (7, 0)
    dist.init_process_group("nccl")
    assert dist.get_world_size() == 4
    group = dist.new_group(backend="gloo")
    comm = CustomAllreduce(group, device=rank)
    assert not comm.disabled and comm.fully_connected
    reports = []
    for rows in (1, 4, 5, 16, 128):
        pair = torch.empty(rows, 2, device="cuda", dtype=torch.float32)
        peers = [torch.empty_like(pair) for _ in range(4)]

        def nccl(pair=pair, peers=peers):
            dist.all_gather(peers, pair)
            gathered = torch.stack(peers, dim=1)
            winner = gathered[..., 0].argmax(dim=1, keepdim=True)
            return gathered[..., 1].gather(1, winner).squeeze(1).long()

        def ipc(pair=pair):
            result = comm.custom_top1_argmax(pair)
            assert result is not None
            return result

        pair[:, 0] = rank
        pair[:, 1] = rank * 62080 + torch.arange(rows, device="cuda")
        for _ in range(3):
            nccl()
            ipc()
        graphs, outputs = {}, {}
        for name, launch in (("nccl", nccl), ("ipc", ipc)):
            torch.cuda.synchronize()
            dist.barrier()
            graph = torch.cuda.CUDAGraph()
            saved = []
            with comm.capture(), torch.cuda.graph(graph):
                for _ in range(4):
                    saved.append(launch())
            graphs[name], outputs[name] = graph, saved
        # Replays must read changing values, preserve shard-ID ties and handle
        # width changes and repeated signaling epochs, including concurrent C4.
        for case in range(12):
            torch.manual_seed(20261005 + rank + case * 4)
            pair[:, 0].normal_()
            if case in (0, 1, 2):
                pair[:, 0].fill_((0.0, float("inf"), -float("inf"))[case])
            if (case == 3 and rank == 2) or (case == 4 and rank in (1, 3)) or case == 5:
                pair[:, 0].fill_(float("nan"))
            for name in ("nccl", "ipc"):
                for _ in range(3):
                    graphs[name].replay()
                torch.cuda.synchronize()
            for control, candidate in zip(outputs["nccl"], outputs["ipc"]):
                assert torch.equal(control, candidate), (rows, case, rank)
        samples = {name: [] for name in graphs}
        for trial in range(7):
            for name in ("nccl", "ipc") if trial % 2 else ("ipc", "nccl"):
                dist.barrier()
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                start.record()
                for _ in range(200):
                    graphs[name].replay()
                end.record()
                end.synchronize()
                times = [None] * 4
                dist.all_gather_object(times, start.elapsed_time(end) / 200, group)
                samples[name].append(max(times))
        reports.append(
            dict(
                rows=rows,
                calls=4,
                correct=True,
                median_ms={n: statistics.median(s) for n, s in samples.items()},
                rank_max_samples_ms=samples,
            )
        )
        if rank == 0:
            print(json.dumps(reports[-1]), flush=True)
    dist.barrier()
    # Captured NCCL graph references must die before ProcessGroupNCCL shuts
    # down; otherwise its graph-destruction callback can wait indefinitely.
    del graphs, outputs, graph, saved
    torch.cuda.synchronize()
    comm.close()
    dist.destroy_process_group(group)
    dist.destroy_process_group()
    if rank == 0:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(reports, indent=2) + "\n")
        print(json.dumps(reports), flush=True)
    faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
