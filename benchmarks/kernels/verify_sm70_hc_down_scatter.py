# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TP4 raw-bit and alignment oracle for the HC down all-gather; no model load.

With four exclusively available V100s and the source-matched extension:
VLLM_SM70_TP4_PUSH_ALLREDUCE=1 CUDA_VISIBLE_DEVICES=0,1,2,3 \
  .venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=4 \
  benchmarks/kernels/verify_sm70_hc_down_scatter.py --out scatter.json
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rank, local = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    torch.accelerator.set_device_index(local)
    if int(os.environ["WORLD_SIZE"]) != 4 or torch.cuda.get_device_capability() != (
        7,
        0,
    ):
        raise RuntimeError("Requires TP4 on four exclusively available SM70 GPUs")
    dist.init_process_group("nccl")
    group = dist.new_group(backend="gloo")
    comm = CustomAllreduce(group=group, device=local, max_size=8 * 1024 * 1024)
    try:
        if not comm.supports_sm70_qwen38_hc_output_allgather():
            raise RuntimeError("Load the source-matched HC custom-AR extension")
        gen = torch.Generator(device="cuda").manual_seed(20260905 + rank)
        inp = torch.empty(88, dtype=torch.int16, device="cuda")
        peers = [torch.empty(176, dtype=torch.uint8, device="cuda") for _ in range(4)]
        storage = [
            torch.full((344,), 0x1234, dtype=torch.int16, device="cuda")
            for _ in range(8)
        ]
        graphs = []
        for offset, buffer in enumerate(storage):
            graph = torch.cuda.CUDAGraph()
            inp.zero_()
            with comm.capture(), torch.cuda.graph(graph):
                comm.sm70_qwen38_hc_down_allgather(
                    inp.view(torch.float16),
                    buffer[offset : offset + 336].view(torch.float16),
                )
            graphs.append(graph)
        mismatches = 0
        for case in range(16):
            inp.random_(-32768, 32768, generator=gen)
            # Exercise the protocol's existing reserved-NaN canonicalization
            # in low-rank, injection, and padding positions on every rank.
            inp[case % 80] = inp[80] = inp[81 + case % 3] = 0x7F7F
            dist.all_gather(peers, inp.view(torch.uint8))
            bits = torch.stack(peers).view(torch.int16).reshape(4, 88)
            bits = torch.where(bits == 0x7F7F, 0x7E00, bits)
            expected = torch.cat(
                (bits[:, :80].reshape(-1), bits[:, 80], bits[:, 81:84].reshape(-1))
            )
            offset = case % 8
            buffer = storage[offset]
            for _ in range(16):
                graphs[offset].replay()
            torch.accelerator.synchronize()
            mismatches += int(
                torch.count_nonzero(buffer[offset : offset + 336] != expected)
            )
            assert bool(torch.all(buffer[:offset] == 0x1234))
            assert bool(torch.all(buffer[offset + 336 :] == 0x1234))
        results = [None] * 4
        dist.all_gather_object(
            results, {"rank": rank, "bit_mismatches": mismatches}, group=group
        )
        if any(row["bit_mismatches"] for row in results):
            raise RuntimeError(f"HC raw-bit gather mismatch: {results}")
        if rank == 0:
            report = {
                "scope": "HC down scatter only, not full-model quality or speed",
                "quality": results,
                "cases": 16,
                "fp16_output_offsets": list(range(8)),
                "graph_replays": 256,
            }
            args.out.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report), flush=True)
    finally:
        comm.close()
        dist.destroy_process_group(group)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
