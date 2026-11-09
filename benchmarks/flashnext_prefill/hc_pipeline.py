# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Screen overlapping TP4 HC projections and gathers, without a model restart.

Launch with torchrun on four idle V100s. This uses the production HC arithmetic
and NCCL provider with synthetic checkpoint-shaped inputs. It is a throughput
screen, not a model quality or runtime-state qualification.
"""

import json
import os

import torch
import torch.distributed as dist
from collectives import measure
from moe_tiles import relative_l2

from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.models.qwen4_exp.nvidia.ops.hc import (
    hc_assemble_ranks,
    hc_gate_mix_local,
    hc_packet_silu,
)

ORDER = (0, 1, 2, 3)


def sequential(nccl, x, down, up, rank):
    local = torch.mm(x, down.T)
    packet = local.new_empty((4 * x.shape[0], 88))
    nccl.all_gather(packet, local)
    lora, injection = hc_packet_silu(packet, ORDER, 4)
    local_gate = torch.mm(lora, up.T)
    mixed = hc_gate_mix_local(x, local_gate, rank, 4)
    gathered = mixed.new_empty((4 * x.shape[0], 640))
    nccl.all_gather(gathered, mixed)
    return hc_assemble_ranks(gathered, ORDER), injection


class Pipeline:
    def __init__(self, nccl, x, down, up, rank, chunks, overlap=True):
        self.nccl, self.down, self.up, self.rank = nccl, down, up, rank
        self.x = x.chunk(chunks)
        self.comm = torch.cuda.Stream() if overlap else torch.cuda.current_stream()
        self.ready = [torch.cuda.Event() for _ in self.x]
        self.packet_done = [torch.cuda.Event() for _ in self.x]
        self.mixed_ready = [torch.cuda.Event() for _ in self.x]
        self.gather_done = [torch.cuda.Event() for _ in self.x]
        self.packet = [v.new_empty((4 * v.shape[0], 88)) for v in self.x]
        self.gathered = [v.new_empty((4 * v.shape[0], 640)) for v in self.x]

    def __call__(self):
        main = torch.cuda.current_stream()
        # Retain the inputs until all their asynchronous gathers are queued.
        locals_, mixed, injections = [], [], []
        for i, x in enumerate(self.x):
            locals_.append(torch.mm(x, self.down.T))
            self.ready[i].record(main)
            with torch.cuda.stream(self.comm):
                self.comm.wait_event(self.ready[i])
                self.nccl.all_gather(self.packet[i], locals_[-1])
                locals_[-1].record_stream(self.comm)
                self.packet_done[i].record(self.comm)
        for i, x in enumerate(self.x):
            main.wait_event(self.packet_done[i])
            lora, injection = hc_packet_silu(self.packet[i], ORDER, 4)
            injections.append(injection)
            gate = torch.mm(lora, self.up.T)
            mixed.append(hc_gate_mix_local(x, gate, self.rank, 4))
            self.mixed_ready[i].record(main)
            with torch.cuda.stream(self.comm):
                self.comm.wait_event(self.mixed_ready[i])
                self.nccl.all_gather(self.gathered[i], mixed[-1])
                mixed[-1].record_stream(self.comm)
                self.gather_done[i].record(self.comm)
        outputs = []
        for i, gathered in enumerate(self.gathered):
            main.wait_event(self.gather_done[i])
            outputs.append(hc_assemble_ranks(gathered, ORDER))
        return torch.cat(outputs), torch.cat(injections)


def main():
    rank = int(os.environ["LOCAL_RANK"])
    torch.accelerator.set_device_index(rank)
    dist.init_process_group("gloo")
    assert dist.get_world_size() == 4
    nccl = PyNcclCommunicator(dist.group.WORLD, rank)
    assert not nccl.disabled
    passed = True
    try:
        for rows in (7344, 5347):
            torch.manual_seed(rows)
            x = torch.randn(rows, 10240, device="cuda", dtype=torch.half) * 0.3
            torch.manual_seed(100 + rank)
            down = torch.randn(88, 10240, device="cuda", dtype=torch.half) * 0.01
            up = torch.randn(2560, 320, device="cuda", dtype=torch.half) * 0.01

            def control(x=x, down=down, up=up):
                return sequential(nccl, x, down, up, rank)

            expected = control()
            result = {"tokens": rows, "sequential": measure(control)}
            for chunks in (2, 4):
                candidate = Pipeline(nccl, x, down, up, rank, chunks)
                actual = candidate()
                same_tiles = Pipeline(nccl, x, down, up, rank, chunks, overlap=False)()
                error = [relative_l2(a, b) for a, b in zip(actual, expected)]
                serial_error = max(
                    relative_l2(a, b) for a, b in zip(actual, same_tiles)
                )
                finite = all(bool(torch.isfinite(a).all()) for a in actual)
                errors = [None] * 4
                dist.all_gather_object(errors, (error, serial_error, finite))
                numerical_pass = all(
                    max(e) <= 0.001 and s == 0 and f for e, s, f in errors
                )
                passed &= numerical_pass
                # Failed numerical screens remain failed. Timing them decides
                # whether further diagnosis is worthwhile, not admission.
                result[f"chunks_{chunks}"] = measure(candidate) | {
                    "max_rank_output_relative_l2": max(e[0] for e, _, _ in errors),
                    "max_rank_injection_relative_l2": max(e[1] for e, _, _ in errors),
                    "overlap_vs_serial_same_tiles": max(s for _, s, _ in errors),
                    "finite_all_ranks": all(f for _, _, f in errors),
                    "numerical_pass": numerical_pass,
                }
            if rank == 0:
                print(json.dumps(result), flush=True)
    finally:
        torch.accelerator.synchronize()
        dist.destroy_process_group()
    if not passed:
        raise SystemExit("Numerical screen failed; candidate is not qualified")


if __name__ == "__main__":
    main()
