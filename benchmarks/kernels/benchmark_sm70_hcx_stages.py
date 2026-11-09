# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attribute the packaged HCX kernel's stages on real TP4 HC weights.

Run with torchrun and all GPU ownership locks. Device-local globaltimer deltas
identify internal stages; graph events measure the whole chain separately.
Timestamp overhead and synthetic activations preclude endpoint speed claims.
"""

import argparse
import hashlib
import json
import os
import statistics
from pathlib import Path

import torch
import torch.distributed as dist
import vllm._C as core

import vllm
from vllm.models.qwen4_exp.nvidia.sm70_hcx import (
    Sm70HcxRuntime,
    pack_down,
    pack_up,
)

STAGES = {
    "producer_weight_prefetch_and_tp_sum": (0, 1),
    "combine_and_square_sum": (1, 2),
    "down_partial": (2, 3),
    "first_grid_barrier": (3, 4),
    "norm_and_partial_prefetch": (4, 9),
    "down_reduce_and_lora_publish": (9, 5),
    "lora_exchange_up_prefetch_and_grid_barrier": (5, 6),
    "up_and_gate_mix": (6, 7),
    "output_exchange": (7, 8),
}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--pairs", type=int, default=8)
    parser.add_argument("--rows", type=int, default=5, choices=range(1, 9))
    parser.add_argument("--repeats", type=int, default=16)
    parser.add_argument("--samples", type=int, default=8)
    args = parser.parse_args()
    if "site-packages" not in vllm.__file__:
        raise RuntimeError("An installed source-complete wheel is required")
    rank = int(os.environ["LOCAL_RANK"])
    torch.accelerator.set_device_index(rank)
    torch.set_num_threads(1)
    dist.init_process_group("gloo")
    if dist.get_world_size() != 4:
        raise RuntimeError("TP4 required")
    runtime = Sm70HcxRuntime(dist.group.WORLD, torch.device("cuda", rank))
    if not runtime.enabled:
        raise RuntimeError(runtime.reason)
    weights = torch.load(args.weights, map_location="cpu", weights_only=True)
    packed = []
    for record in weights[: args.pairs]:
        down = (
            torch.cat(
                (record["down"].float(), record["inj"].float(), torch.zeros(12, 10240))
            )
            .half()
            .cuda()
        )
        up = record["up"].half().cuda()
        packed.append(
            (
                pack_down(down, runtime.logical_rank),
                pack_up(up, runtime.logical_rank),
                record["nw"].half().cuda(),
            )
        )
    if len(packed) != args.pairs:
        raise RuntimeError("Insufficient weight pairs")
    m = args.rows
    torch.manual_seed(20261007)
    hidden = torch.randn(m, 10240, device="cuda").half()
    injection = torch.randn(m, 4, device="cuda").half()
    torch.manual_seed(20261007 + rank)
    partial = (torch.randn(m, 2560, device="cuda") * 0.5).half()
    output = torch.empty_like(hidden)
    block = torch.empty_like(partial)
    inj_out = torch.empty_like(injection)
    debug = torch.zeros(80, 16, dtype=torch.int64, device="cuda")

    def run(weight, instrument):
        down, up, norm = weight
        torch.ops._C.sm70_hcx_out(
            partial,
            None,
            hidden,
            injection,
            norm,
            1e-6,
            down,
            up,
            output,
            block,
            inj_out,
            runtime.xn,
            runtime.sq,
            runtime.dpart,
            runtime.bar,
            runtime.seq,
            runtime.ar,
            runtime.lora,
            runtime.hb,
            runtime.logical_rank,
            debug if instrument else None,
            int(runtime.full),
            None,
            None,
            None,
            None,
            -1,
            None,
            None,
            1e-6,
            None,
        )

    for weight in packed:
        run(weight, False)
    torch.accelerator.synchronize()
    dist.barrier()
    reference = runtime.run(
        partial,
        hidden,
        injection,
        packed[-1][2],
        1e-6,
        packed[-1][0],
        packed[-1][1],
    )
    torch.accelerator.synchronize()
    graphs = {}
    for instrument in (False, True):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(args.repeats):
                for weight in packed:
                    run(weight, instrument)
        graphs[instrument] = graph
        for _ in range(3):
            graph.replay()
        torch.accelerator.synchronize()
        for actual, expected in zip((output, block, inj_out), reference):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    times = {False: [], True: []}
    phases = []
    for instrument in (False, True, True, False):
        for _ in range(args.samples):
            dist.barrier()
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            graphs[instrument].replay()
            end.record()
            end.synchronize()
            times[instrument].append(
                start.elapsed_time(end) * 1000 / args.repeats / len(packed)
            )
            if instrument:
                stamp = debug.cpu()
                phases.append(
                    {
                        name: ((stamp[:, b] - stamp[:, a]).float() / 1000).tolist()
                        for name, (a, b) in STAGES.items()
                    }
                )
    record = dict(rank=rank, samples_us=times, phase_samples_us=phases)
    records = [None] * 4
    dist.all_gather_object(records, record)
    if rank == 0:
        result = dict(
            scope="packaged HCX stage attribution; not model latency",
            wheel_version=vllm.__version__,
            core_sha256=hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
            M=m,
            weight_pairs=len(packed),
            full_mesh=runtime.full,
            rank_records=records,
            max_rank_median_us={
                str(key): max(statistics.median(r["samples_us"][key]) for r in records)
                for key in times
            },
            interpretation=(
                "Each phase sample is 80 device-local CTA durations from the "
                "last weight pair at a coupled graph tail. Stage medians are "
                "not additive; no cross-device timestamp subtraction is used."
            ),
        )
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result["max_rank_median_us"]), flush=True)
    del graphs
    torch.accelerator.synchronize()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
