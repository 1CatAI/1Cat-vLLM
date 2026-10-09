# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare HCX direct broadcast and XOR exchange on one full NVLink mesh.

Use torchrun with all four GPU ownership locks. Both protocols use the same
installed native module, weights, workspace and captured chain. This is a
communication screen, not a model latency measurement.
"""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
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


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--pairs", type=int, default=8)
    parser.add_argument("--rows", type=int, default=5, choices=range(1, 9))
    parser.add_argument("--repeats", type=int, default=64)
    parser.add_argument("--samples", type=int, default=8)
    args = parser.parse_args()
    if args.pairs < 1 or args.repeats < 1 or args.samples < 1:
        raise ValueError("Positive pairs, repeats and samples required")
    if "site-packages" not in vllm.__file__:
        raise RuntimeError("An installed source-complete wheel is required")
    rank = int(os.environ["LOCAL_RANK"])
    torch.accelerator.set_device_index(rank)
    torch.set_num_threads(1)
    dist.init_process_group("gloo")
    runtime = Sm70HcxRuntime(dist.group.WORLD, torch.device("cuda", rank))
    if not runtime.enabled or not runtime.full:
        raise RuntimeError("This comparison requires a full admitted TP4 NVLink mesh")
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
        packed.append(
            (
                pack_down(down, runtime.logical_rank),
                pack_up(record["up"].half().cuda(), runtime.logical_rank),
                record["nw"].half().cuda(),
            )
        )
    if len(packed) != args.pairs:
        raise RuntimeError("Insufficient weight pairs")
    m = args.rows
    hidden = torch.empty(m, 10240, device="cuda", dtype=torch.float16)
    injection = torch.empty(m, 4, device="cuda", dtype=torch.float16)
    partial = torch.empty(m, 2560, device="cuda", dtype=torch.float16)
    output, block, inj_out = map(torch.empty_like, (hidden, partial, injection))

    def refresh(seed):
        torch.manual_seed(seed)
        hidden.copy_(torch.randn_like(hidden))
        injection.copy_(torch.randn_like(injection))
        torch.manual_seed(seed + rank)
        partial.copy_(torch.randn_like(partial) * 0.5)

    def run(weight, full):
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
            None,
            int(full),
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

    def check(reference):
        for value, expected in zip((output, block, inj_out), reference, strict=True):
            torch.testing.assert_close(value, expected, rtol=0, atol=0)

    # Change inputs at fixed addresses before each protocol switch. The
    # sequence counter advances across both protocols; no peer flags reset.
    for seed in range(3):
        refresh(20261009 + seed * 11)
        for weight in packed:
            run(weight, True)
            torch.accelerator.synchronize()
            reference = tuple(v.clone() for v in (output, block, inj_out))
            run(weight, False)
            torch.accelerator.synchronize()
            check(reference)
    graphs = {}
    refresh(20261009)
    for full in (True, False):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(args.repeats):
                for weight in packed:
                    run(weight, full)
        graphs[full] = graph
    for seed in range(3):
        refresh(20261019 + seed * 11)
        graphs[True].replay()
        torch.accelerator.synchronize()
        reference = tuple(v.clone() for v in (output, block, inj_out))
        graphs[False].replay()
        torch.accelerator.synchronize()
        check(reference)
    times = {True: [], False: []}
    for epoch in range(4):
        for full in (True, False) if epoch % 2 == 0 else (False, True):
            graphs[full].replay()
            torch.accelerator.synchronize()
            for _ in range(args.samples):
                dist.barrier()
                start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                start.record()
                graphs[full].replay()
                end.record()
                end.synchronize()
                times[full].append(
                    start.elapsed_time(end) * 1000 / args.repeats / len(packed)
                )
    records = [None] * 4
    dist.all_gather_object(records, dict(rank=rank, samples_us=times))
    if rank == 0:
        critical = {
            str(full): [
                max(record["samples_us"][full][i] for record in records)
                for i in range(len(times[full]))
            ]
            for full in (True, False)
        }
        medians = {key: statistics.median(values) for key, values in critical.items()}
        result = dict(
            scope="HCX full-mesh communication screen; not model latency",
            wheel_version=vllm.__version__,
            core_sha256=hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
            benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            weights_sha256=hashlib.sha256(args.weights.read_bytes()).hexdigest(),
            topology=subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True),
            shape=dict(m=m, pairs=len(packed), chain_repeats=args.repeats, tp=4),
            exact_output_checks="three eager and three graph input seeds, all TP ranks",
            rank_records=records,
            critical_samples_us=critical,
            critical_median_us=medians,
            xor_delta_us=medians["True"] - medians["False"],
            interpretation=(
                "True is direct three-peer exchange; False is recursive-doubling "
                "and XOR forwarding. Each paired sample uses the maximum rank "
                "graph envelope. CPU entry skew is amortized across the chain. "
                "No source dispatch or model performance change is implied."
            ),
            clocks=subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,clocks.sm,clocks.mem",
                    "--format=csv",
                ],
                text=True,
            ),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result["critical_median_us"]), flush=True)
    del graphs
    torch.accelerator.synchronize()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
