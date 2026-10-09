# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check and time both HCX schedules in the same installed SM70 wheel.

Run with torchrun on four idle V100s and the GPU ownership locks. The input
file contains eight actual HC weight records (down, inj, up, nw). Activations
are synthetic; these complete-boundary timings are not model round latency.
No auxiliary kernel library is loaded.
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
from vllm.models.qwen4_exp.nvidia.sm70_hcx import Sm70HcxRuntime, pack_down, pack_up


def assert_bitwise(actual, expected):
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(
            result.view(torch.int16), reference.view(torch.int16), rtol=0, atol=0
        )


@torch.inference_mode()
def measure(runtime, packed, rank, m, args):
    partial = torch.empty(m, 2560, device="cuda", dtype=torch.float16)
    secondary = torch.empty_like(partial)
    hidden = torch.empty(m, 10240, device="cuda", dtype=torch.float16)
    injection = torch.empty(m, 4, device="cuda", dtype=torch.float16)
    outputs = tuple(torch.empty_like(x) for x in (hidden, partial, injection))

    def run(weight, schedule, second=None, norm=None):
        down, up, nw = weight
        arguments = (
            partial,
            second,
            hidden,
            injection,
            nw if norm is None else norm,
            1e-6,
            down,
            up,
            *outputs,
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
        if schedule is None:
            torch.ops._C.sm70_hcx_out(*arguments)
        else:
            torch.ops._C.sm70_hcx_out(*arguments, schedule)

    def change_input(seed):
        torch.manual_seed(seed)
        hidden.normal_()
        injection.normal_()
        torch.manual_seed(seed + rank)
        partial.normal_(0, 0.5)
        secondary.normal_(0, 0.25)
        if seed == 0:
            hidden.zero_()
            partial.zero_()

    checks = 0
    for seed in (20261009, 20261010, 20261011, 0):
        change_input(seed)
        for weight in packed:
            second = secondary if seed == 20261010 else None
            norm = weight[2][:2560].contiguous() if seed == 20261011 else None
            run(weight, False, second, norm)
            reference = tuple(x.clone() for x in outputs)
            for schedule in (True, None):
                run(weight, schedule, second, norm)
                assert_bitwise(outputs, reference)
                checks += 1
    torch.cuda.synchronize()
    graphs = {}
    for schedule in (False, True):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(args.repeats):
                for weight in packed:
                    run(weight, schedule)
        graphs[schedule] = graph
        graph.replay()
    torch.cuda.synchronize()
    for seed in (20261012, 20261013, 20261014):
        change_input(seed)
        graphs[False].replay()
        reference = tuple(x.clone() for x in outputs)
        graphs[True].replay()
        assert_bitwise(outputs, reference)
        checks += 1
    torch.cuda.synchronize()
    for _ in range(4):
        for graph in graphs.values():
            graph.replay()
    torch.cuda.synchronize()
    times = {"reference": [], "candidate": []}
    for sample in range(args.samples):
        order = (
            (False, True, True, False)
            if sample % 2 == 0
            else (True, False, False, True)
        )
        for schedule in order:
            dist.barrier()
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            graphs[schedule].replay()
            end.record()
            end.synchronize()
            us = start.elapsed_time(end) * 1000 / args.repeats / len(packed)
            times["candidate" if schedule else "reference"].append(us)
    local = dict(rank=rank, bitwise_checks=checks, samples_us=times)
    records = [None] * 4
    dist.all_gather_object(records, local)
    critical = {
        key: [
            max(r["samples_us"][key][i] for r in records)
            for i in range(len(times[key]))
        ]
        for key in times
    }
    medians = {key: statistics.median(values) for key, values in critical.items()}
    del graphs
    torch.cuda.synchronize()
    return dict(
        M=m,
        bitwise_checks_per_rank=checks,
        critical_median_us=medians,
        critical_samples_us=critical,
        rank_records=records,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--rows", type=int, nargs="+", default=[1, 5, 8], choices=range(1, 9)
    )
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=64)
    args = parser.parse_args()
    if "site-packages" not in vllm.__file__:
        raise RuntimeError("Run against an installed source-complete wheel")
    if min(args.samples, args.repeats) < 1:
        raise ValueError("Positive sample and repeat counts required")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    dist.init_process_group("gloo")
    if dist.get_world_size() != 4:
        raise RuntimeError("TP4 required")
    runtime = Sm70HcxRuntime(dist.group.WORLD, torch.device("cuda", rank))
    if not runtime.enabled or not runtime.full:
        raise RuntimeError(f"Full mesh HCX required: {runtime.reason}")
    records = torch.load(args.weights, map_location="cpu", weights_only=True)[:8]
    packed = []
    for record in records:
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
    if len(packed) != 8:
        raise RuntimeError("Eight actual weight pairs required")
    results = []
    for m in args.rows:
        result = measure(runtime, packed, rank, m, args)
        results.append(result)
        if rank == 0:
            print(
                json.dumps(
                    {
                        "M": m,
                        **result["critical_median_us"],
                        "bitwise_checks_per_rank": result["bitwise_checks_per_rank"],
                    }
                ),
                flush=True,
            )
    if rank == 0:
        args.output.write_text(
            json.dumps(
                dict(
                    scope="complete installed HCX boundary; not model round latency",
                    torch=torch.__version__,
                    cuda=torch.version.cuda,
                    vllm=vllm.__version__,
                    gpu=torch.cuda.get_device_name(),
                    core_sha256=hashlib.sha256(
                        Path(core.__file__).read_bytes()
                    ).hexdigest(),
                    benchmark_sha256=hashlib.sha256(
                        Path(__file__).read_bytes()
                    ).hexdigest(),
                    weights_sha256=hashlib.sha256(
                        args.weights.read_bytes()
                    ).hexdigest(),
                    weight_pairs=len(packed),
                    boundaries_per_replay=args.repeats * len(packed),
                    full_mesh=runtime.full,
                    results=results,
                ),
                indent=2,
            )
            + "\n"
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
