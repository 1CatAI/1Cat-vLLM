# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Measure full-group and representative peer-link all-reduce crossovers.

Run with torchrun on one node, one process per visible GPU. Results describe
complete dependent graph calls, not isolated wire latency or model throughput.
Use a fresh output directory for each run. The ordinary communicator retains
its serving capture/registration lifecycle, including for pair comparisons.
"""

import argparse
import json
import socket
from datetime import timedelta
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--sizes-kib", type=int, nargs="+", default=[1, 5, 10, 25, 40, 80]
    )
    parser.add_argument(
        "--precisions", choices=("fp16", "fp32"), nargs="+", default=["fp16", "fp32"]
    )
    parser.add_argument("--budget", type=int, default=2048)
    parser.add_argument("--shortlist", type=int, default=8)
    parser.add_argument(
        "--groups", choices=("full", "pairs"), nargs="+", default=["full", "pairs"]
    )
    args = parser.parse_args()
    if any(size <= 0 for size in args.sizes_kib):
        parser.error("Payload sizes must be positive KiB counts")

    import torch
    import torch.distributed as dist

    from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator
    from vllm.distributed.device_communicators.topology_all_reduce import (
        collective_tuner,
        planner_probe,
    )

    torch.set_num_threads(1)
    dist.init_process_group("gloo", timeout=timedelta(seconds=1800))
    rank, world = dist.get_rank(), dist.get_world_size()
    hosts = [None] * world
    dist.all_gather_object(hosts, socket.gethostname())
    if len(set(hosts)) != 1 or world != torch.accelerator.device_count():
        raise ValueError("Use one process per visible GPU on a single node")
    torch.accelerator.set_device_index(rank)
    device = torch.device("cuda", rank)
    visible = planner_probe.visible_devices()
    topology = planner_probe.discover()
    groups = [("full", list(range(world)))] if "full" in args.groups else []
    representatives = {}
    for a in range(world):
        for b in range(a + 1, world):
            label = topology.labels.get((visible[a], visible[b]))
            if label and (visible[b], visible[a]) in topology.labels:
                representatives.setdefault(label, [a, b])
    if "pairs" in args.groups:
        groups.extend(sorted(representatives.items()))
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "manifest.json").write_text(
            json.dumps(
                dict(
                    groups=groups,
                    sizes_kib=args.sizes_kib,
                    precisions=args.precisions,
                    budget=args.budget,
                    shortlist=args.shortlist,
                    grids="automatic",
                    torch=str(torch.__version__),
                    cuda=torch.version.cuda,
                ),
                indent=2,
            )
            + "\n"
        )
    dist.barrier()
    rows = []
    try:
        for label, members in groups:
            group = dist.new_group(members, backend="gloo")
            if rank in members:
                backend = CudaCommunicator(
                    group, device=device, unique_name="tp-benchmark"
                )
                try:
                    for precision in args.precisions:
                        for size in args.sizes_kib:
                            cache = args.output / f"{label}-{precision}-{size}.json"
                            for repeat in range(2):
                                prepared = collective_tuner.prepare_best(
                                    group,
                                    device,
                                    topology,
                                    existing=backend,
                                    payload_bytes=size * 1024,
                                    precision=precision,
                                    budget=args.budget,
                                    shortlist=args.shortlist,
                                    cache=cache,
                                    context=dict(group=members, link=label),
                                )
                                try:
                                    if (
                                        prepared.channel is not None
                                        and world > 1
                                        and prepared.receipt["cache_hit"]
                                        != bool(repeat)
                                    ):
                                        raise RuntimeError("Unexpected cache reuse")
                                    rows.append(
                                        dict(
                                            group=label,
                                            ranks=members,
                                            precision=precision,
                                            kib=size,
                                            repeat=repeat,
                                            receipt=prepared.receipt,
                                        )
                                    )
                                finally:
                                    prepared.close()
                            (args.output / f"rank{rank}.json").write_text(
                                json.dumps(rows, indent=2) + "\n"
                            )
                finally:
                    backend.destroy()
                    dist.destroy_process_group(group)
            dist.barrier()
            if rank == 0:
                print(f"Completed {label}", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
