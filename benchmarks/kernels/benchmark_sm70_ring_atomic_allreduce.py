# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Research-only verified-NVLink-edge FP16/FP32 allreduce screen.

Hold /tmp/gpu0-3.lock and use idle GPUs. Compare complete graph collectives
against the installed PyNCCL dispatcher, not against extrapolated timings.
No production routing, topology override or environment switch is added.
"""

import argparse
import ctypes
import datetime
import hashlib
import itertools
import json
import os
import socket
import statistics
import subprocess
from pathlib import Path

import pynvml
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def check_exclusive():
    pids = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            "0,1,2,3",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    foreign = {int(p) for p in pids.split()} - {os.getpid()}
    if foreign:
        raise RuntimeError(f"Foreign GPU processes invalidate this run: {foreign}")


def discover_order(devices):
    """Find a hypercube embedding using only direct NVLink P2P edges."""
    world = len(devices)
    if world not in (2, 4, 8):
        raise ValueError("Require 2/4/8 devices")
    pynvml.nvmlInit()
    try:
        handles = [pynvml.nvmlDeviceGetHandleByIndex(d) for d in devices]
        direct = [
            [
                i == j
                or pynvml.nvmlDeviceGetP2PStatus(
                    handles[i], handles[j], pynvml.NVML_P2P_CAPS_INDEX_NVLINK
                )
                == pynvml.NVML_P2P_STATUS_OK
                for j in range(world)
            ]
            for i in range(world)
        ]
        uuids = [pynvml.nvmlDeviceGetUUID(h) for h in handles]
    finally:
        pynvml.nvmlShutdown()
    stages = world.bit_length() - 1
    for tail in itertools.permutations(range(1, world)):
        order = (0, *tail)
        if all(
            direct[order[i]][order[i ^ (1 << bit)]]
            for i in range(world)
            for bit in range(stages)
        ):
            return list(order), direct, uuids
    raise RuntimeError("No verified direct-edge hypercube in this GPU set")


def worker(rank, args, order, port):
    from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

    world = len(order)
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world,
        timeout=datetime.timedelta(seconds=120),
    )
    torch.ops.load_library(str(args.library.resolve()))
    stages = world.bit_length() - 1
    capacity = (16 * args.hidden + 1) // 2
    blocks = (2 * capacity + 255) // 256
    logical = order.index(rank)
    neighbors = {order[logical ^ (1 << bit)] for bit in range(stages)} | {rank}
    pointers = CustomAllreduce.create_shared_buffer(
        2 * stages * capacity * 16 + 2 * stages * blocks * 4,
        group=dist.group.WORLD,
        peer_ranks=neighbors,
    )
    # System-scoped atomic operations on peer device memory require native
    # atomics on every accessed edge, in addition to direct NVLink access.
    cudart = ctypes.CDLL(str(Path(os.environ["CUDA_HOME"]) / "lib64/libcudart.so"))
    for peer in neighbors - {rank}:
        supported = ctypes.c_int()
        status = cudart.cudaDeviceGetP2PAttribute(
            ctypes.byref(supported), 3, rank, peer
        )
        if status or supported.value != 1:
            raise RuntimeError(f"Native peer atomics unavailable: {rank}->{peer}")
    addresses = torch.tensor(pointers, dtype=torch.int64, device="cuda")
    counters = torch.zeros(blocks, dtype=torch.int32, device="cuda")
    nccl = PyNcclCommunicator(dist.group.WORLD, device=rank)
    if not nccl.available or nccl.disabled:
        raise RuntimeError("No PyNCCL control")
    report = {"rank": rank, "cases": [], "replay_checks": []}
    graphs = {}
    try:
        for size in (
            1,
            17,
            257,
            args.hidden,
            2 * args.hidden,
            4 * args.hidden,
            5 * args.hidden,
            8 * args.hidden,
            16 * args.hidden,
        ):
            x = torch.empty(size, device="cuda", dtype=torch.float16)
            candidate = torch.empty_like(x)
            x.fill_(rank + 1)
            dist.barrier()
            torch.ops._C.sm70_ring_atomic_allreduce_out(
                candidate,
                x,
                addresses,
                counters,
                order,
                rank,
                capacity,
                args.block_packets,
            )
            torch.cuda.synchronize()
            expected = world * (world + 1) / 2
            if not bool((candidate == expected).all()):
                raise RuntimeError("Initial transport arithmetic mismatch")
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                torch.ops._C.sm70_ring_atomic_allreduce_out(
                    candidate,
                    x,
                    addresses,
                    counters,
                    order,
                    rank,
                    capacity,
                    args.block_packets,
                )
            graphs[size] = (graph, x, candidate)

        # Exercise odd tails and independently counted inactive blocks when
        # graph widths shrink/grow. Inputs are exactly summable FP16 integers.
        sequence = list(graphs) + list(reversed(graphs)) + list(graphs)
        for step, size in enumerate(sequence):
            graph, x, candidate = graphs[size]
            torch.manual_seed(4000 + step * world + rank)
            x.random_(-7, 8)
            candidate.fill_(float("nan"))
            inputs = [None] * world
            dist.all_gather_object(inputs, x.cpu())
            expected = torch.stack(inputs).double().sum(0).half()
            dist.barrier()
            graph.replay()
            torch.cuda.synchronize()
            if not torch.equal(candidate.cpu(), expected):
                raise RuntimeError(f"Changed/poisoned replay mismatch at {size}")
            report["replay_checks"].append({"elements": size, "integer_exact": True})

        # Half subnormals and noninteger values use an independent FP64 sum.
        for scale in (2**-24, 0.03, 1.0, 3.0):
            graph, x, candidate = graphs[args.hidden]
            if scale == 2**-24:
                x.fill_(scale)
            else:
                x.normal_(0, scale)
            inputs = [None] * world
            dist.all_gather_object(inputs, x.cpu())
            expected = torch.stack(inputs).double().sum(0).half()
            candidate.fill_(float("nan"))
            dist.barrier()
            graph.replay()
            torch.cuda.synchronize()
            delta = candidate.cpu().float() - expected.float()
            valid = bool(torch.isfinite(candidate).all()) and torch.allclose(
                candidate.cpu(), expected, atol=2**-24, rtol=0.001
            )
            if not valid:
                raise RuntimeError("FP64/subnormal oracle mismatch")
            report["replay_checks"].append(
                {
                    "scale": scale,
                    "max_abs": float(delta.abs().max()),
                    "different_half_bits": int(
                        (
                            candidate.cpu().view(torch.int16)
                            != expected.view(torch.int16)
                        ).sum()
                    ),
                }
            )

        for m in (1, 2, 4, 5, 8, 16):
            size = m * args.hidden
            inputs = [
                torch.randn(size, device="cuda", dtype=torch.float16)
                for _ in range(args.calls)
            ]
            outputs = [[torch.empty_like(x) for x in inputs] for _ in range(2)]
            chains = []
            for arm in range(2):

                def launch(arm=arm, inputs=inputs, outputs=outputs):
                    for x, out in zip(inputs, outputs[arm]):
                        if arm == 0:
                            nccl.all_reduce(x, out_tensor=out)
                        else:
                            torch.ops._C.sm70_ring_atomic_allreduce_out(
                                out,
                                x,
                                addresses,
                                counters,
                                order,
                                rank,
                                capacity,
                                args.block_packets,
                            )

                dist.barrier()
                for _ in range(3):
                    launch()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    launch()
                chains.append(graph)
            samples = [[], []]
            for sample in range(6):
                for offset in range(2):
                    arm = (sample + offset) % 2
                    dist.barrier()
                    for _ in range(5):
                        chains[arm].replay()
                    torch.cuda.synchronize()
                    dist.barrier()
                    start, end = [
                        torch.cuda.Event(enable_timing=True) for _ in range(2)
                    ]
                    start.record()
                    for _ in range(30):
                        chains[arm].replay()
                    end.record()
                    end.synchronize()
                    samples[arm].append(
                        start.elapsed_time(end) * 1000 / 30 / args.calls
                    )
            case = {
                "M": m,
                "input_bytes": size * 2,
                "packet_store_bytes_per_rank": size * 8 * stages,
                "grid_blocks": (size + 255) // 256,
                "samples_us": samples,
                "median_us": [statistics.median(s) for s in samples],
            }
            report["cases"].append(case)
            if rank == 0:
                print(json.dumps(case), flush=True)
            del chains
        dist.barrier()
        args.out.with_name(args.out.stem + f".rank{rank}.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
    finally:
        graphs.clear()
        torch.cuda.synchronize()
        dist.barrier()
        CustomAllreduce.free_shared_buffer(pointers, rank=rank)
        nccl.destroy()
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--hidden", type=int, default=2560)
    parser.add_argument("--calls", type=int, default=96)
    parser.add_argument("--block-packets", action="store_true")
    args = parser.parse_args()
    check_exclusive()
    devices = [int(d) for d in args.gpus.split(",")]
    if args.hidden < 257 or args.calls <= 0:
        parser.error("Require hidden >=257 and positive calls")
    order, direct, uuids = discover_order(devices)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "complete": False,
        "research_only": True,
        "endpoint": False,
        "world": len(devices),
        "devices": devices,
        "uuid": uuids,
        "rank_order": order,
        "direct_nvlink": direct,
        "library_sha256": hashlib.sha256(args.library.read_bytes()).hexdigest(),
        "hidden": args.hidden,
        "calls_per_graph": args.calls,
        "block_packets": args.block_packets,
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(worker, args=(args, order, port), nprocs=len(devices), join=True)
    report["ranks"] = [
        json.loads(args.out.with_name(args.out.stem + f".rank{r}.json").read_text())
        for r in range(len(devices))
    ]
    check_exclusive()
    report["complete"] = True
    args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
