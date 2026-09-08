# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare frozen publication, serial chunks, and event-ordered chunk overlap.

Uses four ranks of real consecutive-layer weights with changing synthetic
activations. Each chunk owns a separate two-epoch channel; consumers wait for
their local producer before entering the original peer packet protocol.
GPU 4--7 only. No service route is installed by this benchmark.
"""

import argparse
import hashlib
import json
import os
import statistics
from pathlib import Path

import torch
import torch.distributed as dist

from vllm import _custom_ops as custom_ops
from vllm import _sm70_ops as ops
from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--control-library", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--correctness-only", action="store_true")
    p.add_argument("--cycles", type=int, default=5)
    p.add_argument("--coordination-backend", choices=["nccl", "gloo"], default="nccl")
    args = p.parse_args()
    if args.cycles < 1:
        p.error("--cycles must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "4,5,6,7"
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    assert torch.cuda.get_device_capability() == (7, 0)
    torch.ops.load_library(str(args.control_library))
    torch.ops.load_library(str(args.library))
    dist.init_process_group(args.coordination_backend)
    assert dist.get_world_size() == 4
    group = dist.new_group(backend="gloo")
    ca = CustomAllreduce(group, rank, max_size=128 * 1024)
    assert not ca.disabled and ca.sm70_tp4_push_buffer_ptrs is not None
    peers = ca.sm70_tp4_push_buffer_ptrs
    channels = []
    comm_stream = torch.cuda.Stream(device=rank)
    generator = torch.Generator(device="cuda").manual_seed(20260908 + rank)
    items, provenance, guards = [], [], []

    def output(shape):
        storage = torch.full(
            (shape[0] * shape[1] + 16,), -37, device="cuda", dtype=torch.float16
        )
        guards.append(storage)
        return storage[8:-8].view(shape)

    try:
        for path in sorted((args.root / f"rank{rank}").glob("*.pt")):
            item = torch.load(path, map_location="cpu", weights_only=True)
            assert item["rank"] == rank
            provenance.append(
                dict(
                    file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest()
                )
            )
            item["codes"] = item["codes"].cuda()
            item["scales"] = item["scales"].cuda()
            item["input"] = torch.randn(
                (8, item["k"]), dtype=torch.float16, device="cuda", generator=generator
            )
            width = item["n"] // 2 if item["gated"] else item["n"]
            item["output"] = [output((8, width)) for _ in range(3)]
            item["row_parallel"] = not item["gated"] and item["n"] == 5120
            item["reduced"] = (
                [output((8, width)) for _ in range(3)] if item["row_parallel"] else None
            )
            item["ready_events"] = [torch.cuda.Event() for _ in range(2)]
            items.append(item)
        assert len(items) == 16 and sum(i["row_parallel"] for i in items) == 8
        tail_input = torch.randn(
            (8, 5120), dtype=torch.float16, device="cuda", generator=generator
        )
        tail_outputs = [output((8, 5120)) for _ in range(3)]

        for _ in range(2):
            channel = ca.create_shared_buffer(
                custom_ops.sm70_tp4_push_allreduce_buffer_size(), group=group
            )
            channels.append(channel)
            torch.ops._qpn2_chunked.initialize(tail_input, channel[rank])
        torch.cuda.synchronize()
        dist.barrier()

        def run(arm):
            for item in items:
                arguments = [
                    item["output"][arm],
                    item["input"],
                    item["codes"],
                    item["scales"],
                    item["global_scale"],
                    item["split_k"],
                    item["nacc"],
                ]
                if item["row_parallel"]:
                    if arm == 0:
                        torch.ops._qpn2_candidate.publish(*arguments, peers, rank)
                        torch.ops._qpn2_candidate.consume(
                            item["output"][arm], item["reduced"][arm], peers, rank
                        )
                    else:
                        main_stream = torch.cuda.current_stream()
                        for chunk, channel in enumerate(channels):
                            torch.ops._qpn2_chunked.publish(
                                *arguments, channel, rank, chunk * 2560
                            )
                            if arm == 1:
                                torch.ops._qpn2_chunked.consume(
                                    item["reduced"][arm], channel, rank, chunk * 2560
                                )
                            else:
                                ready = item["ready_events"][chunk]
                                ready.record(main_stream)
                                with torch.cuda.stream(comm_stream):
                                    comm_stream.wait_event(ready)
                                    torch.ops._qpn2_chunked.consume(
                                        item["reduced"][arm],
                                        channel,
                                        rank,
                                        chunk * 2560,
                                    )
                        if arm == 2:
                            main_stream.wait_stream(comm_stream)
                else:
                    op = (
                        ops.nvfp4_qpn2_gated_sm70_out
                        if item["gated"]
                        else ops.nvfp4_qpn2_gemm_sm70_out
                    )
                    op(*arguments)
            # Odd collective count plus an ordinary push call exercises epoch
            # transitions across graphs and the two publication mechanisms.
            ca.all_reduce(tail_input, out=tail_outputs[arm], registered=True)

        graphs = []
        for arm in range(3):
            torch.cuda.synchronize()
            dist.barrier()
            graph = torch.cuda.CUDAGraph()
            with ca.capture(), torch.cuda.graph(graph):
                run(arm)
            graphs.append(graph)

        def check():
            pairs = (
                [i["output"] for i in items]
                + [i["reduced"] for i in items if i["row_parallel"]]
                + [tail_outputs]
            )
            for index, outputs in enumerate(pairs):
                a = outputs[0]
                for arm, b in enumerate(outputs[1:], 1):
                    assert torch.isfinite(a).all() and torch.isfinite(b).all(), (
                        "nonfinite",
                        rank,
                        index,
                        arm,
                    )
                    assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)), (
                        "mismatch",
                        rank,
                        index,
                        arm,
                        int(torch.count_nonzero(a != b)),
                    )
            for g in guards:
                assert (g[:8] == -37).all() and (g[-8:] == -37).all()

        for cycle in range(args.cycles):
            for item in items:
                item["input"].normal_(generator=generator)
            tail_input.normal_(generator=generator)
            for g in guards:
                g[8:-8].fill_(float("nan"))
            dist.barrier()
            for arm in (0, 1, 2) if cycle % 2 == 0 else (2, 1, 0):
                if rank == cycle % 4:
                    torch.cuda._sleep(20000)
                graphs[arm].replay()
            torch.cuda.synchronize()
            check()
        samples = [[], [], []]
        for trial in range(0 if args.correctness_only else 7):
            for arm in (0, 1, 2) if trial % 2 == 0 else (2, 1, 0):
                for _ in range(20):
                    graphs[arm].replay()
                torch.cuda.synchronize()
                dist.barrier()
                start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                start.record()
                for _ in range(50):
                    graphs[arm].replay()
                end.record()
                end.synchronize()
                ranks = [None] * 4
                dist.all_gather_object(ranks, start.elapsed_time(end) / 50, group=group)
                samples[arm].append(max(ranks))
        check()
        ranks = [None] * 4
        dist.all_gather_object(
            ranks, dict(rank=rank, snapshots=provenance), group=group
        )
        if rank == 0:
            result = dict(
                rank_weights=ranks,
                shape="TP4 B1/q8 four consecutive layers",
                projections=16,
                row_parallel=8,
                extra_ordinary_push=1,
                activation_kind="synthetic_fp16_normal",
                all_outputs_bitwise_equal=True,
                canaries_intact=True,
                changing_input_cycles=args.cycles,
                rank_skew_cycles=20000,
                medians_ms=[statistics.median(x) for x in samples]
                if not args.correctness_only
                else None,
                samples_ms=samples,
                arms=["frozen_publisher", "two_chunks_serial", "two_chunks_overlap"],
                paired_saved_ms=[
                    [a - b for a, b in zip(samples[0], candidate)]
                    for candidate in samples[1:]
                ],
                control_library_sha256=hashlib.sha256(
                    args.control_library.read_bytes()
                ).hexdigest(),
                library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
                communicator_sha256=hashlib.sha256(
                    Path(os.environ["VLLM_SM70_CUSTOM_AR_LIBRARY"]).read_bytes()
                ).hexdigest(),
                complete_round_performance=False,
            )
            args.output.write_text(json.dumps(result, indent=2))
            print(
                json.dumps(
                    {k: v for k, v in result.items() if k != "rank_weights"}, indent=2
                ),
                flush=True,
            )
    finally:
        # All streams and ranks have finished reading peer allocations before
        # the owning rank releases either channel.
        torch.cuda.synchronize()
        dist.barrier(group=group)
        for channel in channels:
            ca.free_shared_buffer(channel, rank=rank)
        ca.close()
        dist.destroy_process_group(group)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
