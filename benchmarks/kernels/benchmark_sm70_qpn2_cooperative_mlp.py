# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate cooperative gate/up and down publication on four ranks of real weights.

This screen uses synthetic activations and sixteen projections from consecutive
layers, with eight dependent all-reduces and an extra ordinary push call to
exercise mixed-protocol epoch transitions. It is not a complete model round.
Set VLLM_SM70_CUSTOM_AR_LIBRARY to a communicator built from the same header
as the candidate. Use Gloo process coordination for focused sanitizer runs.
"""

import argparse
import hashlib
import json
import os
import statistics
from pathlib import Path

import torch
import torch.distributed as dist

from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--publisher-library", type=Path, required=True)
    p.add_argument("--capped-library", type=Path, required=True)
    p.add_argument("--correctness-only", action="store_true")
    p.add_argument("--cycles", type=int, default=5)
    p.add_argument("--coordination-backend", choices=["nccl", "gloo"], default="nccl")
    args = p.parse_args()
    if args.cycles < 1:
        p.error("--cycles must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    assert torch.cuda.get_device_capability() == (7, 0)
    torch.ops.load_library(str(args.library))
    torch.ops.load_library(str(args.publisher_library))
    torch.ops.load_library(str(args.capped_library))
    resources = torch.ops._qpn2_coop_mlp.resources()
    assert resources[3] >= 2, resources
    dist.init_process_group(args.coordination_backend)
    assert dist.get_world_size() == 4
    group = dist.new_group(backend="gloo")
    ca = CustomAllreduce(group, rank, max_size=128 * 1024)
    assert not ca.disabled and ca.sm70_tp4_push_buffer_ptrs is not None
    peers = ca.sm70_tp4_push_buffer_ptrs
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
            item["output"] = [output((8, width)) for _ in range(2)]
            item["row_parallel"] = not item["gated"] and item["n"] == 5120
            item["reduced"] = (
                [output((8, width)) for _ in range(2)] if item["row_parallel"] else None
            )
            items.append(item)
        assert len(items) == 16 and sum(i["row_parallel"] for i in items) == 8
        tail_input = torch.randn(
            (8, 5120), dtype=torch.float16, device="cuda", generator=generator
        )
        tail_outputs = [output((8, 5120)) for _ in range(2)]

        gates = {x["layer"]: x for x in items if x["gated"]}
        downs = {x["layer"]: x for x in items if x["row_parallel"] and x["k"] == 4352}
        for layer in range(4):
            assert (
                next(i for i, x in enumerate(items) if x is downs[layer])
                == next(i for i, x in enumerate(items) if x is gates[layer]) + 1
            )

        rejected_rows = []
        gate, down = gates[0], downs[0]
        for rows in (1, 2, 4, 7, 9, 16):
            try:
                torch.ops._qpn2_coop_mlp.mlp(
                    gate["output"][0].new_empty((rows, 4352)),
                    down["output"][0].new_empty((rows, 5120)),
                    gate["input"].new_empty((rows, 5120)),
                    gate["codes"],
                    gate["scales"],
                    down["codes"],
                    down["scales"],
                    gate["global_scale"],
                    down["global_scale"],
                    peers,
                    rank,
                )
            except RuntimeError as error:
                assert "exact TP4 q8 MLP required" in str(error), error
                rejected_rows.append(rows)
            else:
                raise AssertionError(("unexpected non-q8 launch", rows))

        def run(arm):
            for item in items:
                down = item["row_parallel"] and item["k"] == 4352
                input_ = gates[item["layer"]]["output"][arm] if down else item["input"]
                if arm and item["gated"]:
                    paired = downs[item["layer"]]
                    torch.ops._qpn2_coop_mlp.mlp(
                        item["output"][arm],
                        paired["output"][arm],
                        input_,
                        item["codes"],
                        item["scales"],
                        paired["codes"],
                        paired["scales"],
                        item["global_scale"],
                        paired["global_scale"],
                        peers,
                        rank,
                    )
                    continue
                arguments = [
                    item["output"][arm],
                    input_,
                    item["codes"],
                    item["scales"],
                    item["global_scale"],
                    item["split_k"],
                    item["nacc"],
                ]
                if item["row_parallel"]:
                    if not (arm and down):
                        torch.ops._qpn2_candidate.publish(*arguments, peers, rank)
                    torch.ops._qpn2_candidate.consume(
                        item["output"][arm], item["reduced"][arm], peers, rank
                    )
                else:
                    op = (
                        torch.ops._qpn2_capped.gated
                        if item["gated"]
                        else torch.ops._qpn2_capped.gemm
                    )
                    op(*arguments)
            ca.all_reduce(tail_input, out=tail_outputs[arm], registered=True)

        graphs = []
        for arm in range(2):
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
            for index, (a, b) in enumerate(pairs):
                assert torch.isfinite(a).all() and torch.isfinite(b).all(), (
                    "nonfinite",
                    rank,
                    index,
                )
                assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)), (
                    "mismatch",
                    rank,
                    index,
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
            for arm in (0, 1) if cycle % 2 == 0 else (1, 0):
                if rank == cycle % 4:
                    torch.cuda._sleep(20000)
                graphs[arm].replay()
            torch.cuda.synchronize()
            check()
        samples = [[], []]
        for trial in range(0 if args.correctness_only else 7):
            for arm in (0, 1) if trial % 2 == 0 else (1, 0):
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
                cooperative_resources=resources,
                gate_down_dependency=True,
                rejected_rows=rejected_rows,
                publisher_sha256=hashlib.sha256(
                    args.publisher_library.read_bytes()
                ).hexdigest(),
                capped_sha256=hashlib.sha256(
                    args.capped_library.read_bytes()
                ).hexdigest(),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
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
                paired_saved_ms=[a - b for a, b in zip(*samples)],
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
        ca.close()
        dist.destroy_process_group(group)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
