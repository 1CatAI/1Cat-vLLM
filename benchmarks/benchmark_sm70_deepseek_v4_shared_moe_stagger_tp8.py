#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Join router-first shared-MoE scheduling to the SM70 TP8 all-reduce."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
import torch.distributed as dist
from benchmark_sm70_deepseek_v4_shared_moe_stagger import Fixture, Schedule
from benchmark_sm70_tp8_hierarchical_allreduce import _measure_graph, _rank_max

from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce

CONTROL = Schedule("overlap_root_release", True, False)
CANDIDATE = Schedule("overlap_router_first", True, True)


def _capture(
    custom: CustomAllreduce,
    fixture: Fixture,
    schedule: Schedule,
    collectives: int,
) -> torch.cuda.CUDAGraph:
    main = torch.cuda.Stream()
    auxiliary = torch.cuda.Stream()
    reduce_output = fixture.reduce_output

    def layer(registered: bool) -> None:
        if not schedule.router_first:
            auxiliary.wait_stream(main)
        topk_weights, topk_ids = fixture.router_call()
        if schedule.router_first:
            auxiliary.wait_stream(main)
        fixture.routed_call(topk_weights, topk_ids)
        with torch.cuda.stream(auxiliary):
            fixture.shared_call()
        main.wait_stream(auxiliary)
        fixture.combine()
        custom.all_reduce(
            fixture.combined_output.flatten(),
            out=reduce_output.flatten(),
            registered=registered,
        )

    main.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(main):
        for _ in range(4):
            layer(False)
    main.synchronize()

    graph = torch.cuda.CUDAGraph()
    with custom.capture(), torch.cuda.graph(graph, stream=main):
        for _ in range(collectives):
            layer(True)
    return graph


def _compare(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, object]:
    local_equal = torch.equal(candidate, reference)
    equal_flags: list[bool | None] = [None] * dist.get_world_size()
    dist.all_gather_object(equal_flags, local_equal)
    rank_outputs: list[torch.Tensor | None] = [None] * dist.get_world_size()
    dist.all_gather_object(rank_outputs, candidate.detach().cpu())
    first = next(output for output in rank_outputs if output is not None)
    diff = (candidate.float() - reference.float()).abs()
    return {
        "equal_to_control": all(bool(flag) for flag in equal_flags),
        "all_ranks_bitwise_equal": all(
            output is not None and torch.equal(output, first) for output in rank_outputs
        ),
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collectives", type=int, default=43)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    if dist.get_world_size() != 8:
        raise RuntimeError("This benchmark requires eight ranks.")
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    if torch.cuda.get_device_capability(device) != (7, 0):
        raise RuntimeError("This benchmark requires NVIDIA V100/SM70 GPUs.")

    custom = CustomAllreduce(group=dist.group.WORLD, device=device)
    if custom.disabled or not custom.tp8_hierarchical:
        raise RuntimeError("The SM70 TP8 hierarchical all-reduce is unavailable.")

    fixture = Fixture(args.seed + rank)
    fixture.reduce_output = torch.empty_like(fixture.combined_output)
    control_graph = _capture(custom, fixture, CONTROL, args.collectives)
    candidate_graph = _capture(custom, fixture, CANDIDATE, args.collectives)

    control_graph.replay()
    torch.cuda.synchronize()
    control_output = fixture.reduce_output.clone()
    candidate_graph.replay()
    torch.cuda.synchronize()
    initial = _compare(fixture.reduce_output, control_output)

    fixture.x.add_(0.03125)
    control_graph.replay()
    torch.cuda.synchronize()
    control_output.copy_(fixture.reduce_output)
    candidate_graph.replay()
    torch.cuda.synchronize()
    dynamic = _compare(fixture.reduce_output, control_output)

    samples: dict[str, list[float]] = {CONTROL.name: [], CANDIDATE.name: []}
    schedules = ((CONTROL, control_graph), (CANDIDATE, candidate_graph))
    for repeat in range(args.repeats):
        order = schedules if repeat % 2 == 0 else tuple(reversed(schedules))
        for schedule, graph in order:
            samples[schedule.name].append(
                _rank_max(
                    _measure_graph(
                        graph,
                        collectives=args.collectives,
                        warmup=args.warmup,
                        iterations=args.iterations,
                    )
                )
            )

    control_median = statistics.median(samples[CONTROL.name])
    candidate_median = statistics.median(samples[CANDIDATE.name])
    payload = {
        "contract": {
            "world_size": dist.get_world_size(),
            "batch": 1,
            "layers": args.collectives,
            "cuda_graph": True,
            "hierarchical_allreduce": True,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "repeats": args.repeats,
            "seed": args.seed,
        },
        "correctness": {"initial": initial, "dynamic_replay": dynamic},
        CONTROL.name: {
            "rank_max_samples_ms_per_layer": samples[CONTROL.name],
            "rank_max_median_ms_per_layer": control_median,
        },
        CANDIDATE.name: {
            "rank_max_samples_ms_per_layer": samples[CANDIDATE.name],
            "rank_max_median_ms_per_layer": candidate_median,
        },
        "speedup": control_median / candidate_median,
        "projected_saving_ms_per_token": args.collectives
        * (control_median - candidate_median),
    }
    if rank == 0:
        encoded = json.dumps(payload, indent=2)
        print(encoded)
        if args.output_json is not None:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(encoded + "\n", encoding="utf-8")

    passed = all(
        bool(result["equal_to_control"]) and bool(result["all_ranks_bitwise_equal"])
        for result in (initial, dynamic)
    )
    torch.cuda.synchronize()
    dist.barrier()
    custom.close()
    dist.barrier()
    dist.destroy_process_group()
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
