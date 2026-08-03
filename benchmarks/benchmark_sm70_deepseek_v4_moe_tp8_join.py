# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure the DeepSeek-V4 MoE tail joined to hierarchical TP8 all-reduce."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
import torch.distributed as dist
from benchmark_sm70_deepseek_v4_shared_moe_priority import Fixture
from benchmark_sm70_tp8_hierarchical_allreduce import _measure_graph, _rank_max

from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce


def _capture(
    custom: CustomAllreduce,
    fixture: Fixture,
    *,
    collectives: int,
    fused_reduce_add: bool,
) -> torch.cuda.CUDAGraph:
    main = torch.cuda.Stream()
    aux = torch.cuda.Stream()
    reduce_output = fixture.reduce_output

    def layer(registered: bool) -> None:
        aux.wait_stream(main)
        with torch.cuda.stream(aux):
            fixture.shared_call()
        fixture.routed_call(False, fused_reduce_add)
        main.wait_stream(aux)
        if fused_reduce_add:
            fixture.fused_reduce_add()
        else:
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
    diff = (candidate.float() - reference.float()).abs()
    local_equal = torch.equal(candidate, reference)
    equal_flags: list[bool | None] = [None] * dist.get_world_size()
    dist.all_gather_object(equal_flags, local_equal)
    candidate_host = candidate.detach().cpu()
    rank_outputs: list[torch.Tensor | None] = [None] * dist.get_world_size()
    dist.all_gather_object(rank_outputs, candidate_host)
    first = next(item for item in rank_outputs if item is not None)
    return {
        "equal_to_baseline": all(bool(item) for item in equal_flags),
        "all_ranks_bitwise_equal": all(
            item is not None and torch.equal(item, first) for item in rank_outputs
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
    world_size = dist.get_world_size()
    if world_size != 8:
        raise RuntimeError(f"This benchmark requires TP8, got {world_size} ranks.")
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    if torch.cuda.get_device_capability(device) != (7, 0):
        raise RuntimeError("This benchmark requires NVIDIA V100/SM70 GPUs.")

    custom = CustomAllreduce(group=dist.group.WORLD, device=device)
    if custom.disabled or not custom.tp8_hierarchical:
        raise RuntimeError("The hierarchical TP8 custom all-reduce is unavailable.")

    fixture = Fixture(args.seed + rank)
    fixture.reduce_output = torch.empty_like(fixture.combined_output)
    baseline_graph = _capture(
        custom,
        fixture,
        collectives=args.collectives,
        fused_reduce_add=False,
    )
    candidate_graph = _capture(
        custom,
        fixture,
        collectives=args.collectives,
        fused_reduce_add=True,
    )

    baseline_graph.replay()
    torch.cuda.synchronize()
    baseline_output = fixture.reduce_output.clone()
    candidate_graph.replay()
    torch.cuda.synchronize()
    initial = _compare(fixture.reduce_output, baseline_output)

    fixture.x.add_(0.03125)
    baseline_graph.replay()
    torch.cuda.synchronize()
    baseline_output.copy_(fixture.reduce_output)
    candidate_graph.replay()
    torch.cuda.synchronize()
    dynamic = _compare(fixture.reduce_output, baseline_output)

    baseline_samples: list[float] = []
    candidate_samples: list[float] = []
    for repeat in range(args.repeats):
        ordered = (
            ((baseline_graph, baseline_samples), (candidate_graph, candidate_samples))
            if repeat % 2 == 0
            else (
                (candidate_graph, candidate_samples),
                (baseline_graph, baseline_samples),
            )
        )
        for graph, samples in ordered:
            samples.append(
                _rank_max(
                    _measure_graph(
                        graph,
                        collectives=args.collectives,
                        warmup=args.warmup,
                        iterations=args.iterations,
                    )
                )
            )

    baseline_median = statistics.median(baseline_samples)
    candidate_median = statistics.median(candidate_samples)
    payload = {
        "contract": {
            "world_size": world_size,
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
        "baseline_unpermute_add_ar": {
            "rank_max_samples_ms": baseline_samples,
            "rank_max_median_ms": baseline_median,
        },
        "candidate_fused_reduce_add_ar": {
            "rank_max_samples_ms": candidate_samples,
            "rank_max_median_ms": candidate_median,
        },
        "speedup": baseline_median / candidate_median,
        "projected_43_layer_saving_ms": 43 * (baseline_median - candidate_median),
    }
    if rank == 0:
        encoded = json.dumps(payload, indent=2)
        print(encoded)
        if args.output_json is not None:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(encoded + "\n", encoding="utf-8")

    passed = all(
        bool(item["equal_to_baseline"]) and bool(item["all_ranks_bitwise_equal"])
        for item in (initial, dynamic)
    )
    torch.cuda.synchronize()
    dist.barrier()
    custom.close()
    dist.barrier()
    dist.destroy_process_group()
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
