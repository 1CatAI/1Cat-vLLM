# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the opt-in SM70 TP4 push collective for Qwen3.8.

This captures both the existing pull path and the candidate push path in one
``torchrun`` lifetime.  Each graph mirrors one 48-layer verifier round: one
regular all-reduce and one shared+routed sum2 all-reduce per layer. The default
remains the FP16 [5, 2560] MTP4 payload; ``--tokens`` also screens no-MTP
concurrency widths.

Example:
  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
    benchmarks/benchmark_sm70_tp4_mtp5_push_allreduce.py --json-out out.json
"""

import argparse
import json
import os
from pathlib import Path
from statistics import median
from typing import Any

import torch
import torch.distributed as dist

from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce

_HIDDEN_SIZE = 2560
_LAYERS = 48
_MTP5_ENV = "VLLM_SM70_TP4_PUSH_ALLREDUCE_MTP5"
_BATCH_ENV = "VLLM_SM70_TP4_PUSH_ALLREDUCE_QWEN38_BATCH"


def _make_inputs(rank: int, tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(20260828 + rank)
    shape = (tokens, _HIDDEN_SIZE)
    input_a = (
        torch.randn(shape, device="cuda", generator=generator, dtype=torch.float32)
        * 0.03
    ).half()
    input_b = (
        torch.randn(shape, device="cuda", generator=generator, dtype=torch.float32)
        * 0.02
    ).half()
    return input_a, input_b


def _capture_round(
    communicator: CustomAllreduce,
    input_a: torch.Tensor,
    input_b: torch.Tensor,
    *,
    push: bool,
    tokens: int,
) -> tuple[torch.cuda.CUDAGraph, list[torch.Tensor]]:
    os.environ[_MTP5_ENV] = "1" if push and tokens == 5 else "0"
    os.environ[_BATCH_ENV] = "1" if push and tokens in (4, 8, 16) else "0"
    torch.accelerator.synchronize()
    dist.barrier()
    regular_outputs = [torch.empty_like(input_a) for _ in range(_LAYERS)]
    sum2_outputs = [torch.empty_like(input_a) for _ in range(_LAYERS)]
    graph = torch.cuda.CUDAGraph()
    with communicator.capture(), torch.cuda.graph(graph):
        for layer in range(_LAYERS):
            communicator.all_reduce(
                input_a,
                out=regular_outputs[layer],
                registered=True,
            )
            communicator.all_reduce_sum2(
                input_a,
                input_b,
                out=sum2_outputs[layer],
            )
    torch.accelerator.synchronize()
    dist.barrier()
    outputs = [
        output
        for pair in zip(regular_outputs, sum2_outputs, strict=True)
        for output in pair
    ]
    return graph, outputs


def _time_graph(
    graph: torch.cuda.CUDAGraph,
    warmup: int,
    iterations: int,
    object_group: dist.ProcessGroup,
) -> float:
    for _ in range(warmup):
        graph.replay()
    torch.accelerator.synchronize()
    dist.barrier()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        graph.replay()
    end.record()
    end.synchronize()
    local_ms = start.elapsed_time(end) / iterations

    gathered: list[float | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local_ms, group=object_group)
    return max(value for value in gathered if value is not None)


def _compare_outputs(
    baseline: list[torch.Tensor],
    candidate: list[torch.Tensor],
) -> dict[str, Any]:
    mismatch_count = 0
    max_abs_diff = 0.0
    first_mismatch: dict[str, Any] | None = None
    for index, (control, push) in enumerate(zip(baseline, candidate, strict=True)):
        mismatch = control != push
        count = int(mismatch.sum().item())
        mismatch_count += count
        diff = (control.float() - push.float()).abs()
        max_abs_diff = max(max_abs_diff, float(diff.max().item()))
        if count and first_mismatch is None:
            coordinate = tuple(
                int(item)
                for item in torch.nonzero(mismatch, as_tuple=False)[0].tolist()
            )
            first_mismatch = {
                "collective_index": index,
                "coordinate": list(coordinate),
                "baseline": float(control[coordinate].float().item()),
                "candidate": float(push[coordinate].float().item()),
            }
    return {
        "equal": mismatch_count == 0,
        "mismatch_count": mismatch_count,
        "max_abs_diff": max_abs_diff,
        "first_mismatch": first_mismatch,
    }


def _set_pattern(
    input_a: torch.Tensor,
    input_b: torch.Tensor,
    *,
    pattern: str,
    rank: int,
) -> None:
    base = torch.arange(input_a.numel(), device="cuda", dtype=torch.float32).view_as(
        input_a
    )
    if pattern == "exact_int":
        input_a.copy_(((base % 13) + rank).half())
        input_b.copy_((((base * 3) % 17) - rank).half())
    elif pattern == "signed_zero":
        input_a.zero_()
        input_b.zero_()
        input_a[:, 1::2] = -0.0
        input_b[:, ::2] = -0.0
    elif pattern == "model_small":
        generator = torch.Generator(device="cuda")
        generator.manual_seed(20260928 + rank)
        input_a.copy_(
            (
                torch.randn(
                    input_a.shape,
                    device="cuda",
                    generator=generator,
                    dtype=torch.float32,
                )
                * 0.03
            ).half()
        )
        input_b.copy_(
            (
                torch.randn(
                    input_b.shape,
                    device="cuda",
                    generator=generator,
                    dtype=torch.float32,
                )
                * 0.02
            ).half()
        )
    else:
        raise ValueError(f"unsupported pattern: {pattern}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--timing-repeats", type=int, default=4)
    parser.add_argument("--tokens", type=int, choices=(4, 5, 8, 16), default=5)
    parser.add_argument("--json-out")
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations <= 0 or args.timing_repeats <= 0:
        raise ValueError("warmup must be >= 0; iterations/repeats must be > 0")

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 4:
        raise ValueError(f"Qwen3.8 push benchmark requires TP4, got TP{world_size}")

    torch.accelerator.set_device_index(local_rank)
    dist.init_process_group(backend="nccl")
    gloo_group = dist.new_group(backend="gloo")
    communicator = CustomAllreduce(
        group=gloo_group,
        device=local_rank,
        max_size=8 * 1024 * 1024,
    )
    try:
        if communicator.disabled:
            raise RuntimeError("custom all-reduce is disabled")
        if not communicator.fully_connected:
            raise RuntimeError("benchmark requires fully connected TP4")
        if communicator.sm70_tp4_push_buffer_ptrs is None:
            raise RuntimeError("SM70 TP4 push buffer was not registered")

        input_a, input_b = _make_inputs(rank, args.tokens)
        if rank == 0:
            print("[tp4-mtp5-ar] capture baseline pull graph", flush=True)
        baseline_graph, baseline_outputs = _capture_round(
            communicator, input_a, input_b, push=False, tokens=args.tokens
        )
        if rank == 0:
            print("[tp4-mtp5-ar] capture candidate push graph", flush=True)
        candidate_graph, candidate_outputs = _capture_round(
            communicator, input_a, input_b, push=True, tokens=args.tokens
        )
        if rank == 0:
            print("[tp4-mtp5-ar] compare dynamic graph outputs", flush=True)

        comparisons: dict[str, dict[str, Any]] = {}
        for pattern in ("exact_int", "signed_zero", "model_small"):
            # Refill captured addresses to prove dynamic replay and cover the
            # arithmetic/signed-zero contracts without another graph capture.
            _set_pattern(input_a, input_b, pattern=pattern, rank=rank)
            baseline_graph.replay()
            candidate_graph.replay()
            torch.accelerator.synchronize()
            dist.barrier()
            comparisons[pattern] = _compare_outputs(baseline_outputs, candidate_outputs)

        control_samples: list[float] = []
        candidate_samples: list[float] = []
        for repeat in range(args.timing_repeats):
            order = (
                (("baseline", baseline_graph), ("candidate", candidate_graph))
                if repeat % 2 == 0
                else (("candidate", candidate_graph), ("baseline", baseline_graph))
            )
            for name, graph in order:
                value = _time_graph(graph, args.warmup, args.iterations, gloo_group)
                (control_samples if name == "baseline" else candidate_samples).append(
                    value
                )

        control_ms = median(control_samples)
        candidate_ms = median(candidate_samples)
        collectives = 2 * _LAYERS
        savings_ms = control_ms - candidate_ms
        result = {
            "world_size": world_size,
            "shape": [args.tokens, _HIDDEN_SIZE],
            "dtype": "float16",
            "layers": _LAYERS,
            "collectives_per_round": collectives,
            "regular_allreduces_per_round": _LAYERS,
            "sum2_allreduces_per_round": _LAYERS,
            "dynamic_graph_comparisons": comparisons,
            "baseline_pull_ms_per_round": control_ms,
            "candidate_push_ms_per_round": candidate_ms,
            "savings_ms_per_round": savings_ms,
            "speedup": control_ms / candidate_ms,
            "savings_percent": savings_ms / control_ms * 100.0,
            "baseline_us_per_collective": control_ms * 1000.0 / collectives,
            "candidate_us_per_collective": candidate_ms * 1000.0 / collectives,
            "baseline_samples_ms": control_samples,
            "candidate_samples_ms": candidate_samples,
            "warmup_replays_per_sample": args.warmup,
            "timed_replays_per_sample": args.iterations,
        }
        gathered_comparisons: list[dict[str, dict[str, Any]] | None] = [
            None
        ] * world_size
        dist.all_gather_object(gathered_comparisons, comparisons, group=gloo_group)
        result["all_ranks_equal"] = all(
            item is not None
            and all(comparison["equal"] for comparison in item.values())
            for item in gathered_comparisons
        )
        result["rank_comparisons"] = gathered_comparisons

        if rank == 0:
            text = json.dumps(result, indent=2, sort_keys=True)
            print(text)
            if args.json_out:
                path = Path(args.json_out)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text + "\n")
        if not result["all_ranks_equal"]:
            raise SystemExit(1)
    finally:
        os.environ[_MTP5_ENV] = "0"
        os.environ[_BATCH_ENV] = "0"
        communicator.close()
        dist.destroy_process_group(gloo_group)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
