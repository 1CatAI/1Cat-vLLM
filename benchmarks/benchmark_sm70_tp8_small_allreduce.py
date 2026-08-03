# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare graph-captured TP8 NCCL and custom all-reduce for decode shapes."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import torch
import torch.distributed as dist

from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator


def _capture(call) -> torch.cuda.CUDAGraph:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(4):
            call()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        call()
    torch.cuda.synchronize()
    return graph


def _capture_custom(
    custom: CustomAllreduce, inp: torch.Tensor, out: torch.Tensor
) -> torch.cuda.CUDAGraph:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(4):
            custom.all_reduce(inp, out=out, registered=False)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with custom.capture(), torch.cuda.graph(graph, stream=stream):
        custom.all_reduce(inp, out=out, registered=True)
    torch.cuda.synchronize()
    return graph


def _measure_graph(graph: torch.cuda.CUDAGraph, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    dist.barrier()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        graph.replay()
    end.record()
    end.synchronize()
    dist.barrier()
    return start.elapsed_time(end) / iterations


def _gather_rank_max(local_ms: float) -> tuple[list[float], float]:
    gathered: list[float | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local_ms)
    times = [float(value) for value in gathered if value is not None]
    return times, max(times)


def _digest(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _check_output(
    output: torch.Tensor, expected: torch.Tensor
) -> tuple[float, float, str]:
    diff = (output.detach().cpu().float() - expected).abs()
    return float(diff.max().item()), float(diff.mean().item()), _digest(output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--elements", type=int, nargs="+", default=[4096])
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--nccl-only", action="store_true")
    parser.add_argument(
        "--input-pattern",
        choices=("rank-constant", "random"),
        default="rank-constant",
    )
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

    communicator = PyNcclCommunicator(group=dist.group.WORLD, device=device)
    if communicator.disabled:
        raise RuntimeError("PyNcclCommunicator is unavailable.")
    custom = None
    if not args.nccl_only:
        custom = CustomAllreduce(group=dist.group.WORLD, device=device)
        if custom.disabled:
            raise RuntimeError(
                "Custom allreduce is unavailable; enable "
                "VLLM_SM70_TP8_NONFULL_CUSTOM_AR=1 and verify all-pairs P2P."
            )

    results: list[dict[str, object]] = []
    for elements in args.elements:
        if args.input_pattern == "random":
            generator = torch.Generator(device=device).manual_seed(args.seed + rank)
            inp = torch.randn(
                (elements,),
                dtype=torch.float16,
                device=device,
                generator=generator,
            )
        else:
            inp = torch.full((elements,), rank + 1, dtype=torch.float16, device=device)
        gathered_inputs: list[torch.Tensor | None] = [None] * world_size
        dist.all_gather_object(gathered_inputs, inp.detach().cpu())
        expected = torch.stack(
            [tensor.float() for tensor in gathered_inputs if tensor is not None]
        ).sum(dim=0)
        nccl_out = torch.empty_like(inp)

        nccl_graph = _capture(
            lambda inp=inp, out=nccl_out: communicator.all_reduce(inp, out)
        )
        nccl_graph.replay()
        torch.cuda.synchronize()
        nccl_max_abs, nccl_mean_abs, nccl_digest = _check_output(nccl_out, expected)

        nccl_samples = []
        for _ in range(args.repeats):
            nccl_local = _measure_graph(nccl_graph, args.warmup, args.iterations)
            _, nccl_max = _gather_rank_max(nccl_local)
            nccl_samples.append(nccl_max)

        nccl_median = statistics.median(nccl_samples)
        result = {
            "elements": elements,
            "bytes": elements * torch.float16.itemsize,
            "nccl_rank_max_samples_ms": nccl_samples,
            "nccl_rank_max_median_ms": nccl_median,
            "nccl_max_abs": nccl_max_abs,
            "nccl_mean_abs": nccl_mean_abs,
            "nccl_sha256": nccl_digest,
        }
        if custom is not None:
            custom_out = torch.empty_like(inp)
            custom_graph = _capture_custom(custom, inp, custom_out)
            custom_graph.replay()
            torch.cuda.synchronize()
            custom_max_abs, custom_mean_abs, custom_digest = _check_output(
                custom_out, expected
            )
            custom_samples = []
            for _ in range(args.repeats):
                custom_local = _measure_graph(
                    custom_graph, args.warmup, args.iterations
                )
                _, custom_max = _gather_rank_max(custom_local)
                custom_samples.append(custom_max)
            custom_median = statistics.median(custom_samples)
            result.update(
                {
                    "custom_rank_max_samples_ms": custom_samples,
                    "custom_rank_max_median_ms": custom_median,
                    "custom_speedup": nccl_median / custom_median,
                    "projected_87_call_saving_ms": 87 * (nccl_median - custom_median),
                    "custom_max_abs": custom_max_abs,
                    "custom_mean_abs": custom_mean_abs,
                    "custom_sha256": custom_digest,
                }
            )
        results.append(result)

    payload = {
        "world_size": world_size,
        "dtype": "torch.float16",
        "cuda_graph": True,
        "nccl_only": args.nccl_only,
        "input_pattern": args.input_pattern,
        "seed": args.seed,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "repeats": args.repeats,
        "results": results,
    }
    if rank == 0:
        encoded = json.dumps(payload, indent=2)
        print(encoded)
        if args.output_json is not None:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(encoded + "\n", encoding="utf-8")

    torch.cuda.synchronize()
    dist.barrier()
    if custom is not None:
        custom.close()
    communicator.destroy()
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
