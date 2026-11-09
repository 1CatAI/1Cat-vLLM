# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the exact SM70 TP8 hierarchical decode all-reduce route."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections.abc import Callable
from pathlib import Path

import torch
import torch.distributed as dist

from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator


def _capture(call: Callable[[], None]) -> torch.cuda.CUDAGraph:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(4):
            call()
    stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        call()
    return graph


def _capture_custom(
    custom: CustomAllreduce,
    warm_call: Callable[[], None],
    capture_call: Callable[[], None],
) -> torch.cuda.CUDAGraph:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(4):
            warm_call()
    stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with custom.capture(), torch.cuda.graph(graph, stream=stream):
        capture_call()
    return graph


class JoinStage:
    """Reproduce a multi-stream projection join before each collective."""

    def __init__(
        self, *, collectives: int, elements: int, seed: int, rank: int
    ) -> None:
        if elements % 4096 != 0:
            raise ValueError("join-work elements must be a multiple of 4096")
        tokens = elements // 4096
        generator = torch.Generator(device="cuda").manual_seed(seed + rank)
        self.x = torch.randn(
            (tokens, 256), device="cuda", dtype=torch.float16, generator=generator
        )
        self.main_weight = torch.randn(
            (4096, 256), device="cuda", dtype=torch.float16, generator=generator
        )
        self.aux_weights = [
            torch.randn(
                (n, 256), device="cuda", dtype=torch.float16, generator=generator
            )
            for n in (2048, 1024, 512)
        ]
        self.main_out = torch.empty((tokens, 4096), device="cuda", dtype=torch.float16)
        self.aux_outs = [
            torch.empty((tokens, weight.shape[0]), device="cuda", dtype=torch.float16)
            for weight in self.aux_weights
        ]
        self.bias = torch.randn(
            (tokens, 4096),
            device="cuda",
            dtype=torch.float16,
            generator=generator,
        )
        self.reduce_input = torch.empty_like(self.main_out)
        self.reduce_output = torch.empty_like(self.main_out)
        self.collectives = collectives
        self.aux_streams = [torch.cuda.Stream() for _ in range(3)]
        self.start_events = [torch.cuda.Event() for _ in range(collectives)]
        self.done_events = [
            [torch.cuda.Event() for _ in range(3)] for _ in range(collectives)
        ]

    def run(self, reduce: Callable[[torch.Tensor, torch.Tensor], None]) -> None:
        for index in range(self.collectives):
            current = torch.cuda.current_stream()
            self.start_events[index].record(current)
            for stream, done, weight, out in zip(
                self.aux_streams,
                self.done_events[index],
                self.aux_weights,
                self.aux_outs,
                strict=True,
            ):
                with torch.cuda.stream(stream):
                    stream.wait_event(self.start_events[index])
                    torch.mm(self.x, weight.T, out=out)
                    done.record(stream)
            torch.mm(self.x, self.main_weight.T, out=self.main_out)
            for done in self.done_events[index]:
                current.wait_event(done)
            torch.add(self.main_out, self.bias, out=self.reduce_input)
            reduce(self.reduce_input.flatten(), self.reduce_output.flatten())


def _measure_graph(
    graph: torch.cuda.CUDAGraph,
    *,
    collectives: int,
    warmup: int,
    iterations: int,
) -> float:
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
    return start.elapsed_time(end) / (iterations * collectives)


def _rank_max(local_ms: float) -> float:
    gathered: list[float | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local_ms)
    return max(float(value) for value in gathered if value is not None)


def _digest(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _output_stats(
    output: torch.Tensor, expected: torch.Tensor
) -> dict[str, float | str | bool]:
    host_output = output.detach().cpu()
    diff = host_output.float() - expected
    expected_output = expected.to(host_output.dtype)
    digest = _digest(host_output)
    digests: list[str | None] = [None] * dist.get_world_size()
    dist.all_gather_object(digests, digest)
    return {
        "max_abs": float(diff.abs().max().item()),
        "mean_abs": float(diff.abs().mean().item()),
        "sha256": digest,
        "all_ranks_bitwise_equal": len(set(digests)) == 1,
        "bitwise_equal_fixed_tree_reference": torch.equal(host_output, expected_output),
    }


def _expected_sum(inp: torch.Tensor) -> torch.Tensor:
    gathered: list[torch.Tensor | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, inp.detach().cpu())
    tensors = [tensor.float() for tensor in gathered if tensor is not None]
    lower = tensors[0].clone()
    upper = tensors[4].clone()
    for tensor in tensors[1:4]:
        lower.add_(tensor)
    for tensor in tensors[5:8]:
        upper.add_(tensor)
    return lower + upper


def _benchmark_sum2(
    *,
    custom: CustomAllreduce,
    inp_a: torch.Tensor,
    inp_b: torch.Tensor,
    collectives: int,
    warmup: int,
    iterations: int,
    repeats: int,
) -> dict[str, object]:
    local_sum = torch.empty_like(inp_a)
    baseline_out = torch.empty_like(inp_a)
    fused_out = torch.empty_like(inp_a)

    def baseline_warm() -> None:
        for _ in range(collectives):
            torch.add(inp_a, inp_b, out=local_sum)
            custom.all_reduce(local_sum, out=baseline_out, registered=False)

    def baseline_capture() -> None:
        for _ in range(collectives):
            torch.add(inp_a, inp_b, out=local_sum)
            custom.all_reduce(local_sum, out=baseline_out, registered=True)

    baseline_graph = _capture_custom(custom, baseline_warm, baseline_capture)

    def fused_warm() -> None:
        torch.add(inp_a, inp_b, out=local_sum)
        for _ in range(collectives):
            custom.all_reduce(local_sum, out=fused_out, registered=False)

    def fused_capture() -> None:
        for _ in range(collectives):
            custom.all_reduce_sum2(inp_a, inp_b, out=fused_out)

    fused_graph = _capture_custom(custom, fused_warm, fused_capture)

    baseline_graph.replay()
    fused_graph.replay()
    torch.cuda.synchronize()
    expected = _expected_sum(torch.add(inp_a, inp_b))
    baseline_output = _output_stats(baseline_out, expected)
    fused_output = _output_stats(fused_out, expected)

    baseline_samples = []
    fused_samples = []
    for repeat in range(repeats):
        order = (
            ((baseline_graph, baseline_samples), (fused_graph, fused_samples))
            if repeat % 2 == 0
            else ((fused_graph, fused_samples), (baseline_graph, baseline_samples))
        )
        for graph, samples in order:
            samples.append(
                _rank_max(
                    _measure_graph(
                        graph,
                        collectives=collectives,
                        warmup=warmup,
                        iterations=iterations,
                    )
                )
            )

    baseline_median = statistics.median(baseline_samples)
    fused_median = statistics.median(fused_samples)
    return {
        "baseline_add_then_hierarchical": {
            "rank_max_samples_ms": baseline_samples,
            "rank_max_median_ms": baseline_median,
            "output": baseline_output,
        },
        "fused_sum2_hierarchical": {
            "rank_max_samples_ms": fused_samples,
            "rank_max_median_ms": fused_median,
            "output": fused_output,
        },
        "speedup": baseline_median / fused_median,
        "projected_graph_saving_ms": collectives * (baseline_median - fused_median),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collectives", type=int, default=87)
    parser.add_argument("--elements", type=int, choices=(4096, 8 * 4096), default=4096)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--join-work", action="store_true")
    parser.add_argument("--sum2", action="store_true")
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
    custom = CustomAllreduce(group=dist.group.WORLD, device=device)
    if custom.disabled or not custom.tp8_hierarchical:
        raise RuntimeError(
            "Hierarchical custom allreduce is unavailable; set "
            "VLLM_SM70_TP8_HIERARCHICAL_CUSTOM_AR=1 and verify the topology."
        )

    generator = torch.Generator(device=device).manual_seed(args.seed + rank)
    inp = torch.randn(
        (args.elements,), dtype=torch.float16, device=device, generator=generator
    )
    if args.sum2:
        inp_b = torch.randn(
            (args.elements,), dtype=torch.float16, device=device, generator=generator
        )
        payload = {
            "contract": {
                "world_size": world_size,
                "elements": inp.numel(),
                "bytes": inp.numel() * inp.element_size(),
                "dtype": str(inp.dtype),
                "cuda_graph": True,
                "collectives_per_graph": args.collectives,
                "warmup": args.warmup,
                "iterations": args.iterations,
                "repeats": args.repeats,
                "seed": args.seed,
                "sum2": True,
            },
            **_benchmark_sum2(
                custom=custom,
                inp_a=inp,
                inp_b=inp_b,
                collectives=args.collectives,
                warmup=args.warmup,
                iterations=args.iterations,
                repeats=args.repeats,
            ),
        }
        if rank == 0:
            encoded = json.dumps(payload, indent=2)
            print(encoded)
            if args.output_json is not None:
                args.output_json.parent.mkdir(parents=True, exist_ok=True)
                args.output_json.write_text(encoded + "\n", encoding="utf-8")
        torch.cuda.synchronize()
        dist.barrier()
        custom.close()
        communicator.destroy()
        dist.barrier()
        dist.destroy_process_group()
        fused_output = payload["fused_sum2_hierarchical"]["output"]
        return (
            0
            if fused_output["all_ranks_bitwise_equal"]
            and fused_output["bitwise_equal_fixed_tree_reference"]
            else 1
        )
    if args.join_work:
        nccl_stage = JoinStage(
            collectives=args.collectives,
            elements=args.elements,
            seed=args.seed,
            rank=rank,
        )

        def run_nccl() -> None:
            nccl_stage.run(communicator.all_reduce)

        nccl_out = nccl_stage.reduce_output.flatten()
        nccl_input = nccl_stage.reduce_input.flatten()
    else:
        nccl_out = torch.empty_like(inp)
        nccl_input = inp

        def run_nccl() -> None:
            for _ in range(args.collectives):
                communicator.all_reduce(inp, nccl_out)

    nccl_graph = _capture(run_nccl)
    nccl_graph.replay()
    torch.cuda.synchronize()
    nccl_output = _output_stats(nccl_out, _expected_sum(nccl_input))

    if args.join_work:
        custom_stage = JoinStage(
            collectives=args.collectives,
            elements=args.elements,
            seed=args.seed,
            rank=rank,
        )

        def run_custom_warm() -> None:
            custom_stage.run(
                lambda value, out: custom.all_reduce(value, out=out, registered=False)
            )

        def run_custom_capture() -> None:
            custom_stage.run(
                lambda value, out: custom.all_reduce(value, out=out, registered=True)
            )

        custom_out = custom_stage.reduce_output.flatten()
        custom_input = custom_stage.reduce_input.flatten()
    else:
        custom_out = torch.empty_like(inp)
        custom_input = inp

        def run_custom_warm() -> None:
            for _ in range(args.collectives):
                custom.all_reduce(inp, out=custom_out, registered=False)

        def run_custom_capture() -> None:
            for _ in range(args.collectives):
                custom.all_reduce(inp, out=custom_out, registered=True)

    custom_graph = _capture_custom(custom, run_custom_warm, run_custom_capture)
    custom_graph.replay()
    torch.cuda.synchronize()
    custom_output = _output_stats(custom_out, _expected_sum(custom_input))

    nccl_samples = []
    custom_samples = []
    for repeat in range(args.repeats):
        if repeat % 2 == 0:
            nccl_samples.append(
                _rank_max(
                    _measure_graph(
                        nccl_graph,
                        collectives=args.collectives,
                        warmup=args.warmup,
                        iterations=args.iterations,
                    )
                )
            )
            custom_samples.append(
                _rank_max(
                    _measure_graph(
                        custom_graph,
                        collectives=args.collectives,
                        warmup=args.warmup,
                        iterations=args.iterations,
                    )
                )
            )
        else:
            custom_samples.append(
                _rank_max(
                    _measure_graph(
                        custom_graph,
                        collectives=args.collectives,
                        warmup=args.warmup,
                        iterations=args.iterations,
                    )
                )
            )
            nccl_samples.append(
                _rank_max(
                    _measure_graph(
                        nccl_graph,
                        collectives=args.collectives,
                        warmup=args.warmup,
                        iterations=args.iterations,
                    )
                )
            )

    nccl_median = statistics.median(nccl_samples)
    custom_median = statistics.median(custom_samples)
    payload = {
        "contract": {
            "world_size": world_size,
            "elements": inp.numel(),
            "bytes": inp.numel() * inp.element_size(),
            "dtype": str(inp.dtype),
            "cuda_graph": True,
            "collectives_per_graph": args.collectives,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "repeats": args.repeats,
            "seed": args.seed,
            "join_work": args.join_work,
        },
        "nccl": {
            "rank_max_samples_ms": nccl_samples,
            "rank_max_median_ms": nccl_median,
            "output": nccl_output,
        },
        "hierarchical": {
            "rank_max_samples_ms": custom_samples,
            "rank_max_median_ms": custom_median,
            "output": custom_output,
        },
        "speedup": nccl_median / custom_median,
        "projected_graph_saving_ms": args.collectives * (nccl_median - custom_median),
    }
    if rank == 0:
        encoded = json.dumps(payload, indent=2)
        print(encoded)
        if args.output_json is not None:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(encoded + "\n", encoding="utf-8")

    torch.cuda.synchronize()
    dist.barrier()
    custom.close()
    communicator.destroy()
    dist.barrier()
    dist.destroy_process_group()
    return (
        0
        if custom_output["all_ranks_bitwise_equal"]
        and custom_output["bitwise_equal_fixed_tree_reference"]
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
