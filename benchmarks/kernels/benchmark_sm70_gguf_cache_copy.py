# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""TP4 paired screens of the existing expert-cache byte-copy kernel."""

import argparse
import json
import os
import statistics
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist

from vllm import _custom_ops  # noqa: F401 -- register mapped-view operator
from vllm.model_executor.layers.quantization.gguf_expert_cache import _apply_plan

POLICIES = {
    "reference": (16, 8, 1024, 8, False),
    "o2s64b256w4": (2, 64, 256, 4, False),
    "o4s32b256w4": (4, 32, 256, 4, False),
    "o4s64b256w4": (4, 64, 256, 4, False),
    "o8s16b512w4": (8, 16, 512, 4, False),
    "o2s32b512w4": (2, 32, 512, 4, False),
    "o2s32b256w4p3": (2, 32, 256, 4, True),
    "o4s32b256w4p3": (4, 32, 256, 4, True),
    "o4s64b256w4p3": (4, 64, 256, 4, True),
}


def launch(buffers, policy):
    programs, split, block, warps, parallel_shards = POLICIES[policy]
    plan, arena, mapped, device_offsets, host_offsets, words = buffers
    grid = (programs, split, 3) if parallel_shards else (programs, split)
    _apply_plan[grid](
        plan,
        arena,
        mapped,
        device_offsets,
        host_offsets,
        words,
        0,
        OPS_PROGRAMS=programs,
        SPLIT=split,
        BLOCK=block,
        PARALLEL_SHARDS=parallel_shards,
        num_warps=warps,
    )


def reset_data(arena, host, original_arena, original_host):
    arena.copy_(original_arena)
    host.copy_(original_host)


def summarize(values):
    ordered = sorted(values)
    return {
        "mean": statistics.mean(values),
        "p50": statistics.median(values),
        "p90": ordered[round(0.9 * (len(ordered) - 1))],
        "values": values,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=16)
    parser.add_argument("--boundary-only", action="store_true")
    parser.add_argument("--policies", nargs="+", choices=POLICIES)
    args = parser.parse_args()
    policies = (
        {name: POLICIES[name] for name in args.policies} if args.policies else POLICIES
    )
    rank = int(os.environ["LOCAL_RANK"])
    torch.accelerator.set_device_index(rank)
    dist.init_process_group("gloo")
    assert dist.get_world_size() == 4
    device = torch.device("cuda", rank)
    # Flash-Next GSQ-RCO IQ3_S TP4 words per expert. Q2_0's down projection
    # retains its 192-column physical padding for the 160-column local input.
    layouts = {
        0: dict(zip(("w1", "w3", "w2"), (19600, 19600, 28800))),
        1: dict(zip(("w1", "w3", "w2"), (16400, 16400, 17280))),
        17: dict(zip(("w1", "w3", "w2"), (22000, 22000, 28800))),
        47: dict(zip(("w1", "w3", "w2"), (27200, 27200, 28800))),
    }
    if args.shapes:
        layouts = {}
        for entry in json.loads(args.shapes.read_text()):
            layer = int(entry["name"].split(".")[1])
            shard = (
                "w2"
                if "down" in entry["name"]
                else "w1"
                if "gate" in entry["name"]
                else "w3"
            )
            assert entry["rank_bytes"] % 8 == 0
            layouts.setdefault(layer, {})[shard] = entry["rank_bytes"] // 8
    if args.boundary_only:
        layouts = {-1: {"w1": 255, "w3": 256, "w2": 257}}
    output = {
        "world_size": 4,
        "policies": policies,
        "samples": args.samples,
        "repeats": args.repeats,
        "cases": [],
        "correctness": [],
    }

    for layer, shards in sorted(layouts.items()):
        sizes = [shards[k] for k in ("w1", "w3", "w2")]
        # Production slot counts; tail guards separate projection banks.
        slots, host_slots = 410, 136
        offsets, host_offsets, total, host_total = [], [], 0, 0
        for n in sizes:
            total = ((total + 31) // 32) * 32
            offsets.append(total)
            host_offsets.append(host_total)
            total += slots * n + 32
            host_total += host_slots * n
        arena = torch.arange(total, dtype=torch.int64, device=device)
        host = torch.empty(host_total, dtype=torch.int64, pin_memory=True)
        host.copy_(torch.arange(host_total, dtype=torch.int64).mul_(-3).sub_(7))
        original_arena, original_host = arena.clone(), host.clone()
        mapped = torch.ops._C.get_cuda_view_from_cpu_tensor(host)
        assert mapped.device == device
        device_offsets = torch.tensor(offsets, dtype=torch.int64, device=device)
        cpu_offsets = torch.tensor(host_offsets, dtype=torch.int64, device=device)
        words = torch.tensor(sizes, dtype=torch.int64, device=device)
        plan = torch.zeros(8 + 5 * 256, dtype=torch.int32, device=device)

        buffers = (plan, arena, mapped, device_offsets, cpu_offsets, words)
        invoke = partial(launch, buffers)
        reset = partial(reset_data, arena, host, original_arena, original_host)

        # Separate loads/writebacks model the spare-slot policy; a swap is
        # also valid when an incoming expert gives its host slot to a victim.
        cases = (
            ("empty", 0),
            ("paired", 1),
            ("paired", 2),
            ("paired", 4),
            ("paired", 8),
            ("paired", 20),
            ("swap", 1),
            ("swap", 4),
        )
        if args.boundary_only:
            cases = tuple(("swap", n) for n in (0, 1, 2, 3, 4, 5, 15, 16, 17, 128))
        for mode, count in cases:
            cpu_plan = torch.zeros_like(plan, device="cpu")
            ops = count * 2 if mode == "paired" else count
            cpu_plan[0] = ops
            for i in range(count):
                entries = (
                    [(i, i, i)]
                    if mode == "swap"
                    else [(i, i, -1), (slots - count + i, -1, host_slots - count + i)]
                )
                for j, (slot, load, back) in enumerate(entries):
                    op = i * len(entries) + j
                    cpu_plan[8 + op * 5 : 11 + op * 5] = torch.tensor(
                        [slot, load, back]
                    )
            plan.copy_(cpu_plan)
            reset()
            invoke("reference")
            torch.accelerator.synchronize()
            oracle_arena, oracle_host = arena.clone(), host.clone()
            graphs = {}
            for name in policies:
                reset()
                invoke(name)
                torch.accelerator.synchronize()
                assert torch.equal(arena, oracle_arena), (
                    rank,
                    layer,
                    mode,
                    count,
                    name,
                    "GPU",
                )
                assert torch.equal(host, oracle_host), (
                    rank,
                    layer,
                    mode,
                    count,
                    name,
                    "host",
                )
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(args.repeats):
                        invoke(name)
                graphs[name] = graph
                reset()
                graph.replay()
                torch.accelerator.synchronize()
                # Even swap counts restore the original data after a graph.
                expected_arena = (
                    original_arena
                    if mode == "swap" and args.repeats % 2 == 0
                    else oracle_arena
                )
                expected_host = (
                    original_host
                    if mode == "swap" and args.repeats % 2 == 0
                    else oracle_host
                )
                assert torch.equal(arena, expected_arena), (rank, name, "graph GPU")
                assert torch.equal(host, expected_host), (rank, name, "graph host")
                reset()
                arena.add_(19)
                host.add_(23)
                for _ in range(args.repeats):
                    invoke("reference")
                torch.accelerator.synchronize()
                changed_arena, changed_host = arena.clone(), host.clone()
                reset()
                arena.add_(19)
                host.add_(23)
                graph.replay()
                torch.accelerator.synchronize()
                assert torch.equal(arena, changed_arena), (rank, name, "changed GPU")
                assert torch.equal(host, changed_host), (rank, name, "changed host")
                del changed_arena, changed_host

            timings = {name: [] for name in policies}
            for sample in range(args.samples):
                order = list(policies)
                if sample % 2:
                    order.reverse()
                for name in order:
                    reset()
                    dist.barrier()
                    start, end = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    start.record()
                    graphs[name].replay()
                    end.record()
                    end.synchronize()
                    elapsed = torch.tensor(
                        start.elapsed_time(end) * 1000 / args.repeats,
                        dtype=torch.float64,
                    )
                    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
                    timings[name].append(elapsed.item())
            case = {
                "layer": layer,
                "words": sizes,
                "mode": mode,
                "count": count,
                "plan_ops": ops,
                "critical_rank_us": {
                    name: summarize(values) for name, values in timings.items()
                },
            }
            output["cases"].append(case)
            output["correctness"].append(
                {
                    "layer": layer,
                    "mode": mode,
                    "count": count,
                    "eager_and_graph_bit_exact": True,
                }
            )
            if rank == 0:
                print(json.dumps(case), flush=True)
                args.output.write_text(json.dumps(output, indent=2))
            del graphs, oracle_arena, oracle_host
        del invoke, reset, buffers, arena, host, mapped, original_arena, original_host
        torch.accelerator.empty_cache()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
