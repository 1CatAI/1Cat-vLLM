#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the SM70 TP2 tile-runtime all-reduce prototype."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce


def _measure(
    ca: CustomAllreduce,
    tensor: torch.Tensor,
    mode: str,
    tile_numel: int,
    engine_blocks: int,
    producer_blocks: int,
    reducer_blocks: int,
    compute_iters: int,
    replays: int,
    warmup_replays: int,
) -> dict[str, Any]:
    torch.accelerator.synchronize()
    dist.barrier()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        if mode == "inline":
            output = ca.tile_runtime_all_reduce(
                tensor,
                tile_numel=tile_numel,
                engine_blocks=engine_blocks,
                compute_iters=compute_iters,
            )
        elif mode == "engine":
            output = ca.tile_runtime_all_reduce_engine(
                tensor,
                tile_numel=tile_numel,
                producer_blocks=producer_blocks,
                reducer_blocks=reducer_blocks,
                compute_iters=compute_iters,
            )
        else:
            raise ValueError(f"unknown mode: {mode}")

    torch.accelerator.synchronize()
    dist.barrier()

    graph.replay()
    torch.accelerator.synchronize()
    ref = tensor.clone()
    dist.all_reduce(ref)
    max_abs = (output - ref).abs().max().item()
    if max_abs != 0:
        raise RuntimeError(
            "tile runtime all-reduce mismatch: "
            f"tile_numel={tile_numel} engine_blocks={engine_blocks} "
            f"compute_iters={compute_iters} max_abs={max_abs}"
        )

    for _ in range(warmup_replays):
        graph.replay()
    torch.accelerator.synchronize()
    dist.barrier()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    torch.accelerator.synchronize()
    dist.barrier()

    return {
        "avg_ms": start.elapsed_time(end) / replays,
        "mode": mode,
        "tile_numel": tile_numel,
        "engine_blocks": engine_blocks,
        "producer_blocks": producer_blocks,
        "reducer_blocks": reducer_blocks,
        "compute_iters": compute_iters,
        "total_numel": tensor.numel(),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "max_abs": max_abs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-numel", type=int, default=5120)
    parser.add_argument("--modes", default="inline,engine")
    parser.add_argument("--tile-numels", default="256,512,1024,1280,2560,5120")
    parser.add_argument("--engine-blocks", default="0")
    parser.add_argument("--producer-blocks", default="0")
    parser.add_argument("--reducer-blocks", default="0")
    parser.add_argument("--compute-iters", default="0")
    parser.add_argument("--replays", type=int, default=3000)
    parser.add_argument("--warmup-replays", type=int, default=100)
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--max-size-bytes", type=int, default=8 * 1024 * 1024)
    parser.add_argument("--json-out")
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.accelerator.set_device_index(local_rank)
    dist.init_process_group(backend="nccl")
    gloo_group = dist.new_group(backend="gloo")

    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    ca = CustomAllreduce(
        group=gloo_group, device=local_rank, max_size=args.max_size_bytes
    )

    try:
        if world_size != 2:
            raise RuntimeError("SM70 tile runtime prototype benchmark requires TP2")
        if ca.disabled:
            raise RuntimeError("custom allreduce disabled")

        base = torch.arange(args.total_numel, device="cuda", dtype=torch.float32)
        tensor = ((base % 31) * 0.01 + rank).to(dtype)

        tile_numels = [int(item) for item in args.tile_numels.split(",") if item]
        modes = [item for item in args.modes.split(",") if item]
        engine_blocks_list = [
            int(item) for item in args.engine_blocks.split(",") if item
        ]
        producer_blocks_list = [
            int(item) for item in args.producer_blocks.split(",") if item
        ]
        reducer_blocks_list = [
            int(item) for item in args.reducer_blocks.split(",") if item
        ]
        compute_iters_list = [
            int(item) for item in args.compute_iters.split(",") if item
        ]
        for mode in modes:
            if mode not in ("inline", "engine"):
                raise RuntimeError(f"unsupported mode: {mode}")

        results = []
        for mode in modes:
            for compute_iters in compute_iters_list:
                if mode == "inline":
                    for engine_blocks in engine_blocks_list:
                        for tile_numel in tile_numels:
                            if args.total_numel % tile_numel != 0:
                                continue
                            results.append(
                                _measure(
                                    ca,
                                    tensor,
                                    mode,
                                    tile_numel,
                                    engine_blocks,
                                    0,
                                    0,
                                    compute_iters,
                                    args.replays,
                                    args.warmup_replays,
                                )
                            )
                else:
                    for producer_blocks in producer_blocks_list:
                        for reducer_blocks in reducer_blocks_list:
                            for tile_numel in tile_numels:
                                if args.total_numel % tile_numel != 0:
                                    continue
                                results.append(
                                    _measure(
                                        ca,
                                        tensor,
                                        mode,
                                        tile_numel,
                                        0,
                                        producer_blocks,
                                        reducer_blocks,
                                        compute_iters,
                                        args.replays,
                                        args.warmup_replays,
                                    )
                                )

        gathered: list[list[dict[str, Any]]] = [None] * world_size  # type: ignore
        dist.all_gather_object(gathered, results)
        payload = {
            "world_size": world_size,
            "total_numel": args.total_numel,
            "dtype": args.dtype,
            "replays": args.replays,
            "warmup_replays": args.warmup_replays,
            "rank_results": [
                {"rank": rank_id, "results": rank_results}
                for rank_id, rank_results in enumerate(gathered)
            ],
        }
        if rank == 0:
            text = json.dumps(payload, indent=2, sort_keys=True)
            print(text)
            if args.json_out:
                path = Path(args.json_out)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text + "\n")
    finally:
        ca.close()
        dist.destroy_process_group(gloo_group)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
