# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure one-time TP2 row packing and redistribution for a draft LM-head."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-vocab-size", type=int, default=248320)
    parser.add_argument("--hidden-size", type=int, default=5120)
    parser.add_argument("--shortlist-size", type=int, default=131072)
    parser.add_argument("--source-count", type=int, action="append")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def _summary(samples_by_rank: torch.Tensor) -> dict[str, Any]:
    critical = samples_by_rank.max(dim=0).values.cpu().tolist()
    critical.sort()
    return {
        "critical_path_mean_ms": statistics.fmean(critical),
        "critical_path_p50_ms": statistics.median(critical),
        "critical_path_p90_ms": critical[int(0.9 * (len(critical) - 1))],
        "critical_path_min_ms": critical[0],
        "critical_path_max_ms": critical[-1],
        "rank_mean_ms": samples_by_rank.mean(dim=1).cpu().tolist(),
    }


def _gather_samples(samples: list[float], world_size: int) -> torch.Tensor:
    local = torch.tensor(samples, device="cuda", dtype=torch.float32)
    gathered = torch.empty(
        world_size * local.numel(), device="cuda", dtype=torch.float32
    )
    dist.all_gather_into_tensor(gathered, local)
    return gathered.view(world_size, local.numel())


def _time_cuda(
    operation: Callable[[], Any],
    warmup: int,
    iters: int,
    world_size: int,
) -> dict[str, Any]:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    dist.barrier()

    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    dist.barrier()
    return _summary(_gather_samples(samples, world_size))


def _split_matrix(source_counts: list[int], destination_rows: int) -> list[list[int]]:
    remaining = source_counts.copy()
    matrix = [[0 for _ in source_counts] for _ in source_counts]
    for destination in range(len(source_counts)):
        needed = destination_rows
        for source in range(len(source_counts)):
            take = min(needed, remaining[source])
            matrix[source][destination] = take
            remaining[source] -= take
            needed -= take
        if needed:
            raise ValueError("Source rows cannot fill balanced destinations.")
    if any(remaining):
        raise ValueError("Redistribution left unassigned source rows.")
    return matrix


def main() -> None:
    args = _parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2:
        raise ValueError(f"This benchmark is TP2-only, got {world_size} ranks.")
    if args.shortlist_size % world_size:
        raise ValueError("--shortlist-size must be divisible by TP size.")

    source_counts = args.source_count or [111496, 19576]
    if len(source_counts) != world_size or sum(source_counts) != args.shortlist_size:
        raise ValueError("Source counts must match TP size and shortlist size.")
    if args.hidden_size <= 0 or args.model_vocab_size <= 0:
        raise ValueError("Model dimensions must be positive.")

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    try:
        device = torch.device("cuda", local_rank)
        capability = torch.cuda.get_device_capability(device)
        if capability != (7, 0):
            raise RuntimeError(f"Expected SM70, got sm_{capability[0]}{capability[1]}.")

        shard_width = math.ceil(args.model_vocab_size / world_size)
        local_full_rows = min(
            shard_width,
            args.model_vocab_size - rank * shard_width,
        )
        if source_counts[rank] > local_full_rows:
            raise ValueError("A source count exceeds its original vocabulary shard.")

        full_weight = torch.empty(
            (local_full_rows, args.hidden_size),
            dtype=torch.float16,
            device=device,
        )
        selected_indices = torch.arange(
            source_counts[rank], dtype=torch.int64, device=device
        )
        source_packed = torch.empty(
            (source_counts[rank], args.hidden_size),
            dtype=torch.float16,
            device=device,
        )
        destination_rows = args.shortlist_size // world_size
        destination_packed = torch.empty(
            (destination_rows, args.hidden_size),
            dtype=torch.float16,
            device=device,
        )

        split_matrix = _split_matrix(source_counts, destination_rows)
        input_splits = split_matrix[rank]
        output_splits = [split_matrix[source][rank] for source in range(world_size)]

        def gather_rows() -> None:
            torch.index_select(
                full_weight,
                0,
                selected_indices,
                out=source_packed,
            )

        def redistribute() -> None:
            dist.all_to_all_single(
                destination_packed,
                source_packed,
                output_split_sizes=output_splits,
                input_split_sizes=input_splits,
            )

        def gather_and_redistribute() -> None:
            gather_rows()
            redistribute()

        gather_rows()
        redistribute()
        torch.cuda.synchronize()
        dist.barrier()

        timings = {
            "local_selected_row_gather": _time_cuda(
                gather_rows, args.warmup, args.iters, world_size
            ),
            "variable_split_all_to_all": _time_cuda(
                redistribute, args.warmup, args.iters, world_size
            ),
            "gather_plus_all_to_all": _time_cuda(
                gather_and_redistribute,
                args.warmup,
                args.iters,
                world_size,
            ),
        }
        result = {
            "device": torch.cuda.get_device_name(device),
            "device_capability": list(capability),
            "world_size": world_size,
            "model_vocab_size": args.model_vocab_size,
            "hidden_size": args.hidden_size,
            "shortlist_size": args.shortlist_size,
            "source_counts": source_counts,
            "destination_rows_per_rank": destination_rows,
            "split_matrix_source_by_destination": split_matrix,
            "bytes": {
                "packed_head_per_rank": destination_packed.numel()
                * destination_packed.element_size(),
                "cross_rank_payload": sum(
                    split_matrix[source][destination]
                    for source in range(world_size)
                    for destination in range(world_size)
                    if source != destination
                )
                * args.hidden_size
                * source_packed.element_size(),
            },
            "warmup": args.warmup,
            "iters": args.iters,
            "timings": timings,
        }
        if rank == 0:
            text = json.dumps(result, indent=2, sort_keys=True)
            print(text)
            if args.json_out is not None:
                args.json_out.parent.mkdir(parents=True, exist_ok=True)
                args.json_out.write_text(text + "\n", encoding="utf-8")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
