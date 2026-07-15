# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure TP2 full-vocab versus compact top-k verifier-logit transport.

Run on two identical GPUs without loading a model:

  CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
    benchmarks/benchmark_sm70_tp2_compact_verifier_logits.py \
    --json-out bench_results/.../tp2_compact_verifier_logits.json

The compact paths are checked against the full-gather top-k result. This is a
communication and postprocessing microbenchmark, not an end-to-end MTP score.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=5)
    parser.add_argument("--vocab-size", type=int, default=248320)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def _summary(samples_by_rank: torch.Tensor) -> dict[str, Any]:
    # The slowest rank determines the TP critical path for each collective.
    critical_samples = samples_by_rank.max(dim=0).values.cpu().tolist()
    critical_samples.sort()
    rank_means = samples_by_rank.mean(dim=1).cpu().tolist()
    count = len(critical_samples)
    return {
        "rank_mean_ms": rank_means,
        "critical_path_mean_ms": statistics.fmean(critical_samples),
        "critical_path_p50_ms": statistics.median(critical_samples),
        "critical_path_p90_ms": critical_samples[int(0.9 * (count - 1))],
        "critical_path_p99_ms": critical_samples[int(0.99 * (count - 1))],
        "critical_path_min_ms": critical_samples[0],
        "critical_path_max_ms": critical_samples[-1],
    }


def _collect_samples(samples: list[float], world_size: int) -> torch.Tensor:
    local = torch.tensor(samples, device="cuda", dtype=torch.float32)
    gathered = torch.empty(
        (world_size * local.numel(),), device="cuda", dtype=torch.float32
    )
    dist.all_gather_into_tensor(gathered, local)
    return gathered.view(world_size, local.numel())


def _time_cuda(
    operation: Callable[[], Any],
    warmup: int,
    iters: int,
    world_size: int,
) -> tuple[dict[str, Any], Any]:
    for _ in range(warmup):
        result = operation()
    torch.cuda.synchronize()
    dist.barrier()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples: list[float] = []
    for _ in range(iters):
        start.record()
        result = operation()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    dist.barrier()
    return _summary(_collect_samples(samples, world_size)), result


def main() -> None:
    args = _parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2:
        raise ValueError(f"This benchmark is TP2-only, got world_size={world_size}.")
    if args.vocab_size % world_size:
        raise ValueError("--vocab-size must be divisible by TP size.")
    if args.top_k <= 0 or args.top_k > args.vocab_size // world_size:
        raise ValueError("--top-k must be in [1, vocab_size / TP].")

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    try:
        device = torch.device("cuda", local_rank)
        local_vocab_size = args.vocab_size // world_size
        generator = torch.Generator(device=device)
        generator.manual_seed(args.seed + rank)
        local_logits = torch.randn(
            (args.rows, local_vocab_size),
            generator=generator,
            device=device,
            dtype=torch.float16,
        )
        # Random FP16 values have many equal values near the top of the
        # distribution. Seed a unique interleaved global top-k so the
        # correctness check tests token ids rather than PyTorch tie ordering.
        local_topk_positions = torch.arange(args.top_k, device=device)
        local_topk_values = (
            64.0
            - (torch.arange(args.top_k, device=device, dtype=torch.float16) * 2 + rank)
            / 8.0
        )
        local_logits[:, local_topk_positions] = local_topk_values
        vocab_start = rank * local_vocab_size

        gathered_logits = torch.empty(
            (world_size * args.rows, local_vocab_size),
            device=device,
            dtype=local_logits.dtype,
        )
        gathered_values = torch.empty(
            (world_size * args.rows, args.top_k),
            device=device,
            dtype=torch.float32,
        )
        gathered_indices = torch.empty(
            (world_size * args.rows, args.top_k),
            device=device,
            dtype=torch.int64,
        )
        local_pairs = torch.empty(
            (args.rows, args.top_k, 2),
            device=device,
            dtype=torch.float32,
        )
        gathered_pairs = torch.empty(
            (world_size * args.rows, args.top_k, 2),
            device=device,
            dtype=torch.float32,
        )

        def gathered_full_logits() -> torch.Tensor:
            return (
                gathered_logits.view(world_size, args.rows, local_vocab_size)
                .permute(1, 0, 2)
                .reshape(args.rows, args.vocab_size)
            )

        def gathered_compact_candidates() -> tuple[torch.Tensor, torch.Tensor]:
            all_values = (
                gathered_values.view(world_size, args.rows, args.top_k)
                .permute(1, 0, 2)
                .reshape(args.rows, world_size * args.top_k)
            )
            all_indices = (
                gathered_indices.view(world_size, args.rows, args.top_k)
                .permute(1, 0, 2)
                .reshape(args.rows, world_size * args.top_k)
            )
            return all_values, all_indices

        def full_gather_topk() -> tuple[torch.Tensor, torch.Tensor]:
            dist.all_gather_into_tensor(gathered_logits, local_logits)
            return torch.topk(gathered_full_logits().float(), k=args.top_k, dim=-1)

        def compact_topk_two_gathers() -> tuple[torch.Tensor, torch.Tensor]:
            local_values, local_indices = torch.topk(
                local_logits.float(), k=args.top_k, dim=-1
            )
            local_indices = local_indices + vocab_start
            dist.all_gather_into_tensor(gathered_values, local_values)
            dist.all_gather_into_tensor(gathered_indices, local_indices)
            all_values, all_indices = gathered_compact_candidates()
            values, positions = torch.topk(all_values, k=args.top_k, dim=-1)
            return values, all_indices.gather(-1, positions)

        def compact_topk_packed_gather() -> tuple[torch.Tensor, torch.Tensor]:
            local_values, local_indices = torch.topk(
                local_logits.float(), k=args.top_k, dim=-1
            )
            local_pairs[..., 0] = local_values
            local_pairs[..., 1] = (local_indices + vocab_start).float()
            dist.all_gather_into_tensor(gathered_pairs, local_pairs)
            candidates = (
                gathered_pairs.view(world_size, args.rows, args.top_k, 2)
                .permute(1, 0, 2, 3)
                .reshape(args.rows, world_size * args.top_k, 2)
            )
            values, positions = torch.topk(candidates[..., 0], k=args.top_k, dim=-1)
            indices = candidates[..., 1].gather(-1, positions).to(torch.int64)
            return values, indices

        local_values, local_indices = torch.topk(
            local_logits.float(), k=args.top_k, dim=-1
        )
        local_indices = local_indices + vocab_start

        def full_gather_only() -> torch.Tensor:
            dist.all_gather_into_tensor(gathered_logits, local_logits)
            return gathered_logits

        def full_global_topk_only() -> tuple[torch.Tensor, torch.Tensor]:
            return torch.topk(gathered_full_logits().float(), k=args.top_k, dim=-1)

        def local_topk_only() -> tuple[torch.Tensor, torch.Tensor]:
            return torch.topk(local_logits.float(), k=args.top_k, dim=-1)

        def compact_values_gather_only() -> torch.Tensor:
            dist.all_gather_into_tensor(gathered_values, local_values)
            return gathered_values

        def compact_indices_gather_only() -> torch.Tensor:
            dist.all_gather_into_tensor(gathered_indices, local_indices)
            return gathered_indices

        local_pairs[..., 0] = local_values
        local_pairs[..., 1] = local_indices.float()

        def compact_packed_gather_only() -> torch.Tensor:
            dist.all_gather_into_tensor(gathered_pairs, local_pairs)
            return gathered_pairs

        def compact_merge_topk_only() -> tuple[torch.Tensor, torch.Tensor]:
            all_values, all_indices = gathered_compact_candidates()
            values, positions = torch.topk(all_values, k=args.top_k, dim=-1)
            return values, all_indices.gather(-1, positions)

        # Populate static inputs for the two compute-only component probes.
        dist.all_gather_into_tensor(gathered_logits, local_logits)
        dist.all_gather_into_tensor(gathered_values, local_values)
        dist.all_gather_into_tensor(gathered_indices, local_indices)
        torch.cuda.synchronize()
        dist.barrier()

        component_timings = {
            "full_logits_all_gather_only": _time_cuda(
                full_gather_only, args.warmup, args.iters, world_size
            )[0],
            "full_global_topk_only": _time_cuda(
                full_global_topk_only, args.warmup, args.iters, world_size
            )[0],
            "local_topk_only": _time_cuda(
                local_topk_only, args.warmup, args.iters, world_size
            )[0],
            "compact_values_all_gather_only": _time_cuda(
                compact_values_gather_only, args.warmup, args.iters, world_size
            )[0],
            "compact_indices_all_gather_only": _time_cuda(
                compact_indices_gather_only, args.warmup, args.iters, world_size
            )[0],
            "compact_packed_all_gather_only": _time_cuda(
                compact_packed_gather_only, args.warmup, args.iters, world_size
            )[0],
            "compact_merge_topk_only": _time_cuda(
                compact_merge_topk_only, args.warmup, args.iters, world_size
            )[0],
        }

        full_timing, full_result = _time_cuda(
            full_gather_topk, args.warmup, args.iters, world_size
        )
        compact_timing, compact_result = _time_cuda(
            compact_topk_two_gathers, args.warmup, args.iters, world_size
        )
        packed_timing, packed_result = _time_cuda(
            compact_topk_packed_gather, args.warmup, args.iters, world_size
        )
        values_equal = bool(torch.equal(full_result[0], compact_result[0]))
        ids_equal = bool(torch.equal(full_result[1], compact_result[1]))
        packed_values_equal = bool(torch.equal(full_result[0], packed_result[0]))
        packed_ids_equal = bool(torch.equal(full_result[1], packed_result[1]))
        max_value_diff = float((full_result[0] - compact_result[0]).abs().max())

        result = {
            "device": torch.cuda.get_device_name(device),
            "device_capability": list(torch.cuda.get_device_capability(device)),
            "world_size": world_size,
            "shape": {
                "rows": args.rows,
                "vocab_size": args.vocab_size,
                "local_vocab_size": local_vocab_size,
                "top_k": args.top_k,
            },
            "transport_bytes_per_rank": {
                "full_logits_all_gather": local_logits.numel()
                * local_logits.element_size(),
                "compact_values_and_indices_two_gathers": args.rows
                * args.top_k
                * (
                    torch.empty((), dtype=torch.float32).element_size()
                    + torch.empty((), dtype=torch.int64).element_size()
                ),
                "compact_packed_pairs_one_gather": args.rows
                * args.top_k
                * 2
                * torch.empty((), dtype=torch.float32).element_size(),
            },
            "correctness": {
                "topk_values_equal": values_equal,
                "topk_ids_equal": ids_equal,
                "topk_values_max_abs_diff": max_value_diff,
                "packed_topk_values_equal": packed_values_equal,
                "packed_topk_ids_equal": packed_ids_equal,
            },
            "timings": {
                "full_gather_then_global_topk": full_timing,
                "local_topk_then_two_small_gathers": compact_timing,
                "local_topk_then_one_packed_gather": packed_timing,
            },
            "component_timings": component_timings,
        }
        if rank == 0:
            text = json.dumps(result, indent=2, sort_keys=True)
            print(text)
            if args.json_out is not None:
                args.json_out.parent.mkdir(parents=True, exist_ok=True)
                args.json_out.write_text(text + "\n", encoding="utf-8")
        if not (
            values_equal and ids_equal and packed_values_equal and packed_ids_equal
        ):
            raise RuntimeError("Compact TP2 top-k result differs from full gather.")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
