# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate and time the SM70 TP2 dynamic-vocabulary GPU LRU path."""

from __future__ import annotations

import argparse
import json
import os
import statistics
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from vllm import _sm70_ops as sm70_ops
from vllm.v1.spec_decode.static_draft_vocab import (
    compact_dynamic_draft_vocab_tail_state,
    select_dynamic_draft_vocab_shard_seed,
    update_dynamic_draft_vocab_lru_state,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-size", type=int, default=98_304)
    parser.add_argument("--tail-size", type=int, default=512)
    parser.add_argument("--full-vocab-size", type=int, default=248_320)
    parser.add_argument("--hidden-size", type=int, default=5_120)
    parser.add_argument("--correctness-rounds", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--ranking-path", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def _summary(samples_by_rank: torch.Tensor) -> dict[str, Any]:
    critical = samples_by_rank.max(dim=0).values.cpu().tolist()
    critical.sort()
    return {
        "critical_path_mean_ms": statistics.fmean(critical),
        "critical_path_p50_ms": statistics.median(critical),
        "critical_path_p90_ms": critical[int(0.9 * (len(critical) - 1))],
        "critical_path_p99_ms": critical[int(0.99 * (len(critical) - 1))],
        "critical_path_min_ms": critical[0],
        "critical_path_max_ms": critical[-1],
        "rank_mean_ms": samples_by_rank.mean(dim=1).cpu().tolist(),
    }


def _collect_samples(samples: list[float], world_size: int) -> torch.Tensor:
    local = torch.tensor(samples, dtype=torch.float32, device="cuda")
    gathered = torch.empty(
        world_size * local.numel(), dtype=torch.float32, device="cuda"
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
    return _summary(_collect_samples(samples, world_size))


def _make_round_inputs(
    base: list[int],
    shard_tail_tokens: tuple[tuple[int, ...], ...],
    tail_size: int,
    full_vocab_size: int,
    round_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rank0_seed = shard_tail_tokens[0][:tail_size]
    rank1_seed = shard_tail_tokens[1][:tail_size]
    rank0_cold = shard_tail_tokens[0][tail_size:]
    rank1_cold = shard_tail_tokens[1][tail_size:]
    selected_cold = rank0_cold if round_index % 2 == 0 else rank1_cold
    observed = torch.tensor(
        [
            rank0_seed[(round_index * 7) % tail_size],
            rank1_seed[(round_index * 11) % tail_size],
            -1,
            base[(round_index * 13) % len(base)],
            selected_cold[(round_index * 17) % len(selected_cold)],
        ],
        dtype=torch.int32,
    )
    candidate_rows: list[list[int]] = []
    for row_index in range(5):
        filtered_id = (
            base[(round_index * 101 + row_index * 29) % len(base)]
            if row_index % 2 == 0
            else full_vocab_size
        )
        row = [-1, filtered_id]
        for priority_index in range(18):
            rank_index = (row_index + priority_index) % 2
            use_cold_token = priority_index % 3 == 0
            pools = (
                (rank0_seed, rank0_cold),
                (rank1_seed, rank1_cold),
            )
            pool = pools[rank_index][int(use_cold_token)]
            token_index = (
                round_index * 31 + row_index * 19 + priority_index * 7
            ) % len(pool)
            row.append(pool[token_index])
        candidate_rows.append(row)

    # Repeat observed IDs so move-to-end behavior is exercised across inputs.
    candidate_rows[0][2] = int(observed[0])
    candidate_rows[1][2] = int(observed[1])
    candidate_rows[2][2] = int(observed[4])
    return observed, torch.tensor(candidate_rows, dtype=torch.int64)


def main() -> None:
    args = _parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2:
        raise ValueError(f"This benchmark is TP2-only, got {world_size} ranks.")
    if args.tail_size <= 0 or args.correctness_rounds <= 0:
        raise ValueError("Tail size and correctness rounds must be positive.")

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    try:
        device = torch.device("cuda", local_rank)
        capability = torch.cuda.get_device_capability(device)
        if capability != (7, 0):
            raise RuntimeError(f"Expected SM70, got sm_{capability[0]}{capability[1]}.")

        payload = torch.load(args.ranking_path, map_location="cpu", weights_only=True)
        global_ranking = payload.get("global_ranking")
        if not isinstance(global_ranking, torch.Tensor) or global_ranking.ndim != 1:
            raise ValueError(
                "Ranking artifact requires one-dimensional global_ranking."
            )
        ranking = [int(token_id) for token_id in global_ranking.tolist()]
        if len(ranking) != args.full_vocab_size:
            raise ValueError("Ranking and full vocabulary sizes differ.")

        base = ranking[: args.base_size]
        base_token_ids = frozenset(base)
        shard_width = args.full_vocab_size // world_size
        shard_ranges = tuple(
            (
                shard_rank * shard_width,
                min((shard_rank + 1) * shard_width, args.full_vocab_size),
            )
            for shard_rank in range(world_size)
        )
        local_start, local_end = shard_ranges[rank]
        ranked_tail_token_ids = tuple(
            token_id
            for token_id in dict.fromkeys(ranking[args.base_size :])
            if token_id not in base_token_ids
        )
        shard_tail_tokens = tuple(
            tuple(
                token_id
                for token_id in ranked_tail_token_ids
                if shard_start <= token_id < shard_end
            )
            for shard_start, shard_end in shard_ranges
        )
        if any(len(tokens) <= args.tail_size for tokens in shard_tail_tokens):
            raise ValueError(
                "Each vocabulary shard needs more ranked tail tokens than capacity."
            )
        shard_seeds = tuple(
            select_dynamic_draft_vocab_shard_seed(
                ranking[args.base_size :],
                base_token_ids=base_token_ids,
                local_shard_start=shard_start,
                local_shard_end=shard_end,
                tail_size=args.tail_size,
            )
            for shard_start, shard_end in shard_ranges
        )
        host_lrus: list[OrderedDict[int, None]] = [
            OrderedDict.fromkeys(seed) for seed in shard_seeds
        ]
        host_lru = host_lrus[rank]
        gpu_lru = torch.tensor(shard_seeds[rank], dtype=torch.int64, device=device)
        local_tail_ids = torch.empty_like(gpu_lru)
        local_source_rows = torch.empty_like(gpu_lru)
        base_mask = torch.zeros(args.full_vocab_size, dtype=torch.bool, device=device)
        base_mask[torch.tensor(base, dtype=torch.int64, device=device)] = True

        source_weight = torch.zeros(
            (local_end - local_start, args.hidden_size),
            dtype=torch.float16,
            device=device,
        )
        source_rows = torch.arange(
            local_end - local_start, dtype=torch.int64, device=device
        )
        source_weight[:, 0] = (source_rows % 1024).to(torch.float16)
        source_weight[:, 1] = ((source_rows // 1024) % 1024).to(torch.float16)
        local_tail_weight = torch.empty(
            (args.tail_size, args.hidden_size),
            dtype=torch.float16,
            device=device,
        )
        gathered_tail_ids = torch.empty(
            world_size * args.tail_size, dtype=torch.int64, device=device
        )
        tail_active_mask = torch.empty_like(gathered_tail_ids, dtype=torch.bool)
        mapped_tail_ids = torch.empty_like(gathered_tail_ids)
        gathered_lru_ids = torch.empty_like(gathered_tail_ids)

        def update_and_compact(
            observed: torch.Tensor, candidates: torch.Tensor
        ) -> None:
            sm70_ops.sm70_dynamic_draft_vocab_update_tail_out(
                gpu_lru,
                local_tail_ids,
                local_source_rows,
                observed.reshape(-1),
                candidates.reshape(-1),
                base_mask,
                args.full_vocab_size,
                local_start,
                local_end,
            )

        def refresh_weight() -> None:
            sm70_ops.sm70_dynamic_draft_vocab_refresh_tail_weight_out(
                local_tail_weight,
                source_weight,
                local_source_rows,
            )

        def gather_and_map() -> None:
            dist.all_gather_into_tensor(gathered_tail_ids, local_tail_ids)
            torch.ge(gathered_tail_ids, 0, out=tail_active_mask)
            torch.clamp(gathered_tail_ids, min=0, out=mapped_tail_ids)

        empty_observed = torch.empty(0, dtype=torch.int32, device=device)
        empty_candidates = torch.empty(0, dtype=torch.int64, device=device)
        update_and_compact(empty_observed, empty_candidates)
        refresh_weight()
        gather_and_map()
        torch.cuda.synchronize()

        expected_seed = torch.tensor(
            tuple(token_id for seed in shard_seeds for token_id in seed),
            dtype=torch.int64,
        )
        gathered_seed = gathered_tail_ids.cpu()
        if not torch.equal(gathered_seed, expected_seed):
            raise AssertionError("Gathered shard-local seed order is incorrect.")
        gathered_seed_ids = gathered_seed.tolist()
        if len(set(gathered_seed_ids)) != world_size * args.tail_size:
            raise AssertionError("Gathered shard-local seed contains duplicates.")
        if any(token_id in base_token_ids for token_id in gathered_seed_ids):
            raise AssertionError("Gathered shard-local seed contains base tokens.")
        if not bool(tail_active_mask.all().item()):
            raise AssertionError("Shard-local seed did not activate every tail slot.")

        for round_index in range(args.correctness_rounds):
            observed_cpu, candidates_cpu = _make_round_inputs(
                base,
                shard_tail_tokens,
                args.tail_size,
                args.full_vocab_size,
                round_index,
            )
            for host_rank, expected_host_lru in enumerate(host_lrus):
                shard_start, shard_end = shard_ranges[host_rank]
                update_dynamic_draft_vocab_lru_state(
                    expected_host_lru,
                    observed_cpu,
                    candidates_cpu,
                    full_vocab_size=args.full_vocab_size,
                    base_token_ids=base_token_ids,
                    tail_size=args.tail_size,
                    local_shard_start=shard_start,
                    local_shard_end=shard_end,
                )
            observed = observed_cpu.to(device)
            candidates = candidates_cpu.to(device)
            update_and_compact(observed, candidates)
            refresh_weight()
            gather_and_map()
            torch.cuda.synchronize()

            expected_lru = torch.tensor(tuple(host_lru), dtype=torch.int64)
            if not torch.equal(gpu_lru.cpu(), expected_lru):
                raise AssertionError(f"GPU LRU diverged at round {round_index}.")
            dist.all_gather_into_tensor(gathered_lru_ids, gpu_lru)
            expected_all_lrus = torch.tensor(
                tuple(
                    token_id
                    for expected_host_lru in host_lrus
                    for token_id in expected_host_lru
                ),
                dtype=torch.int64,
            )
            if not torch.equal(gathered_lru_ids.cpu(), expected_all_lrus):
                raise AssertionError(
                    f"TP shard-local LRUs diverged at round {round_index}."
                )
            gathered_ids = gathered_tail_ids.cpu().tolist()
            if len(set(gathered_ids)) != world_size * args.tail_size:
                raise AssertionError(
                    f"Gathered tail IDs repeat at round {round_index}."
                )
            if any(token_id in base_token_ids for token_id in gathered_ids):
                raise AssertionError(
                    f"Gathered tail includes base IDs at round {round_index}."
                )

            expected_local_ids, expected_rows = compact_dynamic_draft_vocab_tail_state(
                tuple(host_lru), local_start, local_end, args.tail_size
            )
            expected_local = torch.full((args.tail_size,), -1, dtype=torch.int64)
            expected_source = torch.full((args.tail_size,), -1, dtype=torch.int64)
            expected_local[: len(expected_local_ids)] = torch.tensor(expected_local_ids)
            expected_source[: len(expected_rows)] = torch.tensor(expected_rows)
            if not torch.equal(local_tail_ids.cpu(), expected_local):
                raise AssertionError(f"Local IDs diverged at round {round_index}.")
            if not torch.equal(local_source_rows.cpu(), expected_source):
                raise AssertionError(f"Source rows diverged at round {round_index}.")

            active_count = len(expected_rows)
            if active_count:
                refreshed = local_tail_weight[:active_count, :2].cpu()
                expected_columns = source_weight[
                    torch.tensor(expected_rows, dtype=torch.int64, device=device), :2
                ].cpu()
                if not torch.equal(refreshed, expected_columns):
                    raise AssertionError(
                        f"Weight rows diverged at round {round_index}."
                    )
            if torch.count_nonzero(local_tail_weight[active_count:]).item() != 0:
                raise AssertionError(
                    f"Inactive weight rows are nonzero at {round_index}."
                )

        observed_cpu, candidates_cpu = _make_round_inputs(
            base,
            shard_tail_tokens,
            args.tail_size,
            args.full_vocab_size,
            args.correctness_rounds,
        )
        observed = observed_cpu.to(device)
        candidates = candidates_cpu.to(device)

        def full_refresh() -> None:
            update_and_compact(observed, candidates)
            refresh_weight()
            gather_and_map()

        timings = {
            "lru_update_and_local_compact": _time_cuda(
                lambda: update_and_compact(observed, candidates),
                args.warmup,
                args.iters,
                world_size,
            ),
            "tail_weight_refresh": _time_cuda(
                refresh_weight, args.warmup, args.iters, world_size
            ),
            "tail_ids_all_gather_and_map": _time_cuda(
                gather_and_map, args.warmup, args.iters, world_size
            ),
            "full_gpu_lru_refresh": _time_cuda(
                full_refresh, args.warmup, args.iters, world_size
            ),
        }

        result = {
            "device": torch.cuda.get_device_name(device),
            "device_capability": list(capability),
            "world_size": world_size,
            "shape": {
                "base_size": args.base_size,
                "shard_local_tail_capacity_per_rank": args.tail_size,
                "active_global_tail_capacity": world_size * args.tail_size,
                "physical_tail_rows_per_rank": args.tail_size,
                "rng_width": args.base_size + world_size * args.tail_size,
                "full_vocab_size": args.full_vocab_size,
                "hidden_size": args.hidden_size,
                "observed_ids_per_round": observed.numel(),
                "candidate_rows_per_round": candidates.shape[0],
                "candidate_ids_per_row": candidates.shape[1],
                "candidate_ids_per_round": candidates.numel(),
            },
            "correctness": {
                "rounds": args.correctness_rounds,
                "host_gpu_shard_lru_exact": True,
                "tp_shard_lrus_exact": True,
                "active_seed_tokens": world_size * args.tail_size,
                "gathered_tail_unique_non_base": True,
                "local_stable_compaction_exact": True,
                "source_row_indices_exact": True,
                "tail_weight_refresh_exact": True,
                "observed_dtype": str(observed.dtype),
                "candidate_dtype": str(candidates.dtype),
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
