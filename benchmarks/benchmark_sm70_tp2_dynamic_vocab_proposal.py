# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark four TP2/TP4 FP16 draft proposals with a reduced vocabulary."""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from vllm import _sm70_ops as sm70_ops


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-size", type=int, default=5120)
    parser.add_argument("--base-size", type=int, default=65536)
    parser.add_argument("--tail-size", type=int, default=512)
    parser.add_argument("--active-tail-size", type=int)
    parser.add_argument("--full-vocab-size", type=int, default=248320)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--num-steps", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--skip-cuda-graph", action="store_true")
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


def _capture_cuda_graph(operation: Callable[[], Any]) -> torch.cuda.CUDAGraph:
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        operation()
    capture_stream.synchronize()
    dist.barrier()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        operation()
    torch.cuda.synchronize()
    dist.barrier()
    return graph


def main() -> None:
    args = _parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size not in (2, 4):
        raise ValueError(f"This benchmark requires TP2 or TP4, got {world_size}.")
    if args.base_size % world_size or args.tail_size % world_size:
        raise ValueError("Base and tail sizes must be divisible by TP size.")
    active_tail_size = args.active_tail_size or args.tail_size
    if active_tail_size % world_size or not 20 <= active_tail_size <= args.tail_size:
        raise ValueError("Active tail size must fit and be divisible by TP size.")
    if args.top_k != 20:
        raise ValueError("The experimental fused epilogue requires top-k=20.")
    if not 0.0 < args.top_p <= 1.0:
        raise ValueError("top-p must be in (0, 1].")

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    try:
        device = torch.device("cuda", local_rank)
        capability = torch.cuda.get_device_capability(device)
        if capability != (7, 0):
            raise RuntimeError(f"Expected SM70, got sm_{capability[0]}{capability[1]}.")
        if not hasattr(torch.ops._C, "sm70_f16_lm_head_top20_tc_out"):
            raise RuntimeError("The experimental fused top-20 op is not built.")

        torch.manual_seed(args.seed + rank)
        torch.cuda.manual_seed_all(args.seed + rank)
        local_base_size = args.base_size // world_size
        local_tail_size = args.tail_size // world_size
        local_active_tail_size = active_tail_size // world_size
        local_full_vocab = args.full_vocab_size // world_size
        reduced_vocab_size = args.base_size + args.tail_size
        local_tail_row_start = args.base_size + rank * local_tail_size

        base_weight = torch.randn(
            (local_base_size, args.hidden_size),
            dtype=torch.float16,
            device=device,
        )
        prepared = sm70_ops.sm70_f16_prepare(base_weight)
        tm_base_weight = prepared[0]
        k_ld = int(prepared[1][0].item())
        source_weight = torch.zeros(
            (local_full_vocab, args.hidden_size),
            dtype=torch.float16,
            device=device,
        )
        if rank == 0:
            tail_indices = torch.arange(local_tail_size, device=device)
        else:
            tail_indices = torch.arange(
                local_full_vocab - local_tail_size,
                local_full_vocab,
                device=device,
            )
        tail_weight = torch.empty(
            (local_tail_size, args.hidden_size),
            dtype=torch.float16,
            device=device,
        )
        tail_token_ids = tail_indices.to(torch.int64) + rank * local_full_vocab
        tail_token_ids[local_active_tail_size:] = -1
        base_token_id_map = (
            torch.arange(args.base_size, dtype=torch.int64, device=device) + 90000
        )
        hidden = torch.randn(
            (args.num_steps, args.hidden_size),
            dtype=torch.float16,
            device=device,
        )

        base_values = torch.empty((1, args.top_k), dtype=torch.float32, device=device)
        base_ids = torch.empty((1, args.top_k), dtype=torch.int64, device=device)
        tail_logits = torch.empty(
            (1, local_tail_size), dtype=torch.float16, device=device
        )
        local_pairs = torch.empty((args.top_k, 3), dtype=torch.float32, device=device)
        gathered_pairs = torch.empty(
            (world_size * args.top_k, 3), dtype=torch.float32, device=device
        )
        sampled_tokens = torch.empty(
            (args.num_steps,), dtype=torch.int64, device=device
        )
        sparse_ids = torch.empty(
            (args.num_steps, args.top_k), dtype=torch.int64, device=device
        )
        sparse_probs = torch.empty(
            (args.num_steps, args.top_k), dtype=torch.float32, device=device
        )
        exponentials = torch.empty(
            (args.num_steps, reduced_vocab_size),
            dtype=torch.float32,
            device=device,
        )
        dense_q = torch.empty(
            (args.num_steps, args.full_vocab_size),
            dtype=torch.float32,
            device=device,
        )

        step_index = 0

        def refresh_tail() -> None:
            torch.index_select(source_weight, 0, tail_indices, out=tail_weight)

        def base_top20() -> None:
            step_hidden = hidden[step_index : step_index + 1]
            sm70_ops.sm70_f16_lm_head_top20_tc_out(
                base_values,
                base_ids,
                step_hidden,
                tm_base_weight,
                k_ld,
                rank * local_base_size,
                0,
            )

        def tail_projection() -> None:
            step_hidden = hidden[step_index : step_index + 1]
            torch.mm(step_hidden, tail_weight.t(), out=tail_logits)

        def local_top20_pack() -> None:
            tail_values, tail_positions = torch.topk(
                tail_logits.float().masked_fill(
                    tail_token_ids.view(1, -1) < 0,
                    -torch.inf,
                ),
                k=args.top_k,
                dim=-1,
            )
            tail_ids = tail_token_ids[tail_positions]
            candidate_values = torch.cat((base_values, tail_values), dim=-1)
            candidate_ids = torch.cat((base_token_id_map[base_ids], tail_ids), dim=-1)
            tail_rows = tail_positions + local_tail_row_start
            candidate_rows = torch.cat((base_ids, tail_rows), dim=-1)
            local_values, local_positions = torch.topk(
                candidate_values, k=args.top_k, dim=-1
            )
            local_ids = candidate_ids.gather(-1, local_positions)
            local_rows = candidate_rows.gather(-1, local_positions)
            local_pairs[:, 0] = local_values.view(-1)
            local_pairs[:, 1] = local_ids.view(-1).float()
            local_pairs[:, 2] = local_rows.view(-1).float()

        def local_top20_pack_fused() -> None:
            sm70_ops.sm70_merge_tail_top20_pack_out(
                local_pairs,
                base_values,
                base_ids,
                base_token_id_map,
                tail_logits,
                tail_token_ids,
                local_tail_row_start,
            )

        def packed_all_gather() -> None:
            dist.all_gather_into_tensor(gathered_pairs, local_pairs)

        def global_sample() -> None:
            values, positions = torch.topk(gathered_pairs[:, 0], k=args.top_k, dim=-1)
            ids = gathered_pairs[:, 1][positions].to(torch.int64)
            probs = values.softmax(dim=-1, dtype=torch.float32)
            keep = probs.cumsum(dim=-1) - probs < args.top_p
            probs = probs * keep
            probs = probs / probs.sum(dim=-1)
            exponential = torch.empty_like(probs).exponential_()
            offset = (probs / exponential).argmax(dim=-1)
            sampled_tokens[step_index : step_index + 1].copy_(ids[offset].view(1))
            dist.broadcast(sampled_tokens[step_index : step_index + 1], src=0)

        def global_sample_fused() -> None:
            exponential = exponentials[step_index]
            exponential.exponential_()
            sm70_ops.sm70_sample_packed_top20_out(
                sampled_tokens[step_index : step_index + 1],
                sparse_ids[step_index],
                sparse_probs[step_index],
                gathered_pairs,
                exponential,
                args.top_p,
            )
            dist.broadcast(sampled_tokens[step_index : step_index + 1], src=0)

        def expand_dense_q() -> None:
            dense_q.zero_()
            dense_q.scatter_(1, sparse_ids, sparse_probs)

        def proposal_step() -> None:
            base_top20()
            tail_projection()
            local_top20_pack()
            packed_all_gather()
            global_sample()

        def proposal_step_fused() -> None:
            base_top20()
            tail_projection()
            local_top20_pack_fused()
            packed_all_gather()
            global_sample_fused()

        def four_steps() -> None:
            nonlocal step_index
            for index in range(args.num_steps):
                step_index = index
                proposal_step()
            step_index = 0

        def four_steps_fused() -> None:
            nonlocal step_index
            for index in range(args.num_steps):
                step_index = index
                proposal_step_fused()
            step_index = 0

        def refresh_and_four_steps_fused() -> None:
            refresh_tail()
            four_steps_fused()

        def refresh_four_steps_and_expand_fused() -> None:
            refresh_tail()
            four_steps_fused()
            expand_dense_q()

        refresh_tail()
        base_top20()
        tail_projection()
        local_top20_pack()
        reference_local_pairs = local_pairs.clone()
        local_top20_pack_fused()
        reference_order = reference_local_pairs[:, 1].argsort()
        fused_order = local_pairs[:, 1].argsort()
        ordered_reference_pairs = reference_local_pairs[reference_order]
        ordered_fused_pairs = local_pairs[fused_order]
        local_pair_ids_equal = bool(
            torch.equal(ordered_reference_pairs[:, 1], ordered_fused_pairs[:, 1])
        )
        local_pair_values_equal = bool(
            torch.equal(ordered_reference_pairs[:, 0], ordered_fused_pairs[:, 0])
        )
        local_pair_rows_equal = bool(
            torch.equal(ordered_reference_pairs[:, 2], ordered_fused_pairs[:, 2])
        )

        controlled_offsets = rank * args.top_k + torch.arange(
            args.top_k, dtype=torch.float32, device=device
        )
        local_pairs[:, 0] = 2.0 - 0.05 * controlled_offsets
        local_pairs[:, 1] = 1000.0 + 7.0 * controlled_offsets
        local_pairs[:, 2] = rank * args.top_k + torch.arange(
            args.top_k, dtype=torch.float32, device=device
        )
        packed_all_gather()
        reference_values, reference_positions = torch.topk(
            gathered_pairs[:, 0], k=args.top_k, dim=-1
        )
        reference_ids = gathered_pairs[:, 1][reference_positions].to(torch.int64)
        reference_rows = gathered_pairs[:, 2][reference_positions].to(torch.int64)
        reference_probs = reference_values.softmax(dim=-1, dtype=torch.float32)
        reference_keep = reference_probs.cumsum(dim=-1) - reference_probs < args.top_p
        reference_probs = reference_probs * reference_keep
        reference_probs = reference_probs / reference_probs.sum(dim=-1)
        reference_top_p_nonzero = int((reference_probs != 0).sum().item())
        controlled_exponentials = torch.linspace(
            0.5, 1.5, args.top_k, dtype=torch.float32, device=device
        )
        exponentials[0].fill_(1.0)
        exponentials[0].scatter_(0, reference_rows, controlled_exponentials)
        reference_sample = reference_ids[
            (reference_probs / exponentials[0][reference_rows]).argmax()
        ]
        sm70_ops.sm70_sample_packed_top20_out(
            sampled_tokens[0:1],
            sparse_ids[0],
            sparse_probs[0],
            gathered_pairs,
            exponentials[0],
            args.top_p,
        )
        reference_sparse_order = reference_ids.argsort()
        fused_sparse_order = sparse_ids[0].argsort()
        sparse_ids_equal = bool(
            torch.equal(
                reference_ids[reference_sparse_order],
                sparse_ids[0, fused_sparse_order],
            )
        )
        sparse_probs_max_abs_diff = float(
            (
                reference_probs[reference_sparse_order]
                - sparse_probs[0, fused_sparse_order]
            )
            .abs()
            .max()
            .item()
        )
        sampled_token_equal = bool(reference_sample == sampled_tokens[0])

        four_steps_fused()
        expand_dense_q()
        torch.cuda.synchronize()
        dist.barrier()
        dense_q_row_sums = dense_q.sum(dim=-1)
        dense_q_nonzero_per_row = (dense_q != 0).sum(dim=-1)

        timings = {
            "tail_row_refresh_once": _time_cuda(
                refresh_tail, args.warmup, args.iters, world_size
            ),
            "base_fused_top20_m1": _time_cuda(
                base_top20, args.warmup, args.iters, world_size
            ),
            "tail_projection_m1": _time_cuda(
                tail_projection, args.warmup, args.iters, world_size
            ),
            "local_tail_top20_merge_pack_m1": _time_cuda(
                local_top20_pack, args.warmup, args.iters, world_size
            ),
            "fused_local_tail_top20_merge_pack_m1": _time_cuda(
                local_top20_pack_fused, args.warmup, args.iters, world_size
            ),
            "packed_top20_all_gather_m1": _time_cuda(
                packed_all_gather, args.warmup, args.iters, world_size
            ),
            "global_top20_sample_broadcast_m1": _time_cuda(
                global_sample, args.warmup, args.iters, world_size
            ),
            "fused_global_top20_sample_broadcast_m1": _time_cuda(
                global_sample_fused, args.warmup, args.iters, world_size
            ),
            "expand_four_sparse_q_to_dense": _time_cuda(
                expand_dense_q, args.warmup, args.iters, world_size
            ),
            "proposal_step_m1": _time_cuda(
                proposal_step, args.warmup, args.iters, world_size
            ),
            "proposal_four_steps": _time_cuda(
                four_steps, args.warmup, args.iters, world_size
            ),
            "fused_proposal_step_m1": _time_cuda(
                proposal_step_fused, args.warmup, args.iters, world_size
            ),
            "fused_proposal_four_steps": _time_cuda(
                four_steps_fused, args.warmup, args.iters, world_size
            ),
        }
        if not args.skip_cuda_graph:
            graph = _capture_cuda_graph(four_steps_fused)
            timings["fused_proposal_four_steps_cudagraph_replay"] = _time_cuda(
                graph.replay,
                args.warmup,
                args.iters,
                world_size,
            )
            refresh_graph = _capture_cuda_graph(refresh_and_four_steps_fused)
            timings["fused_refresh_plus_four_steps_cudagraph_replay"] = _time_cuda(
                refresh_graph.replay,
                args.warmup,
                args.iters,
                world_size,
            )
            end_to_end_graph = _capture_cuda_graph(refresh_four_steps_and_expand_fused)
            timings["fused_refresh_four_steps_dense_q_cudagraph_replay"] = _time_cuda(
                end_to_end_graph.replay,
                args.warmup,
                args.iters,
                world_size,
            )

        result = {
            "device": torch.cuda.get_device_name(device),
            "device_capability": list(capability),
            "world_size": world_size,
            "shape": {
                "hidden_size": args.hidden_size,
                "global_base_size": args.base_size,
                "local_base_size": local_base_size,
                "global_tail_size": args.tail_size,
                "local_tail_size": local_tail_size,
                "global_active_tail_size": active_tail_size,
                "local_active_tail_size": local_active_tail_size,
                "reduced_vocab_size": reduced_vocab_size,
                "full_vocab_size": args.full_vocab_size,
                "top_k": args.top_k,
                "top_p": args.top_p,
                "num_steps": args.num_steps,
            },
            "notes": {
                "weights": "synthetic FP16 timing weights",
                "tail_policy": "tail IDs and rows are already GPU-resident",
                "candidate_transport": (
                    "one packed FP32 (value, token_id, reduced_row) gather"
                ),
                "base_token_ids": "non-contiguous reduced-row to full-token map",
                "q_representation": (
                    "baseline-width reduced-vocab RNG, sparse top-20 q, then one "
                    "four-row full-vocab scatter"
                ),
                "baseline_postprocess": (
                    "tail top-k/local merge/pack and global sample use PyTorch ops"
                ),
                "fused_postprocess": (
                    "one local merge/pack kernel and one global softmax/sample kernel"
                ),
            },
            "correctness": {
                "local_pair_ids_equal": local_pair_ids_equal,
                "local_pair_values_equal": local_pair_values_equal,
                "local_pair_rows_equal": local_pair_rows_equal,
                "sparse_ids_equal": sparse_ids_equal,
                "sparse_probs_max_abs_diff": sparse_probs_max_abs_diff,
                "sampled_token_equal": sampled_token_equal,
                "controlled_top_p_nonzero": reference_top_p_nonzero,
                "dense_q_row_sums": dense_q_row_sums.cpu().tolist(),
                "dense_q_nonzero_per_row": dense_q_nonzero_per_row.cpu().tolist(),
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
        if not args.skip_cuda_graph:
            del graph, refresh_graph, end_to_end_graph
            gc.collect()
            torch.cuda.synchronize()
            dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
