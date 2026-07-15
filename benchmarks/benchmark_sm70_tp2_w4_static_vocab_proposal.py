# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark a TP2 TurboMind W4 static-vocabulary draft proposal.

This is a model-free critical-path benchmark for one or four sequential MTP
LM-head plus sampling steps. It measures both the current dense reduced-q
representation and a direct top-k-to-full-q representation. The latter is
mathematically covered by benchmark_sm70_sparse_q_rejection.py.

Run with:

  CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
    benchmarks/benchmark_sm70_tp2_w4_static_vocab_proposal.py
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

from vllm import _sm70_ops as sm70_ops
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-size", type=int, default=5120)
    parser.add_argument("--shortlist-size", type=int, default=131072)
    parser.add_argument("--full-vocab-size", type=int, default=248320)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--num-steps", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260710)
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

    samples: list[float] = []
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


def main() -> None:
    args = _parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2:
        raise ValueError(f"This benchmark is TP2-only, got {world_size} ranks.")
    if args.shortlist_size % world_size:
        raise ValueError("--shortlist-size must be divisible by TP size.")
    if args.hidden_size % args.group_size:
        raise ValueError("--hidden-size must be divisible by group size.")
    if not 0 < args.top_k < args.shortlist_size:
        raise ValueError("--top-k must fit in the shortlist.")
    if args.num_steps <= 0:
        raise ValueError("--num-steps must be positive.")

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    try:
        device = torch.device("cuda", local_rank)
        capability = torch.cuda.get_device_capability(device)
        if capability != (7, 0):
            raise RuntimeError(f"Expected SM70, got sm_{capability[0]}{capability[1]}.")
        for name in ("awq_sm70_prepare", "awq_gemm_sm70_out"):
            if not hasattr(torch.ops._C, name):
                raise RuntimeError(f"Missing torch op _C::{name}.")

        torch.manual_seed(args.seed + rank)
        torch.cuda.manual_seed_all(args.seed + rank)
        local_vocab_size = args.shortlist_size // world_size
        if local_vocab_size % 8:
            raise ValueError("Local vocabulary size must be divisible by 8.")

        qweight = torch.zeros(
            (args.hidden_size, local_vocab_size // 8),
            dtype=torch.int32,
            device=device,
        )
        scales = torch.ones(
            (args.hidden_size // args.group_size, local_vocab_size),
            dtype=torch.float16,
            device=device,
        )
        qzeros = torch.zeros(
            (args.hidden_size // args.group_size, local_vocab_size // 8),
            dtype=torch.int32,
            device=device,
        )
        tm_weight, tm_scales, meta = sm70_ops.awq_sm70_prepare(
            qweight, scales, qzeros, args.group_size
        )
        k_ld = int(meta[0].item())
        q_ld = int(meta[1].item())
        del qweight, scales, qzeros, meta

        hidden = torch.randn((1, args.hidden_size), dtype=torch.float16, device=device)
        local_logits = torch.empty(
            (1, local_vocab_size), dtype=torch.float16, device=device
        )
        gathered_storage = torch.empty(
            (world_size, local_vocab_size), dtype=torch.float16, device=device
        )
        token_id_map = torch.arange(
            args.shortlist_size, dtype=torch.int64, device=device
        )
        temperature = torch.ones(1, dtype=torch.float32, device=device)
        top_k_tensor = torch.full((1,), args.top_k, dtype=torch.int32, device=device)

        def w4_gemm() -> None:
            sm70_ops.awq_gemm_sm70_out(
                local_logits,
                hidden,
                tm_weight,
                tm_scales,
                args.group_size,
                k_ld,
                q_ld,
                False,
            )

        def gather_logits() -> torch.Tensor:
            dist.all_gather_into_tensor(gathered_storage, local_logits)
            return gathered_storage.view(1, args.shortlist_size)

        def dense_proposal_expand() -> tuple[torch.Tensor, torch.Tensor]:
            logits = gathered_storage.view(1, args.shortlist_size).float()
            logits.div_(temperature.view(-1, 1))
            logits = apply_top_k_top_p(logits, top_k_tensor, None)
            reduced_probs = logits.softmax(dim=-1, dtype=torch.float32)
            exponential = torch.empty_like(reduced_probs).exponential_()
            reduced_token_ids = (reduced_probs / exponential).argmax(dim=-1)
            dist.broadcast(reduced_token_ids, src=0)
            global_token_ids = token_id_map[reduced_token_ids]
            full_probs = torch.zeros(
                (1, args.full_vocab_size), dtype=torch.float32, device=device
            )
            full_probs.scatter_(
                1,
                token_id_map.view(1, -1),
                reduced_probs,
            )
            return global_token_ids, full_probs

        def sparse_topk_direct_expand() -> tuple[torch.Tensor, torch.Tensor]:
            logits = gathered_storage.view(1, args.shortlist_size).float()
            logits.div_(temperature.view(-1, 1))
            topk_values, reduced_topk_ids = torch.topk(logits, k=args.top_k, dim=-1)
            topk_probs = topk_values.softmax(dim=-1, dtype=torch.float32)
            exponential = torch.empty_like(topk_probs).exponential_()
            sampled_offset = (topk_probs / exponential).argmax(dim=-1, keepdim=True)
            global_topk_ids = token_id_map[reduced_topk_ids]
            global_token_ids = global_topk_ids.gather(1, sampled_offset).view(-1)
            dist.broadcast(global_token_ids, src=0)
            full_probs = torch.zeros(
                (1, args.full_vocab_size), dtype=torch.float32, device=device
            )
            full_probs.scatter_(1, global_topk_ids, topk_probs)
            return global_token_ids, full_probs

        def dense_step() -> tuple[torch.Tensor, torch.Tensor]:
            w4_gemm()
            gather_logits()
            return dense_proposal_expand()

        def sparse_step() -> tuple[torch.Tensor, torch.Tensor]:
            w4_gemm()
            gather_logits()
            return sparse_topk_direct_expand()

        def repeat(operation: Callable[[], Any]) -> None:
            for _ in range(args.num_steps):
                operation()

        w4_gemm()
        gather_logits()
        torch.cuda.synchronize()
        dist.barrier()

        timings = {
            "w4_local_lm_head_m1": _time_cuda(
                w4_gemm, args.warmup, args.iters, world_size
            ),
            "reduced_logits_all_gather_m1": _time_cuda(
                gather_logits, args.warmup, args.iters, world_size
            ),
            "current_dense_proposal_expand_m1": _time_cuda(
                dense_proposal_expand, args.warmup, args.iters, world_size
            ),
            "sparse_topk_direct_expand_m1": _time_cuda(
                sparse_topk_direct_expand, args.warmup, args.iters, world_size
            ),
            "w4_current_dense_step_m1": _time_cuda(
                dense_step, args.warmup, args.iters, world_size
            ),
            "w4_sparse_topk_step_m1": _time_cuda(
                sparse_step, args.warmup, args.iters, world_size
            ),
            "w4_current_dense_four_steps": _time_cuda(
                lambda: repeat(dense_step),
                args.warmup,
                args.iters,
                world_size,
            ),
            "w4_sparse_topk_four_steps": _time_cuda(
                lambda: repeat(sparse_step),
                args.warmup,
                args.iters,
                world_size,
            ),
        }
        result = {
            "device": torch.cuda.get_device_name(device),
            "device_capability": list(capability),
            "world_size": world_size,
            "shape": {
                "hidden_size": args.hidden_size,
                "shortlist_size": args.shortlist_size,
                "local_vocab_size": local_vocab_size,
                "full_vocab_size": args.full_vocab_size,
                "top_k": args.top_k,
                "num_steps": args.num_steps,
            },
            "quantization": {
                "format": "TurboMind W4A16 prepared AWQ",
                "group_size": args.group_size,
                "k_ld": k_ld,
                "q_ld": q_ld,
                "prepared_weight_bytes": tm_weight.numel() * tm_weight.element_size(),
                "prepared_scale_bytes": tm_scales.numel() * tm_scales.element_size(),
            },
            "notes": {
                "weights": "synthetic zero-point timing weights",
                "dense_path": "matches current dense reduced-q then full-q expansion",
                "sparse_path": (
                    "direct top-k support to full-q; correctness proven separately"
                ),
                "acceptance": "not measured by this model-free benchmark",
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
