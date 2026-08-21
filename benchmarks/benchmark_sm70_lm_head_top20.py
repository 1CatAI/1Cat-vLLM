# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check and time the experimental SM70 TurboMind FP16 top-20 epilogue."""

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
from vllm.utils.torch_utils import set_random_seed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=1)
    parser.add_argument("--n", type=int, default=32768)
    parser.add_argument("--k", type=int, default=5120)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--final-k", type=int, default=16)
    parser.add_argument("--pad", type=int, default=0)
    parser.add_argument("--vocab-start", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def _time_cuda(
    operation: Callable[[], Any],
    device: torch.device,
    warmup: int,
    iters: int,
) -> dict[str, float]:
    if dist.is_initialized():
        dist.barrier()
    for _ in range(warmup):
        operation()
    torch.accelerator.synchronize(device)

    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    ordered = sorted(samples)
    if dist.is_initialized():
        dist.barrier()
    return {
        "mean_ms": statistics.fmean(samples),
        "p50_ms": statistics.median(samples),
        "p90_ms": ordered[int(0.9 * (len(ordered) - 1))],
        "p99_ms": ordered[int(0.99 * (len(ordered) - 1))],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def main() -> int:
    args = _parse_args()
    if args.top_k != 20:
        raise ValueError("The prototype epilogue is fixed to top-k=20.")
    if not 1 <= args.m <= 8:
        raise ValueError("The fused top-20 decode kernel supports M in [1, 8].")
    if not 1 <= args.final_k <= args.top_k:
        raise ValueError("--final-k must be in [1, top-k].")
    if not 0 <= args.pad < args.n:
        raise ValueError("--pad must be in [0, n).")
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if world_size > 1:
        torch.accelerator.set_device_index(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(args.device)
    torch.accelerator.set_device_index(device)
    capability = torch.cuda.get_device_capability(device)
    if capability != (7, 0):
        raise RuntimeError(f"Expected SM70, got sm_{capability[0]}{capability[1]}.")
    if not hasattr(torch.ops._C, "sm70_f16_lm_head_top20_tc_workspace_out"):
        raise RuntimeError("The graph-safe experimental top-20 op is not built.")

    vocab_start = args.vocab_start + rank * args.n
    local_padding = args.pad if rank == world_size - 1 else 0
    set_random_seed(args.seed)
    hidden = torch.randn((args.m, args.k), dtype=torch.float16, device=device)
    set_random_seed(args.seed + rank + 1)
    weight = torch.randn((args.n, args.k), dtype=torch.float16, device=device)
    prepared = sm70_ops.sm70_f16_prepare(weight)
    tm_weight = prepared[0]
    k_ld = int(prepared[1][0].item())

    dense_logits = torch.empty((args.m, args.n), dtype=torch.float16, device=device)
    torch_logits = torch.empty_like(dense_logits)
    fused_values = torch.empty((args.m, args.top_k), dtype=torch.float32, device=device)
    fused_indices = torch.empty((args.m, args.top_k), dtype=torch.int64, device=device)
    num_vocab_tiles = (args.n - local_padding + 255) // 256
    partial_values = torch.empty(
        (args.m, num_vocab_tiles, args.top_k),
        dtype=torch.float32,
        device=device,
    )
    partial_indices = torch.empty(
        (args.m, num_vocab_tiles, args.top_k),
        dtype=torch.int64,
        device=device,
    )
    fused_values_fp16 = torch.empty(
        (args.m, args.top_k), dtype=torch.float16, device=device
    )
    sparse_logits = torch.empty_like(dense_logits)
    fused_local_values = torch.empty(
        (args.m, args.final_k), dtype=torch.float16, device=device
    )
    fused_local_indices = torch.empty(
        (args.m, args.final_k), dtype=torch.int64, device=device
    )

    def dense_gemm() -> None:
        sm70_ops.sm70_f16_gemm_out(
            dense_logits,
            hidden,
            tm_weight,
            k_ld,
            False,
        )
        if local_padding:
            dense_logits[:, -local_padding:] = -float("inf")

    def dense_gemm_topk() -> tuple[torch.Tensor, torch.Tensor]:
        dense_gemm()
        values, indices = torch.topk(dense_logits, k=args.final_k, dim=-1)
        return values, indices.to(torch.int64) + vocab_start

    def torch_mm() -> None:
        torch.mm(hidden, weight.t(), out=torch_logits)
        if local_padding:
            torch_logits[:, -local_padding:] = -float("inf")

    def torch_mm_topk() -> tuple[torch.Tensor, torch.Tensor]:
        torch_mm()
        values, indices = torch.topk(torch_logits, k=args.final_k, dim=-1)
        return values, indices.to(torch.int64) + vocab_start

    def fused_top20() -> None:
        sm70_ops.sm70_f16_lm_head_top20_tc_workspace_out(
            fused_values,
            fused_indices,
            partial_values,
            partial_indices,
            hidden,
            tm_weight,
            k_ld,
            vocab_start,
            local_padding,
        )

    def fused_dense_order_local_topk() -> tuple[torch.Tensor, torch.Tensor]:
        fused_top20()
        fused_values_fp16.copy_(fused_values)
        fused_indices.sub_(vocab_start)
        sparse_logits.fill_(-float("inf"))
        sparse_logits.scatter_(1, fused_indices, fused_values_fp16)
        torch.topk(
            sparse_logits,
            k=args.final_k,
            dim=-1,
            sorted=True,
            out=(fused_local_values, fused_local_indices),
        )
        fused_local_indices.add_(vocab_start)
        return fused_local_values, fused_local_indices

    def gather_last_dim(tensor: torch.Tensor) -> torch.Tensor:
        if world_size == 1:
            return tensor
        output = torch.empty(
            (world_size * tensor.shape[0], tensor.shape[1]),
            dtype=tensor.dtype,
            device=tensor.device,
        )
        dist.all_gather_into_tensor(output, tensor.contiguous())
        return (
            output.view(world_size, *tensor.shape)
            .movedim(0, -2)
            .reshape(tensor.shape[0], world_size * tensor.shape[1])
        )

    def global_topk(
        local_values: torch.Tensor, local_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        values = gather_last_dim(local_values)
        indices = gather_last_dim(local_indices)
        if values.shape[-1] == args.final_k:
            return values, indices
        selected_values, positions = torch.topk(values, k=args.final_k, dim=-1)
        return selected_values, indices.gather(-1, positions)

    def torch_mm_global_topk() -> tuple[torch.Tensor, torch.Tensor]:
        return global_topk(*torch_mm_topk())

    def fused_final_topk() -> tuple[torch.Tensor, torch.Tensor]:
        return global_topk(*fused_dense_order_local_topk())

    reference_values, reference_indices = torch_mm_global_topk()
    candidate_values, candidate_indices = fused_final_topk()
    torch.accelerator.synchronize(device)
    values_equal = bool(torch.equal(reference_values, candidate_values))
    indices_equal = bool(torch.equal(reference_indices, candidate_indices))
    support_indices_equal = indices_equal
    support_values_equal = values_equal
    max_abs_diff = float((reference_values - candidate_values).abs().max().item())

    result = {
        "device": torch.cuda.get_device_name(device),
        "device_capability": list(capability),
        "shape": {
            "m": args.m,
            "n": args.n,
            "k": args.k,
            "top_k": args.top_k,
            "final_k": args.final_k,
        },
        "pad": local_padding,
        "vocab_start": vocab_start,
        "tp_size": world_size,
        "correctness": {
            "values_equal": values_equal,
            "indices_equal": indices_equal,
            "support_indices_equal": support_indices_equal,
            "support_values_equal": support_values_equal,
            "values_max_abs_diff": max_abs_diff,
            "reference_indices": reference_indices.cpu().tolist(),
            "fused_indices": candidate_indices.cpu().tolist(),
        },
        "timings": {
            "torch_mm_only": _time_cuda(torch_mm, device, args.warmup, args.iters),
            "torch_mm_then_topk": _time_cuda(
                torch_mm_topk, device, args.warmup, args.iters
            ),
            "torch_mm_local16_tp_global16": _time_cuda(
                torch_mm_global_topk, device, args.warmup, args.iters
            ),
            "sm70_f16_gemm_only": _time_cuda(
                dense_gemm, device, args.warmup, args.iters
            ),
            "sm70_f16_gemm_then_torch_topk": _time_cuda(
                dense_gemm_topk, device, args.warmup, args.iters
            ),
            "sm70_f16_fused_top20": _time_cuda(
                fused_top20, device, args.warmup, args.iters
            ),
            "sm70_f16_fused_top20_dense_order_local16_tp_global16": _time_cuda(
                fused_final_topk, device, args.warmup, args.iters
            ),
        },
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    if rank == 0:
        print(text)
    if rank == 0 and args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    return 0 if support_indices_equal and support_values_equal else 1


if __name__ == "__main__":
    raise SystemExit(main())
