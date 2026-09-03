# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark batched compact DFlash2 top-k/top-p rejection on one SM70 GPU.

This isolates target sampling after the TP merge. The separate TP4 compact
logit benchmark measures local top-k and candidate transport.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
    dflash2_sparse_topk_rejection_sample,
    rejection_sample,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-size", type=int, default=248320)
    parser.add_argument("--num-reqs", type=int, default=1)
    parser.add_argument("--num-speculative-steps", type=int, default=7)
    parser.add_argument("--target-top-k", type=int, default=20)
    parser.add_argument("--draft-top-k", type=int, default=16)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def _time_cuda(
    operation: Callable[[], Any],
    device: torch.device,
    warmup: int,
    iters: int,
) -> dict[str, float]:
    for _ in range(warmup):
        operation()
    torch.accelerator.synchronize(device)

    events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(iters)
    ]
    for start, end in events:
        start.record()
        operation()
        end.record()
    events[-1][1].synchronize()
    samples = [float(start.elapsed_time(end)) for start, end in events]
    ordered = sorted(samples)
    return {
        "mean_ms": statistics.fmean(samples),
        "p50_ms": statistics.median(samples),
        "p90_ms": ordered[int(0.9 * (iters - 1))],
        "p99_ms": ordered[int(0.99 * (iters - 1))],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def main() -> int:
    args = _parse_args()
    if not 0.0 < args.top_p <= 1.0:
        raise ValueError("--top-p must be in (0, 1]")
    if not 0 < args.target_top_k <= 64:
        raise ValueError("--target-top-k must be in [1, 64]")
    if not 0 < args.draft_top_k <= 64:
        raise ValueError("--draft-top-k must be in [1, 64]")
    if args.num_reqs < 1:
        raise ValueError("--num-reqs must be positive")

    device = torch.device(args.device)
    torch.accelerator.set_device_index(device)
    capability = torch.cuda.get_device_capability(device)
    if capability != (7, 0):
        raise RuntimeError(f"Expected SM70, got sm_{capability[0]}{capability[1]}.")
    torch.manual_seed(args.seed)

    num_reqs = args.num_reqs
    num_steps = args.num_speculative_steps
    rows_per_req = num_steps + 1
    num_logits = num_reqs * rows_per_req
    raw_target = torch.randn(
        num_logits,
        args.vocab_size,
        dtype=torch.float16,
        device=device,
    )
    target_topk_logits_fp16, target_topk_ids = torch.topk(
        raw_target,
        k=args.target_top_k,
        dim=-1,
    )
    target_topk_logits = target_topk_logits_fp16.float()
    target_k = torch.full(
        (num_logits,),
        args.target_top_k,
        dtype=torch.int32,
        device=device,
    )
    target_p_rows = torch.full(
        (num_logits,), args.top_p, dtype=torch.float32, device=device
    )
    processed_target = apply_top_k_top_p(raw_target.float(), target_k, target_p_rows)

    target_topk_ids_by_req = target_topk_ids.view(
        num_reqs, rows_per_req, args.target_top_k
    )
    target_topk_logits_by_req = target_topk_logits.view(
        num_reqs, rows_per_req, args.target_top_k
    )
    draft_topk_ids = target_topk_ids_by_req[
        :, :num_steps, : args.draft_top_k
    ].contiguous()
    draft_topk_logits = (
        target_topk_logits_by_req[:, :num_steps, : args.draft_top_k]
        + torch.randn(
            num_reqs,
            num_steps,
            args.draft_top_k,
            dtype=torch.float32,
            device=device,
        )
        * 0.2
    ).contiguous()
    dense_draft = torch.full(
        (num_reqs, num_steps, args.vocab_size),
        -float("inf"),
        dtype=torch.float32,
        device=device,
    )
    dense_draft.scatter_(2, draft_topk_ids, draft_topk_logits)

    draft_sampled_2d = torch.zeros(
        num_reqs, rows_per_req, dtype=torch.int64, device=device
    )
    draft_sampled_2d[:, 1:] = draft_topk_ids[:, :, 0]
    draft_sampled = draft_sampled_2d.flatten()
    cu_num_logits = (
        torch.arange(num_reqs + 1, dtype=torch.int32, device=device) * rows_per_req
    )
    pos = torch.arange(num_logits, dtype=torch.int64, device=device) + 32768
    idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=device)
    expanded_idx_mapping = idx_mapping.repeat_interleave(rows_per_req)
    expanded_local_pos = torch.arange(
        rows_per_req, dtype=torch.int32, device=device
    ).repeat(num_reqs)
    temperature = torch.ones(num_reqs, dtype=torch.float32, device=device)
    top_p_per_req = torch.full(
        (num_reqs,), args.top_p, dtype=torch.float32, device=device
    )
    seeds = torch.arange(
        args.seed, args.seed + num_reqs, dtype=torch.int64, device=device
    )

    def dense_rejection() -> tuple[torch.Tensor, torch.Tensor]:
        return rejection_sample(
            processed_target,
            dense_draft,
            draft_sampled,
            cu_num_logits,
            pos,
            idx_mapping,
            expanded_idx_mapping,
            expanded_local_pos,
            temperature,
            seeds,
            num_steps,
        )

    def sparse_rejection() -> tuple[torch.Tensor, torch.Tensor]:
        return dflash2_sparse_topk_rejection_sample(
            target_topk_ids,
            target_topk_logits,
            draft_topk_ids,
            draft_topk_logits,
            draft_sampled,
            cu_num_logits,
            pos,
            idx_mapping,
            temperature,
            top_p_per_req,
            seeds,
            num_steps,
        )

    def dense_topk_topp() -> torch.Tensor:
        return apply_top_k_top_p(raw_target.float(), target_k, target_p_rows)

    def dense_finalize() -> tuple[torch.Tensor, torch.Tensor]:
        target = dense_topk_topp()
        return rejection_sample(
            target,
            dense_draft,
            draft_sampled,
            cu_num_logits,
            pos,
            idx_mapping,
            expanded_idx_mapping,
            expanded_local_pos,
            temperature,
            seeds,
            num_steps,
        )

    dense_out, dense_count = dense_rejection()
    sparse_out, sparse_count = sparse_rejection()
    torch.accelerator.synchronize(device)
    counts_equal = bool(torch.equal(dense_count, sparse_count))
    valid = (
        torch.arange(rows_per_req, device=device).unsqueeze(0) < dense_count[:, None]
    )
    tokens_equal = bool(torch.equal(dense_out[valid], sparse_out[valid]))

    result = {
        "device": torch.cuda.get_device_name(device),
        "device_capability": list(capability),
        "shape": {
            "num_reqs": num_reqs,
            "num_logits": num_logits,
            "vocab_size": args.vocab_size,
            "target_top_k": args.target_top_k,
            "draft_top_k": args.draft_top_k,
            "top_p": args.top_p,
        },
        "correctness": {
            "num_sampled_equal": counts_equal,
            "valid_tokens_equal": tokens_equal,
            "dense_num_sampled": dense_count.tolist(),
            "sparse_num_sampled": sparse_count.tolist(),
        },
        "timings": {
            "dense_topk_topp_only": _time_cuda(
                dense_topk_topp, device, args.warmup, args.iters
            ),
            "dense_rejection_only": _time_cuda(
                dense_rejection, device, args.warmup, args.iters
            ),
            "sparse_topk_topp_rejection": _time_cuda(
                sparse_rejection, device, args.warmup, args.iters
            ),
            "dense_topk_topp_plus_rejection": _time_cuda(
                dense_finalize, device, args.warmup, args.iters
            ),
        },
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    return 0 if counts_equal and tokens_equal else 1


if __name__ == "__main__":
    raise SystemExit(main())
