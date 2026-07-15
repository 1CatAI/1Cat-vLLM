# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate and time sparse-q rejection for a reduced draft vocabulary."""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-vocab-size", type=int, default=248320)
    parser.add_argument("--draft-vocab-size", type=int, default=131072)
    parser.add_argument("--num-spec-tokens", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--atol", type=float, default=1e-6)
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
    torch.cuda.synchronize(device)

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
        "p90_ms": ordered[int(0.9 * (len(ordered) - 1))],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def _draft_sparse_proposal(
    draft_logits: torch.Tensor,
    draft_id_map: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    topk_values, reduced_ids = torch.topk(draft_logits.float(), k=top_k, dim=-1)
    topk_probs = topk_values.softmax(dim=-1, dtype=torch.float32)
    global_ids = draft_id_map[reduced_ids]
    exponential = torch.empty_like(topk_probs).exponential_()
    sampled_offsets = (topk_probs / exponential).argmax(dim=-1, keepdim=True)
    sampled_ids = global_ids.gather(1, sampled_offsets).view(-1)
    return sampled_ids, global_ids, topk_probs


def _draft_dense_proposal(
    draft_logits: torch.Tensor,
    draft_id_map: torch.Tensor,
    top_k: int,
    full_vocab_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    sampled_ids, global_ids, topk_probs = _draft_sparse_proposal(
        draft_logits, draft_id_map, top_k
    )
    dense_probs = torch.zeros(
        (draft_logits.shape[0], full_vocab_size),
        dtype=torch.float32,
        device=draft_logits.device,
    )
    dense_probs.scatter_(1, global_ids, topk_probs)
    return sampled_ids, dense_probs


def _target_sparse_topk_topp(
    target_logits: torch.Tensor,
    top_k: int,
    top_p: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    topk_values, topk_ids = torch.topk(target_logits.float(), k=top_k, dim=-1)
    unfiltered_probs = topk_values.softmax(dim=-1, dtype=torch.float32)
    remove = unfiltered_probs.cumsum(dim=-1) > top_p
    remove[:, 1:] = remove[:, :-1].clone()
    remove[:, 0] = False
    topk_values = topk_values.masked_fill(remove, -float("inf"))
    topk_probs = topk_values.softmax(dim=-1, dtype=torch.float32)
    return topk_ids, topk_probs


def _lookup_sparse_probs(
    query_ids: torch.Tensor,
    support_ids: torch.Tensor,
    support_probs: torch.Tensor,
) -> torch.Tensor:
    matches = support_ids == query_ids.view(-1, 1)
    return torch.where(matches, support_probs, 0.0).sum(dim=-1)


def _sparse_q_rejection_core(
    draft_token_ids: torch.Tensor,
    draft_ids: torch.Tensor,
    draft_probs: torch.Tensor,
    target_ids: torch.Tensor,
    target_probs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_draft = draft_token_ids.shape[0]
    target_draft_probs = _lookup_sparse_probs(
        draft_token_ids,
        target_ids[:num_draft],
        target_probs[:num_draft],
    )
    sampled_draft_probs = _lookup_sparse_probs(
        draft_token_ids,
        draft_ids,
        draft_probs,
    )
    accept_ratio = (
        target_draft_probs
        / sampled_draft_probs.clamp_min(torch.finfo(torch.float32).tiny)
    ).clamp_max_(1.0)
    accepted = (torch.rand_like(accept_ratio) <= accept_ratio).cumprod(dim=0)

    id_matches = target_ids[:num_draft].unsqueeze(-1) == draft_ids.unsqueeze(1)
    draft_probs_at_target = torch.where(
        id_matches,
        draft_probs.unsqueeze(1),
        0.0,
    ).sum(dim=-1)
    residual = (target_probs[:num_draft] - draft_probs_at_target).clamp_min_(0.0)
    exponential = torch.empty_like(residual).exponential_()
    recovered_offsets = (residual / exponential).argmax(dim=-1, keepdim=True)
    recovered_ids = target_ids[:num_draft].gather(1, recovered_offsets).view(-1)
    return accepted, recovered_ids


def _dense_q_rejection_core(
    draft_token_ids: torch.Tensor,
    dense_draft_probs: torch.Tensor,
    dense_target_probs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_draft = draft_token_ids.shape[0]
    query = draft_token_ids.view(-1, 1)
    target_draft_probs = dense_target_probs[:num_draft].gather(1, query).view(-1)
    sampled_draft_probs = dense_draft_probs.gather(1, query).view(-1)
    accept_ratio = (
        target_draft_probs
        / sampled_draft_probs.clamp_min(torch.finfo(torch.float32).tiny)
    ).clamp_max_(1.0)
    accepted = (torch.rand_like(accept_ratio) <= accept_ratio).cumprod(dim=0)

    residual = (dense_target_probs[:num_draft] - dense_draft_probs).clamp_min_(0.0)
    exponential = torch.empty_like(residual).exponential_()
    recovered_ids = (residual / exponential).argmax(dim=-1)
    return accepted, recovered_ids


def main() -> int:
    args = _parse_args()
    if not 0.0 < args.top_p <= 1.0:
        raise ValueError("--top-p must be in (0, 1].")
    if not 0 < args.top_k <= args.draft_vocab_size:
        raise ValueError("--top-k must fit in the draft vocabulary.")
    if args.draft_vocab_size > args.full_vocab_size:
        raise ValueError("Draft vocabulary cannot exceed the full vocabulary.")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    capability = torch.cuda.get_device_capability(device)
    if capability != (7, 0):
        raise RuntimeError(f"Expected SM70, got sm_{capability[0]}{capability[1]}.")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    num_draft = args.num_spec_tokens
    draft_id_map = torch.randperm(args.full_vocab_size, device=device)[
        : args.draft_vocab_size
    ]
    draft_logits = torch.randn(
        (num_draft, args.draft_vocab_size),
        dtype=torch.float16,
        device=device,
    )
    # Make the top-k unique so correctness does not depend on tie ordering.
    draft_logits[:, : args.top_k] = (
        64.0 - torch.arange(args.top_k, dtype=torch.float16, device=device) / 8.0
    )

    draft_token_ids, draft_ids, draft_probs = _draft_sparse_proposal(
        draft_logits, draft_id_map, args.top_k
    )
    dense_draft_probs = torch.zeros(
        (num_draft, args.full_vocab_size), dtype=torch.float32, device=device
    )
    dense_draft_probs.scatter_(1, draft_ids, draft_probs)

    target_logits = torch.randn(
        (num_draft + 1, args.full_vocab_size),
        dtype=torch.float16,
        device=device,
    )
    overlap_count = args.top_k // 2
    other_count = args.top_k - overlap_count
    for row in range(num_draft):
        other_ids = (
            torch.arange(other_count, dtype=torch.int64, device=device)
            + (row + 1) * 1024
        )
        if torch.isin(other_ids, draft_ids[row]).any():
            raise RuntimeError("Synthetic target-only IDs overlap draft support.")
        target_top_ids = torch.cat([draft_ids[row, :overlap_count], other_ids], dim=0)
        target_logits[row, target_top_ids] = (
            64.0 - torch.arange(args.top_k, dtype=torch.float16, device=device) / 8.0
        )
    bonus_ids = torch.arange(args.top_k, dtype=torch.int64, device=device) + 1024
    target_logits[-1, bonus_ids] = (
        64.0 - torch.arange(args.top_k, dtype=torch.float16, device=device) / 8.0
    )

    target_ids, target_probs = _target_sparse_topk_topp(
        target_logits, args.top_k, args.top_p
    )
    top_k_tensor = torch.full(
        (num_draft + 1,), args.top_k, dtype=torch.int32, device=device
    )
    top_p_tensor = torch.full(
        (num_draft + 1,), args.top_p, dtype=torch.float32, device=device
    )
    dense_target_logits = apply_top_k_top_p(
        target_logits.float().clone(), top_k_tensor, top_p_tensor
    )
    dense_target_probs = dense_target_logits.softmax(dim=-1, dtype=torch.float32)

    sparse_target_at_ids = dense_target_probs.gather(1, target_ids)
    target_prob_max_abs = float(
        (sparse_target_at_ids - target_probs).abs().max().item()
    )
    target_mass_max_abs = float(
        (dense_target_probs.sum(dim=-1) - target_probs.sum(dim=-1)).abs().max().item()
    )

    query = draft_token_ids.view(-1, 1)
    dense_p_at_draft = dense_target_probs[:num_draft].gather(1, query).view(-1)
    dense_q_at_draft = dense_draft_probs.gather(1, query).view(-1)
    sparse_p_at_draft = _lookup_sparse_probs(
        draft_token_ids, target_ids[:num_draft], target_probs[:num_draft]
    )
    sparse_q_at_draft = _lookup_sparse_probs(draft_token_ids, draft_ids, draft_probs)

    id_matches = target_ids[:num_draft].unsqueeze(-1) == draft_ids.unsqueeze(1)
    sparse_q_at_target = torch.where(id_matches, draft_probs.unsqueeze(1), 0.0).sum(
        dim=-1
    )
    sparse_residual = (target_probs[:num_draft] - sparse_q_at_target).clamp_min(0.0)
    dense_residual = (dense_target_probs[:num_draft] - dense_draft_probs).clamp_min(0.0)
    dense_residual_at_target = dense_residual.gather(1, target_ids[:num_draft])

    correctness = {
        "target_topk_ids_equal": bool(
            torch.equal(
                target_logits.float().topk(args.top_k, dim=-1).indices,
                target_ids,
            )
        ),
        "target_prob_max_abs": target_prob_max_abs,
        "target_mass_max_abs": target_mass_max_abs,
        "p_at_draft_max_abs": float(
            (dense_p_at_draft - sparse_p_at_draft).abs().max().item()
        ),
        "q_at_draft_max_abs": float(
            (dense_q_at_draft - sparse_q_at_draft).abs().max().item()
        ),
        "residual_at_target_max_abs": float(
            (dense_residual_at_target - sparse_residual).abs().max().item()
        ),
        "residual_mass_max_abs": float(
            (dense_residual.sum(dim=-1) - sparse_residual.sum(dim=-1))
            .abs()
            .max()
            .item()
        ),
    }
    correctness_pass = correctness["target_topk_ids_equal"] and all(
        value <= args.atol
        for key, value in correctness.items()
        if key != "target_topk_ids_equal"
    )

    timings = {
        "draft_sparse_q_proposal_m1": _time_cuda(
            lambda: _draft_sparse_proposal(draft_logits[:1], draft_id_map, args.top_k),
            device,
            args.warmup,
            args.iters,
        ),
        "draft_dense_q_scatter_proposal_m1": _time_cuda(
            lambda: _draft_dense_proposal(
                draft_logits[:1],
                draft_id_map,
                args.top_k,
                args.full_vocab_size,
            ),
            device,
            args.warmup,
            args.iters,
        ),
        "target_sparse_topk_topp_m5": _time_cuda(
            lambda: _target_sparse_topk_topp(target_logits, args.top_k, args.top_p),
            device,
            args.warmup,
            args.iters,
        ),
        "sparse_q_rejection_core_m4": _time_cuda(
            lambda: _sparse_q_rejection_core(
                draft_token_ids,
                draft_ids,
                draft_probs,
                target_ids,
                target_probs,
            ),
            device,
            args.warmup,
            args.iters,
        ),
        "dense_q_rejection_core_m4": _time_cuda(
            lambda: _dense_q_rejection_core(
                draft_token_ids,
                dense_draft_probs,
                dense_target_probs,
            ),
            device,
            args.warmup,
            args.iters,
        ),
    }
    result = {
        "device": torch.cuda.get_device_name(device),
        "device_capability": list(capability),
        "full_vocab_size": args.full_vocab_size,
        "draft_vocab_size": args.draft_vocab_size,
        "num_spec_tokens": num_draft,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "sparse_q_values_per_step": args.top_k,
        "correctness_atol": args.atol,
        "correctness_pass": correctness_pass,
        "correctness": correctness,
        "warmup": args.warmup,
        "iters": args.iters,
        "timings": timings,
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    return 0 if correctness_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
