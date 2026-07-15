# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Microbenchmark verifier-side stochastic rejection sampling pieces.

This is intentionally model-free. It measures the current MTP verifier
sampling path for the Qwen3.6 27B vocab shape, plus a sparse recovered-token
prototype that exploits official top-k sampling.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from dataclasses import replace
from pathlib import Path

import torch

from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
from vllm.v1.sample.rejection_sampler import (
    RejectionSampler,
    apply_sampling_constraints,
    rejection_sample,
    sample_recovered_tokens,
)
from vllm.v1.sample.sampler import Sampler
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata


def _make_sampling_metadata(device: torch.device, top_k: int, top_p: float):
    return SamplingMetadata(
        temperature=torch.tensor([1.0], dtype=torch.float32, device=device),
        all_greedy=False,
        all_random=True,
        top_p=torch.tensor([top_p], dtype=torch.float32, device=device),
        top_k=torch.tensor([top_k], dtype=torch.int32, device=device),
        generators={},
        max_num_logprobs=None,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=torch.empty(0, device=device),
        presence_penalties=torch.empty(0, device=device),
        repetition_penalties=torch.empty(0, device=device),
        output_token_ids=[[]],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
        spec_token_ids=None,
        top_k_cpu=(top_k,),
    )


def _time_cuda(fn, warmup: int, iters: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples_ms: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples_ms.append(start.elapsed_time(end))

    samples_ms.sort()
    return {
        "mean_ms": statistics.fmean(samples_ms),
        "p50_ms": samples_ms[len(samples_ms) // 2],
        "p90_ms": samples_ms[int(len(samples_ms) * 0.9)],
        "p99_ms": samples_ms[min(len(samples_ms) - 1, int(len(samples_ms) * 0.99))],
        "min_ms": samples_ms[0],
        "max_ms": samples_ms[-1],
    }


def _sparse_recovered_prototype(
    *,
    draft_probs: torch.Tensor,
    target_probs: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    # After apply_top_k_top_p, target_probs is zero outside the target sampling
    # support. The standard residual max(target-draft, 0) therefore also has
    # no support outside these top-k/top-p entries.
    topk_probs, topk_ids = torch.topk(target_probs, k=top_k, dim=-1)
    draft_topk_probs = draft_probs.gather(1, topk_ids)
    residual = (topk_probs - draft_topk_probs).clamp_min_(0.0)
    inv_q = torch.empty_like(residual).exponential_().reciprocal_()
    scores = residual * inv_q
    chosen = topk_ids.gather(1, scores.argmax(dim=-1, keepdim=True)).squeeze(1)
    has_mass = residual.sum(dim=-1) > 0
    return torch.where(has_mass, chosen, torch.zeros_like(chosen))


def _sparse_topk_topp_sample_logits_prototype(
    logits: torch.Tensor,
    *,
    top_k: int,
    top_p: float,
) -> torch.Tensor:
    # Equivalent sampling distribution for the common official Qwen setting
    # top_k=20/top_p=0.95 when there are no penalties or extra processors.
    # This avoids the full-vocab mask + softmax used by the generic sampler.
    topk_values, topk_ids = torch.topk(logits.float(), k=top_k, dim=-1)
    topk_probs = topk_values.softmax(dim=-1, dtype=torch.float32)
    sorted_cumsum = topk_probs.cumsum(dim=-1)
    remove_mask = sorted_cumsum > top_p
    remove_mask[:, 1:] = remove_mask[:, :-1].clone()
    remove_mask[:, 0] = False
    topk_values = topk_values.masked_fill(remove_mask, -float("inf"))
    filtered_probs = topk_values.softmax(dim=-1, dtype=torch.float32)
    q = torch.empty_like(filtered_probs).exponential_()
    offsets = (filtered_probs / q).argmax(dim=-1, keepdim=True)
    return topk_ids.gather(1, offsets).view(-1)


def _dense_draft_topk_proposal_prototype(
    logits: torch.Tensor,
    *,
    top_k_tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    proposal_logits = apply_top_k_top_p(logits.float().clone(), top_k_tensor, None)
    probs = proposal_logits.softmax(dim=-1, dtype=torch.float32)
    q = torch.empty_like(probs).exponential_()
    token_ids = (probs / q).argmax(dim=-1).view(-1)
    return token_ids, probs


def _sparse_draft_topk_no_scatter_prototype(
    logits: torch.Tensor,
    *,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    topk_values, topk_ids = torch.topk(logits.float(), k=top_k, dim=-1)
    topk_probs = topk_values.softmax(dim=-1, dtype=torch.float32)
    q = torch.empty_like(topk_probs).exponential_()
    offsets = (topk_probs / q).argmax(dim=-1, keepdim=True)
    token_ids = topk_ids.gather(1, offsets).view(-1)
    return token_ids, topk_ids, topk_probs


def _sparse_draft_topk_with_scatter_prototype(
    logits: torch.Tensor,
    *,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    token_ids, topk_ids, topk_probs = _sparse_draft_topk_no_scatter_prototype(
        logits,
        top_k=top_k,
    )
    probs = torch.zeros_like(logits, dtype=torch.float32)
    probs.scatter_(1, topk_ids, topk_probs)
    return token_ids, probs


def _sparse_target_topk_topp_rejection_prototype(
    sampled_logits: torch.Tensor,
    *,
    draft_token_ids: torch.Tensor,
    draft_probs: torch.Tensor,
    top_k: int,
    top_p: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Torch prototype for sparse official top-k/top-p verifier sampling."""
    num_tokens = draft_token_ids.shape[0]
    target_logits = sampled_logits[:num_tokens].float()
    bonus_logits = sampled_logits[num_tokens:].float()

    target_topk_values, target_topk_ids = torch.topk(
        target_logits, k=top_k, dim=-1
    )
    target_topk_probs = target_topk_values.softmax(dim=-1, dtype=torch.float32)
    target_remove_mask = target_topk_probs.cumsum(dim=-1) > top_p
    target_remove_mask[:, 1:] = target_remove_mask[:, :-1].clone()
    target_remove_mask[:, 0] = False
    target_topk_values = target_topk_values.masked_fill(
        target_remove_mask, -float("inf")
    )
    target_topk_probs = target_topk_values.softmax(dim=-1, dtype=torch.float32)

    bonus_topk_values, bonus_topk_ids = torch.topk(bonus_logits, k=top_k, dim=-1)
    bonus_topk_probs = bonus_topk_values.softmax(dim=-1, dtype=torch.float32)
    bonus_remove_mask = bonus_topk_probs.cumsum(dim=-1) > top_p
    bonus_remove_mask[:, 1:] = bonus_remove_mask[:, :-1].clone()
    bonus_remove_mask[:, 0] = False
    bonus_topk_values = bonus_topk_values.masked_fill(
        bonus_remove_mask, -float("inf")
    )
    bonus_topk_probs = bonus_topk_values.softmax(dim=-1, dtype=torch.float32)

    draft_ids_i64 = draft_token_ids.to(torch.int64).view(-1, 1)
    target_matches = target_topk_ids == draft_ids_i64
    target_draft_probs = torch.where(
        target_matches,
        target_topk_probs,
        torch.zeros_like(target_topk_probs),
    ).sum(dim=-1)
    draft_token_probs = draft_probs.gather(1, draft_ids_i64).squeeze(1)
    accept_ratio = (
        target_draft_probs
        / draft_token_probs.clamp_min(torch.finfo(torch.float32).tiny)
    ).clamp_max(1.0)
    accepted_prefix = (torch.rand_like(accept_ratio) <= accept_ratio).cumprod(0)

    residual = (
        target_topk_probs - draft_probs.gather(1, target_topk_ids)
    ).clamp_min_(0.0)
    residual_q = torch.empty_like(residual).exponential_()
    recovered_offsets = (residual / residual_q).argmax(dim=-1, keepdim=True)
    recovered_token_ids = target_topk_ids.gather(1, recovered_offsets).view(-1)

    bonus_q = torch.empty_like(bonus_topk_probs).exponential_()
    bonus_offsets = (bonus_topk_probs / bonus_q).argmax(dim=-1, keepdim=True)
    bonus_token_ids = bonus_topk_ids.gather(1, bonus_offsets).view(-1)
    return accepted_prefix, recovered_token_ids, bonus_token_ids


def _full_vocab_topk_topp_probability_reference(
    sampled_logits: torch.Tensor,
    *,
    draft_token_ids: torch.Tensor,
    draft_probs: torch.Tensor,
    top_k: int,
    top_k_tensor: torch.Tensor,
    top_p_tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Full-vocab reference for compact verifier-logits probability checks."""
    constrained = apply_top_k_top_p(
        sampled_logits.float().clone(),
        top_k_tensor,
        top_p_tensor,
    )
    probs = constrained.softmax(dim=-1, dtype=torch.float32)
    num_draft = draft_token_ids.shape[0]
    draft_ids_i64 = draft_token_ids.to(torch.int64).view(-1, 1)
    target_draft_probs = probs[:num_draft].gather(1, draft_ids_i64).squeeze(1)
    target_top_probs, target_top_ids = torch.topk(
        probs[:num_draft],
        k=min(top_k, probs.shape[-1]),
        dim=-1,
    )
    residual = (
        target_top_probs - draft_probs.gather(1, target_top_ids)
    ).clamp_min_(0.0)
    return target_draft_probs, target_top_ids, residual


def _compact_tp2_topk_topp_probability_prototype(
    sampled_logits: torch.Tensor,
    *,
    draft_token_ids: torch.Tensor,
    draft_probs: torch.Tensor,
    top_k: int,
    top_p: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single-GPU TP2 simulation for exact compact verifier top-k/top-p data."""
    if sampled_logits.shape[-1] % 2 != 0:
        raise ValueError("compact TP2 prototype requires even vocab size")
    shard_size = sampled_logits.shape[-1] // 2
    shard0 = sampled_logits[:, :shard_size].float()
    shard1 = sampled_logits[:, shard_size:].float()
    vals0, ids0 = torch.topk(shard0, k=top_k, dim=-1)
    vals1, ids1 = torch.topk(shard1, k=top_k, dim=-1)
    ids1 = ids1 + shard_size
    gathered_vals = torch.cat([vals0, vals1], dim=-1)
    gathered_ids = torch.cat([ids0, ids1], dim=-1)
    top_vals, top_pos = torch.topk(gathered_vals, k=top_k, dim=-1)
    top_ids = gathered_ids.gather(1, top_pos)

    top_probs = top_vals.softmax(dim=-1, dtype=torch.float32)
    remove_mask = top_probs.cumsum(dim=-1) > top_p
    remove_mask[:, 1:] = remove_mask[:, :-1].clone()
    remove_mask[:, 0] = False
    top_vals = top_vals.masked_fill(remove_mask, -float("inf"))
    kept_probs = top_vals.softmax(dim=-1, dtype=torch.float32)

    num_draft = draft_token_ids.shape[0]
    draft_ids_i64 = draft_token_ids.to(torch.int64).view(-1, 1)
    target_ids = top_ids[:num_draft]
    target_probs = kept_probs[:num_draft]
    target_matches = target_ids == draft_ids_i64
    target_draft_probs = torch.where(
        target_matches,
        target_probs,
        torch.zeros_like(target_probs),
    ).sum(dim=-1)
    residual = (
        target_probs - draft_probs.gather(1, target_ids)
    ).clamp_min_(0.0)
    return target_draft_probs, target_ids, residual


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--vocab-size", type=int, default=248320)
    parser.add_argument("--num-spec-tokens", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    num_tokens = args.num_spec_tokens
    cu_num_draft_tokens = torch.tensor([num_tokens], dtype=torch.int32, device=device)
    num_draft_tokens = [num_tokens]
    draft_token_ids = torch.randint(
        0,
        args.vocab_size,
        (num_tokens,),
        dtype=torch.int32,
        device=device,
    )
    bonus_token_ids = torch.randint(
        0,
        args.vocab_size,
        (1, 1),
        dtype=torch.int32,
        device=device,
    )
    target_logits_base = torch.randn(
        (num_tokens, args.vocab_size),
        dtype=torch.float16,
        device=device,
    )
    bonus_logits_base = torch.randn(
        (1, args.vocab_size),
        dtype=torch.float16,
        device=device,
    )
    sampled_logits_base = torch.cat([target_logits_base, bonus_logits_base], dim=0)
    target_prepare_base = torch.randn(
        (num_tokens + 1, args.vocab_size),
        dtype=torch.float16,
        device=device,
    )
    target_prepare_indices = torch.arange(
        num_tokens,
        dtype=torch.int64,
        device=device,
    )
    draft_logits = torch.randn(
        (num_tokens, args.vocab_size),
        dtype=torch.float16,
        device=device,
    )
    draft_top_k_tensor = torch.full(
        (num_tokens,),
        args.top_k,
        dtype=torch.int32,
        device=device,
    )
    sampled_top_k_tensor = torch.full(
        (num_tokens + 1,),
        args.top_k,
        dtype=torch.int32,
        device=device,
    )
    sampled_top_p_tensor = torch.full(
        (num_tokens + 1,),
        args.top_p,
        dtype=torch.float32,
        device=device,
    )
    draft_probs = draft_logits.float().softmax(dim=-1).contiguous()
    sampling_metadata = _make_sampling_metadata(device, args.top_k, args.top_p)
    forced_logprobs_metadata = replace(sampling_metadata, max_num_logprobs=-1)
    sampler = Sampler("raw_logprobs").to(device)
    rejection_sampler = RejectionSampler(sampler).to(device)
    spec_metadata = SpecDecodeMetadata(
        draft_token_ids=draft_token_ids,
        num_draft_tokens=num_draft_tokens,
        cu_num_draft_tokens=cu_num_draft_tokens,
        cu_num_sampled_tokens=torch.tensor(
            [num_tokens + 1], dtype=torch.int32, device=device
        ),
        target_logits_indices=torch.arange(
            num_tokens, dtype=torch.int32, device=device
        ),
        bonus_logits_indices=torch.tensor(
            [num_tokens], dtype=torch.int32, device=device
        ),
        logits_indices=torch.arange(
            num_tokens + 1, dtype=torch.int32, device=device
        ),
    )

    target_logits_constrained = apply_sampling_constraints(
        target_logits_base.float().clone(),
        cu_num_draft_tokens,
        sampling_metadata,
    )
    target_probs = target_logits_constrained.softmax(dim=-1, dtype=torch.float32)
    full_ref = _full_vocab_topk_topp_probability_reference(
        sampled_logits_base,
        draft_token_ids=draft_token_ids,
        draft_probs=draft_probs,
        top_k=args.top_k,
        top_k_tensor=sampled_top_k_tensor,
        top_p_tensor=sampled_top_p_tensor,
    )
    compact_ref = _compact_tp2_topk_topp_probability_prototype(
        sampled_logits_base,
        draft_token_ids=draft_token_ids,
        draft_probs=draft_probs,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    compact_probability_diff = {
        "draft_prob_max_abs": float((full_ref[0] - compact_ref[0]).abs().max().item()),
        "top_ids_equal": bool(torch.equal(full_ref[1], compact_ref[1])),
        "residual_max_abs": float((full_ref[2] - compact_ref[2]).abs().max().item()),
    }
    torch.cuda.synchronize()

    def _set_combined_bonus(enabled: bool) -> None:
        os.environ["VLLM_SM70_REJECTION_COMBINE_BONUS"] = "1" if enabled else "0"

    timings = {
        "rejection_sampler_forward_legacy_bonus": _time_cuda(
            lambda: (
                _set_combined_bonus(False),
                rejection_sampler(
                    spec_metadata,
                    draft_probs,
                    sampled_logits_base,
                    sampling_metadata,
                ),
            )[-1],
            args.warmup,
            args.iters,
        ),
        "rejection_sampler_forward_combined_bonus": _time_cuda(
            lambda: (
                _set_combined_bonus(True),
                rejection_sampler(
                    spec_metadata,
                    draft_probs,
                    sampled_logits_base,
                    sampling_metadata,
                ),
            )[-1],
            args.warmup,
            args.iters,
        ),
        "constraints_only": _time_cuda(
            lambda: apply_sampling_constraints(
                target_logits_base.float().clone(),
                cu_num_draft_tokens,
                sampling_metadata,
            ),
            args.warmup,
            args.iters,
        ),
        "bonus_sample_old_full_logits": _time_cuda(
            lambda: sampler(
                bonus_logits_base.clone(),
                forced_logprobs_metadata,
                predict_bonus_token=True,
                logprobs_mode_override="raw_logits",
            ),
            args.warmup,
            args.iters,
        ),
        "bonus_sample_no_logprobs": _time_cuda(
            lambda: sampler(
                bonus_logits_base.clone(),
                sampling_metadata,
                predict_bonus_token=True,
            ),
            args.warmup,
            args.iters,
        ),
        "sparse_bonus_topk_topp_prototype": _time_cuda(
            lambda: _sparse_topk_topp_sample_logits_prototype(
                bonus_logits_base,
                top_k=args.top_k,
                top_p=args.top_p,
            ),
            args.warmup,
            args.iters,
        ),
        "target_prepare_with_clone": _time_cuda(
            lambda: target_prepare_base[target_prepare_indices].to(
                torch.float32
            ).clone(),
            args.warmup,
            args.iters,
        ),
        "target_prepare_no_clone": _time_cuda(
            lambda: target_prepare_base[target_prepare_indices].to(torch.float32),
            args.warmup,
            args.iters,
        ),
        "dense_draft_topk_proposal": _time_cuda(
            lambda: _dense_draft_topk_proposal_prototype(
                draft_logits,
                top_k_tensor=draft_top_k_tensor,
            ),
            args.warmup,
            args.iters,
        ),
        "sparse_draft_topk_with_scatter": _time_cuda(
            lambda: _sparse_draft_topk_with_scatter_prototype(
                draft_logits,
                top_k=args.top_k,
            ),
            args.warmup,
            args.iters,
        ),
        "sparse_draft_topk_no_scatter": _time_cuda(
            lambda: _sparse_draft_topk_no_scatter_prototype(
                draft_logits,
                top_k=args.top_k,
            ),
            args.warmup,
            args.iters,
        ),
        "sparse_target_topk_topp_rejection_prototype": _time_cuda(
            lambda: _sparse_target_topk_topp_rejection_prototype(
                sampled_logits_base,
                draft_token_ids=draft_token_ids,
                draft_probs=draft_probs,
                top_k=args.top_k,
                top_p=args.top_p,
            ),
            args.warmup,
            args.iters,
        ),
        "full_vocab_topk_topp_probability_reference": _time_cuda(
            lambda: _full_vocab_topk_topp_probability_reference(
                sampled_logits_base,
                draft_token_ids=draft_token_ids,
                draft_probs=draft_probs,
                top_k=args.top_k,
                top_k_tensor=sampled_top_k_tensor,
                top_p_tensor=sampled_top_p_tensor,
            ),
            args.warmup,
            args.iters,
        ),
        "compact_tp2_topk_topp_probability_prototype": _time_cuda(
            lambda: _compact_tp2_topk_topp_probability_prototype(
                sampled_logits_base,
                draft_token_ids=draft_token_ids,
                draft_probs=draft_probs,
                top_k=args.top_k,
                top_p=args.top_p,
            ),
            args.warmup,
            args.iters,
        ),
        "dense_softmax_only": _time_cuda(
            lambda: target_logits_constrained.softmax(dim=-1, dtype=torch.float32),
            args.warmup,
            args.iters,
        ),
        "dense_recovered_only": _time_cuda(
            lambda: sample_recovered_tokens(
                num_tokens,
                num_draft_tokens,
                cu_num_draft_tokens,
                draft_token_ids,
                draft_probs,
                target_probs,
                sampling_metadata,
                device,
            ),
            args.warmup,
            args.iters,
        ),
        "sparse_recovered_prototype": _time_cuda(
            lambda: _sparse_recovered_prototype(
                draft_probs=draft_probs,
                target_probs=target_probs,
                top_k=args.top_k,
            ),
            args.warmup,
            args.iters,
        ),
        "full_current_rejection": _time_cuda(
            lambda: rejection_sample(
                draft_token_ids,
                num_draft_tokens,
                num_tokens,
                cu_num_draft_tokens,
                draft_probs,
                target_logits_constrained,
                bonus_token_ids,
                sampling_metadata,
            ),
            args.warmup,
            args.iters,
        ),
    }

    result = {
        "device": torch.cuda.get_device_name(device),
        "vocab_size": args.vocab_size,
        "num_spec_tokens": args.num_spec_tokens,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "warmup": args.warmup,
        "iters": args.iters,
        "compact_probability_diff": compact_probability_diff,
        "timings": timings,
    }

    text = json.dumps(result, indent=2, sort_keys=True)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
