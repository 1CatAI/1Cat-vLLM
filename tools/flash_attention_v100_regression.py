#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""FlashAttention V100 quality and speed regression harness.

Run from outside the repository root so imports resolve to the tested
installed environment:

  CUDA_VISIBLE_DEVICES=1 python /path/to/tools/flash_attention_v100_regression.py
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import statistics
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

import flash_attn_v100_cuda
from flash_attn_v100 import (
    flash_attn_decode_paged,
    flash_attn_func,
    flash_attn_prefill_paged,
)


@dataclass
class CaseResult:
    kind: str
    name: str
    shape: dict[str, Any]
    passed: bool
    max_abs: float | None = None
    mean_abs: float | None = None
    max_rel: float | None = None
    message: str | None = None


@dataclass
class SpeedResult:
    name: str
    shape: dict[str, Any]
    flash_ms_mean: float
    flash_ms_median: float
    sdpa_ms_mean: float
    sdpa_ms_median: float
    speedup_vs_sdpa_median: float
    passed_speed_guard: bool


def set_seed(seed: int = 1234) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compare(
    kind: str,
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    shape: dict[str, Any],
    atol: float = 2.5e-2,
    rtol: float = 2.5e-2,
) -> CaseResult:
    diff = (actual.float() - expected.float()).abs()
    denom = expected.float().abs().clamp_min(1e-6)
    rel = diff / denom
    return CaseResult(
        kind=kind,
        name=name,
        shape=shape,
        passed=bool(torch.allclose(actual, expected, atol=atol, rtol=rtol)),
        max_abs=float(diff.max().item()),
        mean_abs=float(diff.mean().item()),
        max_rel=float(rel.max().item()),
    )


def repeat_kv_for_gqa(k: torch.Tensor, q_heads: int) -> torch.Tensor:
    if k.shape[-2] == q_heads:
        return k
    repeat = q_heads // k.shape[-2]
    return torch.repeat_interleave(k, repeat, dim=-2)


def ref_dense_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    scale: float | None = None,
) -> torch.Tensor:
    hq = q.shape[2]
    k = repeat_kv_for_gqa(k, hq)
    v = repeat_kv_for_gqa(v, hq)
    return F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        is_causal=causal,
        scale=scale,
    ).transpose(1, 2).contiguous()


def rms_norm(x: torch.Tensor, weight: torch.Tensor,
             eps: float = 1e-6) -> torch.Tensor:
    y = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps)
    return (y * weight.float()).to(x.dtype)


def apply_rope(x: torch.Tensor, positions: torch.Tensor,
               base: float = 10000.0) -> torch.Tensor:
    """Apply standard RoPE to [B, S, H, D] or [B, H, D] tensors."""
    orig_shape = x.shape
    if x.dim() == 3:
        x = x.unsqueeze(1)
        pos_shape = (x.shape[0], 1, 1, -1)
    else:
        pos_shape = (1, x.shape[1], 1, -1)
    head_dim = x.shape[-1]
    half = head_dim // 2
    inv_freq = 1.0 / (
        base**(torch.arange(0, half, device=x.device, dtype=torch.float32) / half)
    )
    freqs = positions.to(torch.float32).unsqueeze(-1) * inv_freq
    cos = freqs.cos().reshape(pos_shape)
    sin = freqs.sin().reshape(pos_shape)
    x1 = x[..., :half].float()
    x2 = x[..., half:].float()
    out = torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos),
                    dim=-1).to(x.dtype)
    return out.reshape(orig_shape)


def scaled_weight(rows: int, cols: int, device: torch.device) -> torch.Tensor:
    return (torch.randn(rows, cols, device=device, dtype=torch.float16) *
            (cols**-0.5))


def linear_project(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return F.linear(hidden, weight)


def top1_agreement_metrics(
    actual_logits: torch.Tensor,
    expected_logits: torch.Tensor,
    diff: torch.Tensor,
) -> dict[str, float | int]:
    actual_top1 = actual_logits.argmax(dim=-1)
    expected_top1 = expected_logits.argmax(dim=-1)
    top1_match = actual_top1 == expected_top1
    top1_agree = float(top1_match.float().mean().item())

    expected_top2 = torch.topk(expected_logits.float(), k=2, dim=-1).values
    expected_margin = expected_top2[..., 0] - expected_top2[..., 1]
    max_token_diff = diff.reshape(*diff.shape[:-1], -1).max(dim=-1).values
    stable_mask = expected_margin > torch.clamp(
        max_token_diff * 4.0,
        min=1.0e-2,
    )
    if bool(stable_mask.any()):
        stable_agree = float(top1_match[stable_mask].float().mean().item())
    else:
        stable_agree = 1.0
    unstable_mismatches = int(((~stable_mask) & (~top1_match)).sum().item())
    stable_positions = int(stable_mask.sum().item())
    return {
        "top1_agreement": top1_agree,
        "stable_top1_agreement": stable_agree,
        "stable_top1_positions": stable_positions,
        "unstable_top1_mismatches": unstable_mismatches,
        "min_expected_top1_margin": float(expected_margin.min().item()),
    }


@torch.inference_mode()
def model_prefill_logits_case(
    name: str,
    bsz: int,
    seqlen: int,
    hidden_size: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    vocab_size: int,
    device: torch.device,
) -> CaseResult:
    hidden = torch.randn(bsz, seqlen, hidden_size, device=device,
                         dtype=torch.float16)
    norm_w = torch.randn(hidden_size, device=device, dtype=torch.float16) * 0.1 + 1
    q_w = scaled_weight(q_heads * head_dim, hidden_size, device)
    k_w = scaled_weight(kv_heads * head_dim, hidden_size, device)
    v_w = scaled_weight(kv_heads * head_dim, hidden_size, device)
    o_w = scaled_weight(hidden_size, q_heads * head_dim, device)
    lm_w = scaled_weight(vocab_size, hidden_size, device)

    hidden_n = rms_norm(hidden, norm_w)
    q = linear_project(hidden_n, q_w).view(bsz, seqlen, q_heads, head_dim)
    k = linear_project(hidden_n, k_w).view(bsz, seqlen, kv_heads, head_dim)
    v = linear_project(hidden_n, v_w).view(bsz, seqlen, kv_heads, head_dim)
    positions = torch.arange(seqlen, device=device)
    q = apply_rope(q, positions)
    k = apply_rope(k, positions)

    actual_attn = flash_attn_func(q, k, v, causal=True)
    expected_attn = ref_dense_attention(q, k, v, causal=True)
    actual_hidden = F.linear(actual_attn.reshape(bsz, seqlen, -1), o_w) + hidden
    expected_hidden = F.linear(expected_attn.reshape(bsz, seqlen, -1), o_w) + hidden
    actual_logits = F.linear(rms_norm(actual_hidden, norm_w), lm_w)
    expected_logits = F.linear(rms_norm(expected_hidden, norm_w), lm_w)
    torch.cuda.synchronize()

    diff = (actual_logits.float() - expected_logits.float()).abs()
    attn_diff = (actual_attn.float() - expected_attn.float()).abs()
    hidden_diff = (actual_hidden.float() - expected_hidden.float()).abs()
    top1_metrics = top1_agreement_metrics(actual_logits, expected_logits, diff)
    passed = (
        torch.allclose(actual_logits, expected_logits, atol=8e-2, rtol=5e-2)
        and top1_metrics["stable_top1_agreement"] >= 0.999
    )
    return CaseResult(
        kind="model_output",
        name=name,
        shape={
            "batch": bsz,
            "seq_len": seqlen,
            "hidden_size": hidden_size,
            "q_heads": q_heads,
            "kv_heads": kv_heads,
            "head_dim": head_dim,
            "vocab_size": vocab_size,
            "attn_max_abs": float(attn_diff.max().item()),
            "hidden_max_abs": float(hidden_diff.max().item()),
            **top1_metrics,
        },
        passed=bool(passed),
        max_abs=float(diff.max().item()),
        mean_abs=float(diff.mean().item()),
        max_rel=float((diff / expected_logits.float().abs().clamp_min(1e-6)).
                      max().item()),
    )


def fill_paged_cache_from_contiguous(
    k: torch.Tensor,
    v: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    bsz, seqlen, kv_heads, head_dim = k.shape
    blocks_per_seq = (seqlen + block_size - 1) // block_size
    block_table = torch.arange(bsz * blocks_per_seq, device=k.device,
                               dtype=torch.int32).view(bsz, blocks_per_seq)
    k_cache = torch.zeros(bsz * blocks_per_seq, block_size, kv_heads, head_dim,
                          device=k.device, dtype=k.dtype)
    v_cache = torch.zeros_like(k_cache)
    for batch_idx in range(bsz):
        for block_idx in range(blocks_per_seq):
            start = block_idx * block_size
            end = min(start + block_size, seqlen)
            physical = int(block_table[batch_idx, block_idx].item())
            k_cache[physical, :end - start].copy_(k[batch_idx, start:end])
            v_cache[physical, :end - start].copy_(v[batch_idx, start:end])
    seq_lens_t = torch.full((bsz,), seqlen, device=k.device, dtype=torch.int32)
    return k_cache, v_cache, block_table, seq_lens_t


def fp8_torch_dtype(kv_cache_dtype: str) -> torch.dtype:
    if kv_cache_dtype in ("fp8", "fp8_e4m3"):
        return torch.float8_e4m3fn
    if kv_cache_dtype == "fp8_e5m2":
        return torch.float8_e5m2
    raise ValueError(f"unsupported fp8 kv_cache_dtype: {kv_cache_dtype}")


def quantize_paged_cache(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    kv_cache_dtype: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
    if not kv_cache_dtype.startswith("fp8"):
        return k_cache, v_cache, k_cache, v_cache, 1.0, 1.0

    dtype = fp8_torch_dtype(kv_cache_dtype)
    fp8_max = torch.finfo(dtype).max
    k_scale = max(float(k_cache.abs().max().item()) / fp8_max, 1.0e-8)
    v_scale = max(float(v_cache.abs().max().item()) / fp8_max, 1.0e-8)
    k_q = (k_cache / k_scale).to(dtype).view(torch.uint8)
    v_q = (v_cache / v_scale).to(dtype).view(torch.uint8)
    k_deq = k_q.view(dtype).to(torch.float16) * k_scale
    v_deq = v_q.view(dtype).to(torch.float16) * v_scale
    return k_q, v_q, k_deq, v_deq, k_scale, v_scale


@torch.inference_mode()
def model_decode_logits_case(
    name: str,
    bsz: int,
    context_len: int,
    block_size: int,
    hidden_size: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    vocab_size: int,
    device: torch.device,
) -> CaseResult:
    context = torch.randn(bsz, context_len, hidden_size, device=device,
                          dtype=torch.float16)
    token = torch.randn(bsz, hidden_size, device=device, dtype=torch.float16)
    norm_w = torch.randn(hidden_size, device=device, dtype=torch.float16) * 0.1 + 1
    q_w = scaled_weight(q_heads * head_dim, hidden_size, device)
    k_w = scaled_weight(kv_heads * head_dim, hidden_size, device)
    v_w = scaled_weight(kv_heads * head_dim, hidden_size, device)
    o_w = scaled_weight(hidden_size, q_heads * head_dim, device)
    lm_w = scaled_weight(vocab_size, hidden_size, device)

    context_n = rms_norm(context, norm_w)
    token_n = rms_norm(token, norm_w)
    k = linear_project(context_n, k_w).view(bsz, context_len, kv_heads, head_dim)
    v = linear_project(context_n, v_w).view(bsz, context_len, kv_heads, head_dim)
    q = linear_project(token_n, q_w).view(bsz, q_heads, head_dim)
    positions = torch.arange(context_len, device=device)
    k = apply_rope(k, positions)
    q = apply_rope(q, torch.full((bsz,), context_len - 1, device=device))
    k_cache, v_cache, block_table, seq_lens_t = fill_paged_cache_from_contiguous(
        k, v, block_size)

    out = torch.empty_like(q)
    actual_attn = flash_attn_decode_paged(q, k_cache, v_cache, block_table,
                                          seq_lens_t, out=out)
    if actual_attn is None:
        actual_attn = out
    expected_attn = ref_decode_paged(q, k_cache, v_cache, block_table,
                                     [context_len] * bsz)
    actual_hidden = F.linear(actual_attn.reshape(bsz, -1), o_w) + token
    expected_hidden = F.linear(expected_attn.reshape(bsz, -1), o_w) + token
    actual_logits = F.linear(rms_norm(actual_hidden, norm_w), lm_w)
    expected_logits = F.linear(rms_norm(expected_hidden, norm_w), lm_w)
    torch.cuda.synchronize()

    diff = (actual_logits.float() - expected_logits.float()).abs()
    attn_diff = (actual_attn.float() - expected_attn.float()).abs()
    hidden_diff = (actual_hidden.float() - expected_hidden.float()).abs()
    top1_metrics = top1_agreement_metrics(actual_logits, expected_logits, diff)
    passed = (
        torch.allclose(actual_logits, expected_logits, atol=8e-2, rtol=5e-2)
        and top1_metrics["stable_top1_agreement"] >= 0.999
    )
    return CaseResult(
        kind="model_output",
        name=name,
        shape={
            "batch": bsz,
            "context_len": context_len,
            "block_size": block_size,
            "hidden_size": hidden_size,
            "q_heads": q_heads,
            "kv_heads": kv_heads,
            "head_dim": head_dim,
            "vocab_size": vocab_size,
            "attn_max_abs": float(attn_diff.max().item()),
            "hidden_max_abs": float(hidden_diff.max().item()),
            **top1_metrics,
        },
        passed=bool(passed),
        max_abs=float(diff.max().item()),
        mean_abs=float(diff.mean().item()),
        max_rel=float((diff / expected_logits.float().abs().clamp_min(1e-6)).
                      max().item()),
    )


@torch.inference_mode()
def dense_quality_case(
    name: str,
    bsz: int,
    q_len: int,
    k_len: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    causal: bool,
    device: torch.device,
) -> CaseResult:
    q = torch.randn(bsz, q_len, q_heads, head_dim, device=device,
                    dtype=torch.float16)
    k = torch.randn(bsz, k_len, kv_heads, head_dim, device=device,
                    dtype=torch.float16)
    v = torch.randn_like(k)
    actual = flash_attn_func(q, k, v, causal=causal)
    expected = ref_dense_attention(q, k, v, causal)
    torch.cuda.synchronize()
    return compare(
        "dense",
        name,
        actual,
        expected,
        {
            "batch": bsz,
            "q_len": q_len,
            "k_len": k_len,
            "q_heads": q_heads,
            "kv_heads": kv_heads,
            "head_dim": head_dim,
            "causal": causal,
        },
    )


def dense_backward_case(device: torch.device) -> CaseResult:
    set_seed(4321)
    bsz, seqlen, heads, head_dim = 1, 32, 4, 64
    q = torch.randn(bsz, seqlen, heads, head_dim, device=device,
                    dtype=torch.float16, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)

    out = flash_attn_func(q, k, v, causal=True)
    ref = ref_dense_attention(q_ref, k_ref, v_ref, causal=True)
    grad = torch.randn_like(out)
    out.backward(grad)
    ref.backward(grad)
    torch.cuda.synchronize()

    diffs = [
        (q.grad.float() - q_ref.grad.float()).abs(),
        (k.grad.float() - k_ref.grad.float()).abs(),
        (v.grad.float() - v_ref.grad.float()).abs(),
    ]
    max_abs = max(float(d.max().item()) for d in diffs)
    mean_abs = statistics.mean(float(d.mean().item()) for d in diffs)
    passed = all(
        torch.allclose(got, ref_grad, atol=3.5e-2, rtol=3.5e-2)
        for got, ref_grad in (
            (q.grad, q_ref.grad),
            (k.grad, k_ref.grad),
            (v.grad, v_ref.grad),
        )
    )
    return CaseResult(
        kind="backward",
        name="dense_backward_b1_s32_h4_hd64",
        shape={
            "batch": bsz,
            "seq_len": seqlen,
            "heads": heads,
            "head_dim": head_dim,
        },
        passed=bool(passed),
        max_abs=max_abs,
        mean_abs=mean_abs,
    )


def make_paged_cache(
    seq_lens: list[int],
    block_size: int,
    kv_heads: int,
    head_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    max_blocks = max((seq_len + block_size - 1) // block_size
                     for seq_len in seq_lens)
    bsz = len(seq_lens)
    block_table = torch.arange(bsz * max_blocks, device=device,
                               dtype=torch.int32).view(bsz, max_blocks)
    k_cache = torch.randn(bsz * max_blocks, block_size, kv_heads, head_dim,
                          device=device, dtype=torch.float16)
    v_cache = torch.randn_like(k_cache)
    seq_lens_t = torch.tensor(seq_lens, device=device, dtype=torch.int32)
    return k_cache, v_cache, block_table, seq_lens_t


def gather_kv(
    cache: torch.Tensor,
    block_table: torch.Tensor,
    batch_idx: int,
    seq_len: int,
) -> torch.Tensor:
    block_size = cache.shape[1]
    blocks = (seq_len + block_size - 1) // block_size
    return cache[block_table[batch_idx, :blocks].long()].reshape(
        -1, cache.shape[2], cache.shape[3])[:seq_len]


def ref_decode_paged(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: list[int],
) -> torch.Tensor:
    outs: list[torch.Tensor] = []
    scale = q.shape[-1]**-0.5
    for batch_idx, seq_len in enumerate(seq_lens):
        k = gather_kv(k_cache, block_table, batch_idx, seq_len)
        v = gather_kv(v_cache, block_table, batch_idx, seq_len)
        k = repeat_kv_for_gqa(k, q.shape[1])
        v = repeat_kv_for_gqa(v, q.shape[1])
        scores = torch.einsum("hd,khd->hk", q[batch_idx].float(), k.float())
        probs = torch.softmax(scores * scale, dim=-1).to(q.dtype)
        outs.append(torch.einsum("hk,khd->hd", probs, v))
    return torch.stack(outs, dim=0)


@torch.inference_mode()
def decode_quality_case(
    name: str,
    seq_lens: list[int],
    block_size: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    device: torch.device,
    kv_cache_dtype: str = "auto",
) -> CaseResult:
    k_cache, v_cache, block_table, seq_lens_t = make_paged_cache(
        seq_lens, block_size, kv_heads, head_dim, device)
    k_cache_in, v_cache_in, k_ref, v_ref, k_scale, v_scale = quantize_paged_cache(
        k_cache, v_cache, kv_cache_dtype)
    q = torch.randn(len(seq_lens), q_heads, head_dim, device=device,
                    dtype=torch.float16)
    out = torch.empty_like(q)
    actual = flash_attn_decode_paged(q, k_cache_in, v_cache_in, block_table,
                                     seq_lens_t, out=out,
                                     kv_cache_dtype=kv_cache_dtype,
                                     k_scale=k_scale, v_scale=v_scale)
    if actual is None:
        actual = out
    expected = ref_decode_paged(q, k_ref, v_ref, block_table, seq_lens)
    torch.cuda.synchronize()
    return compare(
        "decode_paged",
        name,
        actual,
        expected,
        {
            "batch": len(seq_lens),
            "seq_lens": seq_lens,
            "block_size": block_size,
            "q_heads": q_heads,
            "kv_heads": kv_heads,
            "head_dim": head_dim,
            "kv_cache_dtype": kv_cache_dtype,
            "k_scale": k_scale,
            "v_scale": v_scale,
        },
        atol=3.5e-2,
        rtol=3.5e-2,
    )


def ref_prefill_paged(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: list[int],
) -> torch.Tensor:
    outs: list[torch.Tensor] = []
    scale = q.shape[-1]**-0.5
    q_len = q.shape[1]
    for batch_idx, seq_len in enumerate(seq_lens):
        k = gather_kv(k_cache, block_table, batch_idx, seq_len)
        v = gather_kv(v_cache, block_table, batch_idx, seq_len)
        k = repeat_kv_for_gqa(k, q.shape[2])
        v = repeat_kv_for_gqa(v, q.shape[2])
        scores = torch.einsum("qhd,khd->hqk", q[batch_idx].float(), k.float())
        query_pos = torch.arange(seq_len - q_len, seq_len, device=q.device)
        key_pos = torch.arange(seq_len, device=q.device)
        mask = key_pos.unsqueeze(0) > query_pos.unsqueeze(1)
        scores = scores.masked_fill(mask.unsqueeze(0), float("-inf"))
        probs = torch.softmax(scores * scale, dim=-1).to(q.dtype)
        outs.append(torch.einsum("hqk,khd->qhd", probs, v))
    return torch.stack(outs, dim=0)


@torch.inference_mode()
def prefill_paged_quality_case(
    name: str,
    q_len: int,
    seq_lens: list[int],
    block_size: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    device: torch.device,
    kv_cache_dtype: str = "auto",
) -> CaseResult:
    k_cache, v_cache, block_table, seq_lens_t = make_paged_cache(
        seq_lens, block_size, kv_heads, head_dim, device)
    k_cache_in, v_cache_in, k_ref, v_ref, k_scale, v_scale = quantize_paged_cache(
        k_cache, v_cache, kv_cache_dtype)
    q = torch.randn(len(seq_lens), q_len, q_heads, head_dim, device=device,
                    dtype=torch.float16)
    actual = flash_attn_prefill_paged(q, k_cache_in, v_cache_in, block_table,
                                      seq_lens_t,
                                      kv_cache_dtype=kv_cache_dtype,
                                      k_scale=k_scale, v_scale=v_scale)
    expected = ref_prefill_paged(q, k_ref, v_ref, block_table, seq_lens)
    torch.cuda.synchronize()
    return compare(
        "prefill_paged",
        name,
        actual,
        expected,
        {
            "batch": len(seq_lens),
            "q_len": q_len,
            "seq_lens": seq_lens,
            "block_size": block_size,
            "q_heads": q_heads,
            "kv_heads": kv_heads,
            "head_dim": head_dim,
            "kv_cache_dtype": kv_cache_dtype,
            "k_scale": k_scale,
            "v_scale": v_scale,
        },
        atol=3.5e-2,
        rtol=3.5e-2,
    )


@torch.inference_mode()
def dense_short_tail_sweep_case(device: torch.device) -> CaseResult:
    set_seed(20260511)
    failures = 0
    total = 0
    worst_abs = 0.0
    worst_mean = 0.0
    worst_case = ""
    lengths = [1, 2, 3, 4, 5, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256]
    for head_dim in [16, 32, 64, 128, 256]:
        q_heads = 24 if head_dim == 256 else 8
        kv_heads = 4 if head_dim == 256 else 2
        for seqlen in lengths:
            for causal in [False, True]:
                q = torch.randn(1, seqlen, q_heads, head_dim, device=device,
                                dtype=torch.float16)
                k = torch.randn(1, seqlen, kv_heads, head_dim, device=device,
                                dtype=torch.float16)
                v = torch.randn_like(k)
                actual = flash_attn_func(q, k, v, causal=causal)
                expected = ref_dense_attention(q, k, v, causal)
                torch.cuda.synchronize()
                diff = (actual.float() - expected.float()).abs()
                max_abs = float(diff.max().item())
                mean_abs = float(diff.mean().item())
                total += 1
                if max_abs > worst_abs:
                    worst_abs = max_abs
                    worst_mean = mean_abs
                    worst_case = (
                        f"seqlen={seqlen} head_dim={head_dim} "
                        f"q_heads={q_heads} kv_heads={kv_heads} causal={causal}"
                    )
                if (
                    not torch.isfinite(actual).all()
                    or not torch.allclose(actual, expected, atol=3.5e-2, rtol=3.5e-2)
                ):
                    failures += 1
    return CaseResult(
        kind="dense",
        name="dense_short_tail_sweep_len1_256_hd16_256",
        shape={
            "total_subcases": total,
            "failures": failures,
            "lengths": lengths,
            "head_dims": [16, 32, 64, 128, 256],
            "worst_case": worst_case,
        },
        passed=failures == 0,
        max_abs=worst_abs,
        mean_abs=worst_mean,
    )


@torch.inference_mode()
def paged_prefill_tail_sweep_case(device: torch.device) -> CaseResult:
    set_seed(20260512)
    failures = 0
    total = 0
    worst_abs = 0.0
    worst_mean = 0.0
    worst_case = ""
    q_lens = [1, 2, 3, 4, 5, 7, 8, 15, 16, 31, 32]
    prefixes = [0, 1, 15, 16, 17, 63]
    for q_len in q_lens:
        for prefix in prefixes:
            seq_len = q_len + prefix
            result = prefill_paged_quality_case(
                name=f"prefill_tail_q{q_len}_prefix{prefix}_hd256",
                q_len=q_len,
                seq_lens=[seq_len],
                block_size=16,
                q_heads=24,
                kv_heads=4,
                head_dim=256,
                device=device,
            )
            total += 1
            max_abs = float(result.max_abs or 0.0)
            mean_abs = float(result.mean_abs or 0.0)
            if max_abs > worst_abs:
                worst_abs = max_abs
                worst_mean = mean_abs
                worst_case = f"q_len={q_len} prefix={prefix} seq_len={seq_len}"
            if not result.passed:
                failures += 1
    return CaseResult(
        kind="prefill_paged",
        name="prefill_paged_tail_sweep_qwen36_27b_hd256",
        shape={
            "total_subcases": total,
            "failures": failures,
            "q_lens": q_lens,
            "prefixes": prefixes,
            "worst_case": worst_case,
        },
        passed=failures == 0,
        max_abs=worst_abs,
        mean_abs=worst_mean,
    )


@torch.inference_mode()
def long_decode_qwen36_27b_case(device: torch.device) -> CaseResult:
    set_seed(20260513)
    failures = 0
    total = 0
    worst_abs = 0.0
    worst_mean = 0.0
    worst_case = ""
    seq_lens = [8192, 32768, 131072, 262144]
    for seq_len in seq_lens:
        result = decode_quality_case(
            name=f"decode_paged_qwen36_27b_ctx{seq_len}_hd256",
            seq_lens=[seq_len],
            block_size=16,
            q_heads=24,
            kv_heads=4,
            head_dim=256,
            device=device,
        )
        total += 1
        max_abs = float(result.max_abs or 0.0)
        mean_abs = float(result.mean_abs or 0.0)
        if max_abs > worst_abs:
            worst_abs = max_abs
            worst_mean = mean_abs
            worst_case = f"seq_len={seq_len}"
        if not result.passed:
            failures += 1
    return CaseResult(
        kind="decode_paged",
        name="decode_paged_qwen36_27b_long_context_sweep_hd256",
        shape={
            "total_subcases": total,
            "failures": failures,
            "seq_lens": seq_lens,
            "worst_case": worst_case,
        },
        passed=failures == 0,
        max_abs=worst_abs,
        mean_abs=worst_mean,
    )


def support_surface_case(device: torch.device) -> list[CaseResult]:
    try:
        from vllm.v1.attention.backends.flash_attn_v100 import (
            FlashAttnV100Backend,
        )

        advertised = FlashAttnV100Backend.get_supported_head_sizes()
    except Exception as exc:
        return [
            CaseResult(
                kind="support_surface",
                name="vllm_backend_head_sizes_import",
                shape={},
                passed=False,
                message=f"{type(exc).__name__}: {exc}",
            )
        ]

    results: list[CaseResult] = []
    for head_dim in advertised:
        q = torch.randn(1, 8, 2, head_dim, device=device, dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        captured = io.StringIO()
        try:
            with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(
                    captured):
                flash_attn_func(q, k, v, causal=True)
                torch.cuda.synchronize()
            passed = True
            msg = None
        except Exception as exc:
            passed = False
            msg = f"{type(exc).__name__}: {exc}"
        results.append(
            CaseResult(
                kind="support_surface",
                name=f"advertised_dense_head_dim_{head_dim}",
                shape={"head_dim": head_dim, "advertised_head_sizes": advertised},
                passed=passed,
                message=msg,
            ))
    return results


def event_benchmark(fn: Callable[[], torch.Tensor | None],
                    warmup: int,
                    iters: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(float(start.elapsed_time(end)))
    return times


@torch.inference_mode()
def dense_speed_case(
    name: str,
    bsz: int,
    seqlen: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    device: torch.device,
    warmup: int,
    iters: int,
) -> SpeedResult:
    q = torch.randn(bsz, seqlen, q_heads, head_dim, device=device,
                    dtype=torch.float16)
    k = torch.randn(bsz, seqlen, kv_heads, head_dim, device=device,
                    dtype=torch.float16)
    v = torch.randn_like(k)
    k_ref = repeat_kv_for_gqa(k, q_heads)
    v_ref = repeat_kv_for_gqa(v, q_heads)

    def flash_fn() -> torch.Tensor:
        return flash_attn_func(q, k, v, causal=True)

    def sdpa_fn() -> torch.Tensor:
        return F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k_ref.transpose(1, 2),
            v_ref.transpose(1, 2),
            is_causal=True,
        )

    actual = flash_fn()
    expected = sdpa_fn().transpose(1, 2)
    torch.testing.assert_close(actual, expected, atol=3.5e-2, rtol=3.5e-2)
    flash_ms = event_benchmark(flash_fn, warmup, iters)
    sdpa_ms = event_benchmark(sdpa_fn, warmup, iters)
    flash_median = statistics.median(flash_ms)
    sdpa_median = statistics.median(sdpa_ms)
    return SpeedResult(
        name=name,
        shape={
            "batch": bsz,
            "seq_len": seqlen,
            "q_heads": q_heads,
            "kv_heads": kv_heads,
            "head_dim": head_dim,
        },
        flash_ms_mean=statistics.mean(flash_ms),
        flash_ms_median=flash_median,
        sdpa_ms_mean=statistics.mean(sdpa_ms),
        sdpa_ms_median=sdpa_median,
        speedup_vs_sdpa_median=sdpa_median / flash_median,
        passed_speed_guard=flash_median <= sdpa_median * 1.10,
    )


@torch.inference_mode()
def direct_dense_speed_case(
    name: str,
    bsz: int,
    seqlen: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    device: torch.device,
    warmup: int,
    iters: int,
) -> SpeedResult:
    q = torch.randn(bsz, q_heads, seqlen, head_dim, device=device,
                    dtype=torch.float16)
    k = torch.randn(bsz, kv_heads, seqlen, head_dim, device=device,
                    dtype=torch.float16)
    v = torch.randn_like(k)
    out = torch.empty_like(q)
    scale = head_dim**-0.5
    k_ref = repeat_kv_for_gqa(k.transpose(1, 2), q_heads).transpose(1, 2)
    v_ref = repeat_kv_for_gqa(v.transpose(1, 2), q_heads).transpose(1, 2)

    def flash_fn() -> torch.Tensor:
        result = flash_attn_v100_cuda.fwd(q, k, v, out, None, 0.0, scale, True,
                                          -1, -1, 0.0, False, None)
        return result[0]

    def sdpa_fn() -> torch.Tensor:
        return F.scaled_dot_product_attention(q, k_ref, v_ref, is_causal=True)

    actual = flash_fn()
    expected = sdpa_fn()
    torch.testing.assert_close(actual, expected, atol=3.5e-2, rtol=3.5e-2)
    flash_ms = event_benchmark(flash_fn, warmup, iters)
    sdpa_ms = event_benchmark(sdpa_fn, warmup, iters)
    flash_median = statistics.median(flash_ms)
    sdpa_median = statistics.median(sdpa_ms)
    return SpeedResult(
        name=name,
        shape={
            "layout": "direct_bhsd_no_wrapper_transpose",
            "batch": bsz,
            "seq_len": seqlen,
            "q_heads": q_heads,
            "kv_heads": kv_heads,
            "head_dim": head_dim,
        },
        flash_ms_mean=statistics.mean(flash_ms),
        flash_ms_median=flash_median,
        sdpa_ms_mean=statistics.mean(sdpa_ms),
        sdpa_ms_median=sdpa_median,
        speedup_vs_sdpa_median=sdpa_median / flash_median,
        passed_speed_guard=flash_median <= sdpa_median * 1.10,
    )


@torch.inference_mode()
def decode_speed_case(
    name: str,
    bsz: int,
    seq_len: int,
    block_size: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    device: torch.device,
    warmup: int,
    iters: int,
    kv_cache_dtype: str = "auto",
) -> SpeedResult:
    seq_lens = [seq_len] * bsz
    k_cache, v_cache, block_table, seq_lens_t = make_paged_cache(
        seq_lens, block_size, kv_heads, head_dim, device)
    k_cache_in, v_cache_in, k_ref_cache, v_ref_cache, k_scale, v_scale = (
        quantize_paged_cache(k_cache, v_cache, kv_cache_dtype)
    )
    q = torch.randn(bsz, q_heads, head_dim, device=device, dtype=torch.float16)
    out = torch.empty_like(q)
    k_ref = torch.stack(
        [gather_kv(k_ref_cache, block_table, i, seq_len) for i in range(bsz)],
        dim=0,
    )
    v_ref = torch.stack(
        [gather_kv(v_ref_cache, block_table, i, seq_len) for i in range(bsz)],
        dim=0,
    )
    k_ref = repeat_kv_for_gqa(k_ref, q_heads)
    v_ref = repeat_kv_for_gqa(v_ref, q_heads)

    def flash_fn() -> torch.Tensor:
        result = flash_attn_decode_paged(q, k_cache_in, v_cache_in,
                                         block_table, seq_lens_t, out=out,
                                         kv_cache_dtype=kv_cache_dtype,
                                         k_scale=k_scale, v_scale=v_scale)
        return out if result is None else result

    def sdpa_fn() -> torch.Tensor:
        return F.scaled_dot_product_attention(
            q.unsqueeze(2),
            k_ref.transpose(1, 2),
            v_ref.transpose(1, 2),
        ).squeeze(2)

    actual = flash_fn()
    expected = sdpa_fn()
    torch.testing.assert_close(actual, expected, atol=3.5e-2, rtol=3.5e-2)
    flash_ms = event_benchmark(flash_fn, warmup, iters)
    sdpa_ms = event_benchmark(sdpa_fn, warmup, iters)
    flash_median = statistics.median(flash_ms)
    sdpa_median = statistics.median(sdpa_ms)
    return SpeedResult(
        name=name,
        shape={
            "batch": bsz,
            "seq_len": seq_len,
            "block_size": block_size,
            "q_heads": q_heads,
            "kv_heads": kv_heads,
            "head_dim": head_dim,
            "kv_cache_dtype": kv_cache_dtype,
            "k_scale": k_scale,
            "v_scale": v_scale,
        },
        flash_ms_mean=statistics.mean(flash_ms),
        flash_ms_median=flash_median,
        sdpa_ms_mean=statistics.mean(sdpa_ms),
        sdpa_ms_median=sdpa_median,
        speedup_vs_sdpa_median=sdpa_median / flash_median,
        passed_speed_guard=flash_median <= sdpa_median * 1.10,
    )


def run_quality(device: torch.device,
                include_long_context: bool = False) -> list[CaseResult]:
    cases: list[CaseResult] = []
    set_seed()
    runners = [
        lambda: dense_short_tail_sweep_case(device),
        lambda: paged_prefill_tail_sweep_case(device),
        lambda: dense_quality_case("dense_causal_b1_s64_h4_hd64", 1, 64, 64,
                                   4, 4, 64, True, device),
        lambda: dense_quality_case("dense_causal_b2_s128_h8_hd64", 2, 128, 128,
                                   8, 8, 64, True, device),
        lambda: dense_quality_case("dense_gqa_b1_s128_hq16_hkv4_hd64", 1, 128,
                                   128, 16, 4, 64, True, device),
        lambda: dense_quality_case("dense_noncausal_b2_q96_k128_h8_hd128", 2,
                                   96, 128, 8, 8, 128, False, device),
        lambda: dense_quality_case("dense_hd256_b1_s32_h4", 1, 32, 32, 4, 4,
                                   256, True, device),
        lambda: dense_quality_case(
            "dense_qwen35_moe_full_attn_b1_s128_hq16_hkv2_hd256", 1, 128,
            128, 16, 2, 256, True, device),
        lambda: dense_quality_case(
            "dense_qwen35_27b_full_attn_b1_s128_hq24_hkv4_hd256", 1, 128,
            128, 24, 4, 256, True, device),
        lambda: dense_backward_case(device),
        lambda: decode_quality_case("decode_paged_b4_hd64", [1, 17, 64, 129],
                                    16, 8, 8, 64, device),
        lambda: decode_quality_case("decode_paged_gqa_b3_hd128", [33, 79, 131],
                                    16, 16, 4, 128, device),
        lambda: decode_quality_case(
            "decode_paged_qwen35_moe_hd256", [33, 79, 131], 16, 16, 2,
            256, device),
        lambda: decode_quality_case(
            "decode_paged_qwen35_27b_hd256", [33, 79, 131], 16, 24, 4,
            256, device),
        lambda: decode_quality_case(
            "decode_paged_fp8_e4m3_qwen35_hd256", [33, 79, 131], 16, 16,
            2, 256, device, "fp8_e4m3"),
        lambda: decode_quality_case(
            "decode_paged_fp8_e5m2_qwen35_hd256", [33, 79, 131], 16, 16,
            2, 256, device, "fp8_e5m2"),
        lambda: decode_quality_case("decode_paged_hd80", [7, 31], 16, 4, 4, 80,
                                    device),
        lambda: prefill_paged_quality_case("prefill_paged_no_prefix_hd64", 16,
                                           [16, 16], 16, 8, 8, 64, device),
        lambda: prefill_paged_quality_case("prefill_paged_prefix_hd64", 16,
                                           [64, 47], 16, 8, 8, 64, device),
        lambda: prefill_paged_quality_case("prefill_paged_gqa_hd128", 8,
                                           [40, 33], 16, 16, 4, 128, device),
        lambda: prefill_paged_quality_case(
            "prefill_paged_fp8_e4m3_qwen35_hd256", 8, [40, 33], 16, 16,
            2, 256, device, "fp8_e4m3"),
        lambda: prefill_paged_quality_case(
            "prefill_paged_fp8_e5m2_qwen35_hd256", 8, [40, 33], 16, 16,
            2, 256, device, "fp8_e5m2"),
        lambda: model_prefill_logits_case(
            "model_prefill_logits_qwen7b_like_b1_s128_h3584_hq28_hkv4_hd128",
            1, 128, 3584, 28, 4, 128, 4096, device),
        lambda: model_prefill_logits_case(
            "model_prefill_logits_qwen35_moe_like_b1_s128_h2048_hq16_hkv2_hd256",
            1, 128, 2048, 16, 2, 256, 4096, device),
        lambda: model_prefill_logits_case(
            "model_prefill_logits_qwen35_27b_like_b1_s128_h5120_hq24_hkv4_hd256",
            1, 128, 5120, 24, 4, 256, 4096, device),
        lambda: model_decode_logits_case(
            "model_decode_logits_qwen7b_like_b4_ctx1024_h3584_hq28_hkv4_hd128",
            4, 1024, 16, 3584, 28, 4, 128, 4096, device),
        lambda: model_decode_logits_case(
            "model_decode_logits_qwen35_moe_like_b4_ctx1024_h2048_hq16_hkv2_hd256",
            4, 1024, 16, 2048, 16, 2, 256, 4096, device),
        lambda: model_decode_logits_case(
            "model_decode_logits_qwen35_27b_like_b4_ctx1024_h5120_hq24_hkv4_hd256",
            4, 1024, 16, 5120, 24, 4, 256, 4096, device),
    ]
    if include_long_context:
        runners.append(lambda: long_decode_qwen36_27b_case(device))
    for runner in runners:
        try:
            cases.append(runner())
        except Exception as exc:
            cases.append(
                CaseResult(
                    kind="quality",
                    name=getattr(runner, "__name__", "unknown"),
                    shape={},
                    passed=False,
                    message=f"{type(exc).__name__}: {exc}\n"
                    f"{traceback.format_exc(limit=4)}",
                ))
    cases.extend(support_surface_case(device))
    return cases


def run_speed(device: torch.device, warmup: int,
              iters: int) -> list[SpeedResult]:
    set_seed(5678)
    return [
        dense_speed_case("speed_dense_b1_s512_h8_hd64", 1, 512, 8, 8, 64,
                         device, warmup, iters),
        dense_speed_case("speed_dense_b2_s512_h8_hd64", 2, 512, 8, 8, 64,
                         device, warmup, iters),
        dense_speed_case("speed_dense_gqa_b1_s1024_hq16_hkv4_hd64", 1, 1024,
                         16, 4, 64, device, warmup, iters),
        direct_dense_speed_case("speed_direct_dense_b1_s512_h8_hd64", 1, 512,
                                8, 8, 64, device, warmup, iters),
        direct_dense_speed_case(
            "speed_direct_dense_gqa_b1_s1024_hq16_hkv4_hd64", 1, 1024, 16, 4,
            64, device, warmup, iters),
        dense_speed_case(
            "speed_dense_qwen35_moe_b1_s512_hq16_hkv2_hd256", 1, 512, 16,
            2, 256, device, warmup, iters),
        direct_dense_speed_case(
            "speed_direct_dense_qwen35_moe_b1_s512_hq16_hkv2_hd256", 1,
            512, 16, 2, 256, device, warmup, iters),
        direct_dense_speed_case(
            "speed_direct_dense_qwen35_27b_b1_s512_hq24_hkv4_hd256", 1,
            512, 24, 4, 256, device, warmup, iters),
        decode_speed_case("speed_decode_paged_b16_s1024_h8_hd64", 16, 1024,
                          16, 8, 8, 64, device, warmup, iters),
        decode_speed_case("speed_decode_paged_gqa_b16_s1024_hq16_hkv4_hd64",
                          16, 1024, 16, 16, 4, 64, device, warmup, iters),
        decode_speed_case(
            "speed_decode_paged_qwen35_moe_b16_s1024_hq16_hkv2_hd256", 16,
            1024, 16, 16, 2, 256, device, warmup, iters),
        decode_speed_case(
            "speed_decode_paged_fp8_e4m3_qwen35_moe_b16_s1024_hq16_hkv2_hd256",
            16, 1024, 16, 16, 2, 256, device, warmup, iters, "fp8_e4m3"),
        decode_speed_case(
            "speed_decode_paged_fp8_e5m2_qwen35_moe_b16_s1024_hq16_hkv2_hd256",
            16, 1024, 16, 16, 2, 256, device, warmup, iters, "fp8_e5m2"),
        decode_speed_case(
            "speed_decode_paged_qwen35_27b_b16_s1024_hq24_hkv4_hd256", 16,
            1024, 16, 24, 4, 256, device, warmup, iters),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-quality", action="store_true")
    parser.add_argument("--skip-speed", action="store_true")
    parser.add_argument("--include-long-quality", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    props = torch.cuda.get_device_properties(device)
    if props.major != 7 or props.minor != 0:
        raise RuntimeError(
            f"FlashAttention V100 regression must run on sm70, got "
            f"{props.name} sm{props.major}{props.minor}")

    quality: list[CaseResult] = []
    if not args.skip_quality:
        quality = run_quality(device, args.include_long_quality)
    speed: list[SpeedResult] = []
    if not args.skip_speed:
        speed = run_speed(device, args.warmup, args.iters)

    payload = {
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": {
            "name": props.name,
            "capability": [props.major, props.minor],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "quality": [asdict(result) for result in quality],
        "speed": [asdict(result) for result in speed],
    }
    output = args.output_json
    if output is None:
        output = Path(
            f"/tmp/flash_v100_regression_{time.strftime('%Y%m%d_%H%M%S')}.json"
        )
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    quality_failures = [case for case in quality if not case.passed]
    speed_failures = [case for case in speed if not case.passed_speed_guard]
    for case in quality:
        status = "PASS" if case.passed else "FAIL"
        metric = ""
        if case.max_abs is not None:
            metric = f" max_abs={case.max_abs:.6g} mean_abs={case.mean_abs:.6g}"
        msg = f" message={case.message}" if case.message else ""
        print(f"{status} {case.kind} {case.name}{metric}{msg}")
    for case in speed:
        status = "PASS" if case.passed_speed_guard else "FAIL"
        print(
            f"{status} speed {case.name} flash_median="
            f"{case.flash_ms_median:.4f}ms sdpa_median="
            f"{case.sdpa_ms_median:.4f}ms speedup="
            f"{case.speedup_vs_sdpa_median:.3f}x")
    print(f"wrote {output}")
    if quality_failures or speed_failures:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
