#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Long-context FlashAttention V100 vs Triton operator regression.

Run from outside the source checkout so imports resolve to the installed
0.0.3 environment:

  CUDA_VISIBLE_DEVICES=1 python /path/to/tools/fa2_triton_long_context_regression.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

from flash_attn_v100 import (
    flash_attn_decode_paged,
    flash_attn_func,
    flash_attn_prefill_paged,
)
from vllm.v1.attention.ops.triton_unified_attention import unified_attention


@dataclass
class QualityResult:
    name: str
    kind: str
    shape: dict[str, Any]
    passed: bool
    max_abs: float | None = None
    mean_abs: float | None = None
    max_rel: float | None = None
    message: str | None = None


@dataclass
class SpeedResult:
    name: str
    kind: str
    shape: dict[str, Any]
    fa2_ms_median: float
    triton_ms_median: float
    fa2_ms_mean: float
    triton_ms_mean: float
    ratio_fa2_over_triton: float
    speedup_vs_triton: float
    passed_quality: bool
    max_abs: float
    mean_abs: float


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def repeat_kv_for_gqa(x: torch.Tensor, q_heads: int) -> torch.Tensor:
    if x.shape[-2] == q_heads:
        return x
    return torch.repeat_interleave(x, q_heads // x.shape[-2], dim=-2)


def fill_paged_cache(
    k: torch.Tensor,
    v: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    bsz, seq_len, kv_heads, head_dim = k.shape
    blocks_per_seq = (seq_len + block_size - 1) // block_size
    block_table = torch.arange(
        bsz * blocks_per_seq,
        device=k.device,
        dtype=torch.int32,
    ).view(bsz, blocks_per_seq)
    k_cache = torch.zeros(
        bsz * blocks_per_seq,
        block_size,
        kv_heads,
        head_dim,
        device=k.device,
        dtype=k.dtype,
    )
    v_cache = torch.zeros_like(k_cache)
    for batch_idx in range(bsz):
        for block_idx in range(blocks_per_seq):
            start = block_idx * block_size
            end = min(start + block_size, seq_len)
            physical = int(block_table[batch_idx, block_idx].item())
            k_cache[physical, : end - start].copy_(k[batch_idx, start:end])
            v_cache[physical, : end - start].copy_(v[batch_idx, start:end])
    seq_lens = torch.full((bsz,), seq_len, device=k.device, dtype=torch.int32)
    return k_cache, v_cache, block_table, seq_lens


def gather_seq(
    cache: torch.Tensor,
    block_table: torch.Tensor,
    batch_idx: int,
    seq_len: int,
) -> torch.Tensor:
    block_size = cache.shape[1]
    blocks = (seq_len + block_size - 1) // block_size
    return cache[block_table[batch_idx, :blocks].long()].reshape(
        -1, cache.shape[2], cache.shape[3]
    )[:seq_len]


def compare_tensors(
    name: str,
    kind: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    shape: dict[str, Any],
    atol: float = 5.0e-2,
    rtol: float = 5.0e-2,
) -> QualityResult:
    diff = (actual.float() - expected.float()).abs()
    rel = diff / expected.float().abs().clamp_min(1e-6)
    return QualityResult(
        name=name,
        kind=kind,
        shape=shape,
        passed=bool(torch.allclose(actual, expected, atol=atol, rtol=rtol)),
        max_abs=float(diff.max().item()),
        mean_abs=float(diff.mean().item()),
        max_rel=float(rel.max().item()),
    )


def event_benchmark(fn: Callable[[], torch.Tensor | None], warmup: int,
                    iters: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(float(start.elapsed_time(end)))
    return times


def make_cu_seqlens(lengths: list[int], device: torch.device) -> torch.Tensor:
    values = [0]
    for length in lengths:
        values.append(values[-1] + length)
    return torch.tensor(values, dtype=torch.int32, device=device)


def triton_unified(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    query_lens: list[int],
    out: torch.Tensor,
) -> torch.Tensor:
    bsz = len(query_lens)
    max_query_len = max(query_lens)
    max_seq_len = int(seq_lens.max().item())
    kv_heads = k_cache.shape[2]
    q_heads = q.shape[1]
    head_dim = q.shape[2]
    descale = torch.ones((bsz, kv_heads), dtype=torch.float32, device=q.device)
    seq_threshold_3d = max(1, 128 // kv_heads)
    num_segments = 16
    head_dim_padded = 1 << (head_dim - 1).bit_length()
    softmax_segm_output = torch.empty(
        (seq_threshold_3d, q_heads, num_segments, head_dim_padded),
        dtype=torch.float32,
        device=q.device,
    )
    softmax_segm_max = torch.empty(
        (seq_threshold_3d, q_heads, num_segments),
        dtype=torch.float32,
        device=q.device,
    )
    softmax_segm_expsum = torch.empty_like(softmax_segm_max)
    unified_attention(
        q=q,
        k=k_cache,
        v=v_cache,
        out=out,
        cu_seqlens_q=make_cu_seqlens(query_lens, q.device),
        max_seqlen_q=max_query_len,
        seqused_k=seq_lens,
        max_seqlen_k=max_seq_len,
        softmax_scale=head_dim**-0.5,
        causal=True,
        alibi_slopes=None,
        use_alibi_sqrt=False,
        window_size=(-1, -1),
        block_table=block_table,
        softcap=0.0,
        q_descale=None,
        k_descale=descale,
        v_descale=descale,
        seq_threshold_3D=seq_threshold_3d,
        num_par_softmax_segments=num_segments,
        softmax_segm_output=softmax_segm_output,
        softmax_segm_max=softmax_segm_max,
        softmax_segm_expsum=softmax_segm_expsum,
        sinks=None,
        output_scale=None,
        mm_prefix_range=None,
    )
    return out


def fa2_gather_dense_prefill(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_len: int,
) -> torch.Tensor:
    bsz = q.shape[0]
    out = torch.empty_like(q)
    for batch_idx in range(bsz):
        k_seq = gather_seq(k_cache, block_table, batch_idx, seq_len).unsqueeze(0)
        v_seq = gather_seq(v_cache, block_table, batch_idx, seq_len).unsqueeze(0)
        out_seq = flash_attn_func(
            q[batch_idx:batch_idx + 1],
            k_seq,
            v_seq,
            causal=True,
        )
        out[batch_idx:batch_idx + 1].copy_(out_seq)
    return out


@torch.inference_mode()
def dense_prefill_case(
    name: str,
    bsz: int,
    seq_len: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    block_size: int,
    device: torch.device,
) -> QualityResult:
    q = torch.randn(bsz, seq_len, q_heads, head_dim, device=device,
                    dtype=torch.float16)
    k = torch.randn(bsz, seq_len, kv_heads, head_dim, device=device,
                    dtype=torch.float16)
    v = torch.randn_like(k)
    k_cache, v_cache, block_table, seq_lens = fill_paged_cache(k, v, block_size)
    fa2 = flash_attn_func(q, k, v, causal=True).reshape(-1, q_heads, head_dim)
    triton_out = torch.empty_like(fa2)
    triton = triton_unified(
        q.reshape(-1, q_heads, head_dim),
        k_cache,
        v_cache,
        block_table,
        seq_lens,
        [seq_len] * bsz,
        triton_out,
    )
    torch.cuda.synchronize()
    return compare_tensors(
        name=name,
        kind="dense_prefill",
        actual=fa2,
        expected=triton,
        shape={
            "batch": bsz,
            "seq_len": seq_len,
            "q_heads": q_heads,
            "kv_heads": kv_heads,
            "head_dim": head_dim,
            "block_size": block_size,
        },
    )


@torch.inference_mode()
def prefix_prefill_case(
    name: str,
    bsz: int,
    query_len: int,
    seq_len: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    block_size: int,
    device: torch.device,
    mode: str = "paged",
) -> QualityResult:
    q = torch.randn(bsz, query_len, q_heads, head_dim, device=device,
                    dtype=torch.float16)
    k = torch.randn(bsz, seq_len, kv_heads, head_dim, device=device,
                    dtype=torch.float16)
    v = torch.randn_like(k)
    k_cache, v_cache, block_table, seq_lens = fill_paged_cache(k, v, block_size)
    if mode == "paged":
        fa2 = flash_attn_prefill_paged(q, k_cache, v_cache, block_table, seq_lens)
    elif mode == "gather_dense":
        fa2 = fa2_gather_dense_prefill(q, k_cache, v_cache, block_table, seq_len)
    else:
        raise ValueError(f"unknown prefix prefill mode: {mode}")
    triton_out = torch.empty_like(fa2.reshape(-1, q_heads, head_dim))
    triton = triton_unified(
        q.reshape(-1, q_heads, head_dim),
        k_cache,
        v_cache,
        block_table,
        seq_lens,
        [query_len] * bsz,
        triton_out,
    ).view(bsz, query_len, q_heads, head_dim)
    torch.cuda.synchronize()
    return compare_tensors(
        name=name,
        kind=f"prefix_prefill_{mode}",
        actual=fa2,
        expected=triton,
        shape={
            "batch": bsz,
            "query_len": query_len,
            "seq_len": seq_len,
            "q_heads": q_heads,
            "kv_heads": kv_heads,
            "head_dim": head_dim,
            "block_size": block_size,
            "mode": mode,
        },
    )


@torch.inference_mode()
def decode_case(
    name: str,
    bsz: int,
    seq_len: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    block_size: int,
    device: torch.device,
) -> QualityResult:
    k = torch.randn(bsz, seq_len, kv_heads, head_dim, device=device,
                    dtype=torch.float16)
    v = torch.randn_like(k)
    k_cache, v_cache, block_table, seq_lens = fill_paged_cache(k, v, block_size)
    q = torch.randn(bsz, q_heads, head_dim, device=device, dtype=torch.float16)
    fa2_out = torch.empty_like(q)
    fa2 = flash_attn_decode_paged(q, k_cache, v_cache, block_table, seq_lens,
                                  out=fa2_out)
    if fa2 is None:
        fa2 = fa2_out
    triton_out = torch.empty_like(q)
    triton = triton_unified(
        q,
        k_cache,
        v_cache,
        block_table,
        seq_lens,
        [1] * bsz,
        triton_out,
    )
    torch.cuda.synchronize()
    return compare_tensors(
        name=name,
        kind="decode",
        actual=fa2,
        expected=triton,
        shape={
            "batch": bsz,
            "seq_len": seq_len,
            "q_heads": q_heads,
            "kv_heads": kv_heads,
            "head_dim": head_dim,
            "block_size": block_size,
        },
    )


@torch.inference_mode()
def speed_dense_prefill(
    name: str,
    bsz: int,
    seq_len: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    block_size: int,
    device: torch.device,
    warmup: int,
    iters: int,
) -> SpeedResult:
    q = torch.randn(bsz, seq_len, q_heads, head_dim, device=device,
                    dtype=torch.float16)
    k = torch.randn(bsz, seq_len, kv_heads, head_dim, device=device,
                    dtype=torch.float16)
    v = torch.randn_like(k)
    k_cache, v_cache, block_table, seq_lens = fill_paged_cache(k, v, block_size)
    fa2_out = torch.empty(bsz, seq_len, q_heads, head_dim, device=device,
                          dtype=torch.float16)
    triton_out = torch.empty(bsz * seq_len, q_heads, head_dim, device=device,
                             dtype=torch.float16)

    def fa2_fn() -> torch.Tensor:
        result = flash_attn_func(q, k, v, causal=True)
        fa2_out.copy_(result)
        return fa2_out

    def triton_fn() -> torch.Tensor:
        return triton_unified(
            q.reshape(-1, q_heads, head_dim),
            k_cache,
            v_cache,
            block_table,
            seq_lens,
            [seq_len] * bsz,
            triton_out,
        )

    actual = fa2_fn().reshape(-1, q_heads, head_dim)
    expected = triton_fn()
    check = compare_tensors(name, "dense_prefill", actual, expected, {})
    fa2_ms = event_benchmark(fa2_fn, warmup, iters)
    triton_ms = event_benchmark(triton_fn, warmup, iters)
    return make_speed_result(name, "dense_prefill", check, fa2_ms, triton_ms, {
        "batch": bsz,
        "seq_len": seq_len,
        "q_heads": q_heads,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "block_size": block_size,
    })


@torch.inference_mode()
def speed_prefix_prefill(
    name: str,
    bsz: int,
    query_len: int,
    seq_len: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    block_size: int,
    device: torch.device,
    warmup: int,
    iters: int,
    include_paged: bool = True,
) -> list[SpeedResult]:
    q = torch.randn(bsz, query_len, q_heads, head_dim, device=device,
                    dtype=torch.float16)
    k = torch.randn(bsz, seq_len, kv_heads, head_dim, device=device,
                    dtype=torch.float16)
    v = torch.randn_like(k)
    k_cache, v_cache, block_table, seq_lens = fill_paged_cache(k, v, block_size)
    triton_out = torch.empty(bsz * query_len, q_heads, head_dim, device=device,
                             dtype=torch.float16)
    gather_out = torch.empty_like(q)

    def fa2_paged_fn() -> torch.Tensor:
        return flash_attn_prefill_paged(q, k_cache, v_cache, block_table, seq_lens)

    def fa2_gather_dense_fn() -> torch.Tensor:
        gather_out.copy_(
            fa2_gather_dense_prefill(q, k_cache, v_cache, block_table, seq_len)
        )
        return gather_out

    def triton_fn() -> torch.Tensor:
        return triton_unified(
            q.reshape(-1, q_heads, head_dim),
            k_cache,
            v_cache,
            block_table,
            seq_lens,
            [query_len] * bsz,
            triton_out,
        )

    expected = triton_fn().view_as(q)
    gather = fa2_gather_dense_fn()
    gather_check = compare_tensors(
        name + "_gather_dense",
        "prefix_prefill",
        gather,
        expected,
        {},
    )
    triton_ms = event_benchmark(triton_fn, warmup, iters)
    gather_ms = event_benchmark(fa2_gather_dense_fn, warmup, iters)
    shape = {
        "batch": bsz,
        "query_len": query_len,
        "seq_len": seq_len,
        "q_heads": q_heads,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "block_size": block_size,
    }
    results = [
        make_speed_result(
            name + "_gather_dense_backend_default",
            "prefix_prefill",
            gather_check,
            gather_ms,
            triton_ms,
            shape,
        ),
    ]
    if include_paged:
        paged = fa2_paged_fn()
        paged_check = compare_tensors(
            name + "_paged",
            "prefix_prefill",
            paged,
            expected,
            {},
        )
        paged_ms = event_benchmark(fa2_paged_fn, warmup, iters)
        results.insert(
            0,
            make_speed_result(
                name + "_paged",
                "prefix_prefill",
                paged_check,
                paged_ms,
                triton_ms,
                shape,
            ),
        )
    return results


@torch.inference_mode()
def speed_decode(
    name: str,
    bsz: int,
    seq_len: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    block_size: int,
    device: torch.device,
    warmup: int,
    iters: int,
) -> SpeedResult:
    k = torch.randn(bsz, seq_len, kv_heads, head_dim, device=device,
                    dtype=torch.float16)
    v = torch.randn_like(k)
    k_cache, v_cache, block_table, seq_lens = fill_paged_cache(k, v, block_size)
    q = torch.randn(bsz, q_heads, head_dim, device=device, dtype=torch.float16)
    fa2_out = torch.empty_like(q)
    triton_out = torch.empty_like(q)

    def fa2_fn() -> torch.Tensor:
        result = flash_attn_decode_paged(q, k_cache, v_cache, block_table,
                                         seq_lens, out=fa2_out)
        return fa2_out if result is None else result

    def triton_fn() -> torch.Tensor:
        return triton_unified(
            q,
            k_cache,
            v_cache,
            block_table,
            seq_lens,
            [1] * bsz,
            triton_out,
        )

    actual = fa2_fn()
    expected = triton_fn()
    check = compare_tensors(name, "decode", actual, expected, {})
    fa2_ms = event_benchmark(fa2_fn, warmup, iters)
    triton_ms = event_benchmark(triton_fn, warmup, iters)
    return make_speed_result(name, "decode", check, fa2_ms, triton_ms, {
        "batch": bsz,
        "seq_len": seq_len,
        "q_heads": q_heads,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "block_size": block_size,
    })


def make_speed_result(
    name: str,
    kind: str,
    quality: QualityResult,
    fa2_ms: list[float],
    triton_ms: list[float],
    shape: dict[str, Any],
) -> SpeedResult:
    fa2_median = statistics.median(fa2_ms)
    triton_median = statistics.median(triton_ms)
    return SpeedResult(
        name=name,
        kind=kind,
        shape=shape,
        fa2_ms_median=fa2_median,
        triton_ms_median=triton_median,
        fa2_ms_mean=statistics.mean(fa2_ms),
        triton_ms_mean=statistics.mean(triton_ms),
        ratio_fa2_over_triton=fa2_median / triton_median,
        speedup_vs_triton=triton_median / fa2_median,
        passed_quality=quality.passed,
        max_abs=float(quality.max_abs or 0.0),
        mean_abs=float(quality.mean_abs or 0.0),
    )


def quality_cases(device: torch.device) -> list[Callable[[], QualityResult]]:
    cases = [
        lambda: dense_prefill_case(
            "qwen35_moe_dense_full_prefill_b1_s2048_h16_kv2_hd256",
            1, 2048, 16, 2, 256, 16, device),
        lambda: dense_prefill_case(
            "qwen35_moe_dense_full_prefill_b1_s2048_h16_kv2_hd256_blk528",
            1, 2048, 16, 2, 256, 528, device),
        lambda: dense_prefill_case(
            "qwen35_27b_dense_full_prefill_b1_s2048_h24_kv4_hd256",
            1, 2048, 24, 4, 256, 16, device),
        lambda: prefix_prefill_case(
            "qwen35_moe_prefix_prefill_b1_q512_ctx8192_h16_kv2_hd256",
            1, 512, 8192, 16, 2, 256, 16, device),
        lambda: prefix_prefill_case(
            "qwen35_moe_prefix_prefill_b1_q512_ctx8192_h16_kv2_hd256_blk528_default",
            1, 512, 8192, 16, 2, 256, 528, device, "gather_dense"),
        lambda: prefix_prefill_case(
            "qwen35_27b_prefix_prefill_b1_q512_ctx8192_h24_kv4_hd256",
            1, 512, 8192, 24, 4, 256, 16, device),
        lambda: decode_case(
            "qwen35_moe_decode_b1_ctx16384_h16_kv2_hd256",
            1, 16384, 16, 2, 256, 16, device),
        lambda: decode_case(
            "qwen35_moe_decode_b4_ctx16384_h16_kv2_hd256",
            4, 16384, 16, 2, 256, 16, device),
        lambda: decode_case(
            "qwen35_moe_decode_b4_ctx16384_h16_kv2_hd256_blk528",
            4, 16384, 16, 2, 256, 528, device),
        lambda: decode_case(
            "qwen35_27b_decode_b4_ctx16384_h24_kv4_hd256",
            4, 16384, 24, 4, 256, 16, device),
    ]
    return cases


def quality_cases_256k(device: torch.device) -> list[Callable[[], QualityResult]]:
    return [
        lambda: prefix_prefill_case(
            "qwen35_moe_prefix_prefill_b1_q512_ctx262144_h16_kv2_hd256_blk528_default",
            1, 512, 262144, 16, 2, 256, 528, device, "gather_dense"),
        lambda: prefix_prefill_case(
            "qwen35_moe_prefix_prefill_b1_q1024_ctx262144_h16_kv2_hd256_blk528_default",
            1, 1024, 262144, 16, 2, 256, 528, device, "gather_dense"),
        lambda: decode_case(
            "qwen35_moe_decode_b1_ctx262144_h16_kv2_hd256_blk528",
            1, 262144, 16, 2, 256, 528, device),
        lambda: decode_case(
            "qwen35_moe_decode_b4_ctx262144_h16_kv2_hd256_blk528",
            4, 262144, 16, 2, 256, 528, device),
    ]


def speed_cases(
    device: torch.device,
    warmup: int,
    iters: int,
) -> list[Callable[[], SpeedResult | list[SpeedResult]]]:
    cases = [
        lambda: speed_dense_prefill(
            "qwen35_moe_dense_full_prefill_b1_s4096_h16_kv2_hd256",
            1, 4096, 16, 2, 256, 16, device, warmup, iters),
        lambda: speed_dense_prefill(
            "qwen35_moe_dense_full_prefill_b1_s4096_h16_kv2_hd256_blk528",
            1, 4096, 16, 2, 256, 528, device, warmup, iters),
        lambda: speed_prefix_prefill(
            "qwen35_moe_prefix_prefill_b1_q512_ctx8192_h16_kv2_hd256",
            1, 512, 8192, 16, 2, 256, 16, device, warmup, iters),
        lambda: speed_prefix_prefill(
            "qwen35_moe_prefix_prefill_b1_q512_ctx8192_h16_kv2_hd256_blk528",
            1, 512, 8192, 16, 2, 256, 528, device, warmup, iters, False),
        lambda: speed_prefix_prefill(
            "qwen35_moe_prefix_prefill_b1_q1024_ctx16384_h16_kv2_hd256",
            1, 1024, 16384, 16, 2, 256, 16, device, warmup, max(3, iters // 2)),
        lambda: speed_decode(
            "qwen35_moe_decode_b1_ctx8192_h16_kv2_hd256",
            1, 8192, 16, 2, 256, 16, device, warmup, iters),
        lambda: speed_decode(
            "qwen35_moe_decode_b4_ctx8192_h16_kv2_hd256",
            4, 8192, 16, 2, 256, 16, device, warmup, iters),
        lambda: speed_decode(
            "qwen35_moe_decode_b4_ctx8192_h16_kv2_hd256_blk528",
            4, 8192, 16, 2, 256, 528, device, warmup, iters),
        lambda: speed_decode(
            "qwen35_moe_decode_b1_ctx32768_h16_kv2_hd256",
            1, 32768, 16, 2, 256, 16, device, warmup, max(3, iters // 2)),
        lambda: speed_decode(
            "qwen35_27b_decode_b4_ctx8192_h24_kv4_hd256",
            4, 8192, 24, 4, 256, 16, device, warmup, iters),
    ]
    return cases


def speed_cases_256k(
    device: torch.device,
    warmup: int,
    iters: int,
) -> list[Callable[[], SpeedResult | list[SpeedResult]]]:
    long_iters = max(3, iters // 2)
    return [
        lambda: speed_prefix_prefill(
            "qwen35_moe_prefix_prefill_b1_q512_ctx262144_h16_kv2_hd256_blk528",
            1, 512, 262144, 16, 2, 256, 528, device, warmup, long_iters,
            False),
        lambda: speed_prefix_prefill(
            "qwen35_moe_prefix_prefill_b1_q1024_ctx262144_h16_kv2_hd256_blk528",
            1, 1024, 262144, 16, 2, 256, 528, device, warmup, long_iters,
            False),
        lambda: speed_decode(
            "qwen35_moe_decode_b1_ctx262144_h16_kv2_hd256_blk528",
            1, 262144, 16, 2, 256, 528, device, warmup, long_iters),
        lambda: speed_decode(
            "qwen35_moe_decode_b4_ctx262144_h16_kv2_hd256_blk528",
            4, 262144, 16, 2, 256, 528, device, warmup, long_iters),
    ]


def speed_cases_tp4_local_decode(
    device: torch.device,
    warmup: int,
    iters: int,
) -> list[Callable[[], SpeedResult | list[SpeedResult]]]:
    """Qwen3.5-35B-A3B-AWQ full-attention local shape under TP4.

    The model config has total q_heads=16, kv_heads=2. With TP=4, each rank
    runs q_heads=4 and replicated kv_heads=1 for full-attention decode.
    """
    contexts = [8192, 32768, 65536, 131072, 262144]
    return [
        lambda bsz=bsz, ctx=ctx: speed_decode(
            f"qwen35_tp4local_decode_b{bsz}_ctx{ctx}_h4_kv1_hd256_blk528",
            bsz, ctx, 4, 1, 256, 528, device, warmup, iters)
        for bsz in (1, 4)
        for ctx in contexts
    ]


def run_safely(
    runner: Callable[[], QualityResult | SpeedResult | list[SpeedResult]],
) -> QualityResult | SpeedResult | list[SpeedResult]:
    try:
        return runner()
    except Exception as exc:
        return QualityResult(
            name=getattr(runner, "__name__", "unknown"),
            kind="exception",
            shape={},
            passed=False,
            message=f"{type(exc).__name__}: {exc}\n"
            f"{traceback.format_exc(limit=8)}",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=15)
    parser.add_argument("--skip-speed", action="store_true")
    parser.add_argument("--profile-256k", action="store_true")
    parser.add_argument("--profile-tp4-local-decode", action="store_true")
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    props = torch.cuda.get_device_properties(device)
    if props.major != 7 or props.minor != 0:
        raise RuntimeError(
            f"expected V100/sm70, got {props.name} sm{props.major}{props.minor}")

    set_seed(20260507)
    quality: list[QualityResult] = []
    if args.profile_tp4_local_decode:
        selected_quality_cases = []
    else:
        selected_quality_cases = (
            quality_cases_256k(device) if args.profile_256k else quality_cases(device)
        )
    for runner in selected_quality_cases:
        result = run_safely(runner)
        if isinstance(result, QualityResult):
            quality.append(result)

    speed: list[SpeedResult] = []
    if not args.skip_speed:
        if args.profile_tp4_local_decode:
            selected_speed_cases = speed_cases_tp4_local_decode(
                device, args.warmup, args.iters)
        elif args.profile_256k:
            selected_speed_cases = speed_cases_256k(device, args.warmup, args.iters)
        else:
            selected_speed_cases = speed_cases(device, args.warmup, args.iters)
        for runner in selected_speed_cases:
            result = run_safely(runner)
            if isinstance(result, list):
                speed.extend(result)
            elif isinstance(result, SpeedResult):
                speed.append(result)
            elif isinstance(result, QualityResult):
                quality.append(result)

    payload = {
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": {
            "name": props.name,
            "capability": [props.major, props.minor],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "quality": [asdict(item) for item in quality],
        "speed": [asdict(item) for item in speed],
    }
    output_json = args.output_json
    if output_json is None:
        output_json = Path(
            f"/tmp/fa2_triton_longctx_ops_{time.strftime('%Y%m%d_%H%M%S')}.json"
        )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                           encoding="utf-8")

    failed_quality = [case.name for case in quality if not case.passed]
    failed_speed_quality = [case.name for case in speed if not case.passed_quality]
    for case in quality:
        status = "PASS" if case.passed else "FAIL"
        print(
            f"{status} quality {case.kind} {case.name} "
            f"max_abs={case.max_abs} mean_abs={case.mean_abs}"
        )
        if case.message:
            print(case.message)
    for case in speed:
        status = "PASS" if case.passed_quality else "FAIL"
        print(
            f"{status} speed {case.kind} {case.name}: "
            f"fa2={case.fa2_ms_median:.4f}ms "
            f"triton={case.triton_ms_median:.4f}ms "
            f"ratio={case.ratio_fa2_over_triton:.3f} "
            f"speedup={case.speedup_vs_triton:.3f}x "
            f"max_abs={case.max_abs:.6g}"
        )
    print(f"wrote {output_json}")
    if failed_quality or failed_speed_quality:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
