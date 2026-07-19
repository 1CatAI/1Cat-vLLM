# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Flash-V100 BFLA paged-prefill kernel benchmark.

This is a kernel-only benchmark for the experimental Flash-V100 BFLA port. It
uses the BFLA README recommended torch-mask parameters by default and reports
both prebuilt-mask kernel speed and mask-build-inclusive speed.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import torch

SOURCE_ROOT = Path(__file__).resolve().parents[2]
FLASH_V100_ROOT = SOURCE_ROOT / "flash-attention-v100"
sys.path.insert(0, str(FLASH_V100_ROOT))

from flash_attn_v100 import (  # noqa: E402
    flash_attn_prefill_paged,
    flash_attn_prefill_paged_bfla,
)

PROFILES = {
    "qwen9-tp1": {"heads_q": 16, "heads_kv": 4, "head_dim": 256, "block_size": 528},
    "qwen27-tp2": {"heads_q": 12, "heads_kv": 2, "head_dim": 256, "block_size": 784},
    "qwen35-tp4": {"heads_q": 4, "heads_kv": 1, "head_dim": 256, "block_size": 1056},
}


def _sync() -> None:
    torch.accelerator.synchronize()


def _time_cuda(fn, *, warmup: int, reps: int) -> dict[str, float | int]:
    for _ in range(warmup):
        fn()
    _sync()

    times: list[float] = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        _sync()
        times.append(start.elapsed_time(end))
    return {
        "median_ms": statistics.median(times),
        "mean_ms": statistics.mean(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "warmup": warmup,
        "reps": reps,
    }


def _make_paged_cache(
    kv_len: int,
    *,
    block_size: int,
    heads_kv: int,
    head_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_blocks = math.ceil(kv_len / block_size)
    k_cache = torch.randn(
        (num_blocks, block_size, heads_kv, head_dim),
        device=device,
        dtype=torch.float16,
    )
    v_cache = torch.randn_like(k_cache)
    block_table = torch.arange(num_blocks, device=device, dtype=torch.int32).view(
        1, num_blocks
    )
    seq_lens = torch.tensor([kv_len], device=device, dtype=torch.int32)
    return k_cache, v_cache, block_table, seq_lens


def _build_bfla_mask(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor,
    *,
    seq_len: int,
    block_size: int,
    mask_block_n: int,
    threshold: float,
    keep_mass: float,
    keep_ratio: float,
    min_keep_blocks: int,
    local_blocks: int,
    spec_stride: int,
    spec_prob: float,
    spec_seed: int,
    softmax_scale: float,
) -> torch.Tensor:
    """Single-sequence torch mask builder matching BFLA's recommended path."""
    q_len = int(q.shape[1])
    heads_q = int(q.shape[2])
    head_dim = int(q.shape[3])
    heads_kv = int(k_cache.shape[2])
    queries_per_kv = heads_q // heads_kv
    q_blocks = math.ceil(q_len / mask_block_n)
    kv_tiles = math.ceil(seq_len / mask_block_n)
    flat_group_tokens = 64
    groups = mask_block_n // flat_group_tokens

    q_pad = torch.zeros(
        (q_blocks * mask_block_n, heads_q, head_dim),
        device=q.device,
        dtype=q.dtype,
    )
    q_pad[:q_len].copy_(q.squeeze(0))
    q_low = (
        q_pad.view(q_blocks, groups, flat_group_tokens, heads_q, head_dim)
        .permute(3, 0, 1, 2, 4)
        .reshape(heads_q, q_blocks, groups, flat_group_tokens * head_dim)
    )

    num_pages = math.ceil(seq_len / block_size)
    pages = block_table[0, :num_pages].to(torch.long)
    k_req = k_cache.index_select(0, pages).reshape(-1, heads_kv, head_dim)[:seq_len]
    k_pad = torch.zeros(
        (kv_tiles * mask_block_n, heads_kv, head_dim),
        device=q.device,
        dtype=k_cache.dtype,
    )
    k_pad[:seq_len].copy_(k_req)
    k_low = (
        k_pad.view(kv_tiles, groups, flat_group_tokens, heads_kv, head_dim)
        .permute(3, 0, 1, 2, 4)
        .reshape(heads_kv, kv_tiles, groups, flat_group_tokens * head_dim)
    )

    context_len = seq_len - q_len
    q_block_end = (
        context_len + (torch.arange(q_blocks, device=q.device) + 1) * mask_block_n - 1
    ).clamp(max=seq_len - 1)
    k_block_start = torch.arange(kv_tiles, device=q.device) * mask_block_n
    causal = k_block_start[None, :] <= q_block_end[:, None]

    keep_per_kv = torch.zeros(
        (heads_kv, q_blocks, kv_tiles), device=q.device, dtype=torch.bool
    )
    for kv_h in range(heads_kv):
        q_h0 = kv_h * queries_per_kv
        q_h1 = q_h0 + queries_per_kv
        scores = torch.einsum("hqgf,krf->hqkgr", q_low[q_h0:q_h1], k_low[kv_h])
        scores = scores.amax(dim=(-1, -2))
        scores = scores.masked_fill(~causal[None, :, :], float("-inf"))
        probs = torch.softmax(scores.float() * softmax_scale, dim=-1)
        keep = (probs > threshold).any(dim=0)

        if keep_mass >= 1.0:
            keep |= causal
        elif keep_mass > 0:
            sorted_probs, sorted_idx = torch.sort(
                probs.float(), dim=-1, descending=True
            )
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            mass_keep_sorted = cumsum <= keep_mass
            mass_keep_sorted[..., 0] = True
            first_over = torch.argmax(
                (cumsum >= keep_mass).to(torch.int32), dim=-1, keepdim=True
            )
            mass_keep_sorted.scatter_(-1, first_over, True)
            mass_keep = torch.zeros_like(probs, dtype=torch.bool)
            mass_keep.scatter_(-1, sorted_idx, mass_keep_sorted)
            keep |= mass_keep.any(dim=0)

        if keep_ratio > 0 or min_keep_blocks > 0:
            topk = max(min_keep_blocks, int(kv_tiles * keep_ratio))
            topk = max(1, min(topk, kv_tiles))
            _, topk_idx = torch.topk(scores.float(), k=topk, dim=-1)
            topk_keep = torch.zeros_like(scores, dtype=torch.bool)
            topk_keep.scatter_(-1, topk_idx, True)
            keep |= topk_keep.any(dim=0)
        keep_per_kv[kv_h] = keep

    keep_per_kv &= causal[None, :, :]
    q_tile_abs = (
        context_len + torch.arange(q_blocks, device=q.device) * mask_block_n
    ) // mask_block_n
    k_idx = torch.arange(kv_tiles, device=q.device)
    local = (k_idx[None, :] <= q_tile_abs[:, None]) & (
        k_idx[None, :] >= q_tile_abs[:, None] - local_blocks
    )
    keep_per_kv |= local[None, :, :]
    keep_per_kv[:, :, 0] = True
    if spec_stride > 0:
        dropped = causal[None, :, :] & ~keep_per_kv
        q_idx = torch.arange(q_blocks, device=q.device, dtype=torch.int64)[:, None]
        k_idx_i64 = torch.arange(kv_tiles, device=q.device, dtype=torch.int64)[None, :]
        stride_keep = (
            (q_idx * 131 + k_idx_i64 * 17 + int(spec_seed)) % spec_stride
        ) == 0
        keep_per_kv |= dropped & stride_keep[None, :, :]
    if spec_prob > 0:
        prob = max(0.0, min(float(spec_prob), 1.0))
        dropped = causal[None, :, :] & ~keep_per_kv
        if prob >= 1.0:
            keep_per_kv |= dropped
        else:
            q_idx = torch.arange(q_blocks, device=q.device, dtype=torch.int64)[
                None, :, None
            ]
            k_idx_i64 = torch.arange(kv_tiles, device=q.device, dtype=torch.int64)[
                None, None, :
            ]
            h_idx = torch.arange(heads_kv, device=q.device, dtype=torch.int64)[
                :, None, None
            ]
            hashed = (
                (q_idx + 1) * 1103515245
                + (k_idx_i64 + 1) * 12345
                + (h_idx + 1) * 2654435761
                + int(spec_seed)
            ) & 0x7FFFFFFF
            random_keep = (hashed % 1000000) < int(prob * 1000000)
            keep_per_kv |= dropped & random_keep
    return keep_per_kv.to(torch.int32).unsqueeze(0).contiguous()


def _causal_tile_count(q_len: int, kv_len: int, mask_block_n: int) -> int:
    q_blocks = math.ceil(q_len / mask_block_n)
    kv_tiles = math.ceil(kv_len / mask_block_n)
    context_len = kv_len - q_len
    total = 0
    for q_idx in range(q_blocks):
        q_end = min(context_len + (q_idx + 1) * mask_block_n - 1, kv_len - 1)
        total += min(kv_tiles, q_end // mask_block_n + 1)
    return total


@torch.inference_mode()
def bench_case(args: argparse.Namespace, q_len: int, kv_len: int) -> dict[str, Any]:
    profile = PROFILES[args.profile]
    heads_q = args.heads_q or profile["heads_q"]
    heads_kv = args.heads_kv or profile["heads_kv"]
    head_dim = args.head_dim or profile["head_dim"]
    block_size = args.block_size or profile["block_size"]
    device = torch.device("cuda")
    softmax_scale = 1.0 / math.sqrt(head_dim)

    torch.manual_seed(args.seed + q_len * 17 + kv_len)
    q = torch.randn((1, q_len, heads_q, head_dim), device=device, dtype=torch.float16)
    k_cache, v_cache, block_table, seq_lens = _make_paged_cache(
        kv_len,
        block_size=block_size,
        heads_kv=heads_kv,
        head_dim=head_dim,
        device=device,
    )

    dense_holder: list[torch.Tensor | None] = [None]
    bfla_holder: list[torch.Tensor | None] = [None]
    mask_holder: list[torch.Tensor | None] = [None]

    def build_mask() -> torch.Tensor:
        return _build_bfla_mask(
            q,
            k_cache,
            block_table,
            seq_len=kv_len,
            block_size=block_size,
            mask_block_n=args.mask_block_n,
            threshold=args.threshold,
            keep_mass=args.keep_mass,
            keep_ratio=args.keep_ratio,
            min_keep_blocks=args.min_keep_blocks,
            local_blocks=args.local_blocks,
            spec_stride=args.spec_stride,
            spec_prob=args.spec_prob,
            spec_seed=args.spec_seed,
            softmax_scale=softmax_scale,
        )

    mask = build_mask()
    mask_holder[0] = mask
    _sync()

    def run_dense() -> None:
        dense_holder[0] = flash_attn_prefill_paged(
            q, k_cache, v_cache, block_table, seq_lens, softmax_scale=softmax_scale
        )

    def run_bfla_kernel() -> None:
        bfla_holder[0] = flash_attn_prefill_paged_bfla(
            q,
            k_cache,
            v_cache,
            block_table,
            seq_lens,
            mask_holder[0],
            args.mask_block_n,
            softmax_scale=softmax_scale,
        )

    def run_mask_only() -> None:
        mask_holder[0] = build_mask()

    def run_bfla_total() -> None:
        mask_holder[0] = build_mask()
        run_bfla_kernel()

    dense_time = _time_cuda(run_dense, warmup=args.warmup, reps=args.reps)
    bfla_kernel_time = _time_cuda(run_bfla_kernel, warmup=args.warmup, reps=args.reps)
    mask_time = _time_cuda(run_mask_only, warmup=args.warmup, reps=args.reps)
    bfla_total_time = _time_cuda(run_bfla_total, warmup=args.warmup, reps=args.reps)

    run_dense()
    run_bfla_kernel()
    _sync()
    diff = (dense_holder[0] - bfla_holder[0]).abs()
    causal_tiles = _causal_tile_count(q_len, kv_len, args.mask_block_n) * heads_kv
    kept_tiles = int(mask.sum().item())
    result = {
        "profile": args.profile,
        "q_len": q_len,
        "kv_len": kv_len,
        "heads_q": heads_q,
        "heads_kv": heads_kv,
        "head_dim": head_dim,
        "block_size": block_size,
        "mask_block_n": args.mask_block_n,
        "threshold": args.threshold,
        "keep_mass": args.keep_mass,
        "keep_ratio": args.keep_ratio,
        "min_keep_blocks": args.min_keep_blocks,
        "local_blocks": args.local_blocks,
        "spec_stride": args.spec_stride,
        "spec_prob": args.spec_prob,
        "spec_seed": args.spec_seed,
        "kept_tiles": kept_tiles,
        "causal_tiles": causal_tiles,
        "keep_fraction": kept_tiles / causal_tiles if causal_tiles else None,
        "dense": dense_time,
        "bfla_kernel": bfla_kernel_time,
        "bfla_mask": mask_time,
        "bfla_total": bfla_total_time,
        "kernel_speedup": dense_time["median_ms"] / bfla_kernel_time["median_ms"],
        "total_speedup": dense_time["median_ms"] / bfla_total_time["median_ms"],
        "dense_vs_sparse_maxdiff": float(diff.max().item()),
        "dense_vs_sparse_meandiff": float(diff.mean().item()),
    }
    torch.accelerator.empty_cache()
    return result


def _parse_case(value: str) -> tuple[int, int]:
    try:
        q_len, kv_len = value.lower().split("x", 1)
        return int(q_len), int(kv_len)
    except Exception as exc:
        raise argparse.ArgumentTypeError(
            f"expected QxKV, for example 16384x262144: {value}"
        ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="qwen35-tp4")
    parser.add_argument("--case", type=_parse_case, nargs="+", required=True)
    parser.add_argument("--heads-q", type=int, default=None)
    parser.add_argument("--heads-kv", type=int, default=None)
    parser.add_argument("--head-dim", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--mask-block-n", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=999.0)
    parser.add_argument("--keep-mass", type=float, default=0.99)
    parser.add_argument("--keep-ratio", type=float, default=0.0)
    parser.add_argument("--min-keep-blocks", type=int, default=0)
    parser.add_argument("--local-blocks", type=int, default=8)
    parser.add_argument("--spec-stride", type=int, default=0)
    parser.add_argument("--spec-prob", type=float, default=0.0)
    parser.add_argument("--spec-seed", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260619)
    parser.add_argument("--jsonl", action="store_true")
    args = parser.parse_args()

    for q_len, kv_len in args.case:
        result = bench_case(args, q_len, kv_len)
        if args.jsonl:
            print(json.dumps(result, sort_keys=True), flush=True)
        else:
            print(
                "BFLA {profile} M={q_len} N={kv_len} keep={keep_fraction:.3f} "
                "dense={dense_ms:.3f}ms bfla_kernel={bfla_ms:.3f}ms "
                "bfla_total={total_ms:.3f}ms speedup={kernel_speedup:.2f}x "
                "total={total_speedup:.2f}x maxdiff={maxdiff:.4f}".format(
                    profile=result["profile"],
                    q_len=result["q_len"],
                    kv_len=result["kv_len"],
                    keep_fraction=result["keep_fraction"],
                    dense_ms=result["dense"]["median_ms"],
                    bfla_ms=result["bfla_kernel"]["median_ms"],
                    total_ms=result["bfla_total"]["median_ms"],
                    kernel_speedup=result["kernel_speedup"],
                    total_speedup=result["total_speedup"],
                    maxdiff=result["dense_vs_sparse_maxdiff"],
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
