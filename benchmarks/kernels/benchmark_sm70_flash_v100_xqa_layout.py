#!/usr/bin/env python3
"""Exact-layout A/B microbenchmark for the SM70 q=1 XQA decode kernel."""

from __future__ import annotations

import argparse
import json
import os

import torch


def run_once(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    out: torch.Tensor,
    *,
    padded_smem: bool,
    block784_index: bool,
    aligned_padded_smem: bool,
    g6_dual_cta: str,
    inherited_g6_dual_cta: str | None,
    split_reduce: str,
    inherited_split_reduce: str | None,
    seq_len: int,
    partition_size: int,
) -> torch.Tensor:
    from flash_attn_v100 import flash_attn_decode_paged_xqa

    os.environ["VLLM_FLASH_V100_XQA_PADDED_SMEM"] = "1" if padded_smem else "0"
    os.environ["VLLM_FLASH_V100_XQA_BLOCK784_INDEX"] = "1" if block784_index else "0"
    os.environ["VLLM_FLASH_V100_XQA_ALIGNED_PADDED_SMEM"] = (
        "1" if aligned_padded_smem else "0"
    )
    if g6_dual_cta == "inherit":
        if inherited_g6_dual_cta is None:
            os.environ.pop("VLLM_FLASH_V100_XQA_G6_DUAL_CTA", None)
        else:
            os.environ["VLLM_FLASH_V100_XQA_G6_DUAL_CTA"] = inherited_g6_dual_cta
    else:
        os.environ["VLLM_FLASH_V100_XQA_G6_DUAL_CTA"] = (
            "1" if g6_dual_cta == "on" else "0"
        )
    if split_reduce == "inherit":
        if inherited_split_reduce is None:
            os.environ.pop("VLLM_FLASH_V100_XQA_SPLIT_REDUCE", None)
        else:
            os.environ["VLLM_FLASH_V100_XQA_SPLIT_REDUCE"] = inherited_split_reduce
    else:
        os.environ["VLLM_FLASH_V100_XQA_SPLIT_REDUCE"] = (
            "1" if split_reduce == "on" else "0"
        )
    os.environ["VLLM_FLASH_V100_DECODE_PARTITION_SIZE"] = str(partition_size)
    return flash_attn_decode_paged_xqa(
        q,
        k_cache,
        v_cache,
        block_table,
        seq_lens,
        out=out,
        max_seq_len_hint=seq_len,
    )


def elapsed_ms(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    out: torch.Tensor,
    *,
    padded_smem: bool,
    block784_index: bool,
    aligned_padded_smem: bool,
    g6_dual_cta: str,
    inherited_g6_dual_cta: str | None,
    split_reduce: str,
    inherited_split_reduce: str | None,
    seq_len: int,
    partition_size: int,
    warmup: int,
    iters: int,
) -> float:
    for _ in range(warmup):
        run_once(
            q,
            k_cache,
            v_cache,
            block_table,
            seq_lens,
            out,
            padded_smem=padded_smem,
            block784_index=block784_index,
            aligned_padded_smem=aligned_padded_smem,
            g6_dual_cta=g6_dual_cta,
            inherited_g6_dual_cta=inherited_g6_dual_cta,
            split_reduce=split_reduce,
            inherited_split_reduce=inherited_split_reduce,
            seq_len=seq_len,
            partition_size=partition_size,
        )
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        run_once(
            q,
            k_cache,
            v_cache,
            block_table,
            seq_lens,
            out,
            padded_smem=padded_smem,
            block784_index=block784_index,
            aligned_padded_smem=aligned_padded_smem,
            g6_dual_cta=g6_dual_cta,
            inherited_g6_dual_cta=inherited_g6_dual_cta,
            split_reduce=split_reduce,
            inherited_split_reduce=inherited_split_reduce,
            seq_len=seq_len,
            partition_size=partition_size,
        )
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=65539)
    parser.add_argument(
        "--block-size",
        type=int,
        default=16,
        help="Paged-KV block size; use the model's resolved attention page size.",
    )
    parser.add_argument("--partition-size", type=int, default=256)
    parser.add_argument(
        "--candidate-partition-size",
        type=int,
        help=(
            "Partition size for the padded candidate. Defaults to "
            "--partition-size, allowing an exact p1024-versus-p256 operator "
            "comparison when set to 256."
        ),
    )
    parser.add_argument("--warmup", type=int, default=40)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument(
        "--block-table-layout",
        choices=("sequential", "reverse", "permuted"),
        default="sequential",
        help="Logical-to-physical paged-KV mapping used by the direct check.",
    )
    parser.add_argument(
        "--layout", choices=("compare", "baseline", "padded"), default="compare"
    )
    parser.add_argument(
        "--candidate-block784-index",
        action="store_true",
        help=(
            "Enable the exact 784-token page-index specialization for the "
            "padded candidate only."
        ),
    )
    parser.add_argument(
        "--baseline-padded",
        action="store_true",
        help=(
            "Keep padded shared memory enabled for the baseline too, so an "
            "A/B comparison isolates another candidate such as block784 "
            "page-index specialization."
        ),
    )
    parser.add_argument(
        "--baseline-block784-index",
        action="store_true",
        help="Enable the block784 index specialization for the baseline.",
    )
    parser.add_argument(
        "--candidate-aligned-padded-smem",
        action="store_true",
        help="Enable the aligned Q272/KV144 padded shared layout candidate.",
    )
    parser.add_argument(
        "--baseline-aligned-padded-smem",
        action="store_true",
        help="Enable the aligned Q272/KV144 padded shared layout for the baseline.",
    )
    parser.add_argument(
        "--candidate-dense-smem",
        action="store_true",
        help="Use dense Q256/KV128 shared storage for the candidate.",
    )
    parser.add_argument(
        "--baseline-g6-dual-cta",
        choices=("inherit", "on", "off"),
        default="inherit",
        help="G6 dispatch mode for the baseline call.",
    )
    parser.add_argument(
        "--candidate-g6-dual-cta",
        choices=("inherit", "on", "off"),
        default="inherit",
        help="G6 dispatch mode for the candidate call.",
    )
    parser.add_argument(
        "--baseline-split-reduce",
        choices=("inherit", "on", "off"),
        default="inherit",
        help="Split-reduce dispatch mode for the baseline call.",
    )
    parser.add_argument(
        "--candidate-split-reduce",
        choices=("inherit", "on", "off"),
        default="inherit",
        help="Split-reduce dispatch mode for the candidate call.",
    )
    parser.add_argument("--profile-single", action="store_true")
    args = parser.parse_args()
    inherited_g6_dual_cta = os.environ.get("VLLM_FLASH_V100_XQA_G6_DUAL_CTA")
    inherited_split_reduce = os.environ.get("VLLM_FLASH_V100_XQA_SPLIT_REDUCE")

    if args.partition_size not in (256, 512, 1024):
        raise ValueError("--partition-size must be one of 256, 512, 1024")
    if args.block_size <= 0:
        raise ValueError("--block-size must be positive")
    if (
        args.candidate_block784_index or args.baseline_block784_index
    ) and args.block_size != 784:
        raise ValueError("block784 candidates require --block-size 784")
    if args.candidate_aligned_padded_smem and not args.candidate_block784_index:
        raise ValueError("candidate aligned padding requires block784 index")
    if args.baseline_aligned_padded_smem and not args.baseline_block784_index:
        raise ValueError("baseline aligned padding requires block784 index")
    if args.candidate_dense_smem and args.candidate_aligned_padded_smem:
        raise ValueError("candidate dense and aligned padded layouts are exclusive")
    candidate_partition_size = (
        args.candidate_partition_size
        if args.candidate_partition_size is not None
        else args.partition_size
    )
    if candidate_partition_size not in (256, 512, 1024):
        raise ValueError("--candidate-partition-size must be one of 256, 512, 1024")

    # Qwen3.6-27B-AWQ TP4 full-attention per-rank shape: Hq=6, Hkv=1, D=256.
    torch.manual_seed(20260713)
    torch.cuda.manual_seed_all(20260713)
    block_size, q_heads, kv_heads, head_dim = (
        args.block_size,
        6,
        1,
        256,
    )
    blocks = (args.seq_len + block_size - 1) // block_size
    q = torch.randn((1, q_heads, head_dim), device="cuda", dtype=torch.float16)
    k_cache = torch.randn(
        (blocks, block_size, kv_heads, head_dim), device="cuda", dtype=torch.float16
    )
    v_cache = torch.randn_like(k_cache)
    if args.block_table_layout == "sequential":
        block_ids = torch.arange(blocks, device="cuda", dtype=torch.int32)
    elif args.block_table_layout == "reverse":
        block_ids = torch.arange(blocks - 1, -1, -1, device="cuda", dtype=torch.int32)
    else:
        block_ids = torch.randperm(blocks, device="cuda", dtype=torch.int32)
    block_table = block_ids.unsqueeze(0)
    seq_lens = torch.tensor([args.seq_len], device="cuda", dtype=torch.int32)
    baseline_out = torch.empty_like(q)
    padded_out = torch.empty_like(q)
    baseline_padded = args.baseline_padded
    candidate_padded = not args.candidate_dense_smem

    if args.profile_single:
        padded = candidate_padded if args.layout == "padded" else baseline_padded
        run_once(
            q,
            k_cache,
            v_cache,
            block_table,
            seq_lens,
            padded_out if padded else baseline_out,
            padded_smem=padded,
            block784_index=(
                args.candidate_block784_index
                if args.layout == "padded"
                else args.baseline_block784_index
            ),
            aligned_padded_smem=(
                args.candidate_aligned_padded_smem
                if args.layout == "padded"
                else args.baseline_aligned_padded_smem
            ),
            g6_dual_cta=(
                args.candidate_g6_dual_cta
                if args.layout == "padded"
                else args.baseline_g6_dual_cta
            ),
            inherited_g6_dual_cta=inherited_g6_dual_cta,
            split_reduce=(
                args.candidate_split_reduce
                if args.layout == "padded"
                else args.baseline_split_reduce
            ),
            inherited_split_reduce=inherited_split_reduce,
            seq_len=args.seq_len,
            partition_size=(
                candidate_partition_size if padded else args.partition_size
            ),
        )
        torch.cuda.synchronize()
        return

    baseline = run_once(
        q,
        k_cache,
        v_cache,
        block_table,
        seq_lens,
        baseline_out,
        padded_smem=baseline_padded,
        block784_index=args.baseline_block784_index,
        aligned_padded_smem=args.baseline_aligned_padded_smem,
        g6_dual_cta=args.baseline_g6_dual_cta,
        inherited_g6_dual_cta=inherited_g6_dual_cta,
        split_reduce=args.baseline_split_reduce,
        inherited_split_reduce=inherited_split_reduce,
        seq_len=args.seq_len,
        partition_size=args.partition_size,
    )
    padded = run_once(
        q,
        k_cache,
        v_cache,
        block_table,
        seq_lens,
        padded_out,
        padded_smem=candidate_padded,
        block784_index=args.candidate_block784_index,
        aligned_padded_smem=args.candidate_aligned_padded_smem,
        g6_dual_cta=args.candidate_g6_dual_cta,
        inherited_g6_dual_cta=inherited_g6_dual_cta,
        split_reduce=args.candidate_split_reduce,
        inherited_split_reduce=inherited_split_reduce,
        seq_len=args.seq_len,
        partition_size=candidate_partition_size,
    )
    torch.cuda.synchronize()

    diff_mask = baseline.ne(padded)
    mismatch_indices = diff_mask.nonzero(as_tuple=False)
    mismatch_count = int(mismatch_indices.size(0))
    result: dict[str, object] = {
        "seq_len": args.seq_len,
        "block_size": block_size,
        "block_table_layout": args.block_table_layout,
        "partition_size": args.partition_size,
        "candidate_partition_size": candidate_partition_size,
        "candidate_block784_index": args.candidate_block784_index,
        "baseline_block784_index": args.baseline_block784_index,
        "candidate_aligned_padded_smem": args.candidate_aligned_padded_smem,
        "baseline_aligned_padded_smem": args.baseline_aligned_padded_smem,
        "candidate_g6_dual_cta": args.candidate_g6_dual_cta,
        "baseline_g6_dual_cta": args.baseline_g6_dual_cta,
        "candidate_split_reduce": args.candidate_split_reduce,
        "baseline_split_reduce": args.baseline_split_reduce,
        "baseline_padded": baseline_padded,
        "candidate_padded": candidate_padded,
        "q_heads": q_heads,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "active_partitions": (args.seq_len + args.partition_size - 1)
        // args.partition_size,
        "candidate_active_partitions": (args.seq_len + candidate_partition_size - 1)
        // candidate_partition_size,
        "bitwise_equal": bool(torch.equal(baseline, padded)),
        "mismatch_count": mismatch_count,
        "max_abs_diff": float((baseline.float() - padded.float()).abs().max()),
    }
    if mismatch_count:
        sample_indices = mismatch_indices[:8].cpu().tolist()
        head_counts = (
            torch.bincount(mismatch_indices[:, 1], minlength=q_heads).cpu().tolist()
        )
        result["mismatch_head_counts"] = head_counts
        result["mismatch_indices"] = sample_indices
        result["mismatch_values"] = [
            {
                "baseline": float(baseline[tuple(index)].float()),
                "candidate": float(padded[tuple(index)].float()),
            }
            for index in sample_indices
        ]

    if args.layout in ("compare", "baseline"):
        result["baseline_partition_plus_reduce_ms"] = elapsed_ms(
            q,
            k_cache,
            v_cache,
            block_table,
            seq_lens,
            baseline_out,
            padded_smem=baseline_padded,
            block784_index=args.baseline_block784_index,
            aligned_padded_smem=args.baseline_aligned_padded_smem,
            g6_dual_cta=args.baseline_g6_dual_cta,
            inherited_g6_dual_cta=inherited_g6_dual_cta,
            split_reduce=args.baseline_split_reduce,
            inherited_split_reduce=inherited_split_reduce,
            seq_len=args.seq_len,
            partition_size=args.partition_size,
            warmup=args.warmup,
            iters=args.iters,
        )
    if args.layout in ("compare", "padded"):
        result["padded_partition_plus_reduce_ms"] = elapsed_ms(
            q,
            k_cache,
            v_cache,
            block_table,
            seq_lens,
            padded_out,
            padded_smem=candidate_padded,
            block784_index=args.candidate_block784_index,
            aligned_padded_smem=args.candidate_aligned_padded_smem,
            g6_dual_cta=args.candidate_g6_dual_cta,
            inherited_g6_dual_cta=inherited_g6_dual_cta,
            split_reduce=args.candidate_split_reduce,
            inherited_split_reduce=inherited_split_reduce,
            seq_len=args.seq_len,
            partition_size=candidate_partition_size,
            warmup=args.warmup,
            iters=args.iters,
        )
    if args.layout == "compare":
        baseline_ms = float(result["baseline_partition_plus_reduce_ms"])
        padded_ms = float(result["padded_partition_plus_reduce_ms"])
        result["speedup"] = baseline_ms / padded_ms
        result["delta_ms"] = padded_ms - baseline_ms

    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
