# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Screen the Flash-V100 grouped Page4 kernel at Qwen3.8 MTP4 width.

This benchmark compares the production five-row Triton QSA verifier shape
against the grouped Page4 operator from PR #387. The candidate pads five rows
to the operator's existing eight-row contract and includes those copies plus
GPU planning in its measured cost.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from flash_attn_v100.flash_attn_interface import flash_attn_v100_cuda

import vllm

if binary_package := os.getenv("VLLM_BINARY_PACKAGE_PATH"):
    vllm.__path__.append(binary_package)

from vllm.models.qwen4_exp.nvidia.ops import qsa as qsa_ops


def _measure_ms(call, *, warmups: int, repeats: int) -> float:
    for _ in range(warmups):
        call()
    torch.accelerator.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        call()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeats


def _logical_indices(
    *, rows: int, seq_len: int, overlap: float, seed: int, independent: bool
) -> torch.Tensor:
    """Build 2051-token selections with controlled adjacent-row Page4 overlap."""
    generator = torch.Generator().manual_seed(seed)
    selectable_pages = torch.arange(seq_len // 4 - 1, dtype=torch.int64)
    shared_count = round(512 * overlap)
    shared = selectable_pages[
        torch.randperm(selectable_pages.numel(), generator=generator)[:shared_count]
    ]
    remaining_mask = torch.ones(selectable_pages.numel(), dtype=torch.bool)
    remaining_mask[shared] = False
    remaining = selectable_pages[remaining_mask]

    selections: list[torch.Tensor] = []
    for row in range(rows):
        row_generator = torch.Generator().manual_seed(seed + row + 1)
        unique = remaining[
            torch.randperm(remaining.numel(), generator=row_generator)[
                : 512 - shared_count
            ]
        ]
        pages = torch.cat((shared, unique))
        tokens = (pages[:, None] * 4 + torch.arange(4)).flatten()
        query_position = seq_len - 1 if independent else seq_len - rows + row
        tail = torch.arange(query_position - 2, query_position + 1)
        selections.append(torch.cat((tokens, tail)).to(torch.int32))
    return torch.stack(selections)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--rows", type=int, choices=(5, 8, 16), default=5)
    parser.add_argument("--independent-requests", action="store_true")
    parser.add_argument("--overlap", type=float, default=0.82)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.seq_len < 4096 or args.seq_len % 4:
        raise ValueError("--seq-len must be a multiple of four and at least 4096")
    if not 0.0 <= args.overlap <= 1.0:
        raise ValueError("--overlap must be between zero and one")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        raise RuntimeError("this benchmark requires an SM70 CUDA device")
    required = ("grouped_sparse_page4_plan_fwd", "grouped_sparse_page4_fwd")
    if not all(hasattr(flash_attn_v100_cuda, name) for name in required):
        raise RuntimeError("Flash-V100 was not built with grouped Page4 operators")

    rows = args.rows
    padded_rows = ((rows + 7) // 8) * 8
    heads = 6
    head_dim = 256
    page_size = 784
    blocks_per_request = (args.seq_len + page_size - 1) // page_size
    num_requests = rows if args.independent_requests else 1
    num_cache_blocks = blocks_per_request * num_requests
    torch.manual_seed(args.seed)
    query = torch.randn((rows, heads, head_dim), device="cuda", dtype=torch.float16)
    key_cache = torch.randn(
        (num_cache_blocks, page_size, 1, head_dim),
        device="cuda",
        dtype=torch.float16,
    )
    value_cache = torch.randn_like(key_cache)
    indices = _logical_indices(
        rows=rows,
        seq_len=args.seq_len,
        overlap=args.overlap,
        seed=args.seed,
        independent=args.independent_requests,
    ).to("cuda")
    block_table = torch.arange(num_cache_blocks, dtype=torch.int32, device="cuda").view(
        num_requests, blocks_per_request
    )
    token_to_req = (
        torch.arange(rows, dtype=torch.int32, device="cuda")
        if args.independent_requests
        else torch.zeros(rows, dtype=torch.int32, device="cuda")
    )
    query_positions = (
        torch.full((rows,), args.seq_len - 1, dtype=torch.int64, device="cuda")
        if args.independent_requests
        else torch.arange(
            args.seq_len - rows,
            args.seq_len,
            dtype=torch.int64,
            device="cuda",
        )
    )
    sequence_lengths = torch.full(
        (num_requests,), args.seq_len, dtype=torch.int32, device="cuda"
    )
    reference = torch.empty_like(query)
    xqa_candidate = torch.empty_like(query)

    padded_query = torch.zeros(
        (padded_rows, heads, head_dim), device="cuda", dtype=torch.float16
    )
    padded_indices = torch.full(
        (padded_rows, 2051), -1, device="cuda", dtype=torch.int32
    )
    padded_token_to_req = torch.full(
        (padded_rows,), -1, device="cuda", dtype=torch.int32
    )
    padded_query_positions = torch.zeros(
        (padded_rows,), device="cuda", dtype=torch.int64
    )
    candidate = torch.empty_like(padded_query)

    def reference_call() -> None:
        qsa_ops.qsa_sparse_paged_attention(
            query,
            key_cache,
            value_cache,
            indices,
            block_table,
            token_to_req,
            reference,
            query_positions=query_positions,
            sequence_lengths=sequence_lengths,
        )

    def candidate_call() -> None:
        padded_query[:rows].copy_(query)
        padded_indices[:rows].copy_(indices)
        padded_token_to_req[:rows].copy_(token_to_req)
        padded_query_positions[:rows].copy_(query_positions)
        qsa_ops._qsa_sparse_paged_attention_sm70_grouped_page4(
            padded_query,
            key_cache,
            value_cache,
            padded_indices,
            block_table,
            padded_token_to_req,
            padded_query_positions,
            sequence_lengths,
            candidate,
            "auto",
            1.0,
            1.0,
            flash_attn_v100_cuda,
        )

    def xqa_candidate_call() -> None:
        result = qsa_ops._qsa_sparse_paged_attention_sm70_xqa_page4(
            query,
            key_cache,
            value_cache,
            indices,
            block_table,
            token_to_req,
            query_positions,
            sequence_lengths,
            xqa_candidate,
            "auto",
            1.0,
            1.0,
        )
        if result is None:
            raise RuntimeError("Flash-V100 XQA Page4 route is unavailable")

    reference_call()
    candidate_call()
    xqa_candidate_call()
    torch.accelerator.synchronize()
    first_candidate = candidate.clone()
    candidate_call()
    torch.accelerator.synchronize()

    error = candidate[:rows].float() - reference.float()
    xqa_error = xqa_candidate.float() - reference.float()
    reference_ms = _measure_ms(
        reference_call, warmups=args.warmups, repeats=args.repeats
    )
    candidate_ms = _measure_ms(
        candidate_call, warmups=args.warmups, repeats=args.repeats
    )
    xqa_candidate_ms = _measure_ms(
        xqa_candidate_call, warmups=args.warmups, repeats=args.repeats
    )
    payload = {
        "source": str(Path(qsa_ops.__file__).resolve()),
        "device": torch.cuda.get_device_name(),
        "sm": list(torch.cuda.get_device_capability()),
        "rows": rows,
        "padded_rows": padded_rows,
        "independent_requests": args.independent_requests,
        "seq_len": args.seq_len,
        "selection_width": indices.shape[1],
        "requested_page_overlap": args.overlap,
        "reference_triton_ms_per_layer": reference_ms,
        "candidate_flash_v100_ms_per_layer": candidate_ms,
        "speedup": reference_ms / candidate_ms,
        "saved_ms_per_layer": reference_ms - candidate_ms,
        "projected_saved_ms_per_12_layer_verifier": 12 * (reference_ms - candidate_ms),
        "max_abs_error": error.abs().max().item(),
        "relative_l2_error": (error.norm() / reference.float().norm()).item(),
        "cosine_similarity": torch.nn.functional.cosine_similarity(
            candidate[:rows].float().flatten(), reference.float().flatten(), dim=0
        ).item(),
        "candidate_replay_bitwise_equal": torch.equal(first_candidate, candidate),
        "padded_rows_zero": torch.count_nonzero(candidate[rows:]).item() == 0,
        "xqa_flash_v100_ms_per_layer": xqa_candidate_ms,
        "xqa_speedup": reference_ms / xqa_candidate_ms,
        "xqa_saved_ms_per_layer": reference_ms - xqa_candidate_ms,
        "xqa_projected_saved_ms_per_12_layer_verifier": 12
        * (reference_ms - xqa_candidate_ms),
        "xqa_max_abs_error": xqa_error.abs().max().item(),
        "xqa_relative_l2_error": (xqa_error.norm() / reference.float().norm()).item(),
        "xqa_cosine_similarity": torch.nn.functional.cosine_similarity(
            xqa_candidate.float().flatten(), reference.float().flatten(), dim=0
        ).item(),
        "timing_includes_padding_copies_and_gpu_planner": True,
        "note": (
            "Operator projection only; end-to-end verifier wall remains authoritative."
        ),
    }
    encoded = json.dumps(payload, indent=2)
    print(encoded)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
