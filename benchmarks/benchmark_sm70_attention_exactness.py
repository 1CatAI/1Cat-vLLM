# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Strict exactness checks for the migrated SM70 FlashAttention-V100 path.

This script is intentionally small and model-free. It exercises the external
``flash_attn_v100`` package directly, then compares outputs against a simple
PyTorch GQA attention reference with strict ``torch.equal`` semantics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import triton
import triton.language as tl

from vllm.v1.attention.ops.triton_attention_helpers import softmax_step

FP16_NOMINAL_OUTPUT_BOUND = 2.0e-3


@dataclass
class CaseResult:
    name: str
    reference: str
    equal: bool
    max_diff: float
    mean_diff: float
    num_different: int
    first_mismatch_flat_index: int | None
    max_diff_flat_index: int | None
    actual_at_max_diff: float | None
    expected_at_max_diff: float | None
    actual_nan_count: int
    expected_nan_count: int
    shape: list[int]
    dtype: str
    seq_len: int
    query_len: int
    q_heads: int
    kv_heads: int
    head_dim: int
    block_size: int | None
    causal: bool
    seed: int


@triton.jit
def _triton_qk_scores_kernel(
    q_ptr,
    k_ptr,
    scores_ptr,
    softmax_scale: tl.constexpr,
    q_stride_b: tl.constexpr,
    q_stride_m: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_d: tl.constexpr,
    k_stride_b: tl.constexpr,
    k_stride_n: tl.constexpr,
    k_stride_h: tl.constexpr,
    k_stride_d: tl.constexpr,
    H: tl.constexpr,
    H_KV: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_bh = tl.program_id(2)

    batch = pid_bh // H
    q_head = pid_bh - batch * H
    kv_group = H // H_KV
    kv_head = q_head // kv_group

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    q_offsets = (
        batch * q_stride_b
        + offs_m[:, None] * q_stride_m
        + q_head * q_stride_h
        + offs_d[None, :] * q_stride_d
    )
    k_offsets = (
        batch * k_stride_b
        + offs_n[None, :] * k_stride_n
        + kv_head * k_stride_h
        + offs_d[:, None] * k_stride_d
    )
    q = tl.load(
        q_ptr + q_offsets,
        mask=(offs_m[:, None] < M) & (offs_d[None, :] < D),
        other=0.0,
    )
    k = tl.load(
        k_ptr + k_offsets,
        mask=(offs_n[None, :] < N) & (offs_d[:, None] < D),
        other=0.0,
    )
    scores = tl.dot(q, k) * softmax_scale
    valid = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    if CAUSAL:
        q_pos = offs_m[:, None] + tl.maximum(N - M, 0)
        valid = valid & (offs_n[None, :] <= q_pos)
    scores = tl.where(valid, scores, -1.0e30)

    out_offsets = ((pid_bh * M + offs_m[:, None]) * N) + offs_n[None, :]
    tl.store(scores_ptr + out_offsets, scores, mask=valid)


@triton.jit
def _triton_qk_lse_kernel(
    q_ptr,
    k_ptr,
    lse_ptr,
    softmax_scale: tl.constexpr,
    q_stride_b: tl.constexpr,
    q_stride_m: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_d: tl.constexpr,
    k_stride_b: tl.constexpr,
    k_stride_n: tl.constexpr,
    k_stride_h: tl.constexpr,
    k_stride_d: tl.constexpr,
    H: tl.constexpr,
    H_KV: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    batch = pid_bh // H
    q_head = pid_bh - batch * H
    kv_group = H // H_KV
    kv_head = q_head // kv_group

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    q_offsets = (
        batch * q_stride_b
        + offs_m[:, None] * q_stride_m
        + q_head * q_stride_h
        + offs_d[None, :] * q_stride_d
    )
    q = tl.load(
        q_ptr + q_offsets,
        mask=(offs_m[:, None] < M) & (offs_d[None, :] < D),
        other=0.0,
    )

    row_max = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    row_sum = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    for start_n in range(0, N, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k_offsets = (
            batch * k_stride_b
            + offs_n[None, :] * k_stride_n
            + kv_head * k_stride_h
            + offs_d[:, None] * k_stride_d
        )
        k = tl.load(
            k_ptr + k_offsets,
            mask=(offs_n[None, :] < N) & (offs_d[:, None] < D),
            other=0.0,
        )
        scores = tl.dot(q, k) * softmax_scale
        valid = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        if CAUSAL:
            q_pos = offs_m[:, None] + tl.maximum(N - M, 0)
            valid = valid & (offs_n[None, :] <= q_pos)
        scores = tl.where(valid, scores, -float("inf"))

        new_max = tl.maximum(row_max, tl.max(scores, axis=1))
        new_max = tl.where(new_max > -float("inf"), new_max, 0.0)
        probs = tl.exp(scores - new_max[:, None])
        tile_sum = tl.sum(probs, axis=1)
        alpha = tl.exp(row_max - new_max)
        row_sum = row_sum * alpha + tile_sum
        row_max = new_max

    out = row_max + tl.log(row_sum)
    out_offsets = pid_bh * M + offs_m
    tl.store(lse_ptr + out_offsets, out, mask=offs_m < M)


@triton.jit
def _triton_single_key_probability_kernel(
    q_ptr,
    k_ptr,
    out_ptr,
    key_idx,
    softmax_scale: tl.constexpr,
    q_stride_b: tl.constexpr,
    q_stride_m: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_d: tl.constexpr,
    k_stride_b: tl.constexpr,
    k_stride_n: tl.constexpr,
    k_stride_h: tl.constexpr,
    k_stride_d: tl.constexpr,
    H: tl.constexpr,
    H_KV: tl.constexpr,
    M_TOTAL: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    batch = pid_bh // H
    q_head = pid_bh - batch * H
    kv_group = H // H_KV
    kv_head = q_head // kv_group

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    q_offsets = (
        batch * q_stride_b
        + offs_m[:, None] * q_stride_m
        + q_head * q_stride_h
        + offs_d[None, :] * q_stride_d
    )
    q = tl.load(
        q_ptr + q_offsets,
        mask=(offs_m[:, None] < M_TOTAL) & (offs_d[None, :] < D),
        other=0.0,
    )

    M_run = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    L_run = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M], dtype=tl.float32)

    for start_n in range(0, N, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k_offsets = (
            batch * k_stride_b
            + offs_n[None, :] * k_stride_n
            + kv_head * k_stride_h
            + offs_d[:, None] * k_stride_d
        )
        k = tl.load(
            k_ptr + k_offsets,
            mask=(offs_n[None, :] < N) & (offs_d[:, None] < D),
            other=0.0,
        )
        scores = tl.dot(q, k) * softmax_scale
        valid = (offs_m[:, None] < M_TOTAL) & (offs_n[None, :] < N)
        if CAUSAL:
            q_pos = offs_m[:, None] + tl.maximum(N - M_TOTAL, 0)
            valid = valid & (offs_n[None, :] <= q_pos)
        scores = tl.where(valid, scores, -float("inf"))

        M_run, L_run, P, alpha = softmax_step(scores, M_run, L_run)
        acc = acc * alpha
        v = tl.where(offs_n == key_idx, 1.0, 0.0).to(tl.float16)
        acc += tl.sum(P.to(tl.float16).to(tl.float32) * v[None, :], axis=1)

    prob = tl.where(L_run == 0.0, 0.0, acc / L_run)
    out_offsets = (batch * M_TOTAL + offs_m) * H + q_head
    tl.store(out_ptr + out_offsets, prob, mask=offs_m < M_TOTAL)


_SOURCE_SUFFIXES = {
    ".cpp",
    ".cu",
    ".cuh",
    ".h",
    ".hpp",
    ".py",
    ".toml",
}
_SOURCE_EXCLUDE_PARTS = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    "build",
}


def _is_source_file(path: Path) -> bool:
    if path.name.endswith((".so", ".pyc", ".pyo")):
        return False
    if path.name.endswith(".egg-info"):
        return False
    return path.suffix in _SOURCE_SUFFIXES or path.name in {"setup.py"}


def _iter_source_files(root: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if any(part in _SOURCE_EXCLUDE_PARTS for part in rel.parts):
            continue
        if not path.is_file() or not _is_source_file(path):
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        files[str(rel)] = digest
    return files


def run_source_parity(reference_dir: Path, candidate_dir: Path) -> dict[str, Any]:
    reference_dir = reference_dir.resolve()
    candidate_dir = candidate_dir.resolve()
    if not reference_dir.is_dir():
        raise FileNotFoundError(f"reference source dir not found: {reference_dir}")
    if not candidate_dir.is_dir():
        raise FileNotFoundError(f"candidate source dir not found: {candidate_dir}")

    reference = _iter_source_files(reference_dir)
    candidate = _iter_source_files(candidate_dir)
    all_paths = sorted(set(reference) | set(candidate))
    mismatched = [
        path
        for path in all_paths
        if path in reference
        and path in candidate
        and reference[path] != candidate[path]
    ]
    reference_only = [path for path in all_paths if path not in candidate]
    candidate_only = [path for path in all_paths if path not in reference]
    kernel_paths = [
        path
        for path in all_paths
        if path.startswith("kernel/") or path.startswith("include/")
    ]
    kernel_mismatched = [path for path in mismatched if path in kernel_paths]
    kernel_reference_only = [
        path for path in reference_only if path in kernel_paths
    ]
    kernel_candidate_only = [
        path for path in candidate_only if path in kernel_paths
    ]
    return {
        "reference_dir": str(reference_dir),
        "candidate_dir": str(candidate_dir),
        "reference_file_count": len(reference),
        "candidate_file_count": len(candidate),
        "passed": not mismatched and not reference_only and not candidate_only,
        "kernel_passed": (
            not kernel_mismatched
            and not kernel_reference_only
            and not kernel_candidate_only
        ),
        "kernel_mismatched": kernel_mismatched,
        "kernel_reference_only": kernel_reference_only,
        "kernel_candidate_only": kernel_candidate_only,
        "mismatched": mismatched,
        "reference_only": reference_only,
        "candidate_only": candidate_only,
        "reference_sha256": reference,
        "candidate_sha256": candidate,
    }


def _diff_stats(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    diff = (actual - expected).abs()
    finite = diff[torch.isfinite(diff)]
    if finite.numel() == 0:
        return float("nan"), float("nan")
    return float(finite.max().item()), float(finite.mean().item())


def _mismatch_stats(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    mismatch = torch.ne(actual, expected)
    num_different = int(mismatch.sum().item())
    first_mismatch = None
    if num_different:
        first_mismatch = int(torch.nonzero(mismatch.flatten(), as_tuple=False)[0])

    diff = (actual - expected).abs().flatten()
    if diff.numel() == 0:
        return {
            "num_different": num_different,
            "first_mismatch_flat_index": first_mismatch,
            "max_diff_flat_index": None,
            "actual_at_max_diff": None,
            "expected_at_max_diff": None,
        }
    max_idx = int(torch.argmax(diff).item())
    return {
        "num_different": num_different,
        "first_mismatch_flat_index": first_mismatch,
        "max_diff_flat_index": max_idx,
        "actual_at_max_diff": float(actual.flatten()[max_idx].item()),
        "expected_at_max_diff": float(expected.flatten()[max_idx].item()),
    }


def _ref_gqa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    causal: bool,
) -> torch.Tensor:
    q_heads = q.shape[2]
    kv_heads = k.shape[2]
    if q_heads != kv_heads:
        if q_heads % kv_heads != 0:
            raise ValueError(
                f"q_heads={q_heads} is not divisible by kv_heads={kv_heads}"
            )
        repeat = q_heads // kv_heads
        k = k.repeat_interleave(repeat, dim=2)
        v = v.repeat_interleave(repeat, dim=2)

    q_ref = q.float().transpose(1, 2)
    k_ref = k.float().transpose(1, 2)
    v_ref = v.float().transpose(1, 2)
    query_len = q.shape[1]
    seq_len = k.shape[1]
    scores = torch.matmul(q_ref, k_ref.transpose(-1, -2)) * softmax_scale
    if causal:
        q_pos = torch.arange(query_len, device=q.device) + max(seq_len - query_len, 0)
        k_pos = torch.arange(seq_len, device=q.device)
        mask = k_pos.view(1, 1, 1, seq_len) > q_pos.view(1, 1, query_len, 1)
        scores = scores.masked_fill(mask, float("-inf"))
    out = torch.matmul(torch.softmax(scores, dim=-1), v_ref)
    return out.transpose(1, 2).to(torch.float16)


def _repeat_kv_for_gqa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_heads = q.shape[2]
    kv_heads = k.shape[2]
    if q_heads == kv_heads:
        return k, v
    if q_heads % kv_heads != 0:
        raise ValueError(f"q_heads={q_heads} is not divisible by kv_heads={kv_heads}")
    repeat = q_heads // kv_heads
    return k.repeat_interleave(repeat, dim=2), v.repeat_interleave(repeat, dim=2)


def _ref_gqa_attention_p_half(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    causal: bool,
) -> torch.Tensor:
    """Diagnostic reference: full softmax, then FP16 P before P @ V."""
    k, v = _repeat_kv_for_gqa(q, k, v)
    q_ref = q.float().transpose(1, 2)
    k_ref = k.float().transpose(1, 2)
    v_ref = v.transpose(1, 2)
    query_len = q.shape[1]
    seq_len = k.shape[1]
    scores = torch.matmul(q_ref, k_ref.transpose(-1, -2)) * softmax_scale
    if causal:
        q_pos = torch.arange(query_len, device=q.device) + max(seq_len - query_len, 0)
        k_pos = torch.arange(seq_len, device=q.device)
        mask = k_pos.view(1, 1, 1, seq_len) > q_pos.view(1, 1, query_len, 1)
        scores = scores.masked_fill(mask, float("-inf"))
    p = torch.softmax(scores, dim=-1).to(torch.float16)
    out = torch.matmul(p, v_ref)
    return out.transpose(1, 2).to(torch.float16)


def _ref_gqa_attention_online(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    causal: bool,
    *,
    tile_size: int,
    p_dtype: torch.dtype,
    clamp_exp_min: float | None = None,
) -> torch.Tensor:
    """Diagnostic online-softmax reference with configurable tile order.

    This is not an acceptance oracle. It helps split whether Flash-V100 is
    closer to latest Triton's tile/update shape or to the current Flash family.
    """
    k, v = _repeat_kv_for_gqa(q, k, v)
    q_ref = q.float().transpose(1, 2)
    k_ref = k.float().transpose(1, 2)
    v_ref = v.transpose(1, 2)
    batch, q_heads, query_len, head_dim = q_ref.shape
    seq_len = k.shape[1]
    acc = torch.zeros(
        (batch, q_heads, query_len, head_dim),
        device=q.device,
        dtype=torch.float32,
    )
    row_max = torch.full(
        (batch, q_heads, query_len),
        -float("inf"),
        device=q.device,
        dtype=torch.float32,
    )
    row_sum = torch.zeros_like(row_max)
    q_pos = torch.arange(query_len, device=q.device) + max(seq_len - query_len, 0)
    k_pos_all = torch.arange(seq_len, device=q.device)

    for tile_start in range(0, seq_len, tile_size):
        tile_end = min(tile_start + tile_size, seq_len)
        k_tile = k_ref[:, :, tile_start:tile_end, :]
        scores = torch.matmul(q_ref, k_tile.transpose(-1, -2)) * softmax_scale
        if causal:
            k_pos = k_pos_all[tile_start:tile_end]
            mask = k_pos.view(1, 1, 1, -1) > q_pos.view(1, 1, query_len, 1)
            scores = scores.masked_fill(mask, float("-inf"))

        tile_max = torch.max(scores, dim=-1).values
        new_max = torch.maximum(row_max, tile_max)
        new_max = torch.where(
            torch.isfinite(new_max), new_max, torch.zeros_like(new_max)
        )
        exp_arg = scores - new_max.unsqueeze(-1)
        if clamp_exp_min is not None:
            exp_arg = torch.maximum(exp_arg, torch.full_like(exp_arg, clamp_exp_min))
        p = torch.exp(exp_arg)
        alpha = torch.exp(row_max - new_max)
        alpha = torch.where(torch.isfinite(alpha), alpha, torch.zeros_like(alpha))
        acc = acc * alpha.unsqueeze(-1)
        v_tile = v_ref[:, :, tile_start:tile_end, :]
        if p_dtype is torch.float32:
            v_tile = v_tile.float()
        pv = torch.matmul(p.to(p_dtype), v_tile)
        acc = acc + pv.float()
        row_sum = row_sum * alpha + p.sum(dim=-1)
        row_max = new_max

    out = acc / torch.clamp(row_sum, min=1e-24).unsqueeze(-1)
    return out.transpose(1, 2).to(torch.float16)


def _ref_decode_steps(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    query_len = q.shape[1]
    seq_len = k.shape[1]
    outs = []
    for i in range(query_len):
        prefix_len = seq_len - query_len + i + 1
        outs.append(
            _ref_gqa_attention(
                q[:, i : i + 1],
                k[:, :prefix_len],
                v[:, :prefix_len],
                softmax_scale,
                causal=False,
            )
        )
    return torch.cat(outs, dim=1)


def _flash_decode_steps(
    flash_attn_v100: Any,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    query_len = q.shape[1]
    seq_len = k.shape[1]
    outs = []
    for i in range(query_len):
        prefix_len = seq_len - query_len + i + 1
        outs.append(
            flash_attn_v100.flash_attn_func(
                q[:, i : i + 1],
                k[:, :prefix_len],
                v[:, :prefix_len],
                softmax_scale=softmax_scale,
                causal=False,
            )
        )
    return torch.cat(outs, dim=1)


def _flash_partitioned_decode_reference(
    flash_attn_v100: Any,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    *,
    partition_size: int,
) -> torch.Tensor:
    """Flash-family matched partition reference for single-query decode."""
    if q.shape[0] != 1 or q.shape[1] != 1:
        raise ValueError("partitioned decode diagnostic expects q shape [1, 1, H, D]")
    seq_len = k.shape[1]
    part_outputs = []
    part_lses = []
    for start in range(0, seq_len, partition_size):
        end = min(start + partition_size, seq_len)
        k_part = k[:, start:end]
        v_part = v[:, start:end]
        part_outputs.append(
            flash_attn_v100.flash_attn_func(
                q,
                k_part,
                v_part,
                softmax_scale=softmax_scale,
                causal=False,
            ).squeeze(0).squeeze(0)
        )
        lse = flash_attn_v100.flash_attn_lse(
            q,
            k_part,
            v_part,
            softmax_scale=softmax_scale,
            causal=False,
        ).float()
        if lse.dim() == 3:
            part_lses.append(lse[0, :, 0])
        elif lse.dim() == 2:
            part_lses.append(lse[0])
        else:
            raise ValueError(f"Unexpected Flash LSE shape: {list(lse.shape)}")

    outputs = torch.stack(part_outputs, dim=0).float()
    lses = torch.stack(part_lses, dim=0)
    global_lse = torch.logsumexp(lses, dim=0)
    weights = torch.exp(lses - global_lse.unsqueeze(0))
    acc = torch.zeros_like(outputs[0], dtype=torch.float32)
    for idx in range(outputs.shape[0]):
        acc += weights[idx].view(-1, 1) * outputs[idx]
    return acc.unsqueeze(0).unsqueeze(0).to(torch.float16)


def _decode_workspace_reduce_reference(
    tmp_out: torch.Tensor,
    max_logits: torch.Tensor,
    exp_sums: torch.Tensor,
    *,
    seq_len: int,
    partition_size: int,
) -> torch.Tensor:
    num_partitions = (seq_len + partition_size - 1) // partition_size
    tmp = tmp_out[0, :, :num_partitions, :].float()
    maxes = max_logits[0, :, :num_partitions].float()
    sums = exp_sums[0, :, :num_partitions].float()
    global_max = torch.max(maxes, dim=-1).values
    weights = sums * torch.exp(maxes - global_max.unsqueeze(-1))
    global_sum = torch.sum(weights, dim=-1)
    acc = torch.zeros_like(tmp[:, 0, :], dtype=torch.float32)
    for idx in range(num_partitions):
        acc += weights[:, idx].view(-1, 1) * tmp[:, idx, :]
    out = acc / torch.clamp(global_sum, min=1e-24).view(-1, 1)
    return out.unsqueeze(0).unsqueeze(0).to(torch.float16)


def _flash_dense_partition_outputs(
    flash_attn_v100: Any,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    *,
    partition_size: int,
) -> torch.Tensor:
    outputs = []
    for start in range(0, k.shape[1], partition_size):
        end = min(start + partition_size, k.shape[1])
        out = flash_attn_v100.flash_attn_func(
            q,
            k[:, start:end],
            v[:, start:end],
            softmax_scale=softmax_scale,
            causal=False,
        )
        outputs.append(out.squeeze(0).squeeze(0))
    return torch.stack(outputs, dim=1)


def _flash_dense_partition_qk_scores(
    flash_attn_v100: Any,
    q: torch.Tensor,
    k: torch.Tensor,
    softmax_scale: float,
    *,
    partition_size: int,
) -> torch.Tensor:
    scores = flash_attn_v100.flash_attn_qk_scores(
        q,
        k,
        softmax_scale=softmax_scale,
        causal=False,
    )
    num_partitions = (k.shape[1] + partition_size - 1) // partition_size
    return scores[0, :, 0, :].reshape(q.shape[2], num_partitions, partition_size)


def _partition_outputs_from_qk_scores(
    qk_scores: torch.Tensor,
    v: torch.Tensor,
    *,
    partition_size: int,
    q_heads: int,
    kv_heads: int,
) -> torch.Tensor:
    if q_heads % kv_heads != 0:
        raise ValueError(f"q_heads={q_heads} is not divisible by kv_heads={kv_heads}")
    repeat = q_heads // kv_heads
    v_heads = v.squeeze(0).repeat_interleave(repeat, dim=1).transpose(0, 1)
    outputs = []
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for part_idx in range(qk_scores.shape[1]):
            start = part_idx * partition_size
            end = min(start + partition_size, v.shape[1])
            valid = end - start
            if valid <= 0:
                break
            scores = qk_scores[:, part_idx, :valid].float()
            row_max = torch.max(scores, dim=-1).values
            probs = torch.exp(scores - row_max.unsqueeze(-1))
            row_sum = torch.sum(probs, dim=-1)
            v_part = v_heads[:, start:end, :].float()
            acc = torch.bmm(probs.unsqueeze(1), v_part).squeeze(1)
            out = acc / torch.clamp(row_sum, min=1e-24).unsqueeze(-1)
            outputs.append(out.to(torch.float16))
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
    return torch.stack(outputs, dim=1)


def _outputs_from_qk_scores(
    qk_scores: torch.Tensor,
    v: torch.Tensor,
    *,
    q_heads: int,
    kv_heads: int,
    p_dtype: torch.dtype,
) -> torch.Tensor:
    if q_heads % kv_heads != 0:
        raise ValueError(f"q_heads={q_heads} is not divisible by kv_heads={kv_heads}")
    repeat = q_heads // kv_heads
    v_heads = v.repeat_interleave(repeat, dim=2).permute(0, 2, 1, 3)
    probs = torch.softmax(qk_scores.float(), dim=-1)
    if p_dtype is torch.float32:
        out = torch.matmul(probs, v_heads.float())
    else:
        out = torch.matmul(probs.to(p_dtype), v_heads)
    return out.permute(0, 2, 1, 3).to(torch.float16)


def _outputs_from_qk_scores_online(
    qk_scores: torch.Tensor,
    v: torch.Tensor,
    *,
    q_heads: int,
    kv_heads: int,
    tile_size: int,
    p_dtype: torch.dtype,
    clamp_exp_min: float | None = None,
) -> torch.Tensor:
    """Online-softmax/PV replay from precomputed QK scores.

    This keeps the QK source fixed and varies only the softmax/PV update order.
    It is a diagnostic for Type-B source proof, not a replacement oracle.
    """
    if q_heads % kv_heads != 0:
        raise ValueError(f"q_heads={q_heads} is not divisible by kv_heads={kv_heads}")
    repeat = q_heads // kv_heads
    v_heads = v.repeat_interleave(repeat, dim=2).permute(0, 2, 1, 3)
    batch, heads, query_len, seq_len = qk_scores.shape
    head_dim = v.shape[-1]
    acc = torch.zeros(
        (batch, heads, query_len, head_dim),
        device=qk_scores.device,
        dtype=torch.float32,
    )
    row_max = torch.full(
        (batch, heads, query_len),
        -float("inf"),
        device=qk_scores.device,
        dtype=torch.float32,
    )
    row_sum = torch.zeros_like(row_max)

    for tile_start in range(0, seq_len, tile_size):
        tile_end = min(tile_start + tile_size, seq_len)
        scores = qk_scores[..., tile_start:tile_end].float()
        tile_max = torch.max(scores, dim=-1).values
        new_max = torch.maximum(row_max, tile_max)
        new_max = torch.where(
            torch.isfinite(new_max), new_max, torch.zeros_like(new_max)
        )
        exp_arg = scores - new_max.unsqueeze(-1)
        if clamp_exp_min is not None:
            exp_arg = torch.maximum(exp_arg, torch.full_like(exp_arg, clamp_exp_min))
        p = torch.exp(exp_arg)
        alpha = torch.exp(row_max - new_max)
        alpha = torch.where(torch.isfinite(alpha), alpha, torch.zeros_like(alpha))
        acc = acc * alpha.unsqueeze(-1)
        v_tile = v_heads[:, :, tile_start:tile_end, :]
        if p_dtype is torch.float32:
            v_tile = v_tile.float()
        acc = acc + torch.matmul(p.to(p_dtype), v_tile).float()
        row_sum = row_sum * alpha + p.sum(dim=-1)
        row_max = new_max

    out = acc / torch.clamp(row_sum, min=1e-24).unsqueeze(-1)
    return out.permute(0, 2, 1, 3).to(torch.float16)


def _lse_from_qk_scores(qk_scores: torch.Tensor) -> torch.Tensor:
    return torch.logsumexp(qk_scores.float(), dim=-1)


def _unravel_flat_index(flat_index: int, shape: list[int]) -> list[int]:
    indices: list[int] = []
    remainder = flat_index
    for dim in reversed(shape):
        indices.append(remainder % dim)
        remainder //= dim
    return list(reversed(indices))


def _as_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def _single_qk_diff_detail(
    q: torch.Tensor,
    k: torch.Tensor,
    flash_qk_scores: torch.Tensor,
    triton_qk_scores: torch.Tensor,
    *,
    flat_index: int | None,
    softmax_scale: float,
    causal: bool,
) -> dict[str, Any] | None:
    if flat_index is None:
        return None

    shape = list(flash_qk_scores.shape)
    batch_idx, q_head_idx, query_idx, key_idx = _unravel_flat_index(
        flat_index,
        shape,
    )
    q_heads = q.shape[2]
    kv_heads = k.shape[2]
    kv_group = q_heads // kv_heads
    kv_head_idx = q_head_idx // kv_group
    causal_q_pos = query_idx + max(k.shape[1] - q.shape[1], 0)
    causal_valid = (not causal) or key_idx <= causal_q_pos

    q_vec = q[batch_idx, query_idx, q_head_idx, :]
    k_vec = k[batch_idx, key_idx, kv_head_idx, :]
    product_half = q_vec * k_vec
    q_float = q_vec.float()
    k_float = k_vec.float()
    fp32_dot = torch.sum(q_float * k_float, dtype=torch.float32)
    half_product_fp32_sum = torch.sum(product_half.float(), dtype=torch.float32)
    half_product_fp16_sum = torch.sum(product_half, dtype=torch.float16)

    chunk_acc = torch.zeros((), device=q.device, dtype=torch.float32)
    for start in range(0, q_vec.numel(), 16):
        end = min(start + 16, q_vec.numel())
        chunk = torch.sum(
            q_float[start:end] * k_float[start:end],
            dtype=torch.float32,
        )
        chunk_acc = chunk_acc + chunk

    flash_value = flash_qk_scores[
        batch_idx,
        q_head_idx,
        query_idx,
        key_idx,
    ]
    triton_value = triton_qk_scores[
        batch_idx,
        q_head_idx,
        query_idx,
        key_idx,
    ]
    references = {
        "torch_fp32_sum_scaled": _as_float(fp32_dot * softmax_scale),
        "torch_half_product_fp32_sum_scaled": _as_float(
            half_product_fp32_sum * softmax_scale
        ),
        "torch_half_product_fp16_sum_scaled": _as_float(
            half_product_fp16_sum.float() * softmax_scale
        ),
        "torch_k16_chunk_fp32_sum_scaled": _as_float(
            chunk_acc * softmax_scale
        ),
    }
    flash_float = _as_float(flash_value)
    triton_float = _as_float(triton_value)
    return {
        "flat_index": flat_index,
        "score_shape": shape,
        "decoded_index": {
            "batch": batch_idx,
            "q_head": q_head_idx,
            "kv_head": kv_head_idx,
            "query": query_idx,
            "key": key_idx,
            "head_dim": int(q_vec.numel()),
        },
        "causal": causal,
        "causal_q_pos": causal_q_pos,
        "causal_valid": causal_valid,
        "flash_score": flash_float,
        "triton_score": triton_float,
        "flash_minus_triton": flash_float - triton_float,
        "references": references,
        "abs_diff_to_flash": {
            name: abs(value - flash_float) for name, value in references.items()
        },
        "abs_diff_to_triton": {
            name: abs(value - triton_float) for name, value in references.items()
        },
        "q_abs_max": _as_float(q_vec.abs().max()),
        "k_abs_max": _as_float(k_vec.abs().max()),
        "product_abs_max": _as_float(product_half.abs().max()),
    }


def _half_neighbor_detail(left: float, right: float) -> dict[str, Any]:
    left_half = torch.tensor(left, dtype=torch.float16)
    right_half = torch.tensor(right, dtype=torch.float16)
    pos_inf = torch.tensor(float("inf"), dtype=torch.float16)
    neg_inf = torch.tensor(-float("inf"), dtype=torch.float16)
    right_next_up = torch.nextafter(right_half, pos_inf)
    left_next_down = torch.nextafter(left_half, neg_inf)
    return {
        "left": float(left_half.item()),
        "right": float(right_half.item()),
        "abs_diff": abs(float(left_half.item()) - float(right_half.item())),
        "right_next_up": float(right_next_up.item()),
        "right_next_up_delta": float((right_next_up - right_half).item()),
        "left_next_down": float(left_next_down.item()),
        "left_next_down_delta": float((left_half - left_next_down).item()),
        "is_one_half_ulp": bool(
            torch.equal(right_next_up, left_half)
            or torch.equal(left_next_down, right_half)
        ),
    }


def _single_output_diff_detail(
    v: torch.Tensor,
    saved_flash: torch.Tensor,
    saved_triton: torch.Tensor,
    flash_qk_scores: torch.Tensor,
    triton_qk_scores: torch.Tensor,
    qk_output_refs: dict[str, torch.Tensor],
    lse_refs: dict[str, torch.Tensor] | None = None,
    *,
    flat_index: int | None,
    causal: bool,
) -> dict[str, Any] | None:
    if flat_index is None:
        return None

    shape = list(saved_flash.shape)
    batch_idx, query_idx, q_head_idx, dim_idx = _unravel_flat_index(
        flat_index,
        shape,
    )
    q_heads = flash_qk_scores.shape[1]
    kv_heads = v.shape[2]
    kv_group = q_heads // kv_heads
    kv_head_idx = q_head_idx // kv_group
    seq_len = flash_qk_scores.shape[-1]
    query_len = flash_qk_scores.shape[-2]
    causal_q_pos = query_idx + max(seq_len - query_len, 0)
    valid_end = seq_len if not causal else min(seq_len, causal_q_pos + 1)

    flash_row = flash_qk_scores[batch_idx, q_head_idx, query_idx, :valid_end]
    triton_row = triton_qk_scores[batch_idx, q_head_idx, query_idx, :valid_end]
    values = v[batch_idx, :valid_end, kv_head_idx, dim_idx]
    flash_probs = torch.softmax(flash_row.float(), dim=-1)
    triton_probs = torch.softmax(triton_row.float(), dim=-1)

    def reduce_row(probs: torch.Tensor) -> dict[str, float]:
        fp32 = torch.sum(probs.float() * values.float(), dtype=torch.float32)
        p_half = torch.sum(
            probs.to(torch.float16).float() * values.float(),
            dtype=torch.float32,
        )
        pv_half = torch.sum(
            probs.to(torch.float16) * values.to(torch.float16),
            dtype=torch.float16,
        )
        return {
            "pv_fp32": _as_float(fp32),
            "p_half_v_fp32": _as_float(p_half),
            "p_half_v_half_sum": _as_float(pv_half.float()),
            "pv_fp32_as_half": _as_float(fp32.to(torch.float16)),
        }

    def top_keys(scores: torch.Tensor, probs: torch.Tensor) -> list[dict[str, float]]:
        count = min(4, int(scores.numel()))
        indices = torch.topk(probs, k=count).indices.tolist()
        return [
            {
                "key": int(idx),
                "score": _as_float(scores[idx]),
                "probability": _as_float(probs[idx]),
                "value": _as_float(values[idx]),
                "contribution": _as_float(probs[idx].float() * values[idx].float()),
            }
            for idx in indices
        ]

    prob_diff = (flash_probs - triton_probs).abs()
    score_diff = (flash_row - triton_row).abs()
    max_prob_idx = int(torch.argmax(prob_diff).item()) if prob_diff.numel() else None
    max_score_idx = int(torch.argmax(score_diff).item()) if score_diff.numel() else None
    flash_value = _as_float(saved_flash.flatten()[flat_index])
    triton_value = _as_float(saved_triton.flatten()[flat_index])
    return {
        "flat_index": flat_index,
        "output_shape": shape,
        "decoded_index": {
            "batch": batch_idx,
            "query": query_idx,
            "q_head": q_head_idx,
            "kv_head": kv_head_idx,
            "dim": dim_idx,
        },
        "causal": causal,
        "causal_q_pos": causal_q_pos,
        "valid_key_count": valid_end,
        "saved_flash_value": flash_value,
        "saved_triton_value": triton_value,
        "flash_minus_triton": flash_value - triton_value,
        "half_neighbor": _half_neighbor_detail(flash_value, triton_value),
        "qk_output_refs_at_index": {
            name: _as_float(tensor.flatten()[flat_index])
            for name, tensor in qk_output_refs.items()
        },
        "lse_refs_at_row": {}
        if lse_refs is None
        else {
            name: _as_float(tensor[batch_idx, q_head_idx, query_idx])
            for name, tensor in lse_refs.items()
        },
        "flash_row_reduce": reduce_row(flash_probs),
        "triton_row_reduce": reduce_row(triton_probs),
        "max_score_diff": {
            "key": max_score_idx,
            "abs_diff": _as_float(score_diff[max_score_idx])
            if max_score_idx is not None
            else None,
            "flash_score": _as_float(flash_row[max_score_idx])
            if max_score_idx is not None
            else None,
            "triton_score": _as_float(triton_row[max_score_idx])
            if max_score_idx is not None
            else None,
        },
        "max_probability_diff": {
            "key": max_prob_idx,
            "abs_diff": _as_float(prob_diff[max_prob_idx])
            if max_prob_idx is not None
            else None,
            "flash_probability": _as_float(flash_probs[max_prob_idx])
            if max_prob_idx is not None
            else None,
            "triton_probability": _as_float(triton_probs[max_prob_idx])
            if max_prob_idx is not None
            else None,
        },
        "top_keys_by_flash_probability": top_keys(flash_row, flash_probs),
        "top_keys_by_triton_probability": top_keys(triton_row, triton_probs),
    }


def _make_paged_cache(
    k: torch.Tensor,
    v: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    seq_len = k.shape[1]
    num_blocks = (seq_len + block_size - 1) // block_size
    key_cache = torch.zeros(
        (num_blocks, block_size, k.shape[2], k.shape[3]),
        device=k.device,
        dtype=k.dtype,
    )
    value_cache = torch.zeros_like(key_cache)
    for block_idx in range(num_blocks):
        start = block_idx * block_size
        end = min(start + block_size, seq_len)
        key_cache[block_idx, : end - start] = k[0, start:end]
        value_cache[block_idx, : end - start] = v[0, start:end]
    block_table = torch.arange(
        num_blocks, device=k.device, dtype=torch.int32
    ).view(1, num_blocks)
    seq_lens = torch.tensor([seq_len], device=k.device, dtype=torch.int32)
    return key_cache, value_cache, block_table, seq_lens


def _make_batched_paged_cache(
    k_values: list[torch.Tensor],
    v_values: list[torch.Tensor],
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    seq_lens_host = [int(k.shape[1]) for k in k_values]
    blocks_per_seq = [(seq_len + block_size - 1) // block_size
                      for seq_len in seq_lens_host]
    total_blocks = sum(blocks_per_seq)
    max_blocks = max(blocks_per_seq)
    device = k_values[0].device
    dtype = k_values[0].dtype
    kv_heads = k_values[0].shape[2]
    head_dim = k_values[0].shape[3]
    key_cache = torch.zeros(
        (total_blocks, block_size, kv_heads, head_dim),
        device=device,
        dtype=dtype,
    )
    value_cache = torch.zeros_like(key_cache)
    block_table = torch.zeros(
        (len(k_values), max_blocks),
        device=device,
        dtype=torch.int32,
    )

    physical_block = 0
    for batch_idx, (k, v, num_blocks) in enumerate(
        zip(k_values, v_values, blocks_per_seq)
    ):
        for logical_block in range(num_blocks):
            start = logical_block * block_size
            end = min(start + block_size, k.shape[1])
            key_cache[physical_block, : end - start] = k[0, start:end]
            value_cache[physical_block, : end - start] = v[0, start:end]
            block_table[batch_idx, logical_block] = physical_block
            physical_block += 1

    seq_lens = torch.tensor(seq_lens_host, device=device, dtype=torch.int32)
    return key_cache, value_cache, block_table, seq_lens


def _strided_view_along_dim(x: torch.Tensor, dim: int) -> torch.Tensor:
    padded_shape = list(x.shape)
    padded_shape[dim] *= 2
    padded = torch.empty(
        padded_shape,
        device=x.device,
        dtype=x.dtype,
    )
    index = [slice(None)] * x.dim()
    index[dim] = slice(0, None, 2)
    view = padded[tuple(index)]
    view.copy_(x)
    return view


def _triton_unified_attention_varlen(
    q_flat: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    query_lens: list[int],
    softmax_scale: float,
    *,
    causal: bool,
) -> torch.Tensor:
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention
    from vllm.v1.kv_cache_interface import KVQuantMode

    q_flat = q_flat.contiguous()
    out = torch.empty_like(q_flat)
    cu_seqlens = [0]
    for query_len in query_lens:
        cu_seqlens.append(cu_seqlens[-1] + query_len)
    query_start_loc = torch.tensor(
        cu_seqlens,
        device=q_flat.device,
        dtype=torch.int32,
    )
    unified_attention(
        q=q_flat,
        k=key_cache,
        v=value_cache,
        out=out,
        cu_seqlens_q=query_start_loc,
        max_seqlen_q=max(query_lens),
        seqused_k=seq_lens,
        max_seqlen_k=int(seq_lens.max().item()),
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=(-1, -1),
        block_table=block_table,
        softcap=0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        kv_quant_mode=KVQuantMode.NONE,
    )
    return out


def _flash_paged_loop_flat(
    flash_attn_v100: Any,
    q_values: list[torch.Tensor],
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    softmax_scale: float,
    *,
    causal: bool,
) -> torch.Tensor:
    chunks = []
    for idx, q in enumerate(q_values):
        out = flash_attn_v100.flash_attn_prefill_paged(
            q,
            key_cache,
            value_cache,
            block_table[idx : idx + 1],
            seq_lens[idx : idx + 1],
            softmax_scale=softmax_scale,
            causal=causal,
        )
        chunks.append(out.squeeze(0))
    return torch.cat(chunks, dim=0)


def _flash_paged_standalone_loop_flat(
    flash_attn_v100: Any,
    q_values: list[torch.Tensor],
    k_values: list[torch.Tensor],
    v_values: list[torch.Tensor],
    block_size: int,
    softmax_scale: float,
    *,
    causal: bool,
) -> torch.Tensor:
    chunks = []
    for q, k, v in zip(q_values, k_values, v_values):
        key_cache, value_cache, block_table, seq_lens = _make_paged_cache(
            k, v, block_size
        )
        out = flash_attn_v100.flash_attn_prefill_paged(
            q,
            key_cache,
            value_cache,
            block_table,
            seq_lens,
            softmax_scale=softmax_scale,
            causal=causal,
        )
        chunks.append(out.squeeze(0))
    return torch.cat(chunks, dim=0)


def _flash_dense_loop_flat(
    flash_attn_v100: Any,
    q_values: list[torch.Tensor],
    k_values: list[torch.Tensor],
    v_values: list[torch.Tensor],
    softmax_scale: float,
    *,
    causal: bool,
) -> torch.Tensor:
    chunks = []
    for q, k, v in zip(q_values, k_values, v_values):
        out = flash_attn_v100.flash_attn_func(
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            causal=causal,
        )
        chunks.append(out.squeeze(0))
    return torch.cat(chunks, dim=0)


def _triton_loop_flat(
    q_values: list[torch.Tensor],
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    softmax_scale: float,
    *,
    causal: bool,
) -> torch.Tensor:
    chunks = []
    for idx, q in enumerate(q_values):
        out = _triton_unified_attention(
            q,
            key_cache,
            value_cache,
            block_table[idx : idx + 1],
            seq_lens[idx : idx + 1],
            softmax_scale,
            causal=causal,
        )
        chunks.append(out.squeeze(0))
    return torch.cat(chunks, dim=0)


def _record_result(
    name: str,
    reference: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    seq_len: int,
    query_len: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    block_size: int | None,
    causal: bool,
    seed: int,
) -> CaseResult:
    torch.cuda.synchronize()
    equal = torch.equal(actual, expected)
    max_diff, mean_diff = _diff_stats(actual, expected)
    mismatch = _mismatch_stats(actual, expected)
    return CaseResult(
        name=name,
        reference=reference,
        equal=equal,
        max_diff=max_diff,
        mean_diff=mean_diff,
        num_different=mismatch["num_different"],
        first_mismatch_flat_index=mismatch["first_mismatch_flat_index"],
        max_diff_flat_index=mismatch["max_diff_flat_index"],
        actual_at_max_diff=mismatch["actual_at_max_diff"],
        expected_at_max_diff=mismatch["expected_at_max_diff"],
        actual_nan_count=int(torch.isnan(actual).sum().item()),
        expected_nan_count=int(torch.isnan(expected).sum().item()),
        shape=list(actual.shape),
        dtype=str(actual.dtype),
        seq_len=seq_len,
        query_len=query_len,
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        causal=causal,
        seed=seed,
    )


def _make_qkv(
    *,
    seq_len: int,
    query_len: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    device = torch.device("cuda")
    q = torch.randn(
        (1, query_len, q_heads, head_dim),
        device=device,
        dtype=torch.float16,
    )
    k = torch.randn(
        (1, seq_len, kv_heads, head_dim),
        device=device,
        dtype=torch.float16,
    )
    v = torch.randn_like(k)
    return q, k, v


def _triton_qk_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    softmax_scale: float,
    causal: bool,
) -> torch.Tensor:
    q = q.contiguous()
    k = k.contiguous()
    batch, query_len, q_heads, head_dim = q.shape
    seq_len = k.shape[1]
    kv_heads = k.shape[2]
    out = torch.full(
        (batch, q_heads, query_len, seq_len),
        -1.0e30,
        device=q.device,
        dtype=torch.float32,
    )
    block_m = 16 if q_heads // kv_heads <= 16 else triton.next_power_of_2(
        q_heads // kv_heads
    )
    block_n = 32
    grid = (
        triton.cdiv(query_len, block_m),
        triton.cdiv(seq_len, block_n),
        batch * q_heads,
    )
    _triton_qk_scores_kernel[grid](
        q,
        k,
        out,
        softmax_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        q_heads,
        kv_heads,
        query_len,
        seq_len,
        head_dim,
        causal,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=head_dim,
        num_warps=4,
    )
    return out


def _triton_qk_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    softmax_scale: float,
    causal: bool,
    *,
    tile_size: int,
) -> torch.Tensor:
    q = q.contiguous()
    k = k.contiguous()
    batch, query_len, q_heads, head_dim = q.shape
    seq_len = k.shape[1]
    kv_heads = k.shape[2]
    out = torch.empty(
        (batch, q_heads, query_len),
        device=q.device,
        dtype=torch.float32,
    )
    block_m = 16
    grid = (triton.cdiv(query_len, block_m), batch * q_heads)
    _triton_qk_lse_kernel[grid](
        q,
        k,
        out,
        softmax_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        q_heads,
        kv_heads,
        query_len,
        seq_len,
        head_dim,
        causal,
        BLOCK_M=block_m,
        BLOCK_N=tile_size,
        BLOCK_D=head_dim,
        num_warps=4,
    )
    return out


def _triton_single_key_probability(
    q: torch.Tensor,
    k: torch.Tensor,
    softmax_scale: float,
    causal: bool,
    *,
    key_idx: int,
    tile_size: int,
) -> torch.Tensor:
    q = q.contiguous()
    k = k.contiguous()
    batch, query_len, q_heads, head_dim = q.shape
    seq_len = k.shape[1]
    kv_heads = k.shape[2]
    out = torch.empty(
        (batch, query_len, q_heads),
        device=q.device,
        dtype=torch.float16,
    )
    block_m = 16
    grid = (triton.cdiv(query_len, block_m), batch * q_heads)
    _triton_single_key_probability_kernel[grid](
        q,
        k,
        out,
        key_idx,
        softmax_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        q_heads,
        kv_heads,
        query_len,
        seq_len,
        head_dim,
        causal,
        BLOCK_M=block_m,
        BLOCK_N=tile_size,
        BLOCK_D=head_dim,
        num_warps=4,
    )
    return out


def _torch_qk_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    softmax_scale: float,
    causal: bool,
) -> torch.Tensor:
    q_heads = q.shape[2]
    kv_heads = k.shape[2]
    if q_heads != kv_heads:
        if q_heads % kv_heads != 0:
            raise ValueError(
                f"q_heads={q_heads} is not divisible by kv_heads={kv_heads}"
            )
        k = k.repeat_interleave(q_heads // kv_heads, dim=2)
    q_ref = q.float().transpose(1, 2)
    k_ref = k.float().transpose(1, 2)
    query_len = q.shape[1]
    seq_len = k.shape[1]
    scores = torch.matmul(q_ref, k_ref.transpose(-1, -2)) * softmax_scale
    if causal:
        q_pos = torch.arange(query_len, device=q.device) + max(seq_len - query_len, 0)
        k_pos = torch.arange(seq_len, device=q.device)
        mask = k_pos.view(1, 1, 1, seq_len) > q_pos.view(1, 1, query_len, 1)
        scores = scores.masked_fill(mask, float("-inf"))
    return torch.logsumexp(scores, dim=-1)


def _simulate_online_single_key_probability(
    scores: torch.Tensor,
    *,
    key_idx: int,
    tile_size: int,
    p_half: bool,
    clamp_exp_min: float | None,
) -> torch.Tensor:
    old_max = torch.full_like(scores[..., 0], -float("inf"))
    old_sum = torch.zeros_like(old_max)
    numerator = torch.zeros_like(old_max)
    numerator_valid = False
    for start in range(0, scores.shape[-1], tile_size):
        end = min(start + tile_size, scores.shape[-1])
        tile = scores[..., start:end]
        row_max = torch.max(tile, dim=-1).values
        new_max = torch.maximum(old_max, row_max)
        new_max = torch.where(
            torch.isfinite(new_max), new_max, torch.zeros_like(new_max)
        )
        alpha = torch.exp(old_max - new_max)
        alpha = torch.where(torch.isfinite(alpha), alpha, torch.zeros_like(alpha))
        exp_arg = tile - new_max.unsqueeze(-1)
        if clamp_exp_min is not None:
            exp_arg = torch.clamp(exp_arg, min=clamp_exp_min)
        probs = torch.exp(exp_arg)
        if numerator_valid:
            numerator = numerator * alpha
        if start <= key_idx < end:
            key_prob = probs[..., key_idx - start]
            numerator = key_prob.to(torch.float16).float() if p_half else key_prob
            numerator_valid = True
        old_sum = old_sum * alpha + probs.sum(dim=-1)
        old_max = new_max
    out = numerator / torch.clamp(old_sum, min=1e-24)
    return out.transpose(1, 2).to(torch.float16)


def _unravel_index(flat_index: int, shape: tuple[int, ...]) -> list[int]:
    index: list[int] = []
    for size in reversed(shape):
        index.append(flat_index % size)
        flat_index //= size
    return list(reversed(index))


def _json_float(value: torch.Tensor | float) -> float | str:
    number = float(value.item()) if isinstance(value, torch.Tensor) else float(value)
    if math.isinf(number):
        return "inf" if number > 0 else "-inf"
    if math.isnan(number):
        return "nan"
    return number


def _trace_online_single_key_scalar(
    scores: torch.Tensor,
    *,
    key_idx: int,
    tile_size: int,
    p_half: bool,
    clamp_exp_min: float | None,
    output_index: list[int],
) -> dict[str, Any]:
    """Trace one scalar probability through the online-softmax update.

    ``scores`` is [B, H, Q, N], while the single-key output tensors used by the
    diagnostics are [B, Q, H].  ``output_index`` follows the output layout.
    """
    batch_idx, query_idx, head_idx = output_index
    row_scores = scores[batch_idx, head_idx, query_idx].float()
    old_max = torch.tensor(-float("inf"), device=scores.device, dtype=torch.float32)
    old_sum = torch.tensor(0.0, device=scores.device, dtype=torch.float32)
    numerator = torch.tensor(0.0, device=scores.device, dtype=torch.float32)
    numerator_valid = False
    trace = []
    for start in range(0, row_scores.shape[0], tile_size):
        end = min(start + tile_size, row_scores.shape[0])
        tile = row_scores[start:end]
        old_max_before = old_max
        row_max = tile.max()
        new_max = torch.maximum(old_max, row_max)
        if not bool(torch.isfinite(new_max).item()):
            new_max = torch.tensor(0.0, device=scores.device, dtype=torch.float32)
        alpha = torch.exp(old_max - new_max)
        if not bool(torch.isfinite(alpha).item()):
            alpha = torch.tensor(0.0, device=scores.device, dtype=torch.float32)
        exp_arg = tile - new_max
        if clamp_exp_min is not None:
            exp_arg = torch.clamp(exp_arg, min=clamp_exp_min)
        probs = torch.exp(exp_arg)
        probs_stored = probs.to(torch.float16).float() if p_half else probs
        tile_sum = probs.sum()
        numerator_before = numerator
        row_sum_before = old_sum
        if numerator_valid:
            numerator = numerator * alpha
        key_in_tile = start <= key_idx < end
        key_prob_float = None
        key_prob_stored = None
        if key_in_tile:
            local_key = key_idx - start
            key_prob_float = float(probs[local_key].item())
            key_prob_stored = float(probs_stored[local_key].item())
            numerator = probs_stored[local_key]
            numerator_valid = True
        old_sum = old_sum * alpha + tile_sum
        old_max = new_max
        probability_float = numerator / torch.clamp(old_sum, min=1e-24)
        trace.append(
            {
                "tile_start": start,
                "tile_end": end,
                "key_in_tile": key_in_tile,
                "old_max_before": _json_float(old_max_before),
                "row_max": _json_float(row_max),
                "new_max": _json_float(new_max),
                "alpha": _json_float(alpha),
                "row_sum_before": _json_float(row_sum_before),
                "tile_sum_float": _json_float(tile_sum),
                "row_sum_after": _json_float(old_sum),
                "numerator_before": _json_float(numerator_before),
                "numerator_after": _json_float(numerator),
                "key_prob_float": key_prob_float,
                "key_prob_stored": key_prob_stored,
                "probability_after_float": float(probability_float.item()),
                "probability_after_half": float(
                    probability_float.to(torch.float16).item()
                ),
            }
        )
    final = numerator / torch.clamp(old_sum, min=1e-24)
    return {
        "tile_size": tile_size,
        "p_half": p_half,
        "clamp_exp_min": clamp_exp_min,
        "output_index_b_q_h": output_index,
        "key_idx": key_idx,
        "final_probability_float": float(final.item()),
        "final_probability_half": float(final.to(torch.float16).item()),
        "trace": trace,
    }


def _triton_unified_attention(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    softmax_scale: float,
    *,
    causal: bool = True,
) -> torch.Tensor:
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention
    from vllm.v1.kv_cache_interface import KVQuantMode

    q_flat = q.squeeze(0).contiguous()
    out = torch.empty_like(q_flat)
    query_len = q_flat.shape[0]
    query_start_loc = torch.tensor(
        [0, query_len],
        device=q.device,
        dtype=torch.int32,
    )
    unified_attention(
        q=q_flat,
        k=key_cache,
        v=value_cache,
        out=out,
        cu_seqlens_q=query_start_loc,
        max_seqlen_q=query_len,
        seqused_k=seq_lens,
        max_seqlen_k=int(seq_lens.max().item()),
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=(-1, -1),
        block_table=block_table,
        softcap=0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        kv_quant_mode=KVQuantMode.NONE,
    )
    return out.unsqueeze(0)


def run_dense_prefill_cases(flash_attn_v100: Any) -> list[CaseResult]:
    cases = [
        (5, 5, 6, 1, 64, False),
        (9, 9, 6, 1, 256, True),
        (69, 16, 6, 1, 256, True),
        (181, 32, 6, 1, 128, True),
    ]
    results = []
    for seq_len, query_len, q_heads, kv_heads, head_dim, causal in cases:
        seed = seq_len * 1000 + query_len * 10 + head_dim + int(causal)
        q, k, v = _make_qkv(
            seq_len=seq_len,
            query_len=query_len,
            q_heads=q_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            seed=seed,
        )
        softmax_scale = 1.0 / math.sqrt(head_dim)
        actual = flash_attn_v100.flash_attn_func(
            q,
            k,
            v,
            causal=causal,
            softmax_scale=softmax_scale,
        )
        expected = _ref_gqa_attention(q, k, v, softmax_scale, causal)
        results.append(
            _record_result(
                "dense_prefill",
                "torch_float32",
                actual,
                expected,
                seq_len=seq_len,
                query_len=query_len,
                q_heads=q_heads,
                kv_heads=kv_heads,
                head_dim=head_dim,
                block_size=None,
                causal=causal,
                seed=seed,
            )
        )
    return results


def run_triton_compare_cases(flash_attn_v100: Any) -> list[CaseResult]:
    cases = [
        (9, 9, 1, 1, 256, 16, True),
        (9, 9, 2, 1, 256, 16, True),
        (9, 9, 4, 1, 256, 16, True),
        (9, 9, 6, 1, 256, 16, True),
        (69, 1, 6, 1, 256, 16, True),
        (69, 16, 6, 1, 256, 16, True),
        (181, 32, 6, 1, 128, 16, True),
        (4096, 1, 6, 1, 256, 832, True),
        (4096, 8, 6, 1, 256, 832, True),
    ]
    results = []
    for seq_len, query_len, q_heads, kv_heads, head_dim, block_size, causal in cases:
        seed = seq_len * 1000 + query_len * 10 + head_dim + block_size + 313
        q, k, v = _make_qkv(
            seq_len=seq_len,
            query_len=query_len,
            q_heads=q_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            seed=seed,
        )
        softmax_scale = 1.0 / math.sqrt(head_dim)
        key_cache, value_cache, block_table, seq_lens = _make_paged_cache(
            k, v, block_size
        )
        triton_out = _triton_unified_attention(
            q,
            key_cache,
            value_cache,
            block_table,
            seq_lens,
            softmax_scale,
        )
        dense_flash = flash_attn_v100.flash_attn_func(
            q,
            k,
            v,
            causal=causal,
            softmax_scale=softmax_scale,
        )
        paged_flash = flash_attn_v100.flash_attn_prefill_paged(
            q,
            key_cache,
            value_cache,
            block_table,
            seq_lens,
            softmax_scale=softmax_scale,
            causal=causal,
        )
        results.append(
            _record_result(
                "dense_flash_v100",
                "triton_unified_attention",
                dense_flash,
                triton_out,
                seq_len=seq_len,
                query_len=query_len,
                q_heads=q_heads,
                kv_heads=kv_heads,
                head_dim=head_dim,
                block_size=block_size,
                causal=causal,
                seed=seed,
            )
        )
        results.append(
            _record_result(
                "paged_prefill",
                "triton_unified_attention",
                paged_flash,
                triton_out,
                seq_len=seq_len,
                query_len=query_len,
                q_heads=q_heads,
                kv_heads=kv_heads,
                head_dim=head_dim,
                block_size=block_size,
                causal=causal,
                seed=seed,
            )
        )
    return results


def _compare_payload(
    name: str,
    reference: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> dict[str, Any]:
    max_diff, mean_diff = _diff_stats(actual, expected)
    mismatch = _mismatch_stats(actual, expected)
    return {
        "name": name,
        "reference": reference,
        "equal": bool(torch.equal(actual, expected)),
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "num_different": mismatch["num_different"],
        "first_mismatch_flat_index": mismatch["first_mismatch_flat_index"],
        "max_diff_flat_index": mismatch["max_diff_flat_index"],
        "actual_at_max_diff": mismatch["actual_at_max_diff"],
        "expected_at_max_diff": mismatch["expected_at_max_diff"],
    }


def run_arithmetic_diagnostic_cases(flash_attn_v100: Any) -> list[dict[str, Any]]:
    cases = [
        (9, 9, 1, 1, 256, 16, True),
        (9, 9, 6, 1, 256, 16, True),
        (69, 16, 6, 1, 256, 16, True),
        (4096, 8, 6, 1, 256, 832, True),
    ]
    diagnostics = []
    for seq_len, query_len, q_heads, kv_heads, head_dim, block_size, causal in cases:
        seed = seq_len * 1000 + query_len * 10 + head_dim + block_size + 313
        q, k, v = _make_qkv(
            seq_len=seq_len,
            query_len=query_len,
            q_heads=q_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            seed=seed,
        )
        softmax_scale = 1.0 / math.sqrt(head_dim)
        key_cache, value_cache, block_table, seq_lens = _make_paged_cache(
            k, v, block_size
        )
        triton_out = _triton_unified_attention(
            q,
            key_cache,
            value_cache,
            block_table,
            seq_lens,
            softmax_scale,
        )
        dense_flash = flash_attn_v100.flash_attn_func(
            q,
            k,
            v,
            causal=causal,
            softmax_scale=softmax_scale,
        )
        paged_flash = flash_attn_v100.flash_attn_prefill_paged(
            q,
            key_cache,
            value_cache,
            block_table,
            seq_lens,
            softmax_scale=softmax_scale,
            causal=causal,
        )
        variants = {
            "torch_full_softmax_fp32_pv_fp32": _ref_gqa_attention(
                q, k, v, softmax_scale, causal
            ),
            "torch_full_softmax_p_half": _ref_gqa_attention_p_half(
                q, k, v, softmax_scale, causal
            ),
            "torch_online_tile32_p_half": _ref_gqa_attention_online(
                q,
                k,
                v,
                softmax_scale,
                causal,
                tile_size=32,
                p_dtype=torch.float16,
            ),
            "torch_online_tile64_p_half": _ref_gqa_attention_online(
                q,
                k,
                v,
                softmax_scale,
                causal,
                tile_size=64,
                p_dtype=torch.float16,
            ),
            "torch_online_tile64_p_half_clamp80": _ref_gqa_attention_online(
                q,
                k,
                v,
                softmax_scale,
                causal,
                tile_size=64,
                p_dtype=torch.float16,
                clamp_exp_min=-80.0,
            ),
            "torch_online_tile32_p_float": _ref_gqa_attention_online(
                q,
                k,
                v,
                softmax_scale,
                causal,
                tile_size=32,
                p_dtype=torch.float32,
            ),
        }
        comparisons = [
            _compare_payload(
                "dense_flash_v100",
                "triton_unified_attention",
                dense_flash,
                triton_out,
            ),
            _compare_payload(
                "paged_flash_v100",
                "triton_unified_attention",
                paged_flash,
                triton_out,
            ),
            _compare_payload(
                "paged_flash_v100",
                "dense_flash_v100",
                paged_flash,
                dense_flash,
            ),
        ]
        for name, tensor in variants.items():
            comparisons.append(
                _compare_payload(name, "triton_unified_attention", tensor, triton_out)
            )
            comparisons.append(
                _compare_payload(name, "dense_flash_v100", tensor, dense_flash)
            )
        diagnostics.append(
            {
                "seq_len": seq_len,
                "query_len": query_len,
                "q_heads": q_heads,
                "kv_heads": kv_heads,
                "head_dim": head_dim,
                "block_size": block_size,
                "causal": causal,
                "seed": seed,
                "shape": list(triton_out.shape),
                "comparisons": comparisons,
            }
        )
    return diagnostics


def _make_basis_value(v_template: torch.Tensor) -> tuple[torch.Tensor, int]:
    seq_len = v_template.shape[1]
    head_dim = v_template.shape[3]
    span = min(seq_len, head_dim)
    v = torch.zeros_like(v_template)
    for pos in range(span):
        v[0, pos, :, pos] = 1.0
    return v, span


def _make_single_key_value(v_template: torch.Tensor, key_idx: int) -> torch.Tensor:
    v = torch.zeros_like(v_template)
    v[0, key_idx, :, :] = 1.0
    return v


def run_basis_value_diagnostic_cases(flash_attn_v100: Any) -> list[dict[str, Any]]:
    """Split Flash-vs-Triton diff between probability and P@V accumulation.

    With basis or single-key values, each output dimension mostly exposes one
    attention probability. If those already differ, the next repair target is
    QK/softmax/reduction order. If only random values differ, the mismatch is
    likely in P@V accumulation/output ordering.
    """
    cases = [
        (9, 9, 1, 1, 256, 16, True),
        (9, 9, 6, 1, 256, 16, True),
        (69, 1, 6, 1, 256, 16, True),
        (69, 16, 6, 1, 256, 16, True),
        (181, 32, 6, 1, 256, 16, True),
        (4096, 1, 6, 1, 256, 832, True),
        (4096, 8, 6, 1, 256, 832, True),
    ]
    diagnostics = []
    for seq_len, query_len, q_heads, kv_heads, head_dim, block_size, causal in cases:
        seed = seq_len * 1000 + query_len * 10 + head_dim + block_size + 313
        q, k, random_v = _make_qkv(
            seq_len=seq_len,
            query_len=query_len,
            q_heads=q_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            seed=seed,
        )
        softmax_scale = 1.0 / math.sqrt(head_dim)
        basis_v, basis_span = _make_basis_value(random_v)
        first_query_tail = max(seq_len - query_len, 0)
        value_variants = [
            ("random", random_v, {"kind": "random"}),
            ("ones", torch.ones_like(random_v), {"kind": "normalization"}),
            (
                "basis_prefix",
                basis_v,
                {
                    "kind": "probability_probe",
                    "basis_span": basis_span,
                    "covers_full_sequence": basis_span == seq_len,
                },
            ),
            (
                "single_key_0",
                _make_single_key_value(random_v, 0),
                {"kind": "probability_probe", "key_idx": 0},
            ),
            (
                "single_key_first_query_tail",
                _make_single_key_value(random_v, first_query_tail),
                {
                    "kind": "probability_probe",
                    "key_idx": first_query_tail,
                },
            ),
            (
                "single_key_last",
                _make_single_key_value(random_v, seq_len - 1),
                {"kind": "probability_probe", "key_idx": seq_len - 1},
            ),
        ]
        variant_payloads = []
        for variant_name, v, variant_meta in value_variants:
            key_cache, value_cache, block_table, seq_lens = _make_paged_cache(
                k, v, block_size
            )
            triton_out = _triton_unified_attention(
                q,
                key_cache,
                value_cache,
                block_table,
                seq_lens,
                softmax_scale,
            )
            dense_flash = flash_attn_v100.flash_attn_func(
                q,
                k,
                v,
                causal=causal,
                softmax_scale=softmax_scale,
            )
            paged_flash = flash_attn_v100.flash_attn_prefill_paged(
                q,
                key_cache,
                value_cache,
                block_table,
                seq_lens,
                softmax_scale=softmax_scale,
                causal=causal,
            )
            variant_payloads.append(
                {
                    "name": variant_name,
                    **variant_meta,
                    "comparisons": [
                        _compare_payload(
                            "dense_flash_v100",
                            "triton_unified_attention",
                            dense_flash,
                            triton_out,
                        ),
                        _compare_payload(
                            "paged_flash_v100",
                            "triton_unified_attention",
                            paged_flash,
                            triton_out,
                        ),
                        _compare_payload(
                            "paged_flash_v100",
                            "dense_flash_v100",
                            paged_flash,
                            dense_flash,
                        ),
                    ],
                }
            )
        diagnostics.append(
            {
                "seq_len": seq_len,
                "query_len": query_len,
                "q_heads": q_heads,
                "kv_heads": kv_heads,
                "head_dim": head_dim,
                "block_size": block_size,
                "causal": causal,
                "seed": seed,
                "variants": variant_payloads,
            }
        )
    return diagnostics


def _qk_pattern_variants(
    q: torch.Tensor,
    k: torch.Tensor,
) -> list[tuple[str, torch.Tensor, torch.Tensor, dict[str, Any]]]:
    head_dim = q.shape[-1]
    variants: list[tuple[str, torch.Tensor, torch.Tensor, dict[str, Any]]] = [
        (
            "random_full",
            q,
            k,
            {"kind": "baseline", "nonzero_dims": head_dim},
        )
    ]

    variants.append(
        (
            "zero_scores",
            torch.zeros_like(q),
            torch.zeros_like(k),
            {"kind": "exact_uniform", "nonzero_dims": 0},
        )
    )
    constant_q = torch.zeros_like(q)
    constant_k = torch.zeros_like(k)
    constant_q[..., 0] = 1.0
    constant_k[..., 0] = 1.0
    variants.append(
        (
            "constant_single_dim",
            constant_q,
            constant_k,
            {"kind": "exact_uniform", "nonzero_dims": 1},
        )
    )
    ramp_q = torch.zeros_like(q)
    ramp_k = torch.zeros_like(k)
    ramp_q[..., 0] = 1.0
    positions = torch.arange(k.shape[1], device=k.device, dtype=torch.float32)
    ramp = ((positions % 17) - 8) / 8
    ramp_k[0, :, :, 0] = ramp.to(k.dtype).view(-1, 1)
    variants.append(
        (
            "exact_key_ramp_single_dim",
            ramp_q,
            ramp_k,
            {"kind": "exact_nonuniform_scores", "nonzero_dims": 1},
        )
    )
    random_q_ramp_k = torch.zeros_like(q)
    random_q_ramp_k[..., 0] = q[..., 0]
    variants.append(
        (
            "random_q_exact_key_ramp_single_dim",
            random_q_ramp_k,
            ramp_k,
            {"kind": "partial_qk_dot", "nonzero_dims": 1},
        )
    )
    unit_q_random_k = torch.zeros_like(q)
    unit_q_random_k[..., 0] = 1.0
    unit_k_random = torch.zeros_like(k)
    unit_k_random[..., 0] = k[..., 0]
    variants.append(
        (
            "unit_q_random_key_single_dim",
            unit_q_random_k,
            unit_k_random,
            {"kind": "partial_qk_dot", "nonzero_dims": 1},
        )
    )
    half_q_random_k = torch.zeros_like(q)
    half_q_random_k[..., 0] = 0.5
    variants.append(
        (
            "half_q_random_key_single_dim",
            half_q_random_k,
            unit_k_random,
            {"kind": "partial_qk_dot", "nonzero_dims": 1},
        )
    )

    for dims in (1, 2, 16, 128):
        if dims > head_dim:
            continue
        q_pattern = torch.zeros_like(q)
        k_pattern = torch.zeros_like(k)
        q_pattern[..., :dims] = q[..., :dims]
        k_pattern[..., :dims] = k[..., :dims]
        variants.append(
            (
                f"random_first_{dims}_dims",
                q_pattern,
                k_pattern,
                {"kind": "partial_qk_dot", "nonzero_dims": dims},
            )
        )

    if q.shape[2] > 1:
        variants.append(
            (
                "repeat_query_head0_full_random",
                q[:, :, :1, :].expand(-1, -1, q.shape[2], -1).contiguous(),
                k,
                {"kind": "head_layout_probe", "nonzero_dims": head_dim},
            )
        )
    return variants


def run_qk_pattern_diagnostic_cases(flash_attn_v100: Any) -> list[dict[str, Any]]:
    """Narrow Flash-vs-Triton probability mismatch to QK or softmax shape."""
    cases = [
        (9, 9, 1, 1, 256, 16, True),
        (9, 9, 6, 1, 256, 16, True),
        (69, 1, 6, 1, 256, 16, True),
        (69, 16, 6, 1, 256, 16, True),
        (181, 32, 6, 1, 256, 16, True),
        (4096, 1, 6, 1, 256, 832, True),
        (4096, 8, 6, 1, 256, 832, True),
    ]
    diagnostics = []
    for seq_len, query_len, q_heads, kv_heads, head_dim, block_size, causal in cases:
        seed = seq_len * 1000 + query_len * 10 + head_dim + block_size + 313
        q, k, random_v = _make_qkv(
            seq_len=seq_len,
            query_len=query_len,
            q_heads=q_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            seed=seed,
        )
        softmax_scale = 1.0 / math.sqrt(head_dim)
        basis_v, basis_span = _make_basis_value(random_v)
        first_query_tail = max(seq_len - query_len, 0)
        value_variants = [
            ("random", random_v, {"kind": "random"}),
            ("ones", torch.ones_like(random_v), {"kind": "normalization"}),
            (
                "basis_prefix",
                basis_v,
                {
                    "kind": "probability_probe",
                    "basis_span": basis_span,
                    "covers_full_sequence": basis_span == seq_len,
                },
            ),
            (
                "single_key_0",
                _make_single_key_value(random_v, 0),
                {"kind": "probability_probe", "key_idx": 0},
            ),
            (
                "single_key_first_query_tail",
                _make_single_key_value(random_v, first_query_tail),
                {
                    "kind": "probability_probe",
                    "key_idx": first_query_tail,
                },
            ),
        ]
        qk_payloads = []
        for qk_name, q_case, k_case, qk_meta in _qk_pattern_variants(q, k):
            value_payloads = []
            for variant_name, v, variant_meta in value_variants:
                key_cache, value_cache, block_table, seq_lens = _make_paged_cache(
                    k_case, v, block_size
                )
                triton_out = _triton_unified_attention(
                    q_case,
                    key_cache,
                    value_cache,
                    block_table,
                    seq_lens,
                    softmax_scale,
                )
                dense_flash = flash_attn_v100.flash_attn_func(
                    q_case,
                    k_case,
                    v,
                    causal=causal,
                    softmax_scale=softmax_scale,
                )
                paged_flash = flash_attn_v100.flash_attn_prefill_paged(
                    q_case,
                    key_cache,
                    value_cache,
                    block_table,
                    seq_lens,
                    softmax_scale=softmax_scale,
                    causal=causal,
                )
                value_payloads.append(
                    {
                        "name": variant_name,
                        **variant_meta,
                        "comparisons": [
                            _compare_payload(
                                "dense_flash_v100",
                                "triton_unified_attention",
                                dense_flash,
                                triton_out,
                            ),
                            _compare_payload(
                                "paged_flash_v100",
                                "triton_unified_attention",
                                paged_flash,
                                triton_out,
                            ),
                            _compare_payload(
                                "paged_flash_v100",
                                "dense_flash_v100",
                                paged_flash,
                                dense_flash,
                            ),
                        ],
                    }
                )
            qk_payloads.append(
                {
                    "name": qk_name,
                    **qk_meta,
                    "value_variants": value_payloads,
                }
            )
        diagnostics.append(
            {
                "seq_len": seq_len,
                "query_len": query_len,
                "q_heads": q_heads,
                "kv_heads": kv_heads,
                "head_dim": head_dim,
                "block_size": block_size,
                "causal": causal,
                "seed": seed,
                "qk_patterns": qk_payloads,
            }
        )
    return diagnostics


def _ref_gqa_attention_first_dim_product(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    causal: bool,
    *,
    product_mode: str,
) -> torch.Tensor:
    k, v = _repeat_kv_for_gqa(q, k, v)
    q_ref = q.transpose(1, 2)
    k_ref = k.transpose(1, 2)
    v_ref = v.float().transpose(1, 2)
    q0 = q_ref[..., 0]
    k0 = k_ref[..., 0]
    if product_mode == "fp32":
        scores = q0.float().unsqueeze(-1) * k0.float().unsqueeze(-2)
    elif product_mode == "half":
        scores = (q0.unsqueeze(-1) * k0.unsqueeze(-2)).float()
    else:
        raise ValueError(f"unsupported product_mode={product_mode}")
    scores = scores * softmax_scale
    query_len = q.shape[1]
    seq_len = k.shape[1]
    if causal:
        q_pos = torch.arange(query_len, device=q.device) + max(seq_len - query_len, 0)
        k_pos = torch.arange(seq_len, device=q.device)
        mask = k_pos.view(1, 1, 1, seq_len) > q_pos.view(1, 1, query_len, 1)
        scores = scores.masked_fill(mask, float("-inf"))
    out = torch.matmul(torch.softmax(scores, dim=-1), v_ref)
    return out.transpose(1, 2).to(torch.float16)


def run_qk_reference_diagnostic_cases(flash_attn_v100: Any) -> list[dict[str, Any]]:
    cases = [
        (69, 16, 6, 1, 256, 16, True),
        (181, 32, 6, 1, 256, 16, True),
        (4096, 8, 6, 1, 256, 832, True),
    ]
    diagnostics = []
    for seq_len, query_len, q_heads, kv_heads, head_dim, block_size, causal in cases:
        seed = seq_len * 1000 + query_len * 10 + head_dim + block_size + 313
        q, k, random_v = _make_qkv(
            seq_len=seq_len,
            query_len=query_len,
            q_heads=q_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            seed=seed,
        )
        softmax_scale = 1.0 / math.sqrt(head_dim)
        basis_v, basis_span = _make_basis_value(random_v)
        value_variants = [
            (
                "basis_prefix",
                basis_v,
                {
                    "kind": "probability_probe",
                    "basis_span": basis_span,
                    "covers_full_sequence": basis_span == seq_len,
                },
            ),
            (
                "single_key_0",
                _make_single_key_value(random_v, 0),
                {"kind": "probability_probe", "key_idx": 0},
            ),
            ("random", random_v, {"kind": "random"}),
        ]
        selected_qk = [
            item
            for item in _qk_pattern_variants(q, k)
            if item[0] in ("exact_key_ramp_single_dim", "random_first_1_dims")
        ]
        qk_payloads = []
        for qk_name, q_case, k_case, qk_meta in selected_qk:
            value_payloads = []
            for variant_name, v, variant_meta in value_variants:
                key_cache, value_cache, block_table, seq_lens = _make_paged_cache(
                    k_case, v, block_size
                )
                triton_out = _triton_unified_attention(
                    q_case,
                    key_cache,
                    value_cache,
                    block_table,
                    seq_lens,
                    softmax_scale,
                )
                dense_flash = flash_attn_v100.flash_attn_func(
                    q_case,
                    k_case,
                    v,
                    causal=causal,
                    softmax_scale=softmax_scale,
                )
                ref_fp32_product = _ref_gqa_attention_first_dim_product(
                    q_case,
                    k_case,
                    v,
                    softmax_scale,
                    causal,
                    product_mode="fp32",
                )
                ref_half_product = _ref_gqa_attention_first_dim_product(
                    q_case,
                    k_case,
                    v,
                    softmax_scale,
                    causal,
                    product_mode="half",
                )
                value_payloads.append(
                    {
                        "name": variant_name,
                        **variant_meta,
                        "comparisons": [
                            _compare_payload(
                                "dense_flash_v100",
                                "triton_unified_attention",
                                dense_flash,
                                triton_out,
                            ),
                            _compare_payload(
                                "dense_flash_v100",
                                "torch_first_dim_fp32_product",
                                dense_flash,
                                ref_fp32_product,
                            ),
                            _compare_payload(
                                "triton_unified_attention",
                                "torch_first_dim_fp32_product",
                                triton_out,
                                ref_fp32_product,
                            ),
                            _compare_payload(
                                "dense_flash_v100",
                                "torch_first_dim_half_product",
                                dense_flash,
                                ref_half_product,
                            ),
                            _compare_payload(
                                "triton_unified_attention",
                                "torch_first_dim_half_product",
                                triton_out,
                                ref_half_product,
                            ),
                            _compare_payload(
                                "torch_first_dim_half_product",
                                "torch_first_dim_fp32_product",
                                ref_half_product,
                                ref_fp32_product,
                            ),
                        ],
                    }
                )
            qk_payloads.append(
                {
                    "name": qk_name,
                    **qk_meta,
                    "value_variants": value_payloads,
                }
            )
        diagnostics.append(
            {
                "seq_len": seq_len,
                "query_len": query_len,
                "q_heads": q_heads,
                "kv_heads": kv_heads,
                "head_dim": head_dim,
                "block_size": block_size,
                "causal": causal,
                "seed": seed,
                "qk_patterns": qk_payloads,
            }
        )
    return diagnostics


def run_qk_score_diagnostic_cases(flash_attn_v100: Any) -> list[dict[str, Any]]:
    cases = [
        (69, 1, 6, 1, 256, 16, True),
        (69, 16, 6, 1, 256, 16, True),
        (4096, 1, 6, 1, 256, 832, True),
    ]
    selected_names = {
        "exact_key_ramp_single_dim",
        "random_q_exact_key_ramp_single_dim",
        "unit_q_random_key_single_dim",
        "half_q_random_key_single_dim",
        "random_first_1_dims",
        "random_first_2_dims",
        "random_full",
    }
    diagnostics = []
    for seq_len, query_len, q_heads, kv_heads, head_dim, block_size, causal in cases:
        seed = seq_len * 1000 + query_len * 10 + head_dim + block_size + 313
        q, k, _ = _make_qkv(
            seq_len=seq_len,
            query_len=query_len,
            q_heads=q_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            seed=seed,
        )
        softmax_scale = 1.0 / math.sqrt(head_dim)
        qk_payloads = []
        for qk_name, q_case, k_case, qk_meta in _qk_pattern_variants(q, k):
            if qk_name not in selected_names:
                continue
            flash_scores = flash_attn_v100.flash_attn_qk_scores(
                q_case,
                k_case,
                softmax_scale=softmax_scale,
                causal=causal,
            )
            triton_scores = _triton_qk_scores(
                q_case,
                k_case,
                softmax_scale,
                causal,
            )
            qk_payloads.append(
                {
                    "name": qk_name,
                    **qk_meta,
                    "comparisons": [
                        _compare_payload(
                            "flash_v100_qk_scores",
                            "triton_tl_dot_qk_scores",
                            flash_scores,
                            triton_scores,
                        )
                    ],
                }
            )
        diagnostics.append(
            {
                "seq_len": seq_len,
                "query_len": query_len,
                "q_heads": q_heads,
                "kv_heads": kv_heads,
                "head_dim": head_dim,
                "block_size": block_size,
                "causal": causal,
                "seed": seed,
                "qk_patterns": qk_payloads,
            }
        )
    return diagnostics


def run_softmax_lse_diagnostic_cases(flash_attn_v100: Any) -> list[dict[str, Any]]:
    cases = [
        (69, 1, 6, 1, 256, 16, True),
        (69, 16, 6, 1, 256, 16, True),
        (4096, 1, 6, 1, 256, 832, True),
    ]
    selected_names = {
        "exact_key_ramp_single_dim",
        "random_q_exact_key_ramp_single_dim",
        "unit_q_random_key_single_dim",
        "half_q_random_key_single_dim",
        "random_first_1_dims",
        "random_first_2_dims",
        "random_full",
    }
    diagnostics = []
    for seq_len, query_len, q_heads, kv_heads, head_dim, block_size, causal in cases:
        seed = seq_len * 1000 + query_len * 10 + head_dim + block_size + 313
        q, k, v = _make_qkv(
            seq_len=seq_len,
            query_len=query_len,
            q_heads=q_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            seed=seed,
        )
        softmax_scale = 1.0 / math.sqrt(head_dim)
        qk_payloads = []
        for qk_name, q_case, k_case, qk_meta in _qk_pattern_variants(q, k):
            if qk_name not in selected_names:
                continue
            flash_lse = flash_attn_v100.flash_attn_lse(
                q_case,
                k_case,
                v,
                causal=causal,
                softmax_scale=softmax_scale,
            )
            triton_lse_tile16 = _triton_qk_lse(
                q_case,
                k_case,
                softmax_scale,
                causal,
                tile_size=16,
            )
            triton_lse_tile32 = _triton_qk_lse(
                q_case,
                k_case,
                softmax_scale,
                causal,
                tile_size=32,
            )
            torch_lse = _torch_qk_lse(q_case, k_case, softmax_scale, causal)
            qk_payloads.append(
                {
                    "name": qk_name,
                    **qk_meta,
                    "comparisons": [
                        _compare_payload(
                            "flash_v100_softmax_lse",
                            "triton_online_lse_tile16",
                            flash_lse,
                            triton_lse_tile16,
                        ),
                        _compare_payload(
                            "flash_v100_softmax_lse",
                            "triton_online_lse_tile32",
                            flash_lse,
                            triton_lse_tile32,
                        ),
                        _compare_payload(
                            "flash_v100_softmax_lse",
                            "torch_logsumexp_fp32",
                            flash_lse,
                            torch_lse,
                        ),
                        _compare_payload(
                            "triton_online_lse_tile16",
                            "triton_online_lse_tile32",
                            triton_lse_tile16,
                            triton_lse_tile32,
                        ),
                    ],
                }
            )
        diagnostics.append(
            {
                "seq_len": seq_len,
                "query_len": query_len,
                "q_heads": q_heads,
                "kv_heads": kv_heads,
                "head_dim": head_dim,
                "block_size": block_size,
                "causal": causal,
                "seed": seed,
                "qk_patterns": qk_payloads,
            }
        )
    return diagnostics


def run_softmax_tile_sim_diagnostic_cases(
    flash_attn_v100: Any,
) -> list[dict[str, Any]]:
    cases = [
        (69, 1, 6, 1, 256, 16, True),
        (69, 16, 6, 1, 256, 16, True),
        (4096, 1, 6, 1, 256, 832, True),
    ]
    selected_names = {
        "exact_key_ramp_single_dim",
        "random_first_1_dims",
        "random_first_2_dims",
    }
    diagnostics = []
    for seq_len, query_len, q_heads, kv_heads, head_dim, block_size, causal in cases:
        seed = seq_len * 1000 + query_len * 10 + head_dim + block_size + 313
        q, k, random_v = _make_qkv(
            seq_len=seq_len,
            query_len=query_len,
            q_heads=q_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            seed=seed,
        )
        softmax_scale = 1.0 / math.sqrt(head_dim)
        key_candidates = sorted(
            {
                0,
                min(16, seq_len - 1),
                min(28, seq_len - 1),
                min(63, seq_len - 1),
                min(64, seq_len - 1),
                seq_len - 1,
            }
        )
        qk_payloads = []
        for qk_name, q_case, k_case, qk_meta in _qk_pattern_variants(q, k):
            if qk_name not in selected_names:
                continue
            scores = flash_attn_v100.flash_attn_qk_scores(
                q_case,
                k_case,
                softmax_scale=softmax_scale,
                causal=causal,
            )
            key_payloads = []
            for key_idx in key_candidates:
                v = _make_single_key_value(random_v, key_idx)
                key_cache, value_cache, block_table, seq_lens = _make_paged_cache(
                    k_case, v, block_size
                )
                triton_out = _triton_unified_attention(
                    q_case,
                    key_cache,
                    value_cache,
                    block_table,
                    seq_lens,
                    softmax_scale,
                )[..., 0]
                flash_out = flash_attn_v100.flash_attn_func(
                    q_case,
                    k_case,
                    v,
                    causal=causal,
                    softmax_scale=softmax_scale,
                )[..., 0]
                simulations = {
                    "sim_tile16_p_half": _simulate_online_single_key_probability(
                        scores,
                        key_idx=key_idx,
                        tile_size=16,
                        p_half=True,
                        clamp_exp_min=None,
                    ),
                    "sim_tile32_p_half": _simulate_online_single_key_probability(
                        scores,
                        key_idx=key_idx,
                        tile_size=32,
                        p_half=True,
                        clamp_exp_min=None,
                    ),
                    "sim_tile64_p_half_clamp80": (
                        _simulate_online_single_key_probability(
                            scores,
                            key_idx=key_idx,
                            tile_size=64,
                            p_half=True,
                            clamp_exp_min=-80.0,
                        )
                    ),
                    "sim_tile64_p_float_clamp80": (
                        _simulate_online_single_key_probability(
                            scores,
                            key_idx=key_idx,
                            tile_size=64,
                            p_half=False,
                            clamp_exp_min=-80.0,
                        )
                    ),
                }
                comparisons = [
                    _compare_payload(
                        "dense_flash_v100_single_key",
                        "triton_unified_attention_single_key",
                        flash_out,
                        triton_out,
                    ),
                ]
                for sim_name, sim in simulations.items():
                    comparisons.append(
                        _compare_payload(
                            sim_name,
                            "dense_flash_v100_single_key",
                            sim,
                            flash_out,
                        )
                    )
                    comparisons.append(
                        _compare_payload(
                            sim_name,
                            "triton_unified_attention_single_key",
                            sim,
                            triton_out,
                        )
                    )
                key_payloads.append(
                    {
                        "key_idx": key_idx,
                        "comparisons": comparisons,
                    }
                )
            qk_payloads.append(
                {
                    "name": qk_name,
                    **qk_meta,
                    "keys": key_payloads,
                }
            )
        diagnostics.append(
            {
                "seq_len": seq_len,
                "query_len": query_len,
                "q_heads": q_heads,
                "kv_heads": kv_heads,
                "head_dim": head_dim,
                "block_size": block_size,
                "causal": causal,
                "seed": seed,
                "qk_patterns": qk_payloads,
            }
        )
    return diagnostics


def run_softmax_update_trace_diagnostic_cases(
    flash_attn_v100: Any,
) -> list[dict[str, Any]]:
    cases = [
        (69, 1, 6, 1, 256, 16, True),
        (69, 16, 6, 1, 256, 16, True),
        (4096, 1, 6, 1, 256, 832, True),
    ]
    selected_names = {
        "exact_key_ramp_single_dim",
        "random_first_1_dims",
        "random_first_2_dims",
        "random_full",
    }
    diagnostics = []
    for seq_len, query_len, q_heads, kv_heads, head_dim, block_size, causal in cases:
        seed = seq_len * 1000 + query_len * 10 + head_dim + block_size + 313
        q, k, random_v = _make_qkv(
            seq_len=seq_len,
            query_len=query_len,
            q_heads=q_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            seed=seed,
        )
        softmax_scale = 1.0 / math.sqrt(head_dim)
        # This script calls unified_attention without 3D segment buffers, so
        # even q=1 uses the 2D path's prefill tile size.
        triton_tile = 32
        key_candidates = sorted(
            {
                0,
                min(1, seq_len - 1),
                min(2, seq_len - 1),
                min(3, seq_len - 1),
                min(16, seq_len - 1),
                min(28, seq_len - 1),
                min(31, seq_len - 1),
                min(32, seq_len - 1),
                min(63, seq_len - 1),
                min(64, seq_len - 1),
                seq_len - 1,
            }
        )
        qk_payloads = []
        for qk_name, q_case, k_case, qk_meta in _qk_pattern_variants(q, k):
            if qk_name not in selected_names:
                continue
            scores = flash_attn_v100.flash_attn_qk_scores(
                q_case,
                k_case,
                softmax_scale=softmax_scale,
                causal=causal,
            )
            triton_scores = _triton_qk_scores(
                q_case,
                k_case,
                softmax_scale=softmax_scale,
                causal=causal,
            )
            key_payloads = []
            for key_idx in key_candidates:
                v = _make_single_key_value(random_v, key_idx)
                key_cache, value_cache, block_table, seq_lens = _make_paged_cache(
                    k_case, v, block_size
                )
                triton_out = _triton_unified_attention(
                    q_case,
                    key_cache,
                    value_cache,
                    block_table,
                    seq_lens,
                    softmax_scale,
                )[..., 0]
                flash_out = flash_attn_v100.flash_attn_func(
                    q_case,
                    k_case,
                    v,
                    causal=causal,
                    softmax_scale=softmax_scale,
                )[..., 0]
                triton_debug_prob = _triton_single_key_probability(
                    q_case,
                    k_case,
                    softmax_scale,
                    causal,
                    key_idx=key_idx,
                    tile_size=triton_tile,
                )
                flash_vs_triton = _compare_payload(
                    "dense_flash_v100_single_key",
                    "triton_unified_attention_single_key",
                    flash_out,
                    triton_out,
                )
                max_idx = flash_vs_triton["max_diff_flat_index"]
                output_index = (
                    _unravel_index(int(max_idx), tuple(flash_out.shape))
                    if max_idx is not None
                    else [0, 0, 0]
                )
                simulations = {
                    "triton_expected_tile_p_half": (
                        _simulate_online_single_key_probability(
                            scores,
                            key_idx=key_idx,
                            tile_size=triton_tile,
                            p_half=True,
                            clamp_exp_min=None,
                        )
                    ),
                    "tile16_p_half": _simulate_online_single_key_probability(
                        scores,
                        key_idx=key_idx,
                        tile_size=16,
                        p_half=True,
                        clamp_exp_min=None,
                    ),
                    "tile32_p_half": _simulate_online_single_key_probability(
                        scores,
                        key_idx=key_idx,
                        tile_size=32,
                        p_half=True,
                        clamp_exp_min=None,
                    ),
                    "flash_like_tile64_p_half_clamp80": (
                        _simulate_online_single_key_probability(
                            scores,
                            key_idx=key_idx,
                            tile_size=64,
                            p_half=True,
                            clamp_exp_min=-80.0,
                        )
                    ),
                    "tile64_p_half_no_clamp": _simulate_online_single_key_probability(
                        scores,
                        key_idx=key_idx,
                        tile_size=64,
                        p_half=True,
                        clamp_exp_min=None,
                    ),
                    "tile64_p_float_clamp80": _simulate_online_single_key_probability(
                        scores,
                        key_idx=key_idx,
                        tile_size=64,
                        p_half=False,
                        clamp_exp_min=-80.0,
                    ),
                }
                comparisons = [flash_vs_triton]
                comparisons.append(
                    _compare_payload(
                        "triton_debug_single_key_probability",
                        "triton_unified_attention_single_key",
                        triton_debug_prob,
                        triton_out,
                    )
                )
                comparisons.append(
                    _compare_payload(
                        "triton_debug_single_key_probability",
                        "dense_flash_v100_single_key",
                        triton_debug_prob,
                        flash_out,
                    )
                )
                for sim_name, sim in simulations.items():
                    comparisons.append(
                        _compare_payload(
                            sim_name,
                            "dense_flash_v100_single_key",
                            sim,
                            flash_out,
                        )
                    )
                    comparisons.append(
                        _compare_payload(
                            sim_name,
                            "triton_unified_attention_single_key",
                            sim,
                            triton_out,
                        )
                    )
                traces = {
                    "triton_expected_tile_p_half": _trace_online_single_key_scalar(
                        scores,
                        key_idx=key_idx,
                        tile_size=triton_tile,
                        p_half=True,
                        clamp_exp_min=None,
                        output_index=output_index,
                    ),
                    "flash_like_tile64_p_half_clamp80": (
                        _trace_online_single_key_scalar(
                            scores,
                            key_idx=key_idx,
                            tile_size=64,
                            p_half=True,
                            clamp_exp_min=-80.0,
                            output_index=output_index,
                        )
                    ),
                    "tile64_p_float_clamp80": _trace_online_single_key_scalar(
                        scores,
                        key_idx=key_idx,
                        tile_size=64,
                        p_half=False,
                        clamp_exp_min=-80.0,
                        output_index=output_index,
                    ),
                }
                key_payloads.append(
                    {
                        "key_idx": key_idx,
                        "max_diff_output_index_b_q_h": output_index,
                        "flash_value_at_max": (
                            flash_vs_triton["actual_at_max_diff"]
                        ),
                        "triton_value_at_max": (
                            flash_vs_triton["expected_at_max_diff"]
                        ),
                        "comparisons": comparisons,
                        "traces": traces,
                    }
                )
            qk_payloads.append(
                {
                    "name": qk_name,
                    **qk_meta,
                    "triton_tile": triton_tile,
                    "qk_score_compare": _compare_payload(
                        "flash_v100_qk_scores",
                        "triton_tl_dot_qk_scores",
                        scores,
                        triton_scores,
                    ),
                    "keys": key_payloads,
                }
            )
        diagnostics.append(
            {
                "seq_len": seq_len,
                "query_len": query_len,
                "q_heads": q_heads,
                "kv_heads": kv_heads,
                "head_dim": head_dim,
                "block_size": block_size,
                "causal": causal,
                "seed": seed,
                "qk_patterns": qk_payloads,
            }
        )
    return diagnostics


def run_paged_prefill_cases(flash_attn_v100: Any) -> list[CaseResult]:
    cases = [
        (9, 9, 6, 1, 256, 16, True),
        (69, 16, 6, 1, 256, 16, True),
        (181, 32, 6, 1, 128, 16, True),
        (4096, 8, 6, 1, 256, 832, True),
    ]
    results = []
    for seq_len, query_len, q_heads, kv_heads, head_dim, block_size, causal in cases:
        seed = seq_len * 1000 + query_len * 10 + head_dim + block_size
        q, k, v = _make_qkv(
            seq_len=seq_len,
            query_len=query_len,
            q_heads=q_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            seed=seed,
        )
        softmax_scale = 1.0 / math.sqrt(head_dim)
        key_cache, value_cache, block_table, seq_lens = _make_paged_cache(
            k, v, block_size
        )
        actual = flash_attn_v100.flash_attn_prefill_paged(
            q,
            key_cache,
            value_cache,
            block_table,
            seq_lens,
            softmax_scale=softmax_scale,
            causal=causal,
        )
        expected = _ref_gqa_attention(q, k, v, softmax_scale, causal)
        results.append(
            _record_result(
                "paged_prefill",
                "torch_float32",
                actual,
                expected,
                seq_len=seq_len,
                query_len=query_len,
                q_heads=q_heads,
                kv_heads=kv_heads,
                head_dim=head_dim,
                block_size=block_size,
                causal=causal,
                seed=seed,
            )
        )
        dense_flash = flash_attn_v100.flash_attn_func(
            q,
            k,
            v,
            causal=causal,
            softmax_scale=softmax_scale,
        )
        results.append(
            _record_result(
                "paged_prefill",
                "dense_flash_v100",
                actual,
                dense_flash,
                seq_len=seq_len,
                query_len=query_len,
                q_heads=q_heads,
                kv_heads=kv_heads,
                head_dim=head_dim,
                block_size=block_size,
                causal=causal,
                seed=seed,
            )
        )
    return results


def run_paged_decode_cases(flash_attn_v100: Any) -> list[CaseResult]:
    cases = [
        (9, 1, 6, 1, 256, 16),
        (69, 1, 6, 1, 256, 16),
        (181, 4, 6, 1, 128, 16),
        (4096, 8, 6, 1, 256, 832),
    ]
    results = []
    for seq_len, query_len, q_heads, kv_heads, head_dim, block_size in cases:
        seed = seq_len * 1000 + query_len * 10 + head_dim + block_size + 17
        q, k, v = _make_qkv(
            seq_len=seq_len,
            query_len=query_len,
            q_heads=q_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            seed=seed,
        )
        softmax_scale = 1.0 / math.sqrt(head_dim)
        key_cache, value_cache, block_table, _ = _make_paged_cache(k, v, block_size)
        q_flat = q.squeeze(0).contiguous()
        decode_block_table = block_table.expand(query_len, -1).contiguous()
        seq_lens = (
            seq_len
            - query_len
            + torch.arange(1, query_len + 1, device=q.device, dtype=torch.int32)
        )
        actual_flat = torch.empty_like(q_flat)
        flash_attn_v100.flash_attn_decode_paged(
            q_flat,
            key_cache,
            value_cache,
            decode_block_table,
            seq_lens,
            softmax_scale=softmax_scale,
            out=actual_flat,
            kv_cache_dtype="auto",
        )
        actual = actual_flat.unsqueeze(0)
        expected = _ref_decode_steps(q, k, v, softmax_scale)
        results.append(
            _record_result(
                "paged_decode",
                "torch_float32",
                actual,
                expected,
                seq_len=seq_len,
                query_len=query_len,
                q_heads=q_heads,
                kv_heads=kv_heads,
                head_dim=head_dim,
                block_size=block_size,
                causal=True,
                seed=seed,
            )
        )
        dense_flash = _flash_decode_steps(flash_attn_v100, q, k, v, softmax_scale)
        results.append(
            _record_result(
                "paged_decode",
                "dense_flash_v100",
                actual,
                dense_flash,
                seq_len=seq_len,
                query_len=query_len,
                q_heads=q_heads,
                kv_heads=kv_heads,
                head_dim=head_dim,
                block_size=block_size,
                causal=True,
                seed=seed,
            )
        )
        if hasattr(flash_attn_v100, "flash_attn_decode_paged_wmma"):
            wmma_flat = torch.empty_like(q_flat)
            flash_attn_v100.flash_attn_decode_paged_wmma(
                q_flat,
                key_cache,
                value_cache,
                decode_block_table,
                seq_lens,
                softmax_scale=softmax_scale,
                out=wmma_flat,
                kv_cache_dtype="auto",
            )
            results.append(
                _record_result(
                    "paged_decode_wmma",
                    "dense_flash_v100",
                    wmma_flat.unsqueeze(0),
                    dense_flash,
                    seq_len=seq_len,
                    query_len=query_len,
                    q_heads=q_heads,
                    kv_heads=kv_heads,
                    head_dim=head_dim,
                    block_size=block_size,
                    causal=True,
                    seed=seed,
                )
            )
        prefill_as_decode = flash_attn_v100.flash_attn_prefill_paged(
            q,
            key_cache,
            value_cache,
            block_table,
            torch.tensor([seq_len], device=q.device, dtype=torch.int32),
            softmax_scale=softmax_scale,
            causal=True,
        )
        dense_flash_causal = flash_attn_v100.flash_attn_func(
            q,
            k,
            v,
            causal=True,
            softmax_scale=softmax_scale,
        )
        results.append(
            _record_result(
                "paged_decode_as_prefill",
                "dense_flash_v100_causal",
                prefill_as_decode,
                dense_flash_causal,
                seq_len=seq_len,
                query_len=query_len,
                q_heads=q_heads,
                kv_heads=kv_heads,
                head_dim=head_dim,
                block_size=block_size,
                causal=True,
                seed=seed,
            )
        )
        results.append(
            _record_result(
                "dense_flash_v100_causal",
                "dense_flash_v100_decode_steps",
                dense_flash_causal,
                dense_flash,
                seq_len=seq_len,
                query_len=query_len,
                q_heads=q_heads,
                kv_heads=kv_heads,
                head_dim=head_dim,
                block_size=None,
                causal=True,
                seed=seed,
            )
        )
    return results


def run_decode_cudagraph_replay_diagnostic_cases(
    flash_attn_v100: Any,
) -> list[dict[str, Any]]:
    """Compare scalar paged decode CUDA graph replay against eager decode.

    vLLM FULL graphs capture decode with fixed tensor addresses and then update
    seq_lens/block_table/query contents before replay. This diagnostic isolates
    that contract for the SM70 scalar paged decode op without starting a model.
    """
    cases = [
        {
            "name": "b2_dynamic_seq_lens_capture_seq1",
            "batch_size": 2,
            "base_seq_lens": (96, 160),
            "steps": 1024,
            "q_heads": 6,
            "kv_heads": 1,
            "head_dim": 256,
            "block_size": 16,
            "seed": 70031,
        },
    ]
    diagnostics: list[dict[str, Any]] = []
    for case in cases:
        batch_size = int(case["batch_size"])
        base_seq_lens = tuple(int(x) for x in case["base_seq_lens"])
        steps = int(case["steps"])
        q_heads = int(case["q_heads"])
        kv_heads = int(case["kv_heads"])
        head_dim = int(case["head_dim"])
        block_size = int(case["block_size"])
        seed = int(case["seed"])
        if len(base_seq_lens) != batch_size:
            raise ValueError("base_seq_lens must match batch_size")

        torch.manual_seed(seed)
        device = torch.device("cuda")
        dtype = torch.float16
        softmax_scale = 1.0 / math.sqrt(head_dim)
        q_steps = torch.randn(
            (steps, batch_size, q_heads, head_dim),
            device=device,
            dtype=dtype,
        )
        k_values = []
        v_values = []
        for base_seq_len in base_seq_lens:
            seq_len = base_seq_len + steps
            k = torch.randn(
                (1, seq_len, kv_heads, head_dim),
                device=device,
                dtype=dtype,
            )
            v = torch.randn_like(k)
            k_values.append(k)
            v_values.append(v)

        key_cache, value_cache, full_block_table, _ = _make_batched_paged_cache(
            k_values,
            v_values,
            block_size,
        )
        seq_lens_steps = torch.empty(
            (steps, batch_size),
            device=device,
            dtype=torch.int32,
        )
        for step in range(steps):
            seq_lens_steps[step].copy_(
                torch.tensor(
                    [base_seq_len + step + 1 for base_seq_len in base_seq_lens],
                    device=device,
                    dtype=torch.int32,
                )
            )

        q_static = q_steps[0].clone()
        block_table_static = full_block_table.clone()
        seq_lens_static = torch.ones(batch_size, device=device, dtype=torch.int32)
        out_graph = torch.empty_like(q_static)
        out_eager = torch.empty_like(q_static)

        def decode_graph_target() -> None:
            flash_attn_v100.flash_attn_decode_paged(
                q_static,
                key_cache,
                value_cache,
                block_table_static,
                seq_lens_static,
                softmax_scale=softmax_scale,
                out=out_graph,
                kv_cache_dtype="auto",
            )

        def decode_eager_target() -> None:
            flash_attn_v100.flash_attn_decode_paged(
                q_static,
                key_cache,
                value_cache,
                block_table_static,
                seq_lens_static,
                softmax_scale=softmax_scale,
                out=out_eager,
                kv_cache_dtype="auto",
            )

        # Allocate extension-side workspaces and warm kernels before capture.
        for _ in range(3):
            decode_graph_target()
        torch.cuda.synchronize()

        side_stream = torch.cuda.Stream()
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            for _ in range(3):
                decode_graph_target()
        torch.cuda.current_stream().wait_stream(side_stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            decode_graph_target()

        first_mismatch: dict[str, Any] | None = None
        first_mismatch_by_row: list[dict[str, Any] | None] = [
            None for _ in range(batch_size)
        ]
        max_diff = 0.0
        mean_diff_at_max = 0.0
        num_different_at_max = 0
        for step in range(steps):
            q_static.copy_(q_steps[step])
            seq_lens_static.copy_(seq_lens_steps[step])
            graph.replay()
            decode_eager_target()
            torch.cuda.synchronize()

            diff = torch.abs(out_graph.float() - out_eager.float())
            step_max_diff = float(diff.max().item())
            if step_max_diff > max_diff:
                max_diff = step_max_diff
                mean_diff_at_max = float(diff.mean().item())
                num_different_at_max = int((out_graph != out_eager).sum().item())

            if step_max_diff != 0.0 and first_mismatch is None:
                flat_idx = int(diff.view(-1).argmax().item())
                row_stride = q_heads * head_dim
                row = flat_idx // row_stride
                row_rem = flat_idx - row * row_stride
                head = row_rem // head_dim
                dim = row_rem - head * head_dim
                first_mismatch = {
                    "step": step,
                    "row": int(row),
                    "head": int(head),
                    "dim": int(dim),
                    "seq_lens": [int(x) for x in seq_lens_steps[step].tolist()],
                    "graph_value": float(out_graph[row, head, dim].item()),
                    "eager_value": float(out_eager[row, head, dim].item()),
                    "abs_diff": step_max_diff,
                }

            for row in range(batch_size):
                if first_mismatch_by_row[row] is not None:
                    continue
                row_diff = torch.abs(
                    out_graph[row].float() - out_eager[row].float()
                )
                row_max = float(row_diff.max().item())
                if row_max != 0.0:
                    flat_idx = int(row_diff.view(-1).argmax().item())
                    head = flat_idx // head_dim
                    dim = flat_idx - head * head_dim
                    first_mismatch_by_row[row] = {
                        "step": step,
                        "head": int(head),
                        "dim": int(dim),
                        "seq_len": int(seq_lens_steps[step, row].item()),
                        "graph_value": float(out_graph[row, head, dim].item()),
                        "eager_value": float(out_eager[row, head, dim].item()),
                        "abs_diff": row_max,
                    }

        diagnostics.append(
            {
                **case,
                "max_blocks": int(full_block_table.shape[1]),
                "key_cache_shape": list(key_cache.shape),
                "block_table_shape": list(block_table_static.shape),
                "tensor_addresses": {
                    "q_static": int(q_static.data_ptr()),
                    "block_table_static": int(block_table_static.data_ptr()),
                    "seq_lens_static": int(seq_lens_static.data_ptr()),
                    "out_graph": int(out_graph.data_ptr()),
                    "key_cache": int(key_cache.data_ptr()),
                    "value_cache": int(value_cache.data_ptr()),
                },
                "equal": first_mismatch is None,
                "max_diff": max_diff,
                "mean_diff_at_max": mean_diff_at_max,
                "num_different_at_max": num_different_at_max,
                "first_mismatch": first_mismatch,
                "first_mismatch_by_row": first_mismatch_by_row,
            }
        )
    return diagnostics


def _time_ms(fn: Any, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / iters)


def _run_decode_timing_case(
    flash_attn_v100: Any,
    *,
    seq_len: int,
    query_len: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    block_size: int,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    seed = seq_len * 1000 + query_len * 10 + head_dim + block_size + 101
    q, k, v = _make_qkv(
        seq_len=seq_len,
        query_len=query_len,
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        seed=seed,
    )
    softmax_scale = 1.0 / math.sqrt(head_dim)
    key_cache, value_cache, block_table, _ = _make_paged_cache(k, v, block_size)
    q_flat = q.squeeze(0).contiguous()
    out_flat = torch.empty_like(q_flat)
    out_wmma_flat = torch.empty_like(q_flat)
    decode_block_table = block_table.expand(query_len, -1).contiguous()
    seq_lens = (
        seq_len
        - query_len
        + torch.arange(1, query_len + 1, device=q.device, dtype=torch.int32)
    )
    full_seq_lens = torch.tensor([seq_len], device=q.device, dtype=torch.int32)
    prefix_len = seq_len - query_len
    dense_cache_k = torch.empty_like(k.squeeze(0))
    dense_cache_v = torch.empty_like(v.squeeze(0))
    if prefix_len > 0:
        dense_cache_k[:prefix_len].copy_(k.squeeze(0)[:prefix_len])
        dense_cache_v[:prefix_len].copy_(v.squeeze(0)[:prefix_len])

    def decode_paged() -> torch.Tensor:
        return flash_attn_v100.flash_attn_decode_paged(
            q_flat,
            key_cache,
            value_cache,
            decode_block_table,
            seq_lens,
            softmax_scale=softmax_scale,
            out=out_flat,
            kv_cache_dtype="auto",
        )

    def decode_as_prefill() -> torch.Tensor:
        return flash_attn_v100.flash_attn_prefill_paged(
            q,
            key_cache,
            value_cache,
            block_table,
            full_seq_lens,
            softmax_scale=softmax_scale,
            causal=True,
        )

    def decode_wmma() -> torch.Tensor:
        return flash_attn_v100.flash_attn_decode_paged_wmma(
            q_flat,
            key_cache,
            value_cache,
            decode_block_table,
            seq_lens,
            softmax_scale=softmax_scale,
            out=out_wmma_flat,
            kv_cache_dtype="auto",
        )

    def dense_flash() -> torch.Tensor:
        return flash_attn_v100.flash_attn_func(
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            causal=True,
        )

    def dense_cache_flash() -> torch.Tensor:
        dense_cache_k[prefix_len:seq_len].copy_(k.squeeze(0)[prefix_len:seq_len])
        dense_cache_v[prefix_len:seq_len].copy_(v.squeeze(0)[prefix_len:seq_len])
        return flash_attn_v100.flash_attn_func(
            q,
            dense_cache_k[:seq_len].unsqueeze(0),
            dense_cache_v[:seq_len].unsqueeze(0),
            softmax_scale=softmax_scale,
            causal=True,
        )

    def triton_attention() -> torch.Tensor:
        return _triton_unified_attention(
            q,
            key_cache,
            value_cache,
            block_table,
            full_seq_lens,
            softmax_scale,
            causal=True,
        )

    def record_candidate(
        result: dict[str, Any],
        name: str,
        fn: Any,
        transform: Any,
        dense_reference: torch.Tensor | None,
        triton_reference: torch.Tensor | None,
    ) -> None:
        try:
            actual = transform(fn()).clone()
        except RuntimeError as exc:
            result[f"{name}_error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
            return
        try:
            result[f"{name}_ms"] = _time_ms(fn, warmup, iters)
        except RuntimeError as exc:
            result[f"{name}_timing_error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
        if dense_reference is not None:
            result[f"{name}_equal_dense"] = bool(torch.equal(actual, dense_reference))
            result[f"{name}_max_diff_dense"] = float(
                (actual - dense_reference).abs().max().item()
            )
        if triton_reference is not None:
            result[f"{name}_equal_triton"] = bool(
                torch.equal(actual, triton_reference)
            )
            result[f"{name}_max_diff_triton"] = float(
                (actual - triton_reference).abs().max().item()
            )

    result: dict[str, Any] = {
        "name": "decode_timing",
        "seq_len": seq_len,
        "query_len": query_len,
        "q_heads": q_heads,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "block_size": block_size,
        "warmup": warmup,
        "iters": iters,
        "decode_partition_size": _decode_partition_size(
            block_table.shape[1] * key_cache.shape[1]
        ),
    }
    dense_actual = None
    triton_actual = None
    try:
        dense_actual = dense_flash().clone()
        result["dense_flash_ms"] = _time_ms(dense_flash, warmup, iters)
    except RuntimeError as exc:
        result["dense_flash_error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
    try:
        triton_actual = triton_attention().clone()
        result["triton_unified_ms"] = _time_ms(triton_attention, warmup, iters)
    except RuntimeError as exc:
        result["triton_unified_error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
    if dense_actual is not None and triton_actual is not None:
        result["dense_flash_equal_triton"] = bool(
            torch.equal(dense_actual, triton_actual)
        )
        result["dense_flash_max_diff_triton"] = float(
            (dense_actual - triton_actual).abs().max().item()
        )

    record_candidate(
        result,
        "decode_paged",
        decode_paged,
        lambda tensor: tensor.unsqueeze(0),
        dense_actual,
        triton_actual,
    )
    record_candidate(
        result,
        "decode_as_prefill",
        decode_as_prefill,
        lambda tensor: tensor,
        dense_actual,
        triton_actual,
    )
    record_candidate(
        result,
        "decode_wmma",
        decode_wmma,
        lambda tensor: tensor.unsqueeze(0),
        dense_actual,
        triton_actual,
    )
    record_candidate(
        result,
        "dense_cache",
        dense_cache_flash,
        lambda tensor: tensor,
        dense_actual,
        triton_actual,
    )
    return result


def run_decode_timing_cases(
    flash_attn_v100: Any,
    *,
    warmup: int,
    iters: int,
) -> list[dict[str, Any]]:
    cases = [
        (4096, 1, 6, 1, 256, 16),
        (4096, 8, 6, 1, 256, 16),
        (16384, 1, 6, 1, 256, 16),
        (32768, 1, 6, 1, 256, 16),
    ]
    timings: list[dict[str, Any]] = []
    for seq_len, query_len, q_heads, kv_heads, head_dim, block_size in cases:
        timings.append(
            _run_decode_timing_case(
                flash_attn_v100,
                seq_len=seq_len,
                query_len=query_len,
                q_heads=q_heads,
                kv_heads=kv_heads,
                head_dim=head_dim,
                block_size=block_size,
                warmup=warmup,
                iters=iters,
            )
        )
    return timings


def _record_dict(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return asdict(_record_result(*args, **kwargs))


def _decode_partition_size(max_seq_capacity: int) -> int:
    raw = os.getenv("VLLM_FLASH_V100_DECODE_PARTITION_SIZE")
    if raw is not None:
        return int(raw)
    return 512 if max_seq_capacity > 20480 else 256


def _make_mask_probe_qkv(
    *,
    seq_len: int,
    query_len: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = torch.device("cuda")
    q = torch.zeros(
        (1, query_len, q_heads, head_dim),
        device=device,
        dtype=torch.float16,
    )
    k = torch.zeros(
        (1, seq_len, kv_heads, head_dim),
        device=device,
        dtype=torch.float16,
    )
    v = torch.zeros_like(k)
    values = torch.arange(seq_len, device=device, dtype=torch.float32)
    values = (values / max(seq_len - 1, 1)).to(torch.float16)
    v[0, :, :, 0] = values.view(-1, 1)
    return q, k, v


def run_mask_edge_cases(flash_attn_v100: Any) -> list[dict[str, Any]]:
    cases = [
        (69, 16, 6, 1, 256, 16, True),
        (69, 16, 6, 1, 256, 16, False),
        (4096, 8, 6, 1, 256, 16, True),
        (4096, 8, 6, 1, 256, 16, False),
    ]
    payloads: list[dict[str, Any]] = []
    for seq_len, query_len, q_heads, kv_heads, head_dim, block_size, causal in cases:
        seed = seq_len * 1000 + query_len * 10 + head_dim + block_size + int(causal)
        q, k, v = _make_qkv(
            seq_len=seq_len,
            query_len=query_len,
            q_heads=q_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            seed=seed,
        )
        softmax_scale = 1.0 / math.sqrt(head_dim)
        key_cache, value_cache, block_table, seq_lens = _make_paged_cache(
            k, v, block_size
        )
        if causal:
            expected = _triton_unified_attention(
                q,
                key_cache,
                value_cache,
                block_table,
                seq_lens,
                softmax_scale,
                causal=True,
            )
            reference_name = "triton_unified_attention"
        else:
            expected = _ref_gqa_attention(q, k, v, softmax_scale, causal=False)
            reference_name = "torch_float32"
        dense_flash = flash_attn_v100.flash_attn_func(
            q,
            k,
            v,
            causal=causal,
            softmax_scale=softmax_scale,
        )
        paged_flash = flash_attn_v100.flash_attn_prefill_paged(
            q,
            key_cache,
            value_cache,
            block_table,
            seq_lens,
            softmax_scale=softmax_scale,
            causal=causal,
        )
        payloads.extend(
            [
                _record_dict(
                    "mask_dense_flash_v100",
                    reference_name,
                    dense_flash,
                    expected,
                    seq_len=seq_len,
                    query_len=query_len,
                    q_heads=q_heads,
                    kv_heads=kv_heads,
                    head_dim=head_dim,
                    block_size=block_size,
                    causal=causal,
                    seed=seed,
                ),
                _record_dict(
                    "mask_paged_prefill",
                    reference_name,
                    paged_flash,
                    expected,
                    seq_len=seq_len,
                    query_len=query_len,
                    q_heads=q_heads,
                    kv_heads=kv_heads,
                    head_dim=head_dim,
                    block_size=block_size,
                    causal=causal,
                    seed=seed,
                ),
            ]
        )

    seq_len = 17
    query_len = 5
    q_heads = 6
    kv_heads = 1
    head_dim = 256
    q, k, v = _make_mask_probe_qkv(
        seq_len=seq_len,
        query_len=query_len,
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
    )
    softmax_scale = 1.0 / math.sqrt(head_dim)
    causal_flash = flash_attn_v100.flash_attn_func(
        q,
        k,
        v,
        causal=True,
        softmax_scale=softmax_scale,
    )
    noncausal_flash = flash_attn_v100.flash_attn_func(
        q,
        k,
        v,
        causal=False,
        softmax_scale=softmax_scale,
    )
    payloads.append(
        {
            "name": "mask_probe_causal_vs_noncausal",
            "reference": "torch_float32",
            "causal_matches_reference": _compare_payload(
                "causal_flash_v100",
                "torch_float32",
                causal_flash,
                _ref_gqa_attention(q, k, v, softmax_scale, True),
            ),
            "noncausal_matches_reference": _compare_payload(
                "noncausal_flash_v100",
                "torch_float32",
                noncausal_flash,
                _ref_gqa_attention(q, k, v, softmax_scale, False),
            ),
            "causal_vs_noncausal": _compare_payload(
                "causal_flash_v100",
                "noncausal_flash_v100",
                causal_flash,
                noncausal_flash,
            ),
            "seq_len": seq_len,
            "query_len": query_len,
            "q_heads": q_heads,
            "kv_heads": kv_heads,
            "head_dim": head_dim,
        }
    )
    return payloads


def run_varlen_edge_cases(flash_attn_v100: Any) -> list[dict[str, Any]]:
    q_heads = 6
    kv_heads = 1
    head_dim = 256
    block_size = 16
    softmax_scale = 1.0 / math.sqrt(head_dim)
    payloads: list[dict[str, Any]] = []

    uniform_specs = [(69, 8), (181, 8), (4096, 8)]
    q_values = []
    k_values = []
    v_values = []
    for idx, (seq_len, query_len) in enumerate(uniform_specs):
        q, k, v = _make_qkv(
            seq_len=seq_len,
            query_len=query_len,
            q_heads=q_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            seed=7000 + idx,
        )
        q_values.append(q)
        k_values.append(k)
        v_values.append(v)
    key_cache, value_cache, block_table, seq_lens = _make_batched_paged_cache(
        k_values, v_values, block_size
    )
    q_batch = torch.cat(q_values, dim=0)
    flash_batch = flash_attn_v100.flash_attn_prefill_paged(
        q_batch,
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        softmax_scale=softmax_scale,
        causal=True,
    )
    q_flat = q_batch.reshape(-1, q_heads, head_dim)
    query_lens = [query_len for _, query_len in uniform_specs]
    triton_flat = _triton_unified_attention_varlen(
        q_flat,
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        query_lens,
        softmax_scale,
        causal=True,
    )
    payloads.append(
        {
            "name": "varlen_uniform_query_batched_paged_prefill",
            "seq_lens": [seq for seq, _ in uniform_specs],
            "query_lens": query_lens,
            "block_size": block_size,
            "comparison": _compare_payload(
                "flash_v100_batched_paged_prefill",
                "triton_unified_attention_varlen",
                flash_batch.reshape(-1, q_heads, head_dim),
                triton_flat,
            ),
        }
    )

    ragged_specs = [(69, 1), (181, 3), (4096, 8)]
    q_values = []
    k_values = []
    v_values = []
    for idx, (seq_len, query_len) in enumerate(ragged_specs):
        q, k, v = _make_qkv(
            seq_len=seq_len,
            query_len=query_len,
            q_heads=q_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            seed=9000 + idx,
        )
        q_values.append(q)
        k_values.append(k)
        v_values.append(v)
    key_cache, value_cache, block_table, seq_lens = _make_batched_paged_cache(
        k_values, v_values, block_size
    )
    flash_chunks = []
    for idx, q in enumerate(q_values):
        out = flash_attn_v100.flash_attn_prefill_paged(
            q,
            key_cache,
            value_cache,
            block_table[idx : idx + 1],
            seq_lens[idx : idx + 1],
            softmax_scale=softmax_scale,
            causal=True,
        )
        flash_chunks.append(out.squeeze(0))
    flash_flat = torch.cat(flash_chunks, dim=0)
    q_flat = torch.cat([q.squeeze(0) for q in q_values], dim=0)
    query_lens = [query_len for _, query_len in ragged_specs]
    triton_flat = _triton_unified_attention_varlen(
        q_flat,
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        query_lens,
        softmax_scale,
        causal=True,
    )
    payloads.append(
        {
            "name": "varlen_ragged_query_looped_paged_prefill",
            "seq_lens": [seq for seq, _ in ragged_specs],
            "query_lens": query_lens,
            "block_size": block_size,
            "comparison": _compare_payload(
                "flash_v100_looped_paged_prefill",
                "triton_unified_attention_varlen",
                flash_flat,
                triton_flat,
            ),
        }
    )
    return payloads


def run_varlen_layout_diagnostic_cases(
    flash_attn_v100: Any,
) -> list[dict[str, Any]]:
    """Diagnose whether varlen diffs come from layout/indexing or arithmetic.

    This is intentionally separate from run_varlen_edge_cases. It compares
    same-kernel layout changes first, so a nonzero diff here is a Type-A bug.
    """
    q_heads = 6
    kv_heads = 1
    head_dim = 256
    block_size = 16
    softmax_scale = 1.0 / math.sqrt(head_dim)
    payloads: list[dict[str, Any]] = []

    cases = [
        ("uniform", [(69, 8), (181, 8), (4096, 8)], 7000),
        ("ragged", [(69, 1), (181, 3), (4096, 8)], 9000),
    ]
    for case_name, specs, seed_base in cases:
        q_values = []
        k_values = []
        v_values = []
        for idx, (seq_len, query_len) in enumerate(specs):
            q, k, v = _make_qkv(
                seq_len=seq_len,
                query_len=query_len,
                q_heads=q_heads,
                kv_heads=kv_heads,
                head_dim=head_dim,
                seed=seed_base + idx,
            )
            q_values.append(q)
            k_values.append(k)
            v_values.append(v)

        key_cache, value_cache, block_table, seq_lens = _make_batched_paged_cache(
            k_values, v_values, block_size
        )
        query_lens = [query_len for _, query_len in specs]
        q_flat = torch.cat([q.squeeze(0) for q in q_values], dim=0)
        triton_varlen = _triton_unified_attention_varlen(
            q_flat,
            key_cache,
            value_cache,
            block_table,
            seq_lens,
            query_lens,
            softmax_scale,
            causal=True,
        )
        flash_full_cache_loop = _flash_paged_loop_flat(
            flash_attn_v100,
            q_values,
            key_cache,
            value_cache,
            block_table,
            seq_lens,
            softmax_scale,
            causal=True,
        )
        flash_standalone_loop = _flash_paged_standalone_loop_flat(
            flash_attn_v100,
            q_values,
            k_values,
            v_values,
            block_size,
            softmax_scale,
            causal=True,
        )
        flash_dense_loop = _flash_dense_loop_flat(
            flash_attn_v100,
            q_values,
            k_values,
            v_values,
            softmax_scale,
            causal=True,
        )
        triton_loop = _triton_loop_flat(
            q_values,
            key_cache,
            value_cache,
            block_table,
            seq_lens,
            softmax_scale,
            causal=True,
        )
        comparisons = [
            _compare_payload(
                "flash_v100_full_cache_looped_paged_prefill",
                "flash_v100_standalone_looped_paged_prefill",
                flash_full_cache_loop,
                flash_standalone_loop,
            ),
            _compare_payload(
                "flash_v100_full_cache_looped_paged_prefill",
                "flash_v100_dense_looped_prefill",
                flash_full_cache_loop,
                flash_dense_loop,
            ),
            _compare_payload(
                "triton_unified_attention_varlen",
                "triton_unified_attention_looped",
                triton_varlen,
                triton_loop,
            ),
            _compare_payload(
                "flash_v100_full_cache_looped_paged_prefill",
                "triton_unified_attention_varlen",
                flash_full_cache_loop,
                triton_varlen,
            ),
        ]
        if case_name == "uniform":
            q_batch = torch.cat(q_values, dim=0)
            flash_batched = flash_attn_v100.flash_attn_prefill_paged(
                q_batch,
                key_cache,
                value_cache,
                block_table,
                seq_lens,
                softmax_scale=softmax_scale,
                causal=True,
            ).reshape(-1, q_heads, head_dim)
            comparisons.insert(
                0,
                _compare_payload(
                    "flash_v100_batched_paged_prefill",
                    "flash_v100_full_cache_looped_paged_prefill",
                    flash_batched,
                    flash_full_cache_loop,
                ),
            )
            comparisons.append(
                _compare_payload(
                    "flash_v100_batched_paged_prefill",
                    "triton_unified_attention_varlen",
                    flash_batched,
                    triton_varlen,
                )
            )

        type_a_must_equal = [
            "flash_v100_full_cache_looped_paged_prefill vs "
            "flash_v100_standalone_looped_paged_prefill",
            "triton_unified_attention_varlen vs triton_unified_attention_looped",
        ]
        if case_name == "uniform":
            type_a_must_equal.insert(
                0,
                "flash_v100_batched_paged_prefill vs "
                "flash_v100_full_cache_looped_paged_prefill",
            )
        payloads.append(
            {
                "name": f"varlen_{case_name}_layout_diagnostic",
                "seq_lens": [seq for seq, _ in specs],
                "query_lens": query_lens,
                "block_size": block_size,
                "type_a_must_equal": type_a_must_equal,
                "comparisons": comparisons,
            }
        )
    return payloads


def run_long_sequence_edge_cases(flash_attn_v100: Any) -> list[dict[str, Any]]:
    cases = [
        (4096, 1, 16),
        (4096, 8, 16),
        (8192, 1, 16),
        (8192, 8, 16),
        (16384, 1, 16),
        (16384, 8, 16),
        (32768, 1, 16),
        (32768, 8, 16),
    ]
    q_heads = 6
    kv_heads = 1
    head_dim = 256
    payloads: list[dict[str, Any]] = []
    for seq_len, query_len, block_size in cases:
        seed = seq_len * 1000 + query_len * 10 + block_size + 1201
        q, k, v = _make_qkv(
            seq_len=seq_len,
            query_len=query_len,
            q_heads=q_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            seed=seed,
        )
        softmax_scale = 1.0 / math.sqrt(head_dim)
        key_cache, value_cache, block_table, seq_lens = _make_paged_cache(
            k, v, block_size
        )
        triton_out = _triton_unified_attention(
            q,
            key_cache,
            value_cache,
            block_table,
            seq_lens,
            softmax_scale,
            causal=True,
        )
        dense_flash = flash_attn_v100.flash_attn_func(
            q,
            k,
            v,
            causal=True,
            softmax_scale=softmax_scale,
        )
        case_payload: dict[str, Any] = {
            "name": "long_sequence",
            "seq_len": seq_len,
            "query_len": query_len,
            "q_heads": q_heads,
            "kv_heads": kv_heads,
            "head_dim": head_dim,
            "block_size": block_size,
            "max_num_blocks": int(block_table.shape[1]),
            "comparisons": [
                _compare_payload(
                    "dense_flash_v100",
                    "triton_unified_attention",
                    dense_flash,
                    triton_out,
                ),
            ],
        }
        try:
            paged_flash = flash_attn_v100.flash_attn_prefill_paged(
                q,
                key_cache,
                value_cache,
                block_table,
                seq_lens,
                softmax_scale=softmax_scale,
                causal=True,
            )
            case_payload["comparisons"].append(
                _compare_payload(
                    "paged_prefill",
                    "triton_unified_attention",
                    paged_flash,
                    triton_out,
                )
            )
        except RuntimeError as exc:
            case_payload["paged_prefill_error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
        if query_len == 1:
            q_flat = q.squeeze(0).contiguous()
            out_flat = torch.empty_like(q_flat)
            decode_out = flash_attn_v100.flash_attn_decode_paged(
                q_flat,
                key_cache,
                value_cache,
                block_table,
                seq_lens,
                softmax_scale=softmax_scale,
                out=out_flat,
                kv_cache_dtype="auto",
            ).unsqueeze(0)
            case_payload["decode_partition_size"] = _decode_partition_size(
                block_table.shape[1] * key_cache.shape[1]
            )
            case_payload["comparisons"].append(
                _compare_payload(
                    "paged_decode",
                    "triton_unified_attention",
                    decode_out,
                    triton_out,
                )
            )
        payloads.append(case_payload)
    return payloads


def run_long_decode_matched_diagnostic_cases(
    flash_attn_v100: Any,
) -> list[dict[str, Any]]:
    cases = [
        (4096, 16),
        (8192, 16),
        (16384, 16),
        (32768, 16),
    ]
    q_heads = 6
    kv_heads = 1
    head_dim = 256
    query_len = 1
    payloads: list[dict[str, Any]] = []
    for seq_len, block_size in cases:
        seed = seq_len * 1000 + query_len * 10 + block_size + 1201
        q, k, v = _make_qkv(
            seq_len=seq_len,
            query_len=query_len,
            q_heads=q_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            seed=seed,
        )
        softmax_scale = 1.0 / math.sqrt(head_dim)
        key_cache, value_cache, block_table, seq_lens = _make_paged_cache(
            k, v, block_size
        )
        partition_size = _decode_partition_size(block_table.shape[1] * block_size)
        triton_out = _triton_unified_attention(
            q,
            key_cache,
            value_cache,
            block_table,
            seq_lens,
            softmax_scale,
            causal=True,
        )
        dense_flash = flash_attn_v100.flash_attn_func(
            q,
            k,
            v,
            causal=True,
            softmax_scale=softmax_scale,
        )
        partitioned_flash_ref = _flash_partitioned_decode_reference(
            flash_attn_v100,
            q,
            k,
            v,
            softmax_scale,
            partition_size=partition_size,
        )
        q_flat = q.squeeze(0).contiguous()
        out_flat = torch.empty_like(q_flat)
        decode_workspace = None
        workspace_partition_size = partition_size
        workspace_getter = getattr(flash_attn_v100, "_get_decode_workspace", None)
        if workspace_getter is None:
            from flash_attn_v100 import flash_attn_interface

            workspace_getter = getattr(
                flash_attn_interface,
                "_get_decode_workspace",
                None,
            )
        if workspace_getter is not None:
            decode_workspace, workspace_partition_size = workspace_getter(
                q_flat, key_cache, block_table
            )
        scalar_decode = flash_attn_v100.flash_attn_decode_paged(
            q_flat,
            key_cache,
            value_cache,
            block_table,
            seq_lens,
            softmax_scale=softmax_scale,
            out=out_flat,
            kv_cache_dtype="auto",
        ).unsqueeze(0)
        workspace_reduce_ref = None
        dense_partition_outputs = None
        scalar_qk_scores = None
        dense_partition_qk_scores = None
        triton_partition_qk_scores = None
        if decode_workspace is not None:
            tmp_out, max_logits, exp_sums = decode_workspace
            workspace_reduce_ref = _decode_workspace_reduce_reference(
                tmp_out,
                max_logits,
                exp_sums,
                seq_len=seq_len,
                partition_size=workspace_partition_size,
            )
            dense_partition_outputs = _flash_dense_partition_outputs(
                flash_attn_v100,
                q,
                k,
                v,
                softmax_scale,
                partition_size=workspace_partition_size,
            )
            if hasattr(flash_attn_v100, "flash_attn_decode_qk_scores"):
                scalar_qk_scores = flash_attn_v100.flash_attn_decode_qk_scores(
                    q_flat,
                    key_cache,
                    block_table,
                    seq_lens,
                    softmax_scale=softmax_scale,
                    kv_cache_dtype="auto",
                )
                dense_partition_qk_scores = _flash_dense_partition_qk_scores(
                    flash_attn_v100,
                    q,
                    k,
                    softmax_scale,
                    partition_size=workspace_partition_size,
                )
                triton_qk_scores = _triton_qk_scores(
                    q,
                    k,
                    softmax_scale=softmax_scale,
                    causal=False,
                )
                num_qk_partitions = (
                    seq_len + workspace_partition_size - 1
                ) // workspace_partition_size
                triton_partition_qk_scores = triton_qk_scores[
                    0, :, 0, :
                ].reshape(q_heads, num_qk_partitions, workspace_partition_size)
        wmma_decode = None
        if hasattr(flash_attn_v100, "flash_attn_decode_paged_wmma"):
            wmma_flat = torch.empty_like(q_flat)
            wmma_decode = flash_attn_v100.flash_attn_decode_paged_wmma(
                q_flat,
                key_cache,
                value_cache,
                block_table,
                seq_lens,
                softmax_scale=softmax_scale,
                out=wmma_flat,
                kv_cache_dtype="auto",
            ).unsqueeze(0)

        comparisons = [
            _compare_payload(
                "scalar_paged_decode",
                "flash_partitioned_decode_reference",
                scalar_decode,
                partitioned_flash_ref,
            ),
            _compare_payload(
                "flash_partitioned_decode_reference",
                "dense_flash_v100",
                partitioned_flash_ref,
                dense_flash,
            ),
            _compare_payload(
                "scalar_paged_decode",
                "dense_flash_v100",
                scalar_decode,
                dense_flash,
            ),
            _compare_payload(
                "dense_flash_v100",
                "triton_unified_attention",
                dense_flash,
                triton_out,
            ),
            _compare_payload(
                "scalar_paged_decode",
                "triton_unified_attention",
                scalar_decode,
                triton_out,
            ),
        ]
        if workspace_reduce_ref is not None and dense_partition_outputs is not None:
            num_partitions = (seq_len + workspace_partition_size - 1) // (
                workspace_partition_size
            )
            tmp_out_used = decode_workspace[0][0, :, :num_partitions, :]
            comparisons.extend(
                [
                    _compare_payload(
                        "scalar_paged_decode",
                        "workspace_python_reduce",
                        scalar_decode,
                        workspace_reduce_ref,
                    ),
                    _compare_payload(
                        "workspace_python_reduce",
                        "dense_flash_v100",
                        workspace_reduce_ref,
                        dense_flash,
                    ),
                    _compare_payload(
                        "scalar_decode_partition_tmp_out",
                        "dense_flash_partition_outputs",
                        tmp_out_used,
                        dense_partition_outputs,
                    ),
                ]
            )
            if scalar_qk_scores is not None:
                scalar_qk_used = scalar_qk_scores[0, :, :num_partitions, :]
                scalar_qk_python_outputs = _partition_outputs_from_qk_scores(
                    scalar_qk_used,
                    v,
                    partition_size=workspace_partition_size,
                    q_heads=q_heads,
                    kv_heads=kv_heads,
                )
                dense_qk_python_outputs = _partition_outputs_from_qk_scores(
                    dense_partition_qk_scores,
                    v,
                    partition_size=workspace_partition_size,
                    q_heads=q_heads,
                    kv_heads=kv_heads,
                )
                triton_qk_python_outputs = _partition_outputs_from_qk_scores(
                    triton_partition_qk_scores,
                    v,
                    partition_size=workspace_partition_size,
                    q_heads=q_heads,
                    kv_heads=kv_heads,
                )
                comparisons.extend(
                    [
                        _compare_payload(
                            "scalar_decode_partition_qk_scores",
                            "dense_flash_partition_qk_scores",
                            scalar_qk_used,
                            dense_partition_qk_scores,
                        ),
                        _compare_payload(
                            "scalar_decode_partition_qk_scores",
                            "triton_partition_qk_scores",
                            scalar_qk_used,
                            triton_partition_qk_scores,
                        ),
                        _compare_payload(
                            "dense_flash_partition_qk_scores",
                            "triton_partition_qk_scores",
                            dense_partition_qk_scores,
                            triton_partition_qk_scores,
                        ),
                        _compare_payload(
                            "scalar_qk_python_partition_outputs",
                            "scalar_decode_partition_tmp_out",
                            scalar_qk_python_outputs,
                            tmp_out_used,
                        ),
                        _compare_payload(
                            "scalar_qk_python_partition_outputs",
                            "dense_qk_python_partition_outputs",
                            scalar_qk_python_outputs,
                            dense_qk_python_outputs,
                        ),
                        _compare_payload(
                            "scalar_qk_python_partition_outputs",
                            "dense_flash_partition_outputs",
                            scalar_qk_python_outputs,
                            dense_partition_outputs,
                        ),
                        _compare_payload(
                            "dense_qk_python_partition_outputs",
                            "dense_flash_partition_outputs",
                            dense_qk_python_outputs,
                            dense_partition_outputs,
                        ),
                        _compare_payload(
                            "triton_qk_python_partition_outputs",
                            "dense_qk_python_partition_outputs",
                            triton_qk_python_outputs,
                            dense_qk_python_outputs,
                        ),
                    ]
                )
        if wmma_decode is not None:
            comparisons.append(
                _compare_payload(
                    "wmma_decode",
                    "dense_flash_v100",
                    wmma_decode,
                    dense_flash,
                )
            )
            comparisons.append(
                _compare_payload(
                    "wmma_decode",
                    "triton_unified_attention",
                    wmma_decode,
                    triton_out,
                )
            )
        payloads.append(
            {
                "name": "long_decode_matched_diagnostic",
                "seq_len": seq_len,
                "query_len": query_len,
                "q_heads": q_heads,
                "kv_heads": kv_heads,
                "head_dim": head_dim,
                "block_size": block_size,
                "partition_size": partition_size,
                "num_partitions": (seq_len + partition_size - 1) // partition_size,
                "comparisons": comparisons,
            }
        )
    return payloads


def run_tensor_dump_replay(
    flash_attn_v100: Any,
    tensor_dump: Path,
) -> dict[str, Any]:
    payload = torch.load(tensor_dump, map_location="cpu")
    raw_key_cache_compare = None
    raw_value_cache_compare = None
    if "cache_key" in payload:
        raw_key_cache_compare = _compare_payload(
            "raw_key",
            "cache_key",
            payload["raw_key"],
            payload["cache_key"],
        )
    if "cache_value" in payload:
        raw_value_cache_compare = _compare_payload(
            "raw_value",
            "cache_value",
            payload["raw_value"],
            payload["cache_value"],
        )
    q = payload["query"].unsqueeze(0).cuda().contiguous()
    k = payload["raw_key"].unsqueeze(0).cuda().contiguous()
    v = payload["raw_value"].unsqueeze(0).cuda().contiguous()
    saved_flash = payload["candidate_output"].unsqueeze(0).cuda().contiguous()
    saved_triton = (
        payload["triton_reference_output"].unsqueeze(0).cuda().contiguous()
    )
    softmax_scale = float(payload.get("scale", 1.0 / math.sqrt(q.shape[-1])))
    causal = True

    flash_replay = flash_attn_v100.flash_attn_func(
        q,
        k,
        v,
        softmax_scale=softmax_scale,
        causal=causal,
    )
    compact_block_size = max(16, int(k.shape[1]))
    key_cache, value_cache, block_table, seq_lens = _make_paged_cache(
        k,
        v,
        compact_block_size,
    )
    triton_replay = _triton_unified_attention(
        q,
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        softmax_scale,
        causal=causal,
    )
    q_heads = q.shape[2]
    kv_heads = k.shape[2]
    flash_qk_scores = flash_attn_v100.flash_attn_qk_scores(
        q,
        k,
        softmax_scale=softmax_scale,
        causal=causal,
    )
    triton_qk_scores = _triton_qk_scores(q, k, softmax_scale, causal)
    flash_qk_p_float_out = _outputs_from_qk_scores(
        flash_qk_scores,
        v,
        q_heads=q_heads,
        kv_heads=kv_heads,
        p_dtype=torch.float32,
    )
    triton_qk_p_float_out = _outputs_from_qk_scores(
        triton_qk_scores,
        v,
        q_heads=q_heads,
        kv_heads=kv_heads,
        p_dtype=torch.float32,
    )
    flash_qk_p_half_out = _outputs_from_qk_scores(
        flash_qk_scores,
        v,
        q_heads=q_heads,
        kv_heads=kv_heads,
        p_dtype=torch.float16,
    )
    triton_qk_p_half_out = _outputs_from_qk_scores(
        triton_qk_scores,
        v,
        q_heads=q_heads,
        kv_heads=kv_heads,
        p_dtype=torch.float16,
    )
    qk_online_outputs = {
        "flash_qk_scores_online_tile32_p_float_output": (
            _outputs_from_qk_scores_online(
                flash_qk_scores,
                v,
                q_heads=q_heads,
                kv_heads=kv_heads,
                tile_size=32,
                p_dtype=torch.float32,
            )
        ),
        "flash_qk_scores_online_tile32_p_half_output": (
            _outputs_from_qk_scores_online(
                flash_qk_scores,
                v,
                q_heads=q_heads,
                kv_heads=kv_heads,
                tile_size=32,
                p_dtype=torch.float16,
            )
        ),
        "flash_qk_scores_online_tile64_p_half_output": (
            _outputs_from_qk_scores_online(
                flash_qk_scores,
                v,
                q_heads=q_heads,
                kv_heads=kv_heads,
                tile_size=64,
                p_dtype=torch.float16,
            )
        ),
        "triton_qk_scores_online_tile32_p_float_output": (
            _outputs_from_qk_scores_online(
                triton_qk_scores,
                v,
                q_heads=q_heads,
                kv_heads=kv_heads,
                tile_size=32,
                p_dtype=torch.float32,
            )
        ),
        "triton_qk_scores_online_tile32_p_half_output": (
            _outputs_from_qk_scores_online(
                triton_qk_scores,
                v,
                q_heads=q_heads,
                kv_heads=kv_heads,
                tile_size=32,
                p_dtype=torch.float16,
            )
        ),
        "triton_qk_scores_online_tile64_p_half_output": (
            _outputs_from_qk_scores_online(
                triton_qk_scores,
                v,
                q_heads=q_heads,
                kv_heads=kv_heads,
                tile_size=64,
                p_dtype=torch.float16,
            )
        ),
    }
    flash_lse = flash_attn_v100.flash_attn_lse(
        q,
        k,
        v,
        softmax_scale=softmax_scale,
        causal=causal,
    )
    flash_qk_lse = _lse_from_qk_scores(flash_qk_scores)
    triton_qk_lse = _lse_from_qk_scores(triton_qk_scores)
    triton_lse_tile32 = _triton_qk_lse(
        q,
        k,
        softmax_scale,
        causal,
        tile_size=32,
    )
    triton_lse_tile64 = _triton_qk_lse(
        q,
        k,
        softmax_scale,
        causal,
        tile_size=64,
    )
    torch_lse = _torch_qk_lse(q, k, softmax_scale, causal)
    references = {
        "torch_full_softmax_fp32_pv_fp32": _ref_gqa_attention(
            q, k, v, softmax_scale, causal
        ),
        "torch_full_softmax_p_half": _ref_gqa_attention_p_half(
            q, k, v, softmax_scale, causal
        ),
        "torch_online_tile32_p_half": _ref_gqa_attention_online(
            q,
            k,
            v,
            softmax_scale,
            causal,
            tile_size=32,
            p_dtype=torch.float16,
        ),
        "torch_online_tile32_p_float": _ref_gqa_attention_online(
            q,
            k,
            v,
            softmax_scale,
            causal,
            tile_size=32,
            p_dtype=torch.float32,
        ),
        "torch_online_tile64_p_half": _ref_gqa_attention_online(
            q,
            k,
            v,
            softmax_scale,
            causal,
            tile_size=64,
            p_dtype=torch.float16,
        ),
        "torch_online_tile64_p_half_clamp80": _ref_gqa_attention_online(
            q,
            k,
            v,
            softmax_scale,
            causal,
            tile_size=64,
            p_dtype=torch.float16,
            clamp_exp_min=-80.0,
        ),
    }
    qk_compare = _compare_payload(
        "flash_v100_qk_scores",
        "triton_tl_dot_qk_scores",
        flash_qk_scores,
        triton_qk_scores,
    )
    qk_max_diff_detail = _single_qk_diff_detail(
        q,
        k,
        flash_qk_scores,
        triton_qk_scores,
        flat_index=qk_compare["max_diff_flat_index"],
        softmax_scale=softmax_scale,
        causal=causal,
    )
    output_compare = _compare_payload(
        "saved_flash_candidate",
        "saved_triton_reference",
        saved_flash,
        saved_triton,
    )
    output_max_diff_detail = _single_output_diff_detail(
        v,
        saved_flash,
        saved_triton,
        flash_qk_scores,
        triton_qk_scores,
        {
            "flash_qk_scores_p_float_output": flash_qk_p_float_out,
            "triton_qk_scores_p_float_output": triton_qk_p_float_out,
            "flash_qk_scores_p_half_output": flash_qk_p_half_out,
            "triton_qk_scores_p_half_output": triton_qk_p_half_out,
            **qk_online_outputs,
        },
        {
            "flash_v100_lse": flash_lse,
            "flash_qk_scores_lse": flash_qk_lse,
            "triton_qk_scores_lse": triton_qk_lse,
            "triton_online_lse_tile32": triton_lse_tile32,
            "triton_online_lse_tile64": triton_lse_tile64,
            "torch_logsumexp_fp32": torch_lse,
        },
        flat_index=output_compare["max_diff_flat_index"],
        causal=causal,
    )

    comparisons = [
        output_compare,
        _compare_payload(
            "flash_func_replay",
            "saved_flash_candidate",
            flash_replay,
            saved_flash,
        ),
        _compare_payload(
            "flash_func_replay",
            "saved_triton_reference",
            flash_replay,
            saved_triton,
        ),
        _compare_payload(
            "triton_unified_attention_replay",
            "saved_triton_reference",
            triton_replay,
            saved_triton,
        ),
        _compare_payload(
            "triton_unified_attention_replay",
            "saved_flash_candidate",
            triton_replay,
            saved_flash,
        ),
        qk_compare,
        _compare_payload(
            "flash_qk_scores_p_float_output",
            "saved_flash_candidate",
            flash_qk_p_float_out,
            saved_flash,
        ),
        _compare_payload(
            "flash_qk_scores_p_float_output",
            "saved_triton_reference",
            flash_qk_p_float_out,
            saved_triton,
        ),
        _compare_payload(
            "triton_qk_scores_p_float_output",
            "saved_flash_candidate",
            triton_qk_p_float_out,
            saved_flash,
        ),
        _compare_payload(
            "triton_qk_scores_p_float_output",
            "saved_triton_reference",
            triton_qk_p_float_out,
            saved_triton,
        ),
        _compare_payload(
            "flash_qk_scores_p_half_output",
            "saved_flash_candidate",
            flash_qk_p_half_out,
            saved_flash,
        ),
        _compare_payload(
            "flash_qk_scores_p_half_output",
            "saved_triton_reference",
            flash_qk_p_half_out,
            saved_triton,
        ),
        _compare_payload(
            "triton_qk_scores_p_half_output",
            "saved_flash_candidate",
            triton_qk_p_half_out,
            saved_flash,
        ),
        _compare_payload(
            "triton_qk_scores_p_half_output",
            "saved_triton_reference",
            triton_qk_p_half_out,
            saved_triton,
        ),
        _compare_payload(
            "flash_v100_lse",
            "flash_qk_scores_lse",
            flash_lse,
            flash_qk_lse,
        ),
        _compare_payload(
            "flash_v100_lse",
            "triton_qk_scores_lse",
            flash_lse,
            triton_qk_lse,
        ),
        _compare_payload(
            "flash_v100_lse",
            "triton_online_lse_tile32",
            flash_lse,
            triton_lse_tile32,
        ),
        _compare_payload(
            "flash_v100_lse",
            "triton_online_lse_tile64",
            flash_lse,
            triton_lse_tile64,
        ),
        _compare_payload(
            "flash_v100_lse",
            "torch_logsumexp_fp32",
            flash_lse,
            torch_lse,
        ),
        _compare_payload(
            "triton_online_lse_tile32",
            "triton_qk_scores_lse",
            triton_lse_tile32,
            triton_qk_lse,
        ),
    ]
    for name, ref in references.items():
        comparisons.append(
            _compare_payload(name, "saved_flash_candidate", ref, saved_flash)
        )
        comparisons.append(
            _compare_payload(name, "saved_triton_reference", ref, saved_triton)
        )
    for name, ref in qk_online_outputs.items():
        comparisons.append(
            _compare_payload(name, "saved_flash_candidate", ref, saved_flash)
        )
        comparisons.append(
            _compare_payload(name, "saved_triton_reference", ref, saved_triton)
        )

    return {
        "name": "tensor_dump_replay",
        "tensor_dump": str(tensor_dump),
        "stage": payload.get("stage"),
        "layer": payload.get("layer"),
        "num_actual_tokens": int(payload["num_actual_tokens"]),
        "softmax_scale": softmax_scale,
        "query_shape": list(payload["query"].shape),
        "key_shape": list(payload["raw_key"].shape),
        "value_shape": list(payload["raw_value"].shape),
        "compact_block_size": compact_block_size,
        "input_alignment": {
            "raw_key_vs_cache": raw_key_cache_compare,
            "raw_value_vs_cache": raw_value_cache_compare,
        },
        "qk_max_diff_detail": qk_max_diff_detail,
        "output_max_diff_detail": output_max_diff_detail,
        "comparisons": comparisons,
    }


def _find_comparison(
    replay: dict[str, Any],
    name: str,
    reference: str,
) -> dict[str, Any]:
    for comparison in replay["comparisons"]:
        if comparison["name"] == name and comparison["reference"] == reference:
            return comparison
    raise KeyError(f"missing comparison {name} vs {reference}")


def _compact_tensor_dump_replay(replay: dict[str, Any]) -> dict[str, Any]:
    output = _find_comparison(
        replay,
        "saved_flash_candidate",
        "saved_triton_reference",
    )
    qk = _find_comparison(
        replay,
        "flash_v100_qk_scores",
        "triton_tl_dot_qk_scores",
    )
    flash_replay = _find_comparison(
        replay,
        "flash_func_replay",
        "saved_flash_candidate",
    )
    triton_replay = _find_comparison(
        replay,
        "triton_unified_attention_replay",
        "saved_triton_reference",
    )
    flash_like_clamp80_vs_flash = _find_comparison(
        replay,
        "torch_online_tile64_p_half_clamp80",
        "saved_flash_candidate",
    )
    flash_like_clamp80_vs_triton = _find_comparison(
        replay,
        "torch_online_tile64_p_half_clamp80",
        "saved_triton_reference",
    )
    flash_qk_p_float_vs_flash = _find_comparison(
        replay,
        "flash_qk_scores_p_float_output",
        "saved_flash_candidate",
    )
    flash_qk_p_float_vs_triton = _find_comparison(
        replay,
        "flash_qk_scores_p_float_output",
        "saved_triton_reference",
    )
    triton_qk_p_float_vs_flash = _find_comparison(
        replay,
        "triton_qk_scores_p_float_output",
        "saved_flash_candidate",
    )
    triton_qk_p_float_vs_triton = _find_comparison(
        replay,
        "triton_qk_scores_p_float_output",
        "saved_triton_reference",
    )
    flash_qk_p_half_vs_flash = _find_comparison(
        replay,
        "flash_qk_scores_p_half_output",
        "saved_flash_candidate",
    )
    flash_qk_p_half_vs_triton = _find_comparison(
        replay,
        "flash_qk_scores_p_half_output",
        "saved_triton_reference",
    )
    triton_qk_p_half_vs_flash = _find_comparison(
        replay,
        "triton_qk_scores_p_half_output",
        "saved_flash_candidate",
    )
    triton_qk_p_half_vs_triton = _find_comparison(
        replay,
        "triton_qk_scores_p_half_output",
        "saved_triton_reference",
    )
    flash_lse_vs_flash_qk_lse = _find_comparison(
        replay,
        "flash_v100_lse",
        "flash_qk_scores_lse",
    )
    flash_lse_vs_triton_qk_lse = _find_comparison(
        replay,
        "flash_v100_lse",
        "triton_qk_scores_lse",
    )
    flash_lse_vs_triton_lse_tile32 = _find_comparison(
        replay,
        "flash_v100_lse",
        "triton_online_lse_tile32",
    )
    flash_lse_vs_triton_lse_tile64 = _find_comparison(
        replay,
        "flash_v100_lse",
        "triton_online_lse_tile64",
    )
    flash_lse_vs_torch_lse = _find_comparison(
        replay,
        "flash_v100_lse",
        "torch_logsumexp_fp32",
    )
    triton_lse_tile32_vs_triton_qk_lse = _find_comparison(
        replay,
        "triton_online_lse_tile32",
        "triton_qk_scores_lse",
    )
    output_detail = replay.get("output_max_diff_detail") or {}
    half_neighbor = output_detail.get("half_neighbor") or {}
    max_score_diff = output_detail.get("max_score_diff") or {}
    max_probability_diff = output_detail.get("max_probability_diff") or {}
    qk_detail = replay.get("qk_max_diff_detail") or {}
    qk_output_refs = output_detail.get("qk_output_refs_at_index") or {}
    flash_row_reduce = output_detail.get("flash_row_reduce") or {}
    triton_row_reduce = output_detail.get("triton_row_reduce") or {}
    lse_refs = output_detail.get("lse_refs_at_row") or {}
    saved_flash_value = output_detail.get("saved_flash_value")
    saved_triton_value = output_detail.get("saved_triton_value")
    input_alignment = replay.get("input_alignment") or {}
    key_alignment = input_alignment.get("raw_key_vs_cache")
    value_alignment = input_alignment.get("raw_value_vs_cache")
    layer = replay.get("layer") or {}

    def row_reduce_refs(values: dict[str, Any]) -> dict[str, dict[str, Any]]:
        refs = {}
        for name, value in values.items():
            refs[name] = {
                "value": value,
                "matches_saved_flash": value == saved_flash_value,
                "matches_saved_triton": value == saved_triton_value,
                "abs_diff_to_saved_flash": None
                if value is None or saved_flash_value is None
                else abs(float(value) - float(saved_flash_value)),
                "abs_diff_to_saved_triton": None
                if value is None or saved_triton_value is None
                else abs(float(value) - float(saved_triton_value)),
            }
        return refs

    return {
        "tensor_dump": replay["tensor_dump"],
        "stage": replay.get("stage"),
        "layer_name": layer.get("layer_name"),
        "num_actual_tokens": replay.get("num_actual_tokens"),
        "query_shape": replay.get("query_shape"),
        "key_shape": replay.get("key_shape"),
        "flash_replay_equal_saved": bool(flash_replay["equal"]),
        "triton_replay_equal_saved": bool(triton_replay["equal"]),
        "flash_like_clamp80_equal_saved_flash": bool(
            flash_like_clamp80_vs_flash["equal"]
        ),
        "flash_like_clamp80_vs_saved_flash_max_diff": (
            flash_like_clamp80_vs_flash["max_diff"]
        ),
        "flash_like_clamp80_vs_saved_triton_max_diff": (
            flash_like_clamp80_vs_triton["max_diff"]
        ),
        "flash_qk_p_float_vs_saved_flash_max_diff": (
            flash_qk_p_float_vs_flash["max_diff"]
        ),
        "flash_qk_p_float_vs_saved_triton_max_diff": (
            flash_qk_p_float_vs_triton["max_diff"]
        ),
        "triton_qk_p_float_vs_saved_flash_max_diff": (
            triton_qk_p_float_vs_flash["max_diff"]
        ),
        "triton_qk_p_float_vs_saved_triton_max_diff": (
            triton_qk_p_float_vs_triton["max_diff"]
        ),
        "flash_qk_p_half_vs_saved_flash_max_diff": (
            flash_qk_p_half_vs_flash["max_diff"]
        ),
        "flash_qk_p_half_vs_saved_triton_max_diff": (
            flash_qk_p_half_vs_triton["max_diff"]
        ),
        "triton_qk_p_half_vs_saved_flash_max_diff": (
            triton_qk_p_half_vs_flash["max_diff"]
        ),
        "triton_qk_p_half_vs_saved_triton_max_diff": (
            triton_qk_p_half_vs_triton["max_diff"]
        ),
        "flash_lse_vs_flash_qk_lse_max_diff": (
            flash_lse_vs_flash_qk_lse["max_diff"]
        ),
        "flash_lse_vs_triton_qk_lse_max_diff": (
            flash_lse_vs_triton_qk_lse["max_diff"]
        ),
        "flash_lse_vs_triton_lse_tile32_max_diff": (
            flash_lse_vs_triton_lse_tile32["max_diff"]
        ),
        "flash_lse_vs_triton_lse_tile64_max_diff": (
            flash_lse_vs_triton_lse_tile64["max_diff"]
        ),
        "flash_lse_vs_torch_lse_max_diff": (
            flash_lse_vs_torch_lse["max_diff"]
        ),
        "triton_lse_tile32_vs_triton_qk_lse_max_diff": (
            triton_lse_tile32_vs_triton_qk_lse["max_diff"]
        ),
        "output_equal": bool(output["equal"]),
        "output_max_diff": output["max_diff"],
        "output_mean_diff": output["mean_diff"],
        "output_num_different": output["num_different"],
        "output_is_one_half_ulp": bool(half_neighbor.get("is_one_half_ulp", False)),
        "output_half_abs_diff": half_neighbor.get("abs_diff"),
        "output_valid_key_count": output_detail.get("valid_key_count"),
        "output_max_saved_flash_value": saved_flash_value,
        "output_max_saved_triton_value": saved_triton_value,
        "output_max_qk_refs": qk_output_refs,
        "output_max_qk_ref_matches": row_reduce_refs(qk_output_refs),
        "output_max_flash_row_reduce_refs": row_reduce_refs(flash_row_reduce),
        "output_max_triton_row_reduce_refs": row_reduce_refs(triton_row_reduce),
        "output_max_lse_refs": lse_refs,
        "output_max_flash_qk_p_float_matches_saved_flash": (
            qk_output_refs.get("flash_qk_scores_p_float_output")
            == saved_flash_value
        ),
        "output_max_flash_qk_p_float_matches_saved_triton": (
            qk_output_refs.get("flash_qk_scores_p_float_output")
            == saved_triton_value
        ),
        "output_max_triton_qk_p_float_matches_saved_flash": (
            qk_output_refs.get("triton_qk_scores_p_float_output")
            == saved_flash_value
        ),
        "output_max_triton_qk_p_float_matches_saved_triton": (
            qk_output_refs.get("triton_qk_scores_p_float_output")
            == saved_triton_value
        ),
        "output_max_flash_qk_p_half_matches_saved_flash": (
            qk_output_refs.get("flash_qk_scores_p_half_output")
            == saved_flash_value
        ),
        "output_max_flash_qk_p_half_matches_saved_triton": (
            qk_output_refs.get("flash_qk_scores_p_half_output")
            == saved_triton_value
        ),
        "output_max_triton_qk_p_half_matches_saved_flash": (
            qk_output_refs.get("triton_qk_scores_p_half_output")
            == saved_flash_value
        ),
        "output_max_triton_qk_p_half_matches_saved_triton": (
            qk_output_refs.get("triton_qk_scores_p_half_output")
            == saved_triton_value
        ),
        "output_max_score_diff": max_score_diff.get("abs_diff"),
        "output_max_probability_diff": max_probability_diff.get("abs_diff"),
        "raw_key_cache_equal": None
        if key_alignment is None
        else bool(key_alignment["equal"]),
        "raw_key_cache_max_diff": None
        if key_alignment is None
        else key_alignment["max_diff"],
        "raw_value_cache_equal": None
        if value_alignment is None
        else bool(value_alignment["equal"]),
        "raw_value_cache_max_diff": None
        if value_alignment is None
        else value_alignment["max_diff"],
        "qk_max_diff": qk["max_diff"],
        "qk_mean_diff": qk["mean_diff"],
        "qk_num_different": qk["num_different"],
        "qk_max_causal_valid": qk_detail.get("causal_valid"),
        "qk_max_decoded_index": qk_detail.get("decoded_index"),
    }


def run_tensor_dump_replay_dir(
    flash_attn_v100: Any,
    tensor_dump_dir: Path,
) -> dict[str, Any]:
    paths = sorted(tensor_dump_dir.glob("*.pt"))
    if not paths:
        raise FileNotFoundError(f"no tensor dumps found in {tensor_dump_dir}")

    summaries = []
    aggregate: dict[str, Any] = {
        "tensor_dump_dir": str(tensor_dump_dir),
        "num_dumps": len(paths),
        "num_output_equal": 0,
        "num_output_nonzero": 0,
        "num_nonzero_one_half_ulp": 0,
        "num_nonzero_not_one_half_ulp": 0,
        "num_output_above_fp16_nominal_bound": 0,
        "num_output_above_fp16_nominal_one_half_ulp": 0,
        "num_output_above_fp16_nominal_not_one_half_ulp": 0,
        "num_flash_replay_mismatch": 0,
        "num_triton_replay_mismatch": 0,
        "num_flash_like_clamp80_mismatch_saved_flash": 0,
        "num_output_max_flash_qk_p_float_matches_saved_flash": 0,
        "num_output_max_flash_qk_p_float_matches_saved_triton": 0,
        "num_output_max_triton_qk_p_float_matches_saved_flash": 0,
        "num_output_max_triton_qk_p_float_matches_saved_triton": 0,
        "num_output_max_flash_qk_p_half_matches_saved_flash": 0,
        "num_output_max_flash_qk_p_half_matches_saved_triton": 0,
        "num_output_max_triton_qk_p_half_matches_saved_flash": 0,
        "num_output_max_triton_qk_p_half_matches_saved_triton": 0,
        "num_raw_key_cache_mismatch": 0,
        "num_raw_value_cache_mismatch": 0,
        "num_qk_max_invalid_causal": 0,
        "max_output_diff": 0.0,
        "max_qk_diff": 0.0,
        "max_output_probability_diff": 0.0,
        "max_output_score_diff": 0.0,
        "max_flash_like_clamp80_vs_saved_flash_diff": 0.0,
        "max_flash_like_clamp80_vs_saved_triton_diff": 0.0,
        "max_flash_qk_p_float_vs_saved_flash_diff": 0.0,
        "max_flash_qk_p_float_vs_saved_triton_diff": 0.0,
        "max_triton_qk_p_float_vs_saved_flash_diff": 0.0,
        "max_triton_qk_p_float_vs_saved_triton_diff": 0.0,
        "max_flash_qk_p_half_vs_saved_flash_diff": 0.0,
        "max_flash_qk_p_half_vs_saved_triton_diff": 0.0,
        "max_triton_qk_p_half_vs_saved_flash_diff": 0.0,
        "max_triton_qk_p_half_vs_saved_triton_diff": 0.0,
        "max_flash_lse_vs_flash_qk_lse_diff": 0.0,
        "max_flash_lse_vs_triton_qk_lse_diff": 0.0,
        "max_flash_lse_vs_triton_lse_tile32_diff": 0.0,
        "max_flash_lse_vs_triton_lse_tile64_diff": 0.0,
        "max_flash_lse_vs_torch_lse_diff": 0.0,
        "max_triton_lse_tile32_vs_triton_qk_lse_diff": 0.0,
        "worst_output": None,
        "worst_qk": None,
        "qk_ref_match_counts": {},
        "qk_ref_max_abs_diff": {},
        "row_reduce_match_counts": {},
        "row_reduce_max_abs_diff": {},
    }

    def update_ref_match_aggregate(
        match_counts_key: str,
        max_abs_diff_key: str,
        family: str,
        refs: dict[str, dict[str, Any]],
    ) -> None:
        for name, ref in refs.items():
            key = f"{family}.{name}"
            if key not in aggregate[match_counts_key]:
                aggregate[match_counts_key][key] = {
                    "saved_flash": 0,
                    "saved_triton": 0,
                }
                aggregate[max_abs_diff_key][key] = {
                    "saved_flash": 0.0,
                    "saved_triton": 0.0,
                }
            if ref.get("matches_saved_flash"):
                aggregate[match_counts_key][key]["saved_flash"] += 1
            if ref.get("matches_saved_triton"):
                aggregate[match_counts_key][key]["saved_triton"] += 1
            flash_diff = ref.get("abs_diff_to_saved_flash")
            if flash_diff is not None:
                aggregate[max_abs_diff_key][key]["saved_flash"] = max(
                    aggregate[max_abs_diff_key][key]["saved_flash"],
                    float(flash_diff),
                )
            triton_diff = ref.get("abs_diff_to_saved_triton")
            if triton_diff is not None:
                aggregate[max_abs_diff_key][key]["saved_triton"] = max(
                    aggregate[max_abs_diff_key][key]["saved_triton"],
                    float(triton_diff),
                )

    for path in paths:
        replay = run_tensor_dump_replay(flash_attn_v100, path)
        summary = _compact_tensor_dump_replay(replay)
        summaries.append(summary)
        update_ref_match_aggregate(
            "qk_ref_match_counts",
            "qk_ref_max_abs_diff",
            "qk_output",
            summary["output_max_qk_ref_matches"],
        )
        update_ref_match_aggregate(
            "row_reduce_match_counts",
            "row_reduce_max_abs_diff",
            "flash_row_reduce",
            summary["output_max_flash_row_reduce_refs"],
        )
        update_ref_match_aggregate(
            "row_reduce_match_counts",
            "row_reduce_max_abs_diff",
            "triton_row_reduce",
            summary["output_max_triton_row_reduce_refs"],
        )

        if summary["output_equal"]:
            aggregate["num_output_equal"] += 1
        else:
            aggregate["num_output_nonzero"] += 1
            if summary["output_is_one_half_ulp"]:
                aggregate["num_nonzero_one_half_ulp"] += 1
            else:
                aggregate["num_nonzero_not_one_half_ulp"] += 1
        if summary["output_max_diff"] > 2.0e-3:
            aggregate["num_output_above_fp16_nominal_bound"] += 1
            if summary["output_is_one_half_ulp"]:
                aggregate["num_output_above_fp16_nominal_one_half_ulp"] += 1
            else:
                aggregate["num_output_above_fp16_nominal_not_one_half_ulp"] += 1
        if not summary["flash_replay_equal_saved"]:
            aggregate["num_flash_replay_mismatch"] += 1
        if not summary["triton_replay_equal_saved"]:
            aggregate["num_triton_replay_mismatch"] += 1
        if not summary["flash_like_clamp80_equal_saved_flash"]:
            aggregate["num_flash_like_clamp80_mismatch_saved_flash"] += 1
        if summary["output_max_flash_qk_p_float_matches_saved_flash"]:
            aggregate["num_output_max_flash_qk_p_float_matches_saved_flash"] += 1
        if summary["output_max_flash_qk_p_float_matches_saved_triton"]:
            aggregate["num_output_max_flash_qk_p_float_matches_saved_triton"] += 1
        if summary["output_max_triton_qk_p_float_matches_saved_flash"]:
            aggregate["num_output_max_triton_qk_p_float_matches_saved_flash"] += 1
        if summary["output_max_triton_qk_p_float_matches_saved_triton"]:
            aggregate["num_output_max_triton_qk_p_float_matches_saved_triton"] += 1
        if summary["output_max_flash_qk_p_half_matches_saved_flash"]:
            aggregate["num_output_max_flash_qk_p_half_matches_saved_flash"] += 1
        if summary["output_max_flash_qk_p_half_matches_saved_triton"]:
            aggregate["num_output_max_flash_qk_p_half_matches_saved_triton"] += 1
        if summary["output_max_triton_qk_p_half_matches_saved_flash"]:
            aggregate["num_output_max_triton_qk_p_half_matches_saved_flash"] += 1
        if summary["output_max_triton_qk_p_half_matches_saved_triton"]:
            aggregate["num_output_max_triton_qk_p_half_matches_saved_triton"] += 1
        if summary["raw_key_cache_equal"] is False:
            aggregate["num_raw_key_cache_mismatch"] += 1
        if summary["raw_value_cache_equal"] is False:
            aggregate["num_raw_value_cache_mismatch"] += 1
        if summary["qk_max_causal_valid"] is False:
            aggregate["num_qk_max_invalid_causal"] += 1
        if summary["output_max_diff"] > aggregate["max_output_diff"]:
            aggregate["max_output_diff"] = summary["output_max_diff"]
            aggregate["worst_output"] = summary
        if summary["qk_max_diff"] > aggregate["max_qk_diff"]:
            aggregate["max_qk_diff"] = summary["qk_max_diff"]
            aggregate["worst_qk"] = summary
        probability_diff = summary["output_max_probability_diff"]
        if probability_diff is not None:
            aggregate["max_output_probability_diff"] = max(
                aggregate["max_output_probability_diff"],
                probability_diff,
            )
        score_diff = summary["output_max_score_diff"]
        if score_diff is not None:
            aggregate["max_output_score_diff"] = max(
                aggregate["max_output_score_diff"],
                score_diff,
            )
        aggregate["max_flash_like_clamp80_vs_saved_flash_diff"] = max(
            aggregate["max_flash_like_clamp80_vs_saved_flash_diff"],
            summary["flash_like_clamp80_vs_saved_flash_max_diff"],
        )
        aggregate["max_flash_like_clamp80_vs_saved_triton_diff"] = max(
            aggregate["max_flash_like_clamp80_vs_saved_triton_diff"],
            summary["flash_like_clamp80_vs_saved_triton_max_diff"],
        )
        aggregate["max_flash_qk_p_float_vs_saved_flash_diff"] = max(
            aggregate["max_flash_qk_p_float_vs_saved_flash_diff"],
            summary["flash_qk_p_float_vs_saved_flash_max_diff"],
        )
        aggregate["max_flash_qk_p_float_vs_saved_triton_diff"] = max(
            aggregate["max_flash_qk_p_float_vs_saved_triton_diff"],
            summary["flash_qk_p_float_vs_saved_triton_max_diff"],
        )
        aggregate["max_triton_qk_p_float_vs_saved_flash_diff"] = max(
            aggregate["max_triton_qk_p_float_vs_saved_flash_diff"],
            summary["triton_qk_p_float_vs_saved_flash_max_diff"],
        )
        aggregate["max_triton_qk_p_float_vs_saved_triton_diff"] = max(
            aggregate["max_triton_qk_p_float_vs_saved_triton_diff"],
            summary["triton_qk_p_float_vs_saved_triton_max_diff"],
        )
        aggregate["max_flash_qk_p_half_vs_saved_flash_diff"] = max(
            aggregate["max_flash_qk_p_half_vs_saved_flash_diff"],
            summary["flash_qk_p_half_vs_saved_flash_max_diff"],
        )
        aggregate["max_flash_qk_p_half_vs_saved_triton_diff"] = max(
            aggregate["max_flash_qk_p_half_vs_saved_triton_diff"],
            summary["flash_qk_p_half_vs_saved_triton_max_diff"],
        )
        aggregate["max_triton_qk_p_half_vs_saved_flash_diff"] = max(
            aggregate["max_triton_qk_p_half_vs_saved_flash_diff"],
            summary["triton_qk_p_half_vs_saved_flash_max_diff"],
        )
        aggregate["max_triton_qk_p_half_vs_saved_triton_diff"] = max(
            aggregate["max_triton_qk_p_half_vs_saved_triton_diff"],
            summary["triton_qk_p_half_vs_saved_triton_max_diff"],
        )
        aggregate["max_flash_lse_vs_flash_qk_lse_diff"] = max(
            aggregate["max_flash_lse_vs_flash_qk_lse_diff"],
            summary["flash_lse_vs_flash_qk_lse_max_diff"],
        )
        aggregate["max_flash_lse_vs_triton_qk_lse_diff"] = max(
            aggregate["max_flash_lse_vs_triton_qk_lse_diff"],
            summary["flash_lse_vs_triton_qk_lse_max_diff"],
        )
        aggregate["max_flash_lse_vs_triton_lse_tile32_diff"] = max(
            aggregate["max_flash_lse_vs_triton_lse_tile32_diff"],
            summary["flash_lse_vs_triton_lse_tile32_max_diff"],
        )
        aggregate["max_flash_lse_vs_triton_lse_tile64_diff"] = max(
            aggregate["max_flash_lse_vs_triton_lse_tile64_diff"],
            summary["flash_lse_vs_triton_lse_tile64_max_diff"],
        )
        aggregate["max_flash_lse_vs_torch_lse_diff"] = max(
            aggregate["max_flash_lse_vs_torch_lse_diff"],
            summary["flash_lse_vs_torch_lse_max_diff"],
        )
        aggregate["max_triton_lse_tile32_vs_triton_qk_lse_diff"] = max(
            aggregate["max_triton_lse_tile32_vs_triton_qk_lse_diff"],
            summary["triton_lse_tile32_vs_triton_qk_lse_max_diff"],
        )
        torch.cuda.empty_cache()

    aggregate["all_replays_reproduce_saved"] = (
        aggregate["num_flash_replay_mismatch"] == 0
        and aggregate["num_triton_replay_mismatch"] == 0
    )
    aggregate["all_raw_kv_cache_aligned"] = (
        aggregate["num_raw_key_cache_mismatch"] == 0
        and aggregate["num_raw_value_cache_mismatch"] == 0
    )
    aggregate["all_nonzero_outputs_one_half_ulp"] = (
        aggregate["num_output_nonzero"] == aggregate["num_nonzero_one_half_ulp"]
    )
    aggregate["all_above_bound_outputs_one_half_ulp"] = (
        aggregate["num_output_above_fp16_nominal_bound"]
        == aggregate["num_output_above_fp16_nominal_one_half_ulp"]
    )
    aggregate["diff_decomposition"] = _decompose_tensor_dump_replay_dir_diff(
        aggregate
    )
    aggregate["classification"] = _classify_tensor_dump_replay_dir(aggregate)
    return {
        "name": "tensor_dump_replay_dir",
        "aggregate": aggregate,
        "summaries": summaries,
    }


def _decompose_tensor_dump_replay_dir_diff(
    aggregate: dict[str, Any]
) -> dict[str, Any]:
    input_layout_bug_count = (
        aggregate["num_flash_replay_mismatch"]
        + aggregate["num_triton_replay_mismatch"]
        + aggregate["num_raw_key_cache_mismatch"]
        + aggregate["num_raw_value_cache_mismatch"]
    )
    causal_mask_bug_count = aggregate["num_qk_max_invalid_causal"]
    above_bound_not_half_ulp = aggregate[
        "num_output_above_fp16_nominal_not_one_half_ulp"
    ]
    a_bug_suspect_count = (
        input_layout_bug_count
        + causal_mask_bug_count
        + above_bound_not_half_ulp
    )
    within_bound_nonzero = (
        aggregate["num_output_nonzero"]
        - aggregate["num_output_above_fp16_nominal_bound"]
    )
    return {
        "input_or_layout_bug_suspect_count": input_layout_bug_count,
        "causal_mask_bug_suspect_count": causal_mask_bug_count,
        "above_fp16_bound_not_half_ulp_count": above_bound_not_half_ulp,
        "a_bug_suspect_count": a_bug_suspect_count,
        "within_fp16_bound_nonzero_count": within_bound_nonzero,
        "above_fp16_bound_half_ulp_count": aggregate[
            "num_output_above_fp16_nominal_one_half_ulp"
        ],
        "rounding_noise_candidate_count": (
            within_bound_nonzero
            + aggregate["num_output_above_fp16_nominal_one_half_ulp"]
        ),
    }


def _classify_tensor_dump_replay_dir(aggregate: dict[str, Any]) -> dict[str, Any]:
    cleared = []
    pending = []
    decomposition = aggregate["diff_decomposition"]
    if aggregate["all_replays_reproduce_saved"]:
        cleared.append("replay reproduces both saved Flash and saved Triton outputs")
    else:
        pending.append("replay mismatch makes the diagnostic evidence invalid")
    if aggregate["all_raw_kv_cache_aligned"]:
        cleared.append("raw K/V and extracted cache K/V are aligned")
    else:
        pending.append("raw-vs-cache K/V mismatch remains; treat as A-bug")
    if aggregate["num_qk_max_invalid_causal"] == 0:
        cleared.append("QK max-diff elements are causal-valid")
    else:
        pending.append("QK max-diff includes invalid causal elements")
    if decomposition["a_bug_suspect_count"] == 0:
        cleared.append(
            "bug-suspect component is zero under replay/input/causal/ULP checks"
        )
    else:
        pending.append(
            "bug-suspect component remains nonzero under replay/input/causal/ULP "
            "checks"
        )

    max_output_diff = aggregate["max_output_diff"]
    if max_output_diff > FP16_NOMINAL_OUTPUT_BOUND:
        if aggregate["all_above_bound_outputs_one_half_ulp"]:
            pending.append(
                "some final fp16 outputs exceed the nominal bound, but only as "
                "adjacent fp16 values near rounding boundaries"
            )
        else:
            pending.append("some final fp16 outputs exceed the nominal bound")
    if aggregate["num_output_nonzero"] > 0:
        pending.append(
            "Flash-vs-Triton is cross-implementation nonzero and still needs "
            "broader seqlen/model-level evidence before default acceptance"
        )
    if "num_output_max_flash_qk_p_float_matches_saved_flash" in aggregate:
        pending.append(
            "QK/PV reconstruction at the max-output-diff element is mixed: "
            f"flash_qk_p_float->Flash "
            f"{aggregate['num_output_max_flash_qk_p_float_matches_saved_flash']}/"
            f"{aggregate['num_dumps']}, flash_qk_p_float->Triton "
            f"{aggregate['num_output_max_flash_qk_p_float_matches_saved_triton']}/"
            f"{aggregate['num_dumps']}, triton_qk_p_float->Flash "
            f"{aggregate['num_output_max_triton_qk_p_float_matches_saved_flash']}/"
            f"{aggregate['num_dumps']}, triton_qk_p_float->Triton "
            f"{aggregate['num_output_max_triton_qk_p_float_matches_saved_triton']}/"
            f"{aggregate['num_dumps']}"
        )
    if "num_output_max_flash_qk_p_half_matches_saved_flash" in aggregate:
        pending.append(
            "QK/PV p-half reconstruction at the max-output-diff element is "
            "also mixed: "
            f"flash_qk_p_half->Flash "
            f"{aggregate['num_output_max_flash_qk_p_half_matches_saved_flash']}/"
            f"{aggregate['num_dumps']}, flash_qk_p_half->Triton "
            f"{aggregate['num_output_max_flash_qk_p_half_matches_saved_triton']}/"
            f"{aggregate['num_dumps']}, triton_qk_p_half->Flash "
            f"{aggregate['num_output_max_triton_qk_p_half_matches_saved_flash']}/"
            f"{aggregate['num_dumps']}, triton_qk_p_half->Triton "
            f"{aggregate['num_output_max_triton_qk_p_half_matches_saved_triton']}/"
            f"{aggregate['num_dumps']}"
        )
    if "max_flash_lse_vs_flash_qk_lse_diff" in aggregate:
        pending.append(
            "LSE/normalization remains cross-implementation nonzero: "
            f"flash_lse-vs-flash_qk_lse max_diff="
            f"{aggregate['max_flash_lse_vs_flash_qk_lse_diff']}, "
            f"flash_lse-vs-triton_qk_lse max_diff="
            f"{aggregate['max_flash_lse_vs_triton_qk_lse_diff']}, "
            f"flash_lse-vs-triton_lse_tile32 max_diff="
            f"{aggregate['max_flash_lse_vs_triton_lse_tile32_diff']}"
        )

    def best_match(
        counts_by_name: dict[str, dict[str, int]],
        target: str,
    ) -> tuple[str, int]:
        best_name = ""
        best_count = -1
        for name, counts in counts_by_name.items():
            count = int(counts[target])
            if count > best_count:
                best_name = name
                best_count = count
        return best_name, best_count

    if aggregate.get("qk_ref_match_counts"):
        flash_name, flash_count = best_match(
            aggregate["qk_ref_match_counts"],
            "saved_flash",
        )
        triton_name, triton_count = best_match(
            aggregate["qk_ref_match_counts"],
            "saved_triton",
        )
        pending.append(
            "Precomputed-QK online-softmax/PV reconstruction is still mixed: "
            f"best Flash match {flash_name}={flash_count}/"
            f"{aggregate['num_dumps']}, best Triton match {triton_name}="
            f"{triton_count}/{aggregate['num_dumps']}"
        )
    if aggregate.get("row_reduce_match_counts"):
        flash_name, flash_count = best_match(
            aggregate["row_reduce_match_counts"],
            "saved_flash",
        )
        triton_name, triton_count = best_match(
            aggregate["row_reduce_match_counts"],
            "saved_triton",
        )
        pending.append(
            "PV/final-cast row-reduce reconstruction at the max-output-diff "
            "element is still mixed: best Flash match "
            f"{flash_name}={flash_count}/{aggregate['num_dumps']}, best Triton "
            f"match {triton_name}={triton_count}/{aggregate['num_dumps']}"
        )

    if (
        aggregate["all_replays_reproduce_saved"]
        and aggregate["all_raw_kv_cache_aligned"]
        and aggregate["num_qk_max_invalid_causal"] == 0
        and aggregate["num_output_nonzero"] == 0
    ):
        label = "A-pass"
        default_acceptance = "op-level exact for this replay set"
    elif (
        aggregate["all_replays_reproduce_saved"]
        and aggregate["all_raw_kv_cache_aligned"]
        and aggregate["num_qk_max_invalid_causal"] == 0
    ):
        label = "B-pending"
        default_acceptance = "not default-accepted"
    else:
        label = "A-bug"
        default_acceptance = "blocked until input/layout evidence is repaired"

    return {
        "label": label,
        "path_type": "Flash-vs-Triton cross-implementation replay",
        "fp16_nominal_output_bound": FP16_NOMINAL_OUTPUT_BOUND,
        "default_acceptance": default_acceptance,
        "diff_decomposition": decomposition,
        "cleared_evidence": cleared,
        "pending_evidence": pending,
    }


def _make_partitioned_paged_inputs(
    q: torch.Tensor,
    block_table: torch.Tensor,
    seq_len: int,
    block_size: int,
    partition_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_partitions = (seq_len + partition_size - 1) // partition_size
    blocks_per_partition = (partition_size + block_size - 1) // block_size
    partition_block_table = torch.zeros(
        (num_partitions, blocks_per_partition),
        device=q.device,
        dtype=torch.int32,
    )
    partition_seq_lens = torch.empty(
        (num_partitions,),
        device=q.device,
        dtype=torch.int32,
    )
    for partition_idx in range(num_partitions):
        start = partition_idx * partition_size
        end = min(start + partition_size, seq_len)
        first_block = start // block_size
        num_blocks = (end - start + block_size - 1) // block_size
        partition_block_table[partition_idx, :num_blocks] = block_table[
            0, first_block : first_block + num_blocks
        ]
        partition_seq_lens[partition_idx] = end - start

    q_bhmd = q.permute(0, 2, 1, 3).contiguous()
    q_partitions = q_bhmd.expand(num_partitions, -1, -1, -1).contiguous()
    return q_partitions, partition_block_table, partition_seq_lens


def run_partitioned_wmma_probe_cases(
    flash_attn_v100: Any,
    *,
    warmup: int,
    iters: int,
) -> list[dict[str, Any]]:
    cases = [
        (4096, 16, 256),
        (4096, 16, 512),
        (4096, 16, 1024),
        (8192, 16, 512),
        (16384, 16, 512),
        (32768, 16, 512),
    ]
    q_heads = 6
    kv_heads = 1
    head_dim = 256
    query_len = 1
    payloads: list[dict[str, Any]] = []
    for seq_len, block_size, partition_size in cases:
        seed = seq_len * 1000 + partition_size + 1201
        q, k, v = _make_qkv(
            seq_len=seq_len,
            query_len=query_len,
            q_heads=q_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            seed=seed,
        )
        softmax_scale = 1.0 / math.sqrt(head_dim)
        key_cache, value_cache, block_table, seq_lens = _make_paged_cache(
            k, v, block_size
        )
        q_partitions, partition_block_table, partition_seq_lens = (
            _make_partitioned_paged_inputs(
                q,
                block_table,
                seq_len,
                block_size,
                partition_size,
            )
        )
        partitioned_wmma = flash_attn_v100.flash_attn_prefill_paged_bhmd(
            q_partitions,
            key_cache,
            value_cache,
            partition_block_table,
            partition_seq_lens,
            softmax_scale=softmax_scale,
            causal=False,
        )
        partitioned_wmma = (
            partitioned_wmma.squeeze(2).permute(1, 0, 2).contiguous()
        )
        dense_partition_outputs = _flash_dense_partition_outputs(
            flash_attn_v100,
            q,
            k,
            v,
            softmax_scale,
            partition_size=partition_size,
        )
        dense_flash = flash_attn_v100.flash_attn_func(
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            causal=False,
        )
        partitioned_reference = _flash_partitioned_decode_reference(
            flash_attn_v100,
            q,
            k,
            v,
            softmax_scale,
            partition_size=partition_size,
        )
        q_flat = q.squeeze(0).contiguous()
        scalar_out = torch.empty_like(q_flat)

        def run_partitioned_wmma(
            q_partitions: torch.Tensor = q_partitions,
            key_cache: torch.Tensor = key_cache,
            value_cache: torch.Tensor = value_cache,
            partition_block_table: torch.Tensor = partition_block_table,
            partition_seq_lens: torch.Tensor = partition_seq_lens,
            softmax_scale: float = softmax_scale,
        ) -> None:
            flash_attn_v100.flash_attn_prefill_paged_bhmd(
                q_partitions,
                key_cache,
                value_cache,
                partition_block_table,
                partition_seq_lens,
                softmax_scale=softmax_scale,
                causal=False,
            )

        def run_scalar_decode(
            q_flat: torch.Tensor = q_flat,
            key_cache: torch.Tensor = key_cache,
            value_cache: torch.Tensor = value_cache,
            block_table: torch.Tensor = block_table,
            seq_lens: torch.Tensor = seq_lens,
            softmax_scale: float = softmax_scale,
            scalar_out: torch.Tensor = scalar_out,
        ) -> None:
            flash_attn_v100.flash_attn_decode_paged(
                q_flat,
                key_cache,
                value_cache,
                block_table,
                seq_lens,
                softmax_scale=softmax_scale,
                out=scalar_out,
                kv_cache_dtype="auto",
            )

        def run_wmma_decode(
            q_flat: torch.Tensor = q_flat,
            key_cache: torch.Tensor = key_cache,
            value_cache: torch.Tensor = value_cache,
            block_table: torch.Tensor = block_table,
            seq_lens: torch.Tensor = seq_lens,
            softmax_scale: float = softmax_scale,
            scalar_out: torch.Tensor = scalar_out,
        ) -> None:
            flash_attn_v100.flash_attn_decode_paged_wmma(
                q_flat,
                key_cache,
                value_cache,
                block_table,
                seq_lens,
                softmax_scale=softmax_scale,
                out=scalar_out,
                kv_cache_dtype="auto",
            )

        partitioned_ms = _time_ms(run_partitioned_wmma, warmup, iters)
        scalar_ms = _time_ms(run_scalar_decode, warmup, iters)
        wmma_decode_ms = _time_ms(run_wmma_decode, warmup, iters)
        timing_ms = {
            "partitioned_wmma_prefill_compute_only": partitioned_ms,
            "scalar_decode": scalar_ms,
            "full_wmma_decode": wmma_decode_ms,
            "partitioned_wmma_vs_scalar_ratio": (
                partitioned_ms / scalar_ms
                if scalar_ms > 0.0
                else float("inf")
            ),
            "full_wmma_vs_scalar_ratio": (
                wmma_decode_ms / scalar_ms
                if scalar_ms > 0.0
                else float("inf")
            ),
        }
        comparisons = {
            "partitioned_wmma_vs_dense_partition": _compare_payload(
                "partitioned_wmma_prefill",
                "dense_flash_partition_outputs",
                partitioned_wmma,
                dense_partition_outputs,
            ),
            "partitioned_flash_reference_vs_dense_flash": _compare_payload(
                "flash_partitioned_decode_reference",
                "dense_flash_v100",
                partitioned_reference,
                dense_flash,
            ),
        }
        payloads.append(
            {
                "name": "partitioned_wmma_decode_probe",
                "seq_len": seq_len,
                "query_len": query_len,
                "q_heads": q_heads,
                "kv_heads": kv_heads,
                "head_dim": head_dim,
                "block_size": block_size,
                "partition_size": partition_size,
                "num_partitions": (seq_len + partition_size - 1)
                // partition_size,
                **comparisons,
                "timing_ms": timing_ms,
            }
        )
    return payloads


def run_noncontiguous_edge_cases(flash_attn_v100: Any) -> list[dict[str, Any]]:
    seq_len = 69
    query_len = 16
    q_heads = 6
    kv_heads = 1
    head_dim = 256
    block_size = 16
    seed = 424242
    q, k, v = _make_qkv(
        seq_len=seq_len,
        query_len=query_len,
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        seed=seed,
    )
    softmax_scale = 1.0 / math.sqrt(head_dim)
    q_nc = _strided_view_along_dim(q, 1)
    k_nc = _strided_view_along_dim(k, 1)
    v_nc = _strided_view_along_dim(v, 1)
    dense_contig = flash_attn_v100.flash_attn_func(
        q,
        k,
        v,
        causal=True,
        softmax_scale=softmax_scale,
    )
    dense_nc = flash_attn_v100.flash_attn_func(
        q_nc,
        k_nc,
        v_nc,
        causal=True,
        softmax_scale=softmax_scale,
    )

    key_cache, value_cache, block_table, seq_lens = _make_paged_cache(
        k, v, block_size
    )
    key_cache_nc = _strided_view_along_dim(key_cache, 0)
    value_cache_nc = _strided_view_along_dim(value_cache, 0)
    block_table_nc = _strided_view_along_dim(block_table, 1)
    decode_block_table = block_table.expand(query_len, -1).contiguous()
    decode_block_table_nc = _strided_view_along_dim(decode_block_table, 1)
    decode_seq_lens = (
        seq_len
        - query_len
        + torch.arange(1, query_len + 1, device=q.device, dtype=torch.int32)
    )
    decode_seq_lens_nc = _strided_view_along_dim(decode_seq_lens, 0)

    paged_contig = flash_attn_v100.flash_attn_prefill_paged(
        q,
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        softmax_scale=softmax_scale,
        causal=True,
    )
    paged_nc = flash_attn_v100.flash_attn_prefill_paged(
        q_nc,
        key_cache_nc,
        value_cache_nc,
        block_table_nc,
        seq_lens,
        softmax_scale=softmax_scale,
        causal=True,
    )

    q_flat = q.squeeze(0).contiguous()
    q_flat_nc = _strided_view_along_dim(q_flat, 0)
    decode_contig = flash_attn_v100.flash_attn_decode_paged(
        q_flat,
        key_cache,
        value_cache,
        decode_block_table,
        decode_seq_lens,
        softmax_scale=softmax_scale,
        kv_cache_dtype="auto",
    )
    decode_nc = flash_attn_v100.flash_attn_decode_paged(
        q_flat_nc,
        key_cache_nc,
        value_cache_nc,
        decode_block_table_nc,
        decode_seq_lens_nc,
        softmax_scale=softmax_scale,
        kv_cache_dtype="auto",
    )

    return [
        {
            "name": "noncontiguous_dense_inputs",
            "input_contiguous": {
                "q": q_nc.is_contiguous(),
                "k": k_nc.is_contiguous(),
                "v": v_nc.is_contiguous(),
            },
            "input_strides": {
                "q": list(q_nc.stride()),
                "k": list(k_nc.stride()),
                "v": list(v_nc.stride()),
            },
            "comparison": _compare_payload(
                "dense_flash_v100_noncontiguous",
                "dense_flash_v100_contiguous",
                dense_nc,
                dense_contig,
            ),
        },
        {
            "name": "noncontiguous_paged_prefill_inputs",
            "input_contiguous": {
                "q": q_nc.is_contiguous(),
                "key_cache": key_cache_nc.is_contiguous(),
                "value_cache": value_cache_nc.is_contiguous(),
                "block_table": block_table_nc.is_contiguous(),
                "seq_lens": seq_lens.is_contiguous(),
            },
            "input_strides": {
                "q": list(q_nc.stride()),
                "key_cache": list(key_cache_nc.stride()),
                "value_cache": list(value_cache_nc.stride()),
                "block_table": list(block_table_nc.stride()),
                "seq_lens": list(seq_lens.stride()),
            },
            "comparison": _compare_payload(
                "paged_prefill_noncontiguous",
                "paged_prefill_contiguous",
                paged_nc,
                paged_contig,
            ),
        },
        {
            "name": "noncontiguous_paged_decode_inputs",
            "input_contiguous": {
                "q": q_flat_nc.is_contiguous(),
                "key_cache": key_cache_nc.is_contiguous(),
                "value_cache": value_cache_nc.is_contiguous(),
                "block_table": decode_block_table_nc.is_contiguous(),
                "seq_lens": decode_seq_lens_nc.is_contiguous(),
            },
            "input_strides": {
                "q": list(q_flat_nc.stride()),
                "key_cache": list(key_cache_nc.stride()),
                "value_cache": list(value_cache_nc.stride()),
                "block_table": list(decode_block_table_nc.stride()),
                "seq_lens": list(decode_seq_lens_nc.stride()),
            },
            "comparison": _compare_payload(
                "paged_decode_noncontiguous",
                "paged_decode_contiguous",
                decode_nc,
                decode_contig,
            ),
        },
    ]


def run_edge_case_diagnostics(flash_attn_v100: Any) -> dict[str, Any]:
    return {
        "mask_cases": run_mask_edge_cases(flash_attn_v100),
        "varlen_cases": run_varlen_edge_cases(flash_attn_v100),
        "long_sequence_cases": run_long_sequence_edge_cases(flash_attn_v100),
        "noncontiguous_cases": run_noncontiguous_edge_cases(flash_attn_v100),
    }


def _edge_cases_passed(edge_cases: dict[str, Any]) -> bool:
    def check_comparison(comparison: dict[str, Any]) -> bool:
        return bool(comparison["equal"] and comparison["max_diff"] == 0.0)

    for result in edge_cases["mask_cases"]:
        if "comparison" in result:
            if not check_comparison(result["comparison"]):
                return False
        elif "comparisons" in result:
            if not all(check_comparison(item) for item in result["comparisons"]):
                return False
        elif "causal_matches_reference" in result:
            if not check_comparison(result["causal_matches_reference"]):
                return False
            if not check_comparison(result["noncausal_matches_reference"]):
                return False
            if result["causal_vs_noncausal"]["equal"]:
                return False
        elif not (result["equal"] and result["max_diff"] == 0.0):
            return False
    for category in ("varlen_cases", "long_sequence_cases", "noncontiguous_cases"):
        for result in edge_cases[category]:
            if any(key.endswith("_error") for key in result):
                return False
            if "comparison" in result and not check_comparison(result["comparison"]):
                return False
            if "comparisons" in result and not all(
                check_comparison(item) for item in result["comparisons"]
            ):
                return False
    return True


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _replay_aggregate(payload: dict[str, Any]) -> dict[str, Any]:
    replay_dir = payload.get("replay_dir")
    if isinstance(replay_dir, dict) and isinstance(replay_dir.get("aggregate"), dict):
        return replay_dir["aggregate"]
    if isinstance(payload.get("aggregate"), dict):
        return payload["aggregate"]
    return payload


def _summarize_attention_replay(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    aggregate = _replay_aggregate(payload)
    classification = aggregate.get("classification", {})
    diff_decomposition = aggregate.get("diff_decomposition", {})

    input_aligned = bool(aggregate.get("all_raw_kv_cache_aligned", False))
    input_aligned = input_aligned and aggregate.get("num_raw_key_cache_mismatch") == 0
    input_aligned = input_aligned and aggregate.get("num_raw_value_cache_mismatch") == 0
    causal_clean = aggregate.get("num_qk_max_invalid_causal") == 0
    replay_reproduced = aggregate.get("num_flash_replay_mismatch") == 0
    replay_reproduced = replay_reproduced and (
        aggregate.get("num_triton_replay_mismatch") == 0
    )
    a_bug_suspect = int(diff_decomposition.get("a_bug_suspect_count", 0) or 0)
    layout_suspect = int(
        diff_decomposition.get("input_or_layout_bug_suspect_count", 0) or 0
    )
    causal_suspect = int(
        diff_decomposition.get("causal_mask_bug_suspect_count", 0) or 0
    )

    if (
        not input_aligned
        or not causal_clean
        or not replay_reproduced
        or a_bug_suspect
        or layout_suspect
        or causal_suspect
    ):
        label = "A-bug"
    else:
        label = classification.get("label") or "B-pending"

    return {
        "path": str(path),
        "label": label,
        "reported_label": classification.get("label"),
        "default_acceptance": classification.get("default_acceptance"),
        "input_alignment": {
            "all_raw_kv_cache_aligned": aggregate.get("all_raw_kv_cache_aligned"),
            "num_raw_key_cache_mismatch": aggregate.get(
                "num_raw_key_cache_mismatch"
            ),
            "num_raw_value_cache_mismatch": aggregate.get(
                "num_raw_value_cache_mismatch"
            ),
            "num_qk_max_invalid_causal": aggregate.get("num_qk_max_invalid_causal"),
            "num_flash_replay_mismatch": aggregate.get("num_flash_replay_mismatch"),
            "num_triton_replay_mismatch": aggregate.get(
                "num_triton_replay_mismatch"
            ),
        },
        "diff_decomposition": diff_decomposition,
        "output": {
            "max_output_diff": aggregate.get("max_output_diff"),
            "max_output_probability_diff": aggregate.get(
                "max_output_probability_diff"
            ),
            "max_output_score_diff": aggregate.get("max_output_score_diff"),
            "max_qk_diff": aggregate.get("max_qk_diff"),
            "fp16_nominal_output_bound": classification.get(
                "fp16_nominal_output_bound", FP16_NOMINAL_OUTPUT_BOUND
            ),
        },
        "qk_matched_refs": {
            "qk_ref_match_counts": aggregate.get("qk_ref_match_counts"),
            "qk_ref_max_abs_diff": aggregate.get("qk_ref_max_abs_diff"),
        },
        "pv_final_cast": {
            "row_reduce_match_counts": aggregate.get("row_reduce_match_counts"),
            "row_reduce_max_abs_diff": aggregate.get("row_reduce_max_abs_diff"),
        },
        "cleared_evidence": classification.get("cleared_evidence", []),
        "pending_evidence": classification.get("pending_evidence", []),
    }


def _summarize_model_compare(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    gate = payload.get("model_quality_gate", {})
    checks = gate.get("checks", {})
    return {
        "path": str(path),
        "label": gate.get("label", "B-pending"),
        "default_acceptance": gate.get("default_acceptance"),
        "bounds": gate.get("bounds", {}),
        "checks": checks,
        "token_equal": payload.get("equal", checks.get("token_equal")),
        "pending_evidence": gate.get("pending_evidence", []),
        "failed_evidence": gate.get("failed_evidence", []),
    }


def _build_numeric_gate_summary(
    attention: dict[str, Any],
    model: dict[str, Any] | None,
    selfnoise: dict[str, Any] | None,
) -> dict[str, Any]:
    input_alignment = attention["input_alignment"]
    decomposition = attention["diff_decomposition"]
    output = attention["output"]
    current_input_layer_clean = (
        input_alignment["all_raw_kv_cache_aligned"] is True
        and input_alignment["num_raw_key_cache_mismatch"] == 0
        and input_alignment["num_raw_value_cache_mismatch"] == 0
        and input_alignment["num_qk_max_invalid_causal"] == 0
        and input_alignment["num_flash_replay_mismatch"] == 0
        and input_alignment["num_triton_replay_mismatch"] == 0
        and int(decomposition.get("a_bug_suspect_count", 0) or 0) == 0
    )
    max_output_diff = output.get("max_output_diff")
    fp16_bound = output.get(
        "fp16_nominal_output_bound",
        FP16_NOMINAL_OUTPUT_BOUND,
    )
    dtype_bound_satisfied = (
        max_output_diff is not None and float(max_output_diff) <= float(fp16_bound)
    )
    matched_tiling_proven = attention["label"] == "B-accept"
    model_gate_passed = model is not None and model.get("label") == "model-pass"

    blockers = []
    if not current_input_layer_clean:
        blockers.append(
            "current input/source/mask/replay evidence is not clean; treat as A-bug"
        )
    if attention["label"] == "B-pending":
        blockers.append(
            "matched block/partition replay has not collapsed Flash-vs-Triton "
            "attention diff to zero"
        )
    if not dtype_bound_satisfied:
        blockers.append(
            "max_output_diff exceeds the configured fp16 output bound; adjacent "
            "half-ULP evidence can explain rounding candidates but does not by "
            "itself satisfy B-accept"
        )
    if not model_gate_passed:
        blockers.append(
            "model-level token/logprob/perplexity gate is missing or failed"
        )

    return {
        "policy": "classified max_diff gate",
        "current_input_source_mask": {
            "label": "A-pass" if current_input_layer_clean else "A-bug",
            "evidence": input_alignment,
            "diff_decomposition": decomposition,
        },
        "flash_vs_triton_attention_residual": {
            "label": attention["label"],
            "output": output,
            "dtype_bound_satisfied": dtype_bound_satisfied,
            "matched_tiling_proven": matched_tiling_proven,
            "b_accept_blockers": blockers,
        },
        "model_level": {
            "label": None if model is None else model.get("label"),
            "token_equal": None if model is None else model.get("token_equal"),
            "selfnoise_label": None
            if selfnoise is None
            else selfnoise.get("label"),
            "note": (
                "model-pass is supporting evidence only; it cannot promote an "
                "op-level B-pending attention residual to B-accept."
            ),
        },
        "default_decision": (
            "split-route-only: Triton prefill plus scalar Flash-V100 decode"
            if blockers
            else "eligible-after-route-and-speed-gates"
        ),
    }


def run_classification_summary(
    attention_replay_json: Path,
    model_compare_json: Path | None,
    selfnoise_compare_json: Path | None,
) -> dict[str, Any]:
    attention = _summarize_attention_replay(attention_replay_json)
    model = (
        _summarize_model_compare(model_compare_json)
        if model_compare_json is not None
        else None
    )
    selfnoise = (
        _summarize_model_compare(selfnoise_compare_json)
        if selfnoise_compare_json is not None
        else None
    )

    if attention["label"] == "A-bug":
        overall_label = "A-bug"
    elif attention["label"] != "B-accept" or (
        model is None or model["label"] != "model-pass"
    ):
        overall_label = "B-pending"
    else:
        overall_label = "B-accept"

    return {
        "mode": "classification-summary",
        "overall_label": overall_label,
        "default_acceptance": "not default-accepted"
        if overall_label != "B-accept"
        else "eligible-after-route-and-speed-gates",
        "attention": attention,
        "model_compare": model,
        "selfnoise_compare": selfnoise,
        "numeric_gate": _build_numeric_gate_summary(attention, model, selfnoise),
        "decision": (
            "Input/source and causal bug components are cleared in the supplied "
            "synchronized replay, but the attention route remains B-pending "
            "until the replay itself is promoted to B-accept."
            if overall_label == "B-pending"
            else "See overall_label."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=(
            "all",
            "dense-prefill",
            "paged-prefill",
            "paged-decode",
            "decode-cudagraph-replay-diagnostic",
            "arithmetic-diagnostic",
            "basis-value-diagnostic",
            "edge-cases",
            "varlen-layout-diagnostic",
            "long-decode-matched-diagnostic",
            "partitioned-wmma-probe",
            "qk-pattern-diagnostic",
            "qk-reference-diagnostic",
            "qk-score-diagnostic",
            "softmax-lse-diagnostic",
            "softmax-tile-sim-diagnostic",
            "softmax-update-trace-diagnostic",
            "source-parity",
            "classification-summary",
            "tensor-dump-replay",
            "tensor-dump-replay-dir",
        ),
        default="all",
    )
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument(
        "--reference-source-dir",
        type=Path,
        help="Reference flash-attention-v100 source tree for source-parity mode.",
    )
    parser.add_argument(
        "--candidate-source-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "flash-attention-v100",
        help="Candidate flash-attention-v100 source tree for source-parity mode.",
    )
    parser.add_argument(
        "--tensor-dump",
        type=Path,
        help="Tensor dump produced by VLLM_FLASH_V100_COMPARE_TRITON_TENSOR_DUMP_DIR.",
    )
    parser.add_argument(
        "--tensor-dump-dir",
        type=Path,
        help="Directory of tensor dumps for tensor-dump-replay-dir mode.",
    )
    parser.add_argument(
        "--attention-replay-json",
        type=Path,
        help="Existing tensor-dump replay JSON for classification-summary mode.",
    )
    parser.add_argument(
        "--model-compare-json",
        type=Path,
        help="Existing model compare JSON for classification-summary mode.",
    )
    parser.add_argument(
        "--selfnoise-compare-json",
        type=Path,
        help=(
            "Existing same-backend self-noise compare JSON for "
            "classification-summary mode."
        ),
    )
    parser.add_argument(
        "--allow-nonzero",
        action="store_true",
        help="Exit 0 even when strict equality fails.",
    )
    parser.add_argument(
        "--benchmark-decode",
        action="store_true",
        help="Also record op-level decode timing for scalar/paged-prefill/dense paths.",
    )
    parser.add_argument(
        "--compare-triton",
        action="store_true",
        help=(
            "Also compare dense/paged Flash-V100 outputs with Triton "
            "unified_attention."
        ),
    )
    parser.add_argument("--benchmark-warmup", type=int, default=30)
    parser.add_argument("--benchmark-iters", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_parity = None
    if args.mode == "classification-summary":
        if args.attention_replay_json is None:
            raise ValueError(
                "--attention-replay-json is required for classification-summary"
            )
        payload = run_classification_summary(
            args.attention_replay_json,
            args.model_compare_json,
            args.selfnoise_compare_json,
        )
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if args.allow_nonzero or payload["overall_label"] != "A-bug" else 1

    if args.mode == "source-parity":
        if args.reference_source_dir is None:
            raise ValueError("--reference-source-dir is required for source-parity")
        source_parity = run_source_parity(
            args.reference_source_dir,
            args.candidate_source_dir,
        )
        payload = {
            "mode": args.mode,
            "passed": bool(source_parity["passed"]),
            "source_parity": source_parity,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps(payload, indent=2, sort_keys=True))
        if not payload["passed"] and not args.allow_nonzero:
            return 1
        return 0

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    capability = torch.cuda.get_device_capability()
    if capability != (7, 0):
        raise RuntimeError(f"SM70/V100 is required, got capability={capability}")

    import flash_attn_v100

    if args.mode == "tensor-dump-replay":
        if args.tensor_dump is None:
            raise ValueError("--tensor-dump is required for tensor-dump-replay")
        replay = run_tensor_dump_replay(flash_attn_v100, args.tensor_dump)
        payload = {
            "device": torch.cuda.get_device_name(),
            "device_capability": list(capability),
            "torch": {
                "version": torch.__version__,
                "cuda": torch.version.cuda,
            },
            "mode": args.mode,
            "passed": False,
            "replay": replay,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if args.allow_nonzero else 1

    if args.mode == "tensor-dump-replay-dir":
        if args.tensor_dump_dir is None:
            raise ValueError(
                "--tensor-dump-dir is required for tensor-dump-replay-dir"
            )
        replay_dir = run_tensor_dump_replay_dir(flash_attn_v100, args.tensor_dump_dir)
        payload = {
            "device": torch.cuda.get_device_name(),
            "device_capability": list(capability),
            "torch": {
                "version": torch.__version__,
                "cuda": torch.version.cuda,
            },
            "mode": args.mode,
            "passed": False,
            "replay_dir": replay_dir,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if args.allow_nonzero else 1

    if args.mode == "arithmetic-diagnostic":
        diagnostics = run_arithmetic_diagnostic_cases(flash_attn_v100)
        payload = {
            "device": torch.cuda.get_device_name(),
            "device_capability": list(capability),
            "torch": {
                "version": torch.__version__,
                "cuda": torch.version.cuda,
            },
            "mode": args.mode,
            "passed": False,
            "diagnostics": diagnostics,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if args.allow_nonzero else 1

    if args.mode == "basis-value-diagnostic":
        diagnostics = run_basis_value_diagnostic_cases(flash_attn_v100)
        payload = {
            "device": torch.cuda.get_device_name(),
            "device_capability": list(capability),
            "torch": {
                "version": torch.__version__,
                "cuda": torch.version.cuda,
            },
            "mode": args.mode,
            "passed": False,
            "diagnostics": diagnostics,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if args.allow_nonzero else 1

    if args.mode == "edge-cases":
        edge_cases = run_edge_case_diagnostics(flash_attn_v100)
        passed = _edge_cases_passed(edge_cases)
        payload = {
            "device": torch.cuda.get_device_name(),
            "device_capability": list(capability),
            "torch": {
                "version": torch.__version__,
                "cuda": torch.version.cuda,
            },
            "mode": args.mode,
            "passed": passed,
            "edge_cases": edge_cases,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps(payload, indent=2, sort_keys=True))
        if not passed and not args.allow_nonzero:
            return 1
        return 0

    if args.mode == "varlen-layout-diagnostic":
        diagnostics = run_varlen_layout_diagnostic_cases(flash_attn_v100)
        passed = all(
            comparison["equal"] and comparison["max_diff"] == 0.0
            for result in diagnostics
            for comparison in result["comparisons"]
            if comparison["reference"]
            in (
                "flash_v100_full_cache_looped_paged_prefill",
                "flash_v100_standalone_looped_paged_prefill",
                "triton_unified_attention_looped",
            )
        )
        payload = {
            "device": torch.cuda.get_device_name(),
            "device_capability": list(capability),
            "torch": {
                "version": torch.__version__,
                "cuda": torch.version.cuda,
            },
            "mode": args.mode,
            "passed": passed,
            "diagnostics": diagnostics,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps(payload, indent=2, sort_keys=True))
        if not passed and not args.allow_nonzero:
            return 1
        return 0

    if args.mode == "long-decode-matched-diagnostic":
        diagnostics = run_long_decode_matched_diagnostic_cases(flash_attn_v100)
        payload = {
            "device": torch.cuda.get_device_name(),
            "device_capability": list(capability),
            "torch": {
                "version": torch.__version__,
                "cuda": torch.version.cuda,
            },
            "mode": args.mode,
            "passed": False,
            "diagnostics": diagnostics,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if args.allow_nonzero else 1

    if args.mode == "partitioned-wmma-probe":
        diagnostics = run_partitioned_wmma_probe_cases(
            flash_attn_v100,
            warmup=args.benchmark_warmup,
            iters=args.benchmark_iters,
        )
        passed = all(
            result["partitioned_wmma_vs_dense_partition"]["equal"]
            and result["partitioned_wmma_vs_dense_partition"]["max_diff"] == 0.0
            for result in diagnostics
        )
        payload = {
            "device": torch.cuda.get_device_name(),
            "device_capability": list(capability),
            "torch": {
                "version": torch.__version__,
                "cuda": torch.version.cuda,
            },
            "mode": args.mode,
            "passed": passed,
            "diagnostics": diagnostics,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps(payload, indent=2, sort_keys=True))
        if not passed and not args.allow_nonzero:
            return 1
        return 0

    if args.mode == "decode-cudagraph-replay-diagnostic":
        diagnostics = run_decode_cudagraph_replay_diagnostic_cases(
            flash_attn_v100
        )
        passed = all(result["equal"] for result in diagnostics)
        payload = {
            "device": torch.cuda.get_device_name(),
            "device_capability": list(capability),
            "torch": {
                "version": torch.__version__,
                "cuda": torch.version.cuda,
            },
            "mode": args.mode,
            "passed": passed,
            "diagnostics": diagnostics,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps(payload, indent=2, sort_keys=True))
        if not passed and not args.allow_nonzero:
            return 1
        return 0

    if args.mode == "qk-pattern-diagnostic":
        diagnostics = run_qk_pattern_diagnostic_cases(flash_attn_v100)
        payload = {
            "device": torch.cuda.get_device_name(),
            "device_capability": list(capability),
            "torch": {
                "version": torch.__version__,
                "cuda": torch.version.cuda,
            },
            "mode": args.mode,
            "passed": False,
            "diagnostics": diagnostics,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if args.allow_nonzero else 1

    if args.mode == "qk-reference-diagnostic":
        diagnostics = run_qk_reference_diagnostic_cases(flash_attn_v100)
        payload = {
            "device": torch.cuda.get_device_name(),
            "device_capability": list(capability),
            "torch": {
                "version": torch.__version__,
                "cuda": torch.version.cuda,
            },
            "mode": args.mode,
            "passed": False,
            "diagnostics": diagnostics,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if args.allow_nonzero else 1

    if args.mode == "qk-score-diagnostic":
        diagnostics = run_qk_score_diagnostic_cases(flash_attn_v100)
        payload = {
            "device": torch.cuda.get_device_name(),
            "device_capability": list(capability),
            "torch": {
                "version": torch.__version__,
                "cuda": torch.version.cuda,
            },
            "mode": args.mode,
            "passed": False,
            "diagnostics": diagnostics,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if args.allow_nonzero else 1

    if args.mode == "softmax-lse-diagnostic":
        diagnostics = run_softmax_lse_diagnostic_cases(flash_attn_v100)
        payload = {
            "device": torch.cuda.get_device_name(),
            "device_capability": list(capability),
            "torch": {
                "version": torch.__version__,
                "cuda": torch.version.cuda,
            },
            "mode": args.mode,
            "passed": False,
            "diagnostics": diagnostics,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if args.allow_nonzero else 1

    if args.mode == "softmax-tile-sim-diagnostic":
        diagnostics = run_softmax_tile_sim_diagnostic_cases(flash_attn_v100)
        payload = {
            "device": torch.cuda.get_device_name(),
            "device_capability": list(capability),
            "torch": {
                "version": torch.__version__,
                "cuda": torch.version.cuda,
            },
            "mode": args.mode,
            "passed": False,
            "diagnostics": diagnostics,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if args.allow_nonzero else 1

    if args.mode == "softmax-update-trace-diagnostic":
        diagnostics = run_softmax_update_trace_diagnostic_cases(flash_attn_v100)
        payload = {
            "device": torch.cuda.get_device_name(),
            "device_capability": list(capability),
            "torch": {
                "version": torch.__version__,
                "cuda": torch.version.cuda,
            },
            "mode": args.mode,
            "passed": False,
            "diagnostics": diagnostics,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if args.allow_nonzero else 1

    results: list[CaseResult] = []
    if args.mode in ("all", "dense-prefill"):
        results.extend(run_dense_prefill_cases(flash_attn_v100))
    if args.mode in ("all", "paged-prefill"):
        results.extend(run_paged_prefill_cases(flash_attn_v100))
    if args.mode in ("all", "paged-decode"):
        results.extend(run_paged_decode_cases(flash_attn_v100))
    if args.compare_triton:
        results.extend(run_triton_compare_cases(flash_attn_v100))
    timings = []
    if args.benchmark_decode:
        timings = run_decode_timing_cases(
            flash_attn_v100,
            warmup=args.benchmark_warmup,
            iters=args.benchmark_iters,
        )

    passed = all(result.equal and result.max_diff == 0.0 for result in results)
    payload = {
        "device": torch.cuda.get_device_name(),
        "device_capability": list(capability),
        "torch": {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "mode": args.mode,
        "passed": passed,
        "results": [asdict(result) for result in results],
        "timings": timings,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not passed and not args.allow_nonzero:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
