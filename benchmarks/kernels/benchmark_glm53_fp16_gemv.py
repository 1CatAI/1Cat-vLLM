# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Screen GLM-5.3 FP16 decode GEMV candidates on SM70.

Microbenchmark error bounds are diagnostic only. A candidate is not accepted
until the real-model logits and output-quality gates pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from vllm.triton_utils import tl, triton

_SM70_GLM53_FP16_GEMV_CONFIGS: dict[tuple[int, int], tuple[int, int]] = {
    (6416, 4096): (1024, 2),  # KDA fused q/k/v/b/f_a/g_a
    (4096, 2048): (1024, 4),  # KDA output projection
    (6144, 4096): (1024, 2),  # dense MLP gate/up
    (4096, 3072): (1024, 2),  # dense MLP down
    (1024, 4096): (1024, 4),  # shared-expert gate/up
    (2048, 4096): (1024, 2),  # MLA fused q/kv A projection
    (4096, 1536): (512, 4),  # MLA/indexer B projection
    (4096, 4096): (1024, 4),  # MLA output projection
    (32, 4096): (1024, 4),  # indexer weights
    (128, 4096): (1024, 4),  # indexer key projection
    (1024, 1536): (512, 4),  # TP-sharded indexer query B
}


@triton.jit
def _sm70_glm53_fp16_gemv_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    acc = 0.0
    for block_start in tl.static_range(0, K, BLOCK_K):
        x = tl.load(x_ptr + block_start + offsets).to(tl.float32)
        weight = tl.load(weight_ptr + row * K + block_start + offsets).to(tl.float32)
        acc += tl.sum(x * weight, axis=0)
    tl.store(out_ptr + row, acc)


@triton.jit
def _sm70_glm53_fp16_gemv_evict_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    acc = 0.0
    for block_start in tl.static_range(0, K, BLOCK_K):
        x = tl.load(
            x_ptr + block_start + offsets,
            eviction_policy="evict_last",
        ).to(tl.float32)
        weight = tl.load(
            weight_ptr + row * K + block_start + offsets,
            eviction_policy="evict_first",
        ).to(tl.float32)
        acc += tl.sum(x * weight, axis=0)
    tl.store(out_ptr + row, acc)


@triton.jit
def _sm70_glm53_fp16_gemv_cg_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    acc = 0.0
    for block_start in tl.static_range(0, K, BLOCK_K):
        x = tl.load(
            x_ptr + block_start + offsets,
            cache_modifier=".ca",
        ).to(tl.float32)
        weight = tl.load(
            weight_ptr + row * K + block_start + offsets,
            cache_modifier=".cg",
        ).to(tl.float32)
        acc += tl.sum(x * weight, axis=0)
    tl.store(out_ptr + row, acc)


@triton.jit
def _pair_fold_8x16(values, WIDTH: tl.constexpr):
    lo, hi = tl.split(tl.reshape(values, (8, WIDTH // 2, 2)))
    return lo + hi


@triton.jit
def _sm70_glm53_cublas_match_gemv_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    NUM_CHUNKS,
    CHUNK_TILES,
    N: tl.constexpr,
    K: tl.constexpr,
    VECTOR_WIDTH: tl.constexpr,
    BUTTERFLY_DOWN: tl.constexpr,
):
    """Reconstruct the observed 128-thread cuBLAS gemv2T reduction tree."""
    rows = tl.program_id(0) * 8 + tl.arange(0, 8)
    row_mask = rows < N
    lanes = tl.arange(0, 16)
    if BUTTERFLY_DOWN:
        # A bit-reversed lane order turns shfl_down into adjacent pair folds.
        lane_slots = (
            (lanes & 1) * 8
            + ((lanes >> 1) & 1) * 4
            + ((lanes >> 2) & 1) * 2
            + ((lanes >> 3) & 1)
        )
    else:
        lane_slots = lanes

    lane_offsets = lane_slots.to(tl.int64) * VECTOR_WIDTH
    acc = tl.zeros((8, 16), dtype=tl.float32)
    for chunk in range(0, NUM_CHUNKS):
        chunk_acc = tl.zeros((8, 16), dtype=tl.float32)
        chunk_base = chunk * CHUNK_TILES * VECTOR_WIDTH * 16
        for tile in range(0, CHUNK_TILES):
            tile_base = chunk_base + tile * VECTOR_WIDTH * 16
            for item in tl.static_range(0, VECTOR_WIDTH):
                k = tile_base + lane_offsets + item
                k_mask = k < K
                x = tl.load(x_ptr + k, mask=k_mask, other=0.0)
                weight = tl.load(
                    weight_ptr + rows[:, None] * K + k[None, :],
                    mask=row_mask[:, None] & k_mask[None, :],
                    other=0.0,
                )
                chunk_acc += x[None, :].to(tl.float32) * weight.to(tl.float32)
        acc = tl.where(chunk == 0, chunk_acc, acc + chunk_acc)

    if BUTTERFLY_DOWN:
        reduced = _pair_fold_8x16(acc, 16)
        reduced = _pair_fold_8x16(reduced, 8)
        reduced = _pair_fold_8x16(reduced, 4)
        reduced = _pair_fold_8x16(reduced, 2)
        total = tl.reshape(reduced, (8,))
    else:
        total = tl.zeros((8,), dtype=tl.float32)
        for lane in tl.static_range(0, 16):
            value = tl.sum(tl.where(lanes[None, :] == lane, acc, 0.0), axis=1)
            total = value if lane == 0 else total + value
    tl.store(out_ptr + rows, total, mask=row_mask)


def _measure_graph_us(
    launch: Callable[[], None], *, warmups: int, repeats: int
) -> float:
    for _ in range(warmups):
        launch()
    torch.accelerator.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    for _ in range(warmups):
        graph.replay()
    torch.accelerator.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / repeats


def _benchmark_kernel(
    name: str,
    kernel: Any,
    x: torch.Tensor,
    weight: torch.Tensor,
    reference: torch.Tensor,
    fp32_reference: torch.Tensor,
    *,
    block_k: int,
    num_warps: int,
    warmups: int,
    repeats: int,
) -> dict[str, Any]:
    n, k = weight.shape
    out = torch.empty((1, n), dtype=x.dtype, device=x.device)

    def launch() -> None:
        kernel[(n,)](
            x,
            weight,
            out,
            K=k,
            BLOCK_K=block_k,
            num_warps=num_warps,
        )

    latency_us = _measure_graph_us(launch, warmups=warmups, repeats=repeats)
    error = out.float() - reference.float()
    fp32_error = out.float() - fp32_reference
    cublas_fp32_error = reference.float() - fp32_reference
    traffic_bytes = 2 * (weight.numel() + x.numel() + out.numel())
    return {
        "kernel": name,
        "latency_us": latency_us,
        "effective_gbps": traffic_bytes / (latency_us * 1000.0),
        "exact_equal": torch.equal(out, reference),
        "output_sha256": hashlib.sha256(
            out.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
        ).hexdigest(),
        "max_abs_error": error.abs().max().item(),
        "relative_l2_error": (error.norm() / reference.float().norm()).item(),
        "cosine_similarity": torch.nn.functional.cosine_similarity(
            out.float(), reference.float()
        ).item(),
        "fp32_reference_relative_l2_error": (
            fp32_error.norm() / fp32_reference.norm()
        ).item(),
        "cublas_fp16_fp32_reference_relative_l2_error": (
            cublas_fp32_error.norm() / fp32_reference.norm()
        ).item(),
    }


def _benchmark_cublas_match_kernel(
    x: torch.Tensor,
    weight: torch.Tensor,
    reference: torch.Tensor,
    fp32_reference: torch.Tensor,
    *,
    vector_width: int,
    chunk_tiles: int,
    butterfly_down: bool,
    warmups: int,
    repeats: int,
) -> dict[str, Any]:
    n, k = weight.shape
    out = torch.empty((1, n), dtype=x.dtype, device=x.device)

    def launch() -> None:
        _sm70_glm53_cublas_match_gemv_kernel[(triton.cdiv(n, 8),)](
            x,
            weight,
            out,
            triton.cdiv(k, chunk_tiles * vector_width * 16),
            chunk_tiles,
            N=n,
            K=k,
            VECTOR_WIDTH=vector_width,
            BUTTERFLY_DOWN=butterfly_down,
            num_warps=4,
        )

    latency_us = _measure_graph_us(launch, warmups=warmups, repeats=repeats)
    error = out.float() - reference.float()
    fp32_error = out.float() - fp32_reference
    traffic_bytes = 2 * (weight.numel() + x.numel() + out.numel())
    return {
        "kernel": "cublas_match_gemv2t",
        "vector_width": vector_width,
        "chunk_tiles": chunk_tiles,
        "butterfly_down": butterfly_down,
        "latency_us": latency_us,
        "effective_gbps": traffic_bytes / (latency_us * 1000.0),
        "exact_equal": torch.equal(out, reference),
        "output_sha256": hashlib.sha256(
            out.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
        ).hexdigest(),
        "max_abs_error": error.abs().max().item(),
        "relative_l2_error": (error.norm() / reference.float().norm()).item(),
        "cosine_similarity": torch.nn.functional.cosine_similarity(
            out.float(), reference.float()
        ).item(),
        "fp32_reference_relative_l2_error": (
            fp32_error.norm() / fp32_reference.norm()
        ).item(),
    }


def _load_glm53_cuda_extension() -> None:
    from torch.utils.cpp_extension import load

    source = (
        Path(__file__).resolve().parents[2]
        / "csrc/sm70_turbomind/ops/glm53_fp16_gemv_sm70.cu"
    )
    load(
        name="_C_glm53_gemv_bench",
        sources=[str(source)],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-DVLLM_GLM53_GEMV_STANDALONE"],
        with_cuda=True,
        is_python_module=False,
        verbose=True,
    )


def _benchmark_cuda_chunk_parallel_kernel(
    x: torch.Tensor,
    weight: torch.Tensor,
    reference: torch.Tensor,
    fp32_reference: torch.Tensor,
    *,
    half2_rows: int,
    warmups: int,
    repeats: int,
) -> dict[str, Any]:
    os.environ["VLLM_SM70_GLM53_EXACT_KDA_HALF2_ROWS"] = str(half2_rows)
    out = torch.empty_like(reference)

    def launch() -> None:
        torch.ops._C_glm53_gemv_bench.sm70_glm53_fp16_gemv_out(out, x, weight)

    latency_us = _measure_graph_us(launch, warmups=warmups, repeats=repeats)
    error = out.float() - reference.float()
    fp32_error = out.float() - fp32_reference
    traffic_bytes = 2 * (weight.numel() + x.numel() + out.numel())
    return {
        "kernel": "cuda_cublas_match_gemv2t",
        "half2_rows": half2_rows,
        "latency_us": latency_us,
        "effective_gbps": traffic_bytes / (latency_us * 1000.0),
        "exact_equal": torch.equal(out, reference),
        "output_sha256": hashlib.sha256(
            out.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
        ).hexdigest(),
        "max_abs_error": error.abs().max().item(),
        "relative_l2_error": (error.norm() / reference.float().norm()).item(),
        "cosine_similarity": torch.nn.functional.cosine_similarity(
            out.float().reshape(1, -1), reference.float().reshape(1, -1)
        ).item(),
        "fp32_reference_relative_l2_error": (
            fp32_error.norm() / fp32_reference.norm()
        ).item(),
    }


def _benchmark_turbomind(
    x: torch.Tensor,
    weight: torch.Tensor,
    reference: torch.Tensor,
    fp32_reference: torch.Tensor,
    *,
    warmups: int,
    repeats: int,
) -> dict[str, Any]:
    from vllm import _sm70_ops as sm70_ops

    prepared_weight, meta = sm70_ops.sm70_f16_prepare(weight)
    k_ld = int(meta[0].item())
    out = torch.empty((x.shape[0], weight.shape[0]), dtype=x.dtype, device=x.device)

    def launch() -> None:
        sm70_ops.sm70_f16_gemm_out(out, x, prepared_weight, k_ld, False)

    latency_us = _measure_graph_us(launch, warmups=warmups, repeats=repeats)
    error = out.float() - reference.float()
    fp32_error = out.float() - fp32_reference
    cublas_fp32_error = reference.float() - fp32_reference
    traffic_bytes = 2 * (weight.numel() + x.numel() + out.numel())
    return {
        "kernel": "turbomind_f16_tensorcore",
        "latency_us": latency_us,
        "effective_gbps": traffic_bytes / (latency_us * 1000.0),
        "exact_equal": torch.equal(out, reference),
        "output_sha256": hashlib.sha256(
            out.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
        ).hexdigest(),
        "max_abs_error": error.abs().max().item(),
        "relative_l2_error": (error.norm() / reference.float().norm()).item(),
        "cosine_similarity": torch.nn.functional.cosine_similarity(
            out.float(), reference.float()
        ).item(),
        "fp32_reference_relative_l2_error": (
            fp32_error.norm() / fp32_reference.norm()
        ).item(),
        "cublas_fp16_fp32_reference_relative_l2_error": (
            cublas_fp32_error.norm() / fp32_reference.norm()
        ).item(),
    }


def _benchmark_padded_cublas_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    reference: torch.Tensor,
    fp32_reference: torch.Tensor,
    *,
    padded_m: int,
    warmups: int,
    repeats: int,
) -> dict[str, Any]:
    padded_x = torch.zeros((padded_m, x.shape[1]), dtype=x.dtype, device=x.device)
    padded_x[0].copy_(x[0])
    padded_out = torch.empty(
        (padded_m, weight.shape[0]), dtype=x.dtype, device=x.device
    )
    weight_t = weight.t()

    def launch() -> None:
        torch.mm(padded_x, weight_t, out=padded_out)

    latency_us = _measure_graph_us(launch, warmups=warmups, repeats=repeats)
    out = padded_out[:1]
    error = out.float() - reference.float()
    fp32_error = out.float() - fp32_reference
    cublas_fp32_error = reference.float() - fp32_reference
    traffic_bytes = 2 * (weight.numel() + padded_x.numel() + padded_out.numel())
    return {
        "kernel": f"cublas_gemm_m{padded_m}",
        "latency_us": latency_us,
        "effective_gbps": traffic_bytes / (latency_us * 1000.0),
        "exact_equal": torch.equal(out, reference),
        "output_sha256": hashlib.sha256(
            out.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
        ).hexdigest(),
        "max_abs_error": error.abs().max().item(),
        "relative_l2_error": (error.norm() / reference.float().norm()).item(),
        "cosine_similarity": torch.nn.functional.cosine_similarity(
            out.float(), reference.float()
        ).item(),
        "fp32_reference_relative_l2_error": (
            fp32_error.norm() / fp32_reference.norm()
        ).item(),
        "cublas_fp16_fp32_reference_relative_l2_error": (
            cublas_fp32_error.norm() / fp32_reference.norm()
        ).item(),
    }


def _benchmark_indexer_fp32_paths(
    x: torch.Tensor,
    weight: torch.Tensor,
    fp32_reference: torch.Tensor,
    *,
    warmups: int,
    repeats: int,
) -> list[dict[str, Any]]:
    weight_fp32_t = weight.t().contiguous().float()
    weight_t = weight.t()
    current_out = torch.empty(
        (x.shape[0], weight.shape[0]), dtype=torch.float32, device=x.device
    )
    fp16_out = torch.empty_like(current_out)

    def current_launch() -> None:
        torch.mm(x.float(), weight_fp32_t, out=current_out)

    def fp16_launch() -> None:
        torch.mm(x, weight_t, out_dtype=torch.float32, out=fp16_out)

    current_us = _measure_graph_us(current_launch, warmups=warmups, repeats=repeats)
    fp16_us = _measure_graph_us(fp16_launch, warmups=warmups, repeats=repeats)
    rows = []
    for name, out, latency_us, traffic_bytes in (
        (
            "indexer_current_fp32_cast_mm",
            current_out,
            current_us,
            4 * weight.numel() + 6 * x.numel() + 4 * current_out.numel(),
        ),
        (
            "indexer_fp16_mm_out_fp32",
            fp16_out,
            fp16_us,
            2 * (weight.numel() + x.numel()) + 4 * fp16_out.numel(),
        ),
    ):
        error = out - fp32_reference
        rows.append(
            {
                "kernel": name,
                "latency_us": latency_us,
                "effective_gbps": traffic_bytes / (latency_us * 1000.0),
                "exact_equal": torch.equal(out, fp32_reference),
                "output_sha256": hashlib.sha256(
                    out.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
                ).hexdigest(),
                "max_abs_error": error.abs().max().item(),
                "relative_l2_error": (error.norm() / fp32_reference.norm()).item(),
                "cosine_similarity": torch.nn.functional.cosine_similarity(
                    out, fp32_reference
                ).item(),
                "fp32_reference_relative_l2_error": (
                    error.norm() / fp32_reference.norm()
                ).item(),
                "cublas_fp16_fp32_reference_relative_l2_error": None,
            }
        )
    rows[1]["exact_equal_current_fp32_path"] = torch.equal(fp16_out, current_out)
    rows[1]["max_abs_error_vs_current_fp32_path"] = (
        (fp16_out - current_out).abs().max().item()
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmups", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str)
    parser.add_argument("--cublas-match-kda-sweep", action="store_true")
    parser.add_argument("--cuda-kernel-sweep", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        raise RuntimeError("This benchmark requires an NVIDIA SM70 GPU.")
    if args.cuda_kernel_sweep:
        _load_glm53_cuda_extension()
    torch.manual_seed(args.seed)
    rows: list[dict[str, Any]] = []
    candidates = (
        ("base", _sm70_glm53_fp16_gemv_kernel),
        ("evict", _sm70_glm53_fp16_gemv_evict_kernel),
        ("cg", _sm70_glm53_fp16_gemv_cg_kernel),
    )
    shape_configs = _SM70_GLM53_FP16_GEMV_CONFIGS.items()
    if args.cublas_match_kda_sweep or args.cuda_kernel_sweep:
        shape_configs = [((6416, 4096), _SM70_GLM53_FP16_GEMV_CONFIGS[(6416, 4096)])]
    for (n, k), (block_k, num_warps) in shape_configs:
        x = torch.randn((1, k), dtype=torch.float16, device="cuda")
        weight = torch.randn((n, k), dtype=torch.float16, device="cuda")
        reference = torch.nn.functional.linear(x, weight)
        fp32_reference = torch.nn.functional.linear(x.float(), weight.float())
        for name, kernel in candidates:
            try:
                result = _benchmark_kernel(
                    name,
                    kernel,
                    x,
                    weight,
                    reference,
                    fp32_reference,
                    block_k=block_k,
                    num_warps=num_warps,
                    warmups=args.warmups,
                    repeats=args.repeats,
                )
            except Exception as exc:
                result = {"kernel": name, "error": f"{type(exc).__name__}: {exc}"}
            rows.append(
                {
                    "shape": [n, k],
                    "block_k": block_k,
                    "num_warps": num_warps,
                    **result,
                }
            )
        try:
            result = _benchmark_turbomind(
                x,
                weight,
                reference,
                fp32_reference,
                warmups=args.warmups,
                repeats=args.repeats,
            )
        except Exception as exc:
            result = {
                "kernel": "turbomind_f16_tensorcore",
                "error": f"{type(exc).__name__}: {exc}",
            }
        rows.append(
            {
                "shape": [n, k],
                "block_k": None,
                "num_warps": None,
                **result,
            }
        )
        if args.cublas_match_kda_sweep:
            for vector_width in (1, 2, 4, 8):
                for chunk_tiles in (1, 2, 4, 8, 16, 32, 64):
                    for butterfly_down in (False, True):
                        try:
                            result = _benchmark_cublas_match_kernel(
                                x,
                                weight,
                                reference,
                                fp32_reference,
                                vector_width=vector_width,
                                chunk_tiles=chunk_tiles,
                                butterfly_down=butterfly_down,
                                warmups=args.warmups,
                                repeats=args.repeats,
                            )
                        except Exception as exc:
                            result = {
                                "kernel": "cublas_match_gemv2t",
                                "vector_width": vector_width,
                                "chunk_tiles": chunk_tiles,
                                "butterfly_down": butterfly_down,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        rows.append(
                            {
                                "shape": [n, k],
                                "block_k": None,
                                "num_warps": 4,
                                **result,
                            }
                        )
        if args.cuda_kernel_sweep:
            x8 = torch.randn((8, k), dtype=torch.float16, device="cuda")
            reference8 = torch.cat(
                [torch.nn.functional.linear(x8[i : i + 1], weight) for i in range(8)]
            )
            fp32_reference8 = torch.nn.functional.linear(x8.float(), weight.float())
            for half2_rows in (-5, -4, -3, -2, 4, 0, 1, 2, 8, 16):
                try:
                    result = _benchmark_cuda_chunk_parallel_kernel(
                        x8,
                        weight,
                        reference8,
                        fp32_reference8,
                        half2_rows=half2_rows,
                        warmups=args.warmups,
                        repeats=args.repeats,
                    )
                except Exception as exc:
                    result = {
                        "kernel": "cuda_cublas_match_gemv2t",
                        "half2_rows": half2_rows,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                rows.append(
                    {
                        "shape": [8, n, k],
                        "block_k": None,
                        "num_warps": None,
                        **result,
                    }
                )
        for padded_m in (1, 2, 4, 8, 16):
            try:
                result = _benchmark_padded_cublas_gemm(
                    x,
                    weight,
                    reference,
                    fp32_reference,
                    padded_m=padded_m,
                    warmups=args.warmups,
                    repeats=args.repeats,
                )
            except Exception as exc:
                result = {
                    "kernel": f"cublas_gemm_m{padded_m}",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            rows.append(
                {
                    "shape": [n, k],
                    "block_k": None,
                    "num_warps": None,
                    **result,
                }
            )
        if (n, k) == (32, 4096):
            try:
                indexer_results = _benchmark_indexer_fp32_paths(
                    x,
                    weight,
                    fp32_reference,
                    warmups=args.warmups,
                    repeats=args.repeats,
                )
            except Exception as exc:
                indexer_results = [
                    {
                        "kernel": "indexer_fp32_paths",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                ]
            rows.extend(
                {
                    "shape": [n, k],
                    "block_k": None,
                    "num_warps": None,
                    **result,
                }
                for result in indexer_results
            )

    report = {
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "dtype": "float16",
        "accumulation": "float32",
        "warmups": args.warmups,
        "repeats": args.repeats,
        "rows": rows,
    }
    payload = json.dumps(report, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as output_file:
            output_file.write(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
