# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sweep the exact GLM-5.3 q8 mHC FP32-staging path on SM70."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections.abc import Callable
from pathlib import Path

import torch

import vllm._sm70_ops as sm70_ops
from vllm.model_executor.kernels.mhc.tilelang_kernels import (
    mhc_fused_tilelang,
    sm70_mhc_dot_from_fp32_stage_tilelang,
    sm70_mhc_post_fp32_stage_tilelang,
)
from vllm.model_executor.kernels.mhc.triton import (
    sm70_mhc_pre_norm_from_staging,
)

NUM_TOKENS = 8
HIDDEN_SIZE = 4096
HC_MULT = 4
HC_OUT = HC_MULT * (2 + HC_MULT)
N_SPLITS = 4
CALLS_PER_VERIFICATION = 89


def _capture(call: Callable[[], None]) -> torch.cuda.CUDAGraph:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(4):
            call()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        call()
    graph.replay()
    stream.synchronize()
    return graph


def _time_graph(graph: torch.cuda.CUDAGraph, replays: int, repeats: int) -> list[float]:
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / replays)
    return samples


def _digest(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _error_metrics(
    actual: torch.Tensor, expected: torch.Tensor
) -> dict[str, float | int]:
    absolute_error = (actual.float() - expected.float()).abs()
    return {
        "different_elements": int(torch.count_nonzero(actual != expected).item()),
        "max_abs_error": float(absolute_error.max().item()),
        "mean_abs_error": float(absolute_error.mean().item()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile-n", type=int, nargs="+", default=[3, 6, 8, 12])
    parser.add_argument("--replays", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        raise RuntimeError("This benchmark requires NVIDIA V100/SM70.")
    if any(HC_OUT % tile_n for tile_n in args.tile_n):
        raise ValueError(f"Every tile-n must divide {HC_OUT}.")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    x = torch.randn((NUM_TOKENS, HIDDEN_SIZE), device=device, dtype=torch.float16)
    residual = torch.randn(
        (NUM_TOKENS, HC_MULT, HIDDEN_SIZE), device=device, dtype=torch.float16
    )
    post_mix = torch.sigmoid(
        torch.randn((NUM_TOKENS, HC_MULT), device=device, dtype=torch.float32)
    )
    comb_mix = torch.softmax(
        torch.randn((NUM_TOKENS, HC_MULT, HC_MULT), device=device, dtype=torch.float32),
        dim=1,
    )
    fn = torch.randn(
        (HC_OUT, HC_MULT, HIDDEN_SIZE), device=device, dtype=torch.float32
    ).mul_(1e-4)
    hc_scale = torch.randn((3,), device=device, dtype=torch.float32).mul_(0.1)
    hc_base = torch.randn((HC_OUT,), device=device, dtype=torch.float32).mul_(0.1)
    norm_weight = torch.randn((HIDDEN_SIZE,), device=device, dtype=torch.float16).mul_(
        0.1
    )

    results = []
    reference: tuple[torch.Tensor, ...] | None = None
    all_bitwise = True
    variants = [
        (mode, tile_n)
        for tile_n in args.tile_n
        for mode in ("staged", "fused", "native")
        if mode != "native" or tile_n in (6, 8, 12)
    ]
    for mode, tile_n in variants:
        residual_fp32 = torch.empty(
            (NUM_TOKENS, HC_MULT, HIDDEN_SIZE),
            device=device,
            dtype=torch.float32,
        )
        residual_out = torch.empty_like(residual)
        gemm_out_mul = torch.empty(
            (N_SPLITS, NUM_TOKENS, HC_OUT), device=device, dtype=torch.float32
        )
        gemm_out_sqrsum = torch.empty(
            (N_SPLITS, NUM_TOKENS), device=device, dtype=torch.float32
        )
        post_out = torch.empty(
            (NUM_TOKENS, HC_MULT), device=device, dtype=torch.float32
        )
        comb_out = torch.empty(
            (NUM_TOKENS, HC_MULT, HC_MULT), device=device, dtype=torch.float32
        )
        layer_input = torch.empty_like(x)

        def call(
            *,
            variant_mode: str = mode,
            variant_tile_n: int = tile_n,
            variant_residual_fp32: torch.Tensor = residual_fp32,
            variant_residual_out: torch.Tensor = residual_out,
            variant_gemm_out_mul: torch.Tensor = gemm_out_mul,
            variant_gemm_out_sqrsum: torch.Tensor = gemm_out_sqrsum,
            variant_post_out: torch.Tensor = post_out,
            variant_comb_out: torch.Tensor = comb_out,
            variant_layer_input: torch.Tensor = layer_input,
        ) -> None:
            if variant_mode == "staged":
                sm70_mhc_post_fp32_stage_tilelang(
                    comb_mix,
                    residual,
                    post_mix,
                    x,
                    variant_residual_fp32,
                    variant_residual_out,
                    variant_gemm_out_sqrsum,
                    HIDDEN_SIZE,
                    HC_MULT,
                    n_splits=N_SPLITS,
                )
                sm70_mhc_dot_from_fp32_stage_tilelang(
                    variant_residual_fp32,
                    fn,
                    variant_gemm_out_mul,
                    HIDDEN_SIZE,
                    HC_MULT,
                    HC_OUT,
                    tile_n=variant_tile_n,
                    n_splits=N_SPLITS,
                )
            elif variant_mode == "fused":
                mhc_fused_tilelang(
                    comb_mix,
                    residual,
                    post_mix,
                    x,
                    fn,
                    variant_gemm_out_mul,
                    variant_gemm_out_sqrsum,
                    variant_residual_out,
                    HC_MULT,
                    HIDDEN_SIZE,
                    HC_OUT,
                    tile_n=variant_tile_n,
                    n_splits=N_SPLITS,
                    use_fp16=True,
                )
            else:
                sm70_ops.sm70_glm_mhc_post_dot_q8_out(
                    variant_residual_out,
                    variant_gemm_out_mul,
                    variant_gemm_out_sqrsum,
                    comb_mix,
                    residual,
                    post_mix,
                    x,
                    fn,
                    variant_tile_n,
                )
            sm70_mhc_pre_norm_from_staging(
                variant_gemm_out_mul,
                variant_gemm_out_sqrsum,
                hc_scale,
                hc_base,
                variant_residual_out,
                variant_post_out,
                variant_comb_out,
                variant_layer_input,
                norm_weight,
                1e-6,
                1e-6,
                1e-6,
                2.0,
                20,
                1e-6,
            )

        graph = _capture(call)
        samples = _time_graph(graph, args.replays, args.repeats)
        graph.replay()
        torch.cuda.synchronize()
        outputs = (
            residual_out.clone(),
            gemm_out_mul.clone(),
            gemm_out_sqrsum.clone(),
            post_out.clone(),
            comb_out.clone(),
            layer_input.clone(),
        )
        if reference is None:
            reference = outputs
        bitwise = [
            torch.equal(actual, expected)
            for actual, expected in zip(outputs, reference, strict=True)
        ]
        all_bitwise &= all(bitwise)
        median_us = statistics.median(samples)
        results.append(
            {
                "mode": mode,
                "tile_n": tile_n,
                "samples_us": samples,
                "median_us": median_us,
                "projected_ms_per_verification": (
                    median_us * CALLS_PER_VERIFICATION / 1000.0
                ),
                "all_outputs_bitwise_equal_reference": all(bitwise),
                "output_sha256": [_digest(output) for output in outputs],
                "output_error_vs_reference": [
                    _error_metrics(actual, expected)
                    for actual, expected in zip(outputs, reference, strict=True)
                ],
            }
        )

    baseline = next(
        result
        for result in results
        if result["mode"] == "staged" and result["tile_n"] == 3
    )
    for result in results:
        result["speedup_vs_tile3"] = baseline["median_us"] / result["median_us"]
    payload = {
        "contract": {
            "model": "GLM-5.3-Flash",
            "num_tokens": NUM_TOKENS,
            "hidden_size": HIDDEN_SIZE,
            "hc_mult": HC_MULT,
            "hc_out": HC_OUT,
            "n_splits": N_SPLITS,
            "calls_per_verification": CALLS_PER_VERIFICATION,
            "cuda_graph": True,
        },
        "results": results,
        "all_variants_bitwise_equal": all_bitwise,
    }
    encoded = json.dumps(payload, indent=2)
    print(encoded)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(encoded + "\n", encoding="utf-8")
    return 0 if all_bitwise else 1


if __name__ == "__main__":
    raise SystemExit(main())
