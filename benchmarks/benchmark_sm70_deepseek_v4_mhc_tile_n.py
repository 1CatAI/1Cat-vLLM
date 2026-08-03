# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sweep the exact DeepSeek V4 B1 mHC TileLang output tiling on SM70."""

from __future__ import annotations

import argparse
import json
import statistics

import torch

from vllm.model_executor.kernels.mhc.tilelang_kernels import (
    mhc_fused_tilelang,
    mhc_pre_big_fuse_with_norm_tilelang,
)


def _capture(fn) -> torch.cuda.CUDAGraph:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(4):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        fn()
    graph.replay()
    torch.cuda.synchronize()
    return graph


def _time_graph(graph: torch.cuda.CUDAGraph, replays: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / replays


def benchmark_variant(tile_n: int, replays: int, repeats: int) -> dict[str, object]:
    num_tokens, hidden_size, hc_mult = 1, 4096, 4
    hc_mult3 = hc_mult * (2 + hc_mult)
    n_splits = 8
    x = torch.randn((num_tokens, hidden_size), device="cuda", dtype=torch.float16)
    residual = torch.randn(
        (num_tokens, hc_mult, hidden_size), device="cuda", dtype=torch.float16
    )
    post_mix = torch.randn((num_tokens, hc_mult), device="cuda", dtype=torch.float32)
    comb_mix = torch.randn(
        (num_tokens, hc_mult, hc_mult), device="cuda", dtype=torch.float32
    )
    fn = (
        torch.randn(
            (hc_mult3, hc_mult, hidden_size), device="cuda", dtype=torch.float32
        )
        * 1e-4
    )
    hc_scale = torch.randn((3,), device="cuda", dtype=torch.float32) * 0.1
    hc_base = torch.randn((hc_mult3,), device="cuda", dtype=torch.float32) * 0.1
    norm_weight = torch.randn((hidden_size,), device="cuda", dtype=torch.float16)

    gemm_out_mul = torch.empty(
        (n_splits, num_tokens, hc_mult3), device="cuda", dtype=torch.float32
    )
    gemm_out_sqrsum = torch.empty(
        (n_splits, num_tokens), device="cuda", dtype=torch.float32
    )
    residual_out = torch.empty_like(residual)
    post_mix_out = torch.empty_like(post_mix)
    comb_mix_out = torch.empty(
        (num_tokens, hc_mult * hc_mult), device="cuda", dtype=torch.float32
    )
    layer_input = torch.empty_like(x)

    def call() -> None:
        mhc_fused_tilelang(
            comb_mix,
            residual,
            post_mix,
            x,
            fn,
            gemm_out_mul,
            gemm_out_sqrsum,
            residual_out,
            hc_mult,
            hidden_size,
            hc_mult3,
            tile_n=tile_n,
            n_splits=n_splits,
            use_fp16=True,
        )
        mhc_pre_big_fuse_with_norm_tilelang(
            gemm_out_mul,
            gemm_out_sqrsum,
            hc_scale,
            hc_base,
            residual_out,
            post_mix_out,
            comb_mix_out,
            layer_input,
            norm_weight,
            hidden_size,
            1e-6,
            1e-6,
            1e-6,
            1.0,
            20,
            1e-6,
            n_splits,
            hc_mult,
            use_fp16=True,
        )

    graph = _capture(call)
    samples = [_time_graph(graph, replays) for _ in range(repeats)]
    graph.replay()
    torch.cuda.synchronize()
    return {
        "tile_n": tile_n,
        "samples_ms": samples,
        "median_ms": statistics.median(samples),
        "outputs": {
            "residual": residual_out.clone(),
            "post_mix": post_mix_out.clone(),
            "comb_mix": comb_mix_out.clone(),
            "layer_input": layer_input.clone(),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile-n", type=int, nargs="+", default=[1, 2, 3, 4, 6])
    parser.add_argument("--replays", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260802)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        raise RuntimeError("This benchmark requires an NVIDIA V100 (SM70).")
    if any(24 % tile_n for tile_n in args.tile_n):
        raise ValueError("Every tile_n must divide the 24 mHC outputs.")

    results = []
    for tile_n in args.tile_n:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        results.append(benchmark_variant(tile_n, args.replays, args.repeats))

    baseline = next(result for result in results if result["tile_n"] == 2)
    baseline_outputs = baseline["outputs"]
    serializable = []
    all_bitwise = True
    for result in results:
        outputs = result.pop("outputs")
        parity = {}
        for name, baseline_output in baseline_outputs.items():
            equal = torch.equal(outputs[name], baseline_output)
            parity[f"{name}_bitwise_equal"] = equal
            parity[f"{name}_max_abs"] = float(
                (outputs[name].float() - baseline_output.float()).abs().max().item()
            )
            all_bitwise &= equal
        result["speedup_vs_tile2"] = baseline["median_ms"] / result["median_ms"]
        result["parity_vs_tile2"] = parity
        serializable.append(result)

    print(
        json.dumps(
            {
                "contract": {
                    "model": "DeepSeek-V4-Flash",
                    "m": 1,
                    "hidden_size": 4096,
                    "hc_mult": 4,
                    "n_splits": 8,
                    "use_fp16": True,
                    "cuda_graph": True,
                },
                "results": serializable,
                "all_variants_bitwise_equal": all_bitwise,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
