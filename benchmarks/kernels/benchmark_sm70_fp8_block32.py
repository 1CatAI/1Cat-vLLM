# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Time the SM70 weight-only FP8 32 x 32 block-scale paths.

Every row times, as CUDA-graph replays with weight rotation (each call streams
weights that are not L2 resident; V100 L2 = 6 MB):

* ``fp16_cublas``: the load-time FP16 dequantization (2 B/param) + cuBLAS,
  i.e. ``kernel_config.sm70_fp8.dequant_fallback`` (VLLM_SM70_FP8_TURBOMIND=0);
* ``gemv`` / ``turbomind`` / ``dequant``: the three kernels of
  ``sm70_fp8_block32`` and ``policy``, the dispatch they serve under.

Shapes default to the DeepSeek-V4.1-Flash dense projections at TP4 plus the
replicated ones. Usage:

    python benchmarks/kernels/benchmark_sm70_fp8_block32.py --out results/
"""

from __future__ import annotations

import argparse
import json
import platform
from collections.abc import Callable
from pathlib import Path

import torch

from vllm.model_executor.layers.quantization.utils import sm70_fp8_block32 as b32

# (name, N, K)
SHAPES = [
    ("wq_a+wkv", 1792, 5120),
    ("wq_b_tp4", 8192, 1280),
    ("indexer_wq_b", 4096, 1280),
    ("wo_a_group", 1024, 4096),
    ("wo_b_tp4", 5120, 2048),
    ("shared_w13_tp4", 1152, 5120),
    ("shared_w2_tp4", 5120, 576),
    ("shared_w2_tp8", 5120, 288),
]
TOKENS = [1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512, 4096]


def graph_us(fn: Callable[[], object], reps: int = 20, iters: int = 50) -> float:
    """Mean microseconds of one ``fn()`` from replays of a CUDA graph holding
    ``reps`` back-to-back calls."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(reps):
            fn()
    graph.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / (iters * reps)


def rand_fp8_block32(n: int, k: int, seed: int, exp_lo: int = -13, exp_hi: int = -6):
    """Random E4M3 codes (no NaN) and UE8M0 32 x 32 block exponents."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    codes = torch.randint(0, 256, (n, k), dtype=torch.uint8, device="cuda", generator=g)
    codes[(codes & 0x7F) == 0x7F] = 0x3C
    e = torch.randint(
        exp_lo, exp_hi + 1, (n // 32, k // 32), device="cuda", generator=g
    )
    return codes.view(torch.float8_e4m3fn), (e + 127).to(torch.uint8)


class Rotation:
    """Cycles through weight copies so consecutive calls miss L2."""

    def __init__(self, pool: list) -> None:
        self.pool = pool
        self.i = 0

    def __call__(self):
        self.i = (self.i + 1) % len(self.pool)
        return self.pool[self.i]


def bench_row(name: str, n: int, k: int, m: int, tws: list, fbs: list):
    x = torch.randn(m, k, device="cuda").half()
    tw, fb = Rotation(tws), Rotation(fbs)
    out16 = torch.empty(m, n, dtype=torch.float16, device="cuda")
    out32 = torch.empty(m, n, dtype=torch.float32, device="cuda")
    row: dict = {"name": name, "n": n, "k": k, "m": m}
    row["fp16_cublas_us"] = graph_us(lambda: torch.mm(x, fb().t(), out=out16))
    row["fp16_cublas_f32out_us"] = graph_us(
        lambda: torch.mm(x, fb().t(), out_dtype=torch.float32, out=out32)
    )
    if m <= b32.GEMV_MAX_M:
        row["gemv_us"] = graph_us(
            lambda: b32.fp8_block32_gemv(x, tw(), out_dtype=torch.float16, out=out16)
        )
        row["gemv_f32out_us"] = graph_us(
            lambda: b32.fp8_block32_gemv(x, tw(), out=out32)
        )
    row["turbomind_us"] = graph_us(
        lambda: b32.fp8_block32_turbomind(x, tw(), out=out16)
    )
    row["dequant_us"] = graph_us(lambda: b32.fp8_block32_dequant_mm(x, tw(), out=out16))
    row["policy"] = b32.fp8_block32_path(m, n, k)
    row["policy_us"] = graph_us(lambda: b32.fp8_block32_linear(x, tw(), out=out16))
    row["policy_f32"] = b32.fp8_block32_path(m, n, k, torch.float32)
    row["policy_f32_us"] = graph_us(
        lambda: b32.fp8_block32_linear(x, tw(), out_dtype=torch.float32, out=out32)
    )
    return {key: round(v, 2) if isinstance(v, float) else v for key, v in row.items()}


def bench(args: argparse.Namespace) -> dict:
    # Keep cuBLAS split-K reductions in FP32 for every FP16 GEMM measured here.
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    rows = []
    shapes = [s for s in SHAPES if not args.shapes or s[0] in args.shapes]
    for name, n, k in shapes:
        copies = max(2, (24 << 20) // (n * k) + 1)  # rotate > 4x L2 of FP8 bytes
        raw = [rand_fp8_block32(n, k, seed=i) for i in range(copies)]
        tws = [b32.prepare_fp8_block32(w, s) for w, s in raw]
        fbs = [b32.dequant_fp8_block32_reference(w, s).half() for w, s in raw]
        for m in args.tokens:
            row = bench_row(name, n, k, m, tws, fbs)
            rows.append(row)
            print(json.dumps(row), flush=True)
        del tws, fbs, raw
        torch.cuda.empty_cache()
    props = torch.cuda.get_device_properties(0)
    return {
        "device": props.name,
        "sm": f"{props.major}.{props.minor}",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokens", nargs="+", type=int, default=TOKENS)
    ap.add_argument("--shapes", nargs="*", default=[], help="subset of shape names")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    report = bench(args)
    if args.out:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / "sm70_fp8_block32_bench.json").write_text(
            json.dumps(report, indent=1)
        )
    return report


if __name__ == "__main__":
    main()
