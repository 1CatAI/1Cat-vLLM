# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure native NVFP4 prefill scratch and CUDA Graph replay on an idle V100.

Run each source-built control/candidate in a fresh process on the same GPU:
CUDA_VISIBLE_DEVICES=1 python benchmarks/kernels/benchmark_sm70_nvfp4_prefill_memory.py
"""

import argparse
import gc
import json
import statistics
from pathlib import Path

import torch
import vllm._C  # noqa: F401


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=8192)
    parser.add_argument("--tp-shape", type=int, choices=[1, 2, 4], default=1)
    parser.add_argument("--projection", choices=["gate_up", "down"], default="gate_up")
    parser.add_argument("--result-file", type=Path)
    args = parser.parse_args()
    if torch.cuda.get_device_capability() != (7, 0):
        raise RuntimeError("requires an SM70 GPU")

    torch.manual_seed(7101)
    m = args.rows
    gated = args.projection == "gate_up"
    k, n = (5120, 34816 // args.tp_shape) if gated else (17408 // args.tp_shape, 5120)
    codes = torch.randint(0, 16, (n, k), device="cuda", dtype=torch.uint8)
    raw = torch.randint(1, 8, (n, k // 16), device="cuda").to(torch.float8_e4m3fn)
    scale = 0.00314159
    compact = torch.ops._C.nvfp4_qpn2_prepare_scales_sm70(raw)
    weight, scales, meta = torch.ops._C.nvfp4_sm70_prepare(
        codes.T.contiguous(), (raw.float() * scale).T.half().contiguous(), 16, False
    )
    x = torch.randn(m, k, device="cuda", dtype=torch.float16) * 0.1
    out = torch.empty(m, n // 2 if gated else n, device="cuda", dtype=torch.float16)
    del codes, raw

    def call():
        torch.ops._C.nvfp4_qpn2_tm_dispatch_sm70_out(
            out,
            x,
            weight,
            compact,
            scale,
            8,
            2,
            scales,
            16,
            int(meta[0]),
            int(meta[1]),
            gated,
            1024,
        )

    for _ in range(3):
        call()
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    peak = torch.cuda.max_memory_allocated() - before
    times = []
    for _ in range(9):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(5):
            graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) / 5)
    result = {
        "shape": [m, k, n],
        "projection": args.projection,
        "graph": True,
        "median_ms": statistics.median(times),
        "samples_ms": times,
        "capture_peak_bytes": peak,
        "finite": bool(out.isfinite().all()),
        "native_path": vllm._C.__file__,
        "torch": torch.__version__,
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.result_file:
        args.result_file.write_text(text + "\n")


if __name__ == "__main__":
    main()
