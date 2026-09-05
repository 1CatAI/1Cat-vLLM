# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Screen cuBLASLt heuristics on exact SM70 decode projection shapes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--m", type=int, default=8)
    parser.add_argument("--n", type=int, default=3336)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--workspace-mib", type=int, default=64)
    parser.add_argument("--requested-algorithms", type=int, default=200)
    parser.add_argument("--exhaustive", action="store_true")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--print-summary", action="store_true")
    return parser.parse_args()


def _digest(tensor: torch.Tensor) -> str:
    data = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def _measure_graph_us(
    launch: Callable[[], None], *, warmup: int, iters: int, trials: int
) -> dict[str, Any]:
    for _ in range(warmup):
        launch()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    samples = []
    for _ in range(trials):
        for _ in range(warmup):
            graph.replay()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / iters)
    return {
        "samples_us": samples,
        "median_us": statistics.median(samples),
        "mean_us": statistics.mean(samples),
        "minimum_us": min(samples),
        "maximum_us": max(samples),
    }


def _load_extension() -> Any:
    source = Path(__file__).with_name("csrc") / "benchmark_sm70_cublaslt_algorithms.cpp"
    cuda_home = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    cublas_library = Path(torch.__file__).parent.parent / "nvidia" / "cublas" / "lib"
    return load(
        name="benchmark_sm70_cublaslt_algorithms_ext",
        sources=[str(source)],
        extra_cflags=["-O3"],
        extra_ldflags=[
            f"-L{cuda_home / 'lib'}",
            f"-L{cublas_library}",
            "-lcublasLt",
            "-lcublas",
        ],
        with_cuda=True,
        verbose=True,
    )


def main() -> int:
    args = _parse_args()
    if torch.cuda.get_device_capability() != (7, 0):
        raise RuntimeError("This benchmark requires SM70/V100")
    torch.manual_seed(args.seed)
    extension = _load_extension()
    workspace_bytes = args.workspace_mib * 1024 * 1024
    runner = extension.LtRunner(
        args.m,
        args.n,
        args.k,
        workspace_bytes,
        args.requested_algorithms,
        args.exhaustive,
    )
    algorithm_info = list(runner.algorithm_info())
    if not algorithm_info:
        raise RuntimeError("cuBLASLt returned no heuristic algorithms")

    inputs = torch.randn((args.m, args.k), device="cuda", dtype=torch.float16).mul_(
        0.02
    )
    weight = torch.randn((args.n, args.k), device="cuda", dtype=torch.float16).mul_(
        0.02
    )
    workspace = torch.empty(workspace_bytes, device="cuda", dtype=torch.uint8)
    reference = F.linear(inputs, weight)
    torch.cuda.synchronize()
    reference_hash = _digest(reference)
    baseline_timing = _measure_graph_us(
        lambda: F.linear(inputs, weight),
        warmup=args.warmup,
        iters=args.iters,
        trials=args.trials,
    )

    results: list[dict[str, Any]] = []
    for info in algorithm_info:
        output = torch.empty_like(reference)
        index = int(info["index"])

        def launch(index: int = index, output: torch.Tensor = output) -> None:
            runner.run(index, output, inputs, weight, workspace)

        try:
            launch()
            torch.cuda.synchronize()
            difference = (output.float() - reference.float()).abs()
            quality = {
                "exact": torch.equal(output, reference),
                "different": int((output != reference).sum().item()),
                "max_abs": float(difference.max().item()),
                "mean_abs": float(difference.mean().item()),
                "sha256": _digest(output),
            }
            result = {**dict(info), "quality": quality}
            if quality["exact"] or not args.exhaustive:
                result["timing"] = _measure_graph_us(
                    launch,
                    warmup=args.warmup,
                    iters=args.iters,
                    trials=args.trials,
                )
            results.append(result)
        except RuntimeError as error:
            torch.cuda.synchronize()
            results.append({**dict(info), "error": str(error)})

    results.sort(key=lambda item: item.get("timing", {}).get("median_us", float("inf")))
    exact_results = [
        item for item in results if item.get("quality", {}).get("exact", False)
    ]
    payload = {
        "environment": {
            "device": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
        "contract": {
            "m": args.m,
            "n": args.n,
            "k": args.k,
            "workspace_bytes": workspace_bytes,
            "cuda_graph": True,
            "seed": args.seed,
        },
        "reference": {
            "sha256": reference_hash,
            "timing": baseline_timing,
        },
        "summary": {
            "returned_algorithms": len(algorithm_info),
            "working_algorithms": sum("quality" in item for item in results),
            "exact_algorithms": len(exact_results),
            "fastest_exact": exact_results[0] if exact_results else None,
        },
        "algorithms": results,
    }
    encoded = json.dumps(payload, indent=2)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(encoded + "\n", encoding="utf-8")
    if args.print_summary:
        print(json.dumps(payload["summary"], indent=2))
    else:
        print(encoded)
    return 0 if exact_results else 1


if __name__ == "__main__":
    raise SystemExit(main())
