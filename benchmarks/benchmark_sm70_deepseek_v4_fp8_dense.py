# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Profile the exact DeepSeek V4 TP8 FP8 dense decode shapes on SM70."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from vllm import _sm70_ops as sm70_ops


@dataclass(frozen=True)
class Shape:
    name: str
    k: int
    n: int
    calls_per_token: int
    gated_silu: bool = False


SHAPES = (
    Shape("fused_wqa_wkv", 4096, 1536, 43),
    Shape("wq_b_and_wo_b", 1024, 4096, 86),
    Shape("wo_a_group", 4096, 1024, 43),
    Shape("shared_gate_up", 4096, 512, 43, True),
    Shape("shared_down", 256, 4096, 43),
    Shape("c4_indexer_wq_b", 1024, 8192, 21),
)

DIAGNOSTIC_SHAPES = (Shape("split_wkv", 4096, 512, 43),)


def _digest(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _time_graph(graph: torch.cuda.CUDAGraph, replays: int, repeats: int) -> list[float]:
    samples_ms: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        end.synchronize()
        samples_ms.append(start.elapsed_time(end) / replays)
    return samples_ms


def _profile_graph_replay(graph: torch.cuda.CUDAGraph) -> None:
    cudart = torch.cuda.cudart()
    cudart.cudaProfilerStart()
    graph.replay()
    torch.cuda.synchronize()
    cudart.cudaProfilerStop()


def _capture_graph(run: Callable[[], None]) -> torch.cuda.CUDAGraph:
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        run()
    capture_stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        run()
    return graph


def _measure(
    shape: Shape, replays: int, repeats: int, profile_replay: bool
) -> dict[str, object]:
    qweight = torch.randn((shape.n, shape.k), device="cuda", dtype=torch.float16).to(
        torch.float8_e4m3fn
    )
    scales = torch.ones(
        ((shape.n + 127) // 128, (shape.k + 127) // 128),
        device="cuda",
        dtype=torch.float32,
    )
    weight, tm_scales, meta = sm70_ops.fp8_sm70_prepare(
        qweight, scales, 128, shape.gated_silu
    )
    x = torch.randn((1, shape.k), device="cuda", dtype=torch.float16)
    out_n = shape.n // 2 if shape.gated_silu else shape.n
    direct = torch.empty((1, out_n), device="cuda", dtype=torch.float16)
    graph_out = torch.empty_like(direct)

    for _ in range(4):
        sm70_ops.fp8_gemm_sm70_out_meta(
            direct, x, weight, tm_scales, meta, shape.gated_silu
        )
    torch.cuda.synchronize()

    def run_graph() -> None:
        sm70_ops.fp8_gemm_sm70_out_meta(
            graph_out, x, weight, tm_scales, meta, shape.gated_silu
        )

    graph = _capture_graph(run_graph)
    graph.replay()
    torch.cuda.synchronize()
    equal = torch.equal(direct, graph_out)
    max_abs = float((direct.float() - graph_out.float()).abs().max().item())

    if profile_replay:
        _profile_graph_replay(graph)

    samples_ms = _time_graph(graph, replays, repeats)

    median_ms = statistics.median(samples_ms)
    return {
        **asdict(shape),
        "samples_ms": samples_ms,
        "median_ms": median_ms,
        "projected_ms_per_token": median_ms * shape.calls_per_token,
        "graph_equal": equal,
        "graph_max_abs": max_abs,
        "output_sha256": _digest(graph_out),
    }


def _measure_split_parity(replays: int, repeats: int) -> dict[str, object]:
    k, fused_n, split_n = 4096, 1536, 1024
    qweight = torch.randn((fused_n, k), device="cuda", dtype=torch.float16).to(
        torch.float8_e4m3fn
    )
    scales = torch.ones(
        ((fused_n + 127) // 128, (k + 127) // 128),
        device="cuda",
        dtype=torch.float32,
    )
    fused_weight, fused_scales, fused_meta = sm70_ops.fp8_sm70_prepare(
        qweight, scales, 128, False
    )
    split_prepared = [
        sm70_ops.fp8_sm70_prepare(weight, scale, 128, False)
        for weight, scale in (
            (qweight[:split_n].contiguous(), scales[: split_n // 128].contiguous()),
            (qweight[split_n:].contiguous(), scales[split_n // 128 :].contiguous()),
        )
    ]
    x = torch.randn((1, k), device="cuda", dtype=torch.float16)
    fused_out = torch.empty((1, fused_n), device="cuda", dtype=torch.float16)
    split_out = torch.empty_like(fused_out)

    def run_fused() -> None:
        sm70_ops.fp8_gemm_sm70_out_meta(
            fused_out, x, fused_weight, fused_scales, fused_meta, False
        )

    def run_split() -> None:
        for offset, n, prepared in (
            (0, split_n, split_prepared[0]),
            (split_n, fused_n - split_n, split_prepared[1]),
        ):
            weight, tm_scales, meta = prepared
            sm70_ops.fp8_gemm_sm70_out_meta(
                split_out[:, offset : offset + n],
                x,
                weight,
                tm_scales,
                meta,
                False,
            )

    for _ in range(4):
        run_fused()
        run_split()
    torch.cuda.synchronize()

    fused_graph = _capture_graph(run_fused)
    split_graph = _capture_graph(run_split)
    fused_graph.replay()
    split_graph.replay()
    torch.cuda.synchronize()

    diff = (fused_out.float() - split_out.float()).abs()
    fused_samples = _time_graph(fused_graph, replays, repeats)
    split_samples = _time_graph(split_graph, replays, repeats)
    fused_median = statistics.median(fused_samples)
    split_median = statistics.median(split_samples)
    return {
        "fused_samples_ms": fused_samples,
        "split_samples_ms": split_samples,
        "fused_median_ms": fused_median,
        "split_median_ms": split_median,
        "speedup": fused_median / split_median,
        "projected_savings_ms_per_token": (fused_median - split_median) * 43,
        "equal": torch.equal(fused_out, split_out),
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "fused_sha256": _digest(fused_out),
        "split_sha256": _digest(split_out),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--replays", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--profile-replay", action="store_true")
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument(
        "--name", choices=[shape.name for shape in SHAPES + DIAGNOSTIC_SHAPES]
    )
    parser.add_argument("--split-parity", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        raise RuntimeError("This benchmark requires an NVIDIA V100 (SM70).")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if args.split_parity:
        result = _measure_split_parity(args.replays, args.repeats)
        payload = {"contract": {"m": 1, "tp": 8}, "split_parity": result}
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        return 0 if result["equal"] else 1

    available_shapes = SHAPES if args.name is None else SHAPES + DIAGNOSTIC_SHAPES
    selected_shapes = tuple(
        shape for shape in available_shapes if shape.name == args.name
    )
    if args.name is None:
        selected_shapes = SHAPES
    results = [
        _measure(shape, args.replays, args.repeats, args.profile_replay)
        for shape in selected_shapes
    ]
    projected_ms = sum(float(item["projected_ms_per_token"]) for item in results)
    failures = [item["name"] for item in results if not item["graph_equal"]]
    payload = {
        "contract": {
            "model": "DeepSeek-V4-Flash",
            "tp": 8,
            "m": 1,
            "group_size": 128,
            "cuda_graph": True,
            "replays": args.replays,
            "repeats": args.repeats,
            "seed": args.seed,
        },
        "summary": {
            "shape_count": len(results),
            "calls_per_token": sum(shape.calls_per_token for shape in selected_shapes),
            "projected_ms_per_token": projected_ms,
            "graph_failures": failures,
        },
        "results": results,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    for item in results:
        print(
            f"{item['name']:20s} K={item['k']:5d} N={item['n']:5d} "
            f"{item['median_ms']:.6f} ms x {item['calls_per_token']:3d} = "
            f"{item['projected_ms_per_token']:.3f} ms/token"
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
