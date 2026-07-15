# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate generic custom allreduce on SM70.

This is intentionally model-free.  It isolates the production custom TP
allreduce path, including CUDA graph registered-buffer capture, and compares
it against NCCL/PyTorch all_reduce.

Example:
  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
    benchmarks/benchmark_sm70_custom_all_reduce.py \
    --json-out bench_results/sm70_migration_20260602/custom_ar.json
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from vllm import _custom_ops as ops
from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce

_GEMMA_RMS_NORM_HIDDEN_SIZE = 5120
_GEMMA_RMS_NORM_EPS = 1e-6


def _dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    raise ValueError(f"unsupported dtype: {name}")


def _make_input(
    size: int,
    dtype: torch.dtype,
    rank: int,
    pattern: str,
    seed: int,
) -> torch.Tensor:
    base = torch.arange(size, device="cuda", dtype=torch.float32)
    if pattern == "exact_int":
        tensor = ((base % 23) - 11 + rank).to(dtype)
    elif pattern == "rank_marker":
        tensor = torch.full((size,), rank + 1, device="cuda", dtype=dtype)
    elif pattern == "random_small":
        generator = torch.Generator(device="cuda")
        generator.manual_seed(seed + rank)
        tensor = (torch.rand(size, device="cuda", generator=generator) - 0.5).to(
            dtype
        )
    elif pattern == "model_like":
        generator = torch.Generator(device="cuda")
        generator.manual_seed(seed + rank)
        tensor = (torch.randn(size, device="cuda", generator=generator) * 0.03).to(
            dtype
        )
    else:
        raise ValueError(f"unsupported pattern: {pattern}")
    return tensor


def _reference_all_reduce(inp: torch.Tensor) -> torch.Tensor:
    ref = inp.clone()
    dist.all_reduce(ref, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    return ref


def _custom_allreduce_algo(
    world_size: int,
    fully_connected: bool,
    num_bytes: int,
) -> str:
    env_algo = os.environ.get("VLLM_CUSTOM_ALLREDUCE_ALGO")
    if env_algo in ("1stage", "oneshot"):
        return "1stage"
    if env_algo in ("2stage", "twoshot"):
        return "2stage"
    if env_algo:
        raise ValueError(f"unsupported VLLM_CUSTOM_ALLREDUCE_ALGO={env_algo!r}")
    if world_size == 2:
        return "1stage"
    if fully_connected:
        if (world_size <= 4 and num_bytes < 512 * 1024) or (
            world_size <= 8 and num_bytes < 256 * 1024
        ):
            return "1stage"
        return "2stage"
    raise RuntimeError("custom allreduce should not be active without full P2P")


def _downcast_like(acc: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return acc if dtype == torch.float32 else acc.to(dtype)


def _reference_custom_order(
    inp: torch.Tensor,
    algo: str,
    world_size: int,
) -> torch.Tensor:
    gathered = [torch.empty_like(inp) for _ in range(world_size)]
    dist.all_gather(gathered, inp)
    torch.cuda.synchronize()
    if algo == "1stage":
        acc = gathered[0].float()
        for item in gathered[1:]:
            acc = acc + item.float()
        return _downcast_like(acc, inp.dtype)

    if algo != "2stage":
        raise ValueError(f"unsupported custom allreduce algo: {algo}")

    ref = torch.empty_like(inp)
    part = inp.numel() // world_size
    for partition in range(world_size):
        start = partition * part
        end = inp.numel() if partition == world_size - 1 else start + part
        acc = gathered[partition][start:end].float()
        for offset in range(1, world_size):
            rank = (partition + offset) % world_size
            acc = acc + gathered[rank][start:end].float()
        ref[start:end] = _downcast_like(acc, inp.dtype)
    return ref


def _compare(out: torch.Tensor, ref: torch.Tensor) -> dict[str, Any]:
    diff = (out.float() - ref.float()).abs()
    mismatch = out != ref
    nonzero = int(mismatch.sum().item())
    first_mismatch: dict[str, Any] | None = None
    if nonzero:
        idx = int(torch.nonzero(mismatch, as_tuple=False)[0].item())
        first_mismatch = {
            "index": idx,
            "out": float(out.flatten()[idx].float().item()),
            "ref": float(ref.flatten()[idx].float().item()),
            "diff": float(diff.flatten()[idx].item()),
        }
    return {
        "equal": bool(torch.equal(out, ref)),
        "max_diff": float(diff.max().item()),
        "mean_diff": float(diff.mean().item()),
        "nonzero_count": nonzero,
        "out_checksum": float(out.float().sum().item()),
        "ref_checksum": float(ref.float().sum().item()),
        "first_mismatch": first_mismatch,
    }


def _run_eager(ca: CustomAllreduce, inp: torch.Tensor) -> torch.Tensor:
    torch.cuda.synchronize()
    dist.barrier()
    out = ca.custom_all_reduce(inp)
    if out is None:
        raise RuntimeError("custom allreduce rejected eager input")
    torch.cuda.synchronize()
    dist.barrier()
    return out


def _capture_graph(
    ca: CustomAllreduce,
    inputs: list[torch.Tensor],
) -> tuple[torch.cuda.CUDAGraph, list[torch.Tensor]]:
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with ca.capture(), torch.cuda.graph(graph):
        outputs = [ca.custom_all_reduce(inp) for inp in inputs]
    torch.cuda.synchronize()
    dist.barrier()
    if any(out is None for out in outputs):
        raise RuntimeError("custom allreduce rejected graph input")
    return graph, [out for out in outputs if out is not None]


def _run_graph(
    ca: CustomAllreduce,
    inputs: list[torch.Tensor],
    replays: int,
) -> list[torch.Tensor]:
    graph, outputs = _capture_graph(ca, inputs)
    for _ in range(replays):
        graph.replay()
    torch.cuda.synchronize()
    dist.barrier()
    return outputs


def _time_custom_ar_graph(
    ca: CustomAllreduce,
    inputs: list[torch.Tensor],
    warmup: int,
    iterations: int,
) -> tuple[list[torch.Tensor], dict[str, float]]:
    graph, outputs = _capture_graph(ca, inputs)
    graph_ms = _max_rank_value(_time_graph(graph, warmup, iterations))
    return outputs, {
        "max_rank_ms_per_graph_replay": graph_ms,
        "max_rank_us_per_allreduce": graph_ms * 1000.0 / len(inputs),
        "allreduces_per_graph_replay": float(len(inputs)),
        "warmup_replays": float(warmup),
        "timed_replays": float(iterations),
    }


def _run_warmup(ca: CustomAllreduce, inp: torch.Tensor) -> torch.Tensor:
    torch.cuda.synchronize()
    dist.barrier()
    with ca.capture():
        out = ca.custom_all_reduce(inp)
    if out is None:
        raise RuntimeError("custom allreduce rejected warmup input")
    torch.cuda.synchronize()
    dist.barrier()
    return out


def _run_compile_graph(
    ca: CustomAllreduce,
    inp: torch.Tensor,
    replays: int,
) -> torch.Tensor:
    def _compiled_fn(x: torch.Tensor) -> torch.Tensor:
        return ca.all_reduce(x, registered=False)

    torch.cuda.synchronize()
    dist.barrier()
    compiled_fn = torch.compile(_compiled_fn, fullgraph=True, backend="inductor")
    out = compiled_fn(inp)
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with ca.capture(), torch.cuda.graph(graph):
        out = compiled_fn(inp)
    torch.cuda.synchronize()
    dist.barrier()
    for _ in range(replays):
        graph.replay()
    torch.cuda.synchronize()
    dist.barrier()
    return out


def _strict_compare_tensor(
    candidate: torch.Tensor, baseline: torch.Tensor
) -> dict[str, Any]:
    mismatch = candidate != baseline
    mismatch_count = int(mismatch.sum().item())
    diff = (candidate.float() - baseline.float()).abs()
    first_mismatch: dict[str, Any] | None = None
    if mismatch_count:
        coordinate = torch.nonzero(mismatch, as_tuple=False)[0]
        index = tuple(int(item) for item in coordinate.tolist())
        first_mismatch = {
            "index": list(index),
            "candidate": float(candidate[index].float().item()),
            "baseline": float(baseline[index].float().item()),
            "abs_diff": float(diff[index].item()),
        }
    return {
        "equal": bool(torch.equal(candidate, baseline)),
        "mismatch_count": mismatch_count,
        "max_abs_diff": float(diff.max().item()),
        "first_mismatch": first_mismatch,
    }


def _make_gemma_rms_norm_inputs(
    rank: int,
    seed: int,
    tokens: int,
    residual_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    projection_generator = torch.Generator(device="cuda")
    projection_generator.manual_seed(seed + rank)
    projection = (
        torch.randn(
            (tokens, _GEMMA_RMS_NORM_HIDDEN_SIZE),
            device="cuda",
            dtype=torch.float32,
            generator=projection_generator,
        )
        * 0.03
    ).to(torch.float16)

    shared_generator = torch.Generator(device="cuda")
    shared_generator.manual_seed(seed + 1009)
    residual = torch.randn(
        (tokens, _GEMMA_RMS_NORM_HIDDEN_SIZE),
        device="cuda",
        dtype=torch.float32,
        generator=shared_generator,
    ) * 0.03
    residual = residual.to(residual_dtype)
    weight = (
        torch.randn(
            (_GEMMA_RMS_NORM_HIDDEN_SIZE,),
            device="cuda",
            dtype=torch.float32,
            generator=shared_generator,
        )
        * 0.02
    ).to(torch.float16)
    return projection, residual, weight


def _gemma_rms_norm_baseline(
    ca: CustomAllreduce,
    projection: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    *,
    registered: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    reduced = ca.all_reduce(projection, registered=registered)
    residual_out = reduced.float() + residual.float()
    normalized_fp32 = torch.empty_like(residual_out)
    ops.rms_norm(
        normalized_fp32,
        residual_out,
        weight.float() + 1.0,
        _GEMMA_RMS_NORM_EPS,
    )
    return normalized_fp32.to(projection.dtype), residual_out


def _gemma_rms_norm_candidate(
    ca: CustomAllreduce,
    projection: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    *,
    tp_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if tp_size == 2:
        return ca.sm70_tp2_all_reduce_gemma_rms_norm(
            projection, residual, weight, _GEMMA_RMS_NORM_EPS
        )
    if tp_size == 4:
        return ca.sm70_tp4_all_reduce_gemma_rms_norm(
            projection, residual, weight, _GEMMA_RMS_NORM_EPS
        )
    raise ValueError(f"unsupported Gemma RMSNorm TP size: {tp_size}")


def _capture_gemma_rms_norm_graph(ca: CustomAllreduce, callback):
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.no_grad(), ca.capture(), torch.cuda.graph(graph):
        outputs = callback()
    torch.cuda.synchronize()
    dist.barrier()
    return graph, outputs


def _repeat_gemma_rms_norm(callback, joins: int):
    outputs = None
    for _ in range(joins):
        outputs = callback()
    assert outputs is not None
    return outputs


def _time_graph(graph: torch.cuda.CUDAGraph, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    dist.barrier()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def _max_rank_value(value: float) -> float:
    value_tensor = torch.tensor(value, device="cuda", dtype=torch.float64)
    dist.all_reduce(value_tensor, op=dist.ReduceOp.MAX)
    return float(value_tensor.item())


def _prepare_gemma_rms_norm_prototype(
    ca: CustomAllreduce,
    rank: int,
    seed: int,
    joins: int,
    tokens: int,
    residual_dtype: torch.dtype,
    tp_size: int,
) -> tuple[torch.cuda.CUDAGraph, torch.cuda.CUDAGraph, dict[str, Any]]:
    projection, residual, weight = _make_gemma_rms_norm_inputs(
        rank, seed, tokens, residual_dtype
    )
    baseline_graph, (baseline_normalized, baseline_residual) = (
        _capture_gemma_rms_norm_graph(
            ca,
            lambda: _repeat_gemma_rms_norm(
                lambda: _gemma_rms_norm_baseline(
                    ca,
                    projection,
                    residual,
                    weight,
                    registered=True,
                ),
                joins,
            ),
        )
    )
    candidate_graph, (candidate_normalized, candidate_residual) = (
        _capture_gemma_rms_norm_graph(
            ca,
            lambda: _repeat_gemma_rms_norm(
                lambda: _gemma_rms_norm_candidate(
                    ca,
                    projection,
                    residual,
                    weight,
                    tp_size=tp_size,
                ),
                joins,
            ),
        )
    )

    baseline_graph.replay()
    candidate_graph.replay()
    torch.cuda.synchronize()
    dist.barrier()

    normalized_comparison = _strict_compare_tensor(
        candidate_normalized, baseline_normalized
    )
    residual_comparison = _strict_compare_tensor(candidate_residual, baseline_residual)
    strict_passed = normalized_comparison["equal"] and residual_comparison["equal"]
    first_failure = next(
        (
            {"output": name, "comparison": comparison}
            for name, comparison in (
                ("normalized", normalized_comparison),
                ("residual", residual_comparison),
            )
            if not comparison["equal"]
        ),
        None,
    )
    return (
        baseline_graph,
        candidate_graph,
        {
            "rank": rank,
            "tp_size": tp_size,
            "shape": [tokens, _GEMMA_RMS_NORM_HIDDEN_SIZE],
            "projection_dtype": "float16",
            "residual_dtype": str(residual.dtype).replace("torch.", ""),
            "normalized_dtype": "float16",
            "weight_dtype": str(weight.dtype).replace("torch.", ""),
            "epsilon": _GEMMA_RMS_NORM_EPS,
            "joins_per_graph_replay": joins,
            "baseline": "custom all-reduce + vllm_c Gemma residual RMSNorm sequence",
            "baseline_allreduce_kernel": (
                f"cross_device_reduce_1stage<half,{tp_size}>"
            ),
            "candidate": f"sm70_tp{tp_size}_all_reduce_gemma_rms_norm",
            "cuda_graph": True,
            "normalized_comparison": normalized_comparison,
            "residual_comparison": residual_comparison,
            "strict_passed": strict_passed,
            "first_failure": first_failure,
            "strict_failure_reason": (
                None
                if strict_passed
                else "Candidate did not reproduce the baseline bit pattern; no "
                "tolerance is applied."
            ),
        },
    )


def _time_gemma_rms_norm_graphs(
    baseline_graph: torch.cuda.CUDAGraph,
    candidate_graph: torch.cuda.CUDAGraph,
    warmup: int,
    iterations: int,
    joins: int,
) -> dict[str, float]:
    baseline_ms = _max_rank_value(_time_graph(baseline_graph, warmup, iterations))
    candidate_ms = _max_rank_value(_time_graph(candidate_graph, warmup, iterations))
    return {
        "baseline_ms_max_rank": baseline_ms,
        "candidate_ms_max_rank": candidate_ms,
        "baseline_us_per_join_max_rank": baseline_ms * 1000.0 / joins,
        "candidate_us_per_join_max_rank": candidate_ms * 1000.0 / joins,
        "saved_ms_per_graph_replay": baseline_ms - candidate_ms,
        "speedup": baseline_ms / candidate_ms,
    }


def _check_single(
    ca: CustomAllreduce,
    mode: str,
    size: int,
    dtype: torch.dtype,
    rank: int,
    pattern: str,
    seed: int,
    graph_replays: int,
    timing_warmup: int,
    timing_iterations: int,
) -> dict[str, Any]:
    inp = _make_input(size, dtype, rank, pattern, seed)
    num_bytes = size * inp.element_size()
    algo = _custom_allreduce_algo(ca.world_size, ca.fully_connected, num_bytes)
    nccl_ref = _reference_all_reduce(inp)
    custom_order_ref = _reference_custom_order(inp, algo, ca.world_size)
    graph_timing: dict[str, float] | None = None
    if mode == "eager":
        out = _run_eager(ca, inp)
    elif mode == "warmup":
        out = _run_warmup(ca, inp)
    elif mode == "graph":
        if timing_iterations:
            outputs, graph_timing = _time_custom_ar_graph(
                ca, [inp], timing_warmup, timing_iterations
            )
            out = outputs[0]
        else:
            out = _run_graph(ca, [inp], graph_replays)[0]
    elif mode == "compile_graph":
        out = _run_compile_graph(ca, inp, graph_replays)
    else:
        raise ValueError(f"unsupported single mode: {mode}")
    return {
        "mode": mode,
        "size": size,
        "bytes": num_bytes,
        "dtype": str(dtype).replace("torch.", ""),
        "pattern": pattern,
        "custom_allreduce_algo": algo,
        "comparison": _compare(out, nccl_ref),
        "custom_order_comparison": _compare(out, custom_order_ref),
        "graph_timing": graph_timing,
    }


def _check_graph_multi(
    ca: CustomAllreduce,
    size: int,
    dtype: torch.dtype,
    rank: int,
    pattern: str,
    seed: int,
    graph_replays: int,
    multi_count: int,
    timing_warmup: int,
    timing_iterations: int,
) -> dict[str, Any]:
    inputs = [
        _make_input(size, dtype, rank, pattern, seed + i * 1009)
        for i in range(multi_count)
    ]
    refs = [_reference_all_reduce(inp) for inp in inputs]
    graph_timing: dict[str, float] | None = None
    if timing_iterations:
        outputs, graph_timing = _time_custom_ar_graph(
            ca, inputs, timing_warmup, timing_iterations
        )
    else:
        outputs = _run_graph(ca, inputs, graph_replays)
    comparisons = [_compare(out, ref) for out, ref in zip(outputs, refs)]
    max_diff = max(item["max_diff"] for item in comparisons)
    nonzero_count = sum(item["nonzero_count"] for item in comparisons)
    first_bad = next(
        (
            {"op_index": i, "comparison": item}
            for i, item in enumerate(comparisons)
            if not item["equal"]
        ),
        None,
    )
    return {
        "mode": "graph_multi",
        "size": size,
        "bytes": size * torch.empty((), dtype=dtype).element_size(),
        "dtype": str(dtype).replace("torch.", ""),
        "pattern": pattern,
        "multi_count": multi_count,
        "all_equal": all(item["equal"] for item in comparisons),
        "max_diff": float(max_diff),
        "nonzero_count": int(nonzero_count),
        "first_bad": first_bad,
        "graph_timing": graph_timing,
    }


def _parse_csv_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def _parse_csv(value: str) -> list[str]:
    return [item for item in value.split(",") if item]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sizes",
        default="5120,10240,20480,65536,262144,524288",
        help="Comma-separated tensor numel values. Each must be vector-packable.",
    )
    parser.add_argument("--dtypes", default="float16,float32")
    parser.add_argument(
        "--patterns",
        default="exact_int,rank_marker,random_small,model_like",
    )
    parser.add_argument("--modes", default="eager,graph,graph_multi")
    parser.add_argument("--seed", type=int, default=20260602)
    parser.add_argument("--graph-replays", type=int, default=3)
    parser.add_argument("--multi-count", type=int, default=160)
    parser.add_argument(
        "--graph-timing-warmup",
        type=int,
        default=0,
        help="CUDA-graph warmup replays before the optional timing interval.",
    )
    parser.add_argument(
        "--graph-timing-iters",
        type=int,
        default=0,
        help="Time graph and graph_multi modes when positive; zero disables timing.",
    )
    parser.add_argument("--max-size-bytes", type=int, default=8 * 1024 * 1024)
    parser.add_argument(
        "--require-exact-patterns",
        default="exact_int,rank_marker",
        help="Patterns that must be bitwise equal in every requested mode.",
    )
    parser.add_argument(
        "--gemma-rmsnorm-prototype",
        action="store_true",
        help=(
            "Run only the opt-in TP{2,4} [M,5120] Gemma residual RMSNorm "
            "prototype CUDA-graph benchmark."
        ),
    )
    parser.add_argument(
        "--gemma-rmsnorm-tp",
        choices=[2, 4],
        type=int,
        default=2,
        help="Explicit Gemma RMSNorm prototype tensor-parallel size.",
    )
    parser.add_argument("--gemma-rmsnorm-warmup", type=int, default=100)
    parser.add_argument("--gemma-rmsnorm-iters", type=int, default=1000)
    parser.add_argument("--gemma-rmsnorm-joins", type=int, default=1)
    parser.add_argument("--gemma-rmsnorm-tokens", type=int, default=1)
    parser.add_argument(
        "--gemma-rmsnorm-residual-dtype",
        choices=["float16", "float32"],
        default="float32",
    )
    parser.add_argument("--json-out")
    args = parser.parse_args()
    if args.graph_timing_warmup < 0 or args.graph_timing_iters < 0:
        raise ValueError("graph timing warmup and iterations must be non-negative")

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    gloo_group = dist.new_group(backend="gloo")

    ca = CustomAllreduce(
        group=gloo_group,
        device=local_rank,
        max_size=args.max_size_bytes,
    )
    results: list[dict[str, Any]] = []
    try:
        if ca.disabled:
            payload = {
                "world_size": world_size,
                "rank": rank,
                "custom_allreduce_disabled": True,
            }
            if rank == 0:
                print(json.dumps(payload, indent=2, sort_keys=True))
            raise SystemExit(2)

        if args.gemma_rmsnorm_prototype:
            tp_size = args.gemma_rmsnorm_tp
            if world_size != tp_size:
                raise ValueError(
                    "Gemma RMSNorm prototype requires "
                    f"WORLD_SIZE={tp_size}, got {world_size}"
                )
            if args.gemma_rmsnorm_warmup < 0 or args.gemma_rmsnorm_iters <= 0:
                raise ValueError("Gemma RMSNorm warmup must be >= 0 and iters > 0")
            if args.gemma_rmsnorm_joins <= 0:
                raise ValueError("Gemma RMSNorm joins must be > 0")
            if not 1 <= args.gemma_rmsnorm_tokens <= 64:
                raise ValueError("Gemma RMSNorm tokens must be in [1, 64]")

            residual_dtype = _dtype(args.gemma_rmsnorm_residual_dtype)
            if tp_size == 4:
                if residual_dtype != torch.float32:
                    raise ValueError(
                        "TP4 Gemma RMSNorm prototype requires FP32 residual"
                    )
                configured_algo = os.environ.get("VLLM_CUSTOM_ALLREDUCE_ALGO")
                if configured_algo not in (None, "1stage", "oneshot"):
                    raise ValueError(
                        "TP4 Gemma RMSNorm baseline requires "
                        "VLLM_CUSTOM_ALLREDUCE_ALGO=1stage"
                    )
                os.environ["VLLM_CUSTOM_ALLREDUCE_ALGO"] = "1stage"

            baseline_graph, candidate_graph, local_result = (
                _prepare_gemma_rms_norm_prototype(
                    ca,
                    rank,
                    args.seed,
                    args.gemma_rmsnorm_joins,
                    args.gemma_rmsnorm_tokens,
                    residual_dtype,
                    tp_size,
                )
            )
            gathered: list[dict[str, Any]] = [None] * world_size  # type: ignore
            dist.all_gather_object(gathered, local_result)
            strict_passed = all(result["strict_passed"] for result in gathered)
            payload = {
                "world_size": world_size,
                "custom_allreduce_disabled": False,
                "fully_connected": ca.fully_connected,
                "prototype": f"sm70_tp{tp_size}_all_reduce_gemma_rms_norm",
                "strict_passed": strict_passed,
                "rank_results": gathered,
            }
            if not strict_passed:
                payload["timing"] = None
                if rank == 0:
                    text = json.dumps(payload, indent=2, sort_keys=True)
                    print(text)
                    if args.json_out:
                        path = Path(args.json_out)
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text(text + "\n")
                raise SystemExit(1)

            payload["timing"] = _time_gemma_rms_norm_graphs(
                baseline_graph,
                candidate_graph,
                args.gemma_rmsnorm_warmup,
                args.gemma_rmsnorm_iters,
                args.gemma_rmsnorm_joins,
            )
            if rank == 0:
                text = json.dumps(payload, indent=2, sort_keys=True)
                print(text)
                if args.json_out:
                    path = Path(args.json_out)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(text + "\n")
            raise SystemExit(0)

        sizes = _parse_csv_ints(args.sizes)
        dtypes = [_dtype(name) for name in _parse_csv(args.dtypes)]
        patterns = _parse_csv(args.patterns)
        modes = _parse_csv(args.modes)
        require_exact_patterns = set(_parse_csv(args.require_exact_patterns))

        for dtype in dtypes:
            pack = 16 // torch.empty((), dtype=dtype).element_size()
            for size in sizes:
                if size % pack != 0:
                    raise ValueError(
                        f"size {size} is not a multiple of vector pack {pack}"
                    )
                for pattern in patterns:
                    for mode in modes:
                        if mode in ("eager", "warmup", "graph", "compile_graph"):
                            results.append(
                                _check_single(
                                    ca,
                                    mode,
                                    size,
                                    dtype,
                                    rank,
                                    pattern,
                                    args.seed,
                                    args.graph_replays,
                                    args.graph_timing_warmup,
                                    args.graph_timing_iters,
                                )
                            )
                        elif mode == "graph_multi":
                            results.append(
                                _check_graph_multi(
                                    ca,
                                    size,
                                    dtype,
                                    rank,
                                    pattern,
                                    args.seed,
                                    args.graph_replays,
                                    args.multi_count,
                                    args.graph_timing_warmup,
                                    args.graph_timing_iters,
                                )
                            )
                        else:
                            raise ValueError(f"unsupported mode: {mode}")

        gathered: list[list[dict[str, Any]]] = [None] * world_size  # type: ignore
        dist.all_gather_object(gathered, results)
        all_results = [
            {"rank": rank_id, "results": rank_results}
            for rank_id, rank_results in enumerate(gathered)
        ]
        required_failures = []
        for rank_id, rank_results in enumerate(gathered):
            for item in rank_results:
                if item["pattern"] not in require_exact_patterns:
                    continue
                if item["mode"] == "graph_multi":
                    equal = item["all_equal"]
                    comparison = item
                else:
                    equal = item["comparison"]["equal"]
                    comparison = item["comparison"]
                if not equal:
                    required_failures.append(
                        {
                            "rank": rank_id,
                            "mode": item["mode"],
                            "size": item["size"],
                            "dtype": item["dtype"],
                            "pattern": item["pattern"],
                            "comparison": comparison,
                        }
                    )

        payload = {
            "world_size": world_size,
            "custom_allreduce_disabled": False,
            "fully_connected": ca.fully_connected,
            "max_size_bytes": ca.max_size,
            "sm70_tp4_m5_ar_threads": os.environ.get(
                "VLLM_SM70_TP4_M5_AR_THREADS"
            ),
            "required_exact_patterns": sorted(require_exact_patterns),
            "required_exact_passed": not required_failures,
            "required_failures": required_failures[:16],
            "rank_results": all_results,
        }
        if rank == 0:
            text = json.dumps(payload, indent=2, sort_keys=True)
            print(text)
            if args.json_out:
                path = Path(args.json_out)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text + "\n")
        raise SystemExit(0 if not required_failures else 1)
    finally:
        ca.close()
        dist.destroy_process_group(gloo_group)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
