# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Race experimental QPN8 against TurboMind on real TP-local FP8 weights.

The benchmark is deliberately operator-only: it does not change model
dispatch. It covers the exact Qwen3.8-27B-FP8 TP4 decode shapes at M=1, 2, 4,
and 8, and measures eager launches separately from CUDA Graph replay.

The production gate/up GEMM has a fused SiLU epilogue. ``gate_up_raw`` is
therefore opt-in and diagnostic only; its raw-GEMM timing is not an end-to-end
replacement claim.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import vllm._C  # noqa: F401
from safetensors import safe_open

from vllm import _sm70_ops as sm70_ops

_KORDER8 = [0, 2, 4, 6, 1, 3, 5, 7, 8, 10, 12, 14, 9, 11, 13, 15]
_DEFAULT_CASES = ("down", "gdn_in", "output", "full_qkv")
_ALL_CASES = (*_DEFAULT_CASES, "gate_up_raw", "gate_up_fused")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/home/ymzx/models/Qwen3.8-27B-FP8"),
    )
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument(
        "--qpn8-library",
        type=Path,
        help="Optional standalone library built from fp8_qpn8_sm70.cu.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--m", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument(
        "--cases", choices=_ALL_CASES, nargs="+", default=list(_DEFAULT_CASES)
    )
    parser.add_argument("--split-k", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--nacc", type=int, nargs="+", default=[1, 2])
    parser.add_argument(
        "--decoder", choices=("scalar", "fast"), nargs="+", default=["scalar", "fast"]
    )
    parser.add_argument(
        "--prefetch", choices=("off", "on"), nargs="+", default=["off", "on"]
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument(
        "--cache-state",
        choices=("warm", "cold"),
        nargs="+",
        default=["warm", "cold"],
    )
    parser.add_argument("--cache-scrub-mib", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--relative-l2-limit", type=float, default=5e-3)
    parser.add_argument("--cosine-limit", type=float, default=0.999)
    return parser.parse_args()


def _load_keys(model: Path, keys: list[str]) -> dict[str, torch.Tensor]:
    index = json.loads((model / "model.safetensors.index.json").read_text())
    weight_map: dict[str, str] = index["weight_map"]
    by_file: dict[str, list[str]] = {}
    for key in keys:
        by_file.setdefault(weight_map[key], []).append(key)

    result: dict[str, torch.Tensor] = {}
    for filename, file_keys in by_file.items():
        with safe_open(model / filename, framework="pt", device="cpu") as handle:
            for key in file_keys:
                result[key] = handle.get_tensor(key)
    return result


def _weight_keys(prefix: str) -> tuple[str, str]:
    return f"{prefix}.weight", f"{prefix}.weight_scale_inv"


def _column_shard_raw(
    weight: torch.Tensor,
    scales: torch.Tensor,
    tp_size: int,
    tp_rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    n = weight.shape[0]
    if n % tp_size:
        raise ValueError(f"N={n} is not divisible by TP={tp_size}")
    begin = tp_rank * (n // tp_size)
    end = begin + n // tp_size
    scale_begin = begin // 128
    scale_end = math.ceil(end / 128)
    raw = weight.view(torch.uint8)[begin:end].contiguous()
    return raw, scales[scale_begin:scale_end].contiguous()


def _row_shard_raw(
    weight: torch.Tensor,
    scales: torch.Tensor,
    tp_size: int,
    tp_rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    k = weight.shape[1]
    if k % tp_size:
        raise ValueError(f"K={k} is not divisible by TP={tp_size}")
    begin = tp_rank * (k // tp_size)
    end = begin + k // tp_size
    scale_begin = begin // 128
    scale_end = math.ceil(end / 128)
    raw = weight.view(torch.uint8)[:, begin:end].contiguous()
    return raw, scales[:, scale_begin:scale_end].contiguous()


def _load_case(
    model: Path,
    case: str,
    tp_size: int,
    tp_rank: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    root = "model.language_model.layers"
    if case == "down":
        prefixes = [f"{root}.1.mlp.down_proj"]
        shard = "row"
        note = "production down_proj TP row shard"
    elif case == "gdn_in":
        prefixes = [
            f"{root}.1.linear_attn.in_proj_qkv",
            f"{root}.1.linear_attn.in_proj_z",
        ]
        shard = "column"
        note = "production GDN qkv+z TP column shards concatenated"
    elif case == "output":
        prefixes = [f"{root}.1.linear_attn.out_proj"]
        shard = "row"
        note = "production GDN/full-attention output TP row-shard shape"
    elif case == "full_qkv":
        prefixes = [
            f"{root}.3.self_attn.q_proj",
            f"{root}.3.self_attn.k_proj",
            f"{root}.3.self_attn.v_proj",
        ]
        shard = "column"
        note = "production full-attention q+k+v TP column shards concatenated"
    elif case in ("gate_up_raw", "gate_up_fused"):
        prefixes = [
            f"{root}.1.mlp.gate_proj",
            f"{root}.1.mlp.up_proj",
        ]
        shard = "column"
        note = (
            "production gate+up GEMM with fused SiLU-and-multiply"
            if case == "gate_up_fused"
            else "diagnostic raw gate+up GEMM; production has fused SiLU epilogue"
        )
    else:
        raise ValueError(f"unknown case: {case}")

    keys = [key for prefix in prefixes for key in _weight_keys(prefix)]
    loaded = _load_keys(model, keys)
    raw_parts: list[torch.Tensor] = []
    scale_parts: list[torch.Tensor] = []
    fp8_dtype: torch.dtype | None = None
    for prefix in prefixes:
        weight_key, scale_key = _weight_keys(prefix)
        weight = loaded[weight_key]
        scales = loaded[scale_key].float()
        if fp8_dtype is None:
            fp8_dtype = weight.dtype
        elif weight.dtype != fp8_dtype:
            raise ValueError("concatenated weights have different dtypes")
        if shard == "column":
            raw, scale = _column_shard_raw(weight, scales, tp_size, tp_rank)
        else:
            raw, scale = _row_shard_raw(weight, scales, tp_size, tp_rank)
        raw_parts.append(raw)
        scale_parts.append(scale)

    raw = torch.cat(raw_parts, dim=0) if len(raw_parts) > 1 else raw_parts[0]
    scales = torch.cat(scale_parts, dim=0) if len(scale_parts) > 1 else scale_parts[0]
    assert fp8_dtype is not None
    qweight = raw.to(device, non_blocking=False).view(fp8_dtype).contiguous()
    return qweight, scales.to(device).contiguous(), note


def _qpn8_prepack(raw_weight: torch.Tensor) -> torch.Tensor:
    """[N,K] uint8 -> [tile N/32][group K/16][lane 32][16B]."""
    n, k = raw_weight.shape
    if n % 32 or k % 16:
        raise ValueError(f"QPN8 packing requires N%32=K%16=0, got N={n} K={k}")
    device = raw_weight.device
    tiles, groups = n // 32, k // 16
    lane = torch.arange(32, device=device)
    col = ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) > 0).long() * 4
    korder = torch.tensor(_KORDER8, device=device)
    group = torch.arange(groups, device=device)
    kidx = group[:, None] * 16 + korder[None, :]
    packed = torch.empty((tiles, groups, 32, 16), dtype=torch.uint8, device=device)
    # Bound the temporary expanded index tensors for the 44-MiB projections.
    tiles_per_chunk = max(1, 36864 // max(groups, 1))
    for tile_begin in range(0, tiles, tiles_per_chunk):
        tile_end = min(tile_begin + tiles_per_chunk, tiles)
        tile_count = tile_end - tile_begin
        ncol = (
            torch.arange(tile_begin, tile_end, device=device)[:, None] * 32
            + col[None, :]
        )
        packed[tile_begin:tile_end] = raw_weight[
            ncol.view(tile_count, 1, 32, 1).expand(tile_count, groups, 32, 16),
            kidx.view(1, groups, 1, 16).expand(tile_count, groups, 32, 16),
        ]
    return packed.view(-1).contiguous()


def _qpn8_group_scales(scales: torch.Tensor, n: int, k: int) -> torch.Tensor:
    expected = (math.ceil(n / 128), math.ceil(k / 128))
    if tuple(scales.shape) != expected or n % 128 or k % 128:
        raise ValueError(
            f"QPN8 scales need exact 128x128 blocks, got {tuple(scales.shape)} "
            f"for N={n}, K={k}"
        )
    return (
        scales.t()
        .reshape(k // 128, n // 128)
        .repeat_interleave(4, dim=1)
        .mul(256.0)
        .half()
        .contiguous()
    )


def _dequantized_weight(qweight: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    n, k = qweight.shape
    expanded = scales.repeat_interleave(128, 0).repeat_interleave(128, 1)
    return qweight.float().mul(expanded[:n, :k])


def _error_stats(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    actual_f32 = actual.float()
    reference_f32 = reference.float()
    delta = actual_f32 - reference_f32
    ref_norm = torch.linalg.vector_norm(reference_f32).clamp_min(1e-12)
    relative_l2 = torch.linalg.vector_norm(delta) / ref_norm
    cosine = torch.nn.functional.cosine_similarity(
        actual_f32.flatten(), reference_f32.flatten(), dim=0
    )
    return {
        "finite": bool(torch.isfinite(actual).all().item()),
        "max_abs": float(delta.abs().max().item()),
        "mean_abs": float(delta.abs().mean().item()),
        "relative_l2": float(relative_l2.item()),
        "cosine": float(cosine.item()),
    }


def _event_trials(
    launch: Callable[[], None],
    warmup: int,
    iters: int,
    trials: int,
    cache_scrub: torch.Tensor | None,
) -> dict[str, float]:
    for _ in range(warmup):
        if cache_scrub is not None:
            cache_scrub.add_(1)
        launch()
    torch.cuda.synchronize()
    samples_us: list[float] = []
    for _ in range(trials):
        if cache_scrub is None:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                launch()
            end.record()
            end.synchronize()
            samples_us.append(float(start.elapsed_time(end) * 1000.0 / iters))
        else:
            # Scrub more than V100's L2 immediately before each call, while
            # the event pair measures only the tested operator.
            starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
            ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
            for start, end in zip(starts, ends):
                cache_scrub.add_(1)
                start.record()
                launch()
                end.record()
            ends[-1].synchronize()
            samples_us.append(
                float(
                    statistics.mean(
                        start.elapsed_time(end) * 1000.0
                        for start, end in zip(starts, ends)
                    )
                )
            )
    ordered = sorted(samples_us)
    return {
        "median_us": float(statistics.median(samples_us)),
        "min_us": ordered[0],
        "p10_us": ordered[max(0, math.ceil(0.1 * len(ordered)) - 1)],
        "p90_us": ordered[min(len(ordered) - 1, math.ceil(0.9 * len(ordered)) - 1)],
        "max_us": ordered[-1],
    }


def _benchmark_eager(
    launch: Callable[[], None],
    warmup: int,
    iters: int,
    trials: int,
    cache_scrub: torch.Tensor | None,
) -> dict[str, Any]:
    timing = _event_trials(launch, warmup, iters, trials, cache_scrub)
    launch()
    torch.cuda.synchronize()
    return {"timing": timing, "replay_max_abs": None}


def _benchmark_graph(
    launch: Callable[[], None],
    output: torch.Tensor,
    warmup: int,
    iters: int,
    trials: int,
    cache_scrub: torch.Tensor | None,
) -> dict[str, Any]:
    capture_stream = torch.cuda.Stream(device=output.device)
    capture_stream.wait_stream(torch.cuda.current_stream(output.device))
    with torch.cuda.stream(capture_stream):
        for _ in range(max(warmup, 3)):
            launch()
    capture_stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        launch()
    torch.cuda.synchronize(output.device)

    timing = _event_trials(graph.replay, warmup, iters, trials, cache_scrub)
    graph.replay()
    torch.cuda.synchronize(output.device)
    first = output.clone()
    graph.replay()
    torch.cuda.synchronize(output.device)
    replay_max_abs = float((output - first).abs().max().item())
    del first, graph, capture_stream
    gc.collect()
    return {"timing": timing, "replay_max_abs": replay_max_abs}


def _derived_metrics(
    timing: dict[str, float], m: int, n: int, k: int, bytes_per_call: int
) -> dict[str, float]:
    seconds = timing["median_us"] * 1e-6
    return {
        "effective_gbps": float(bytes_per_call / seconds / 1e9),
        "useful_tflops": float(2 * m * n * k / seconds / 1e12),
    }


def _quality_pass(
    stats: dict[str, Any], relative_l2_limit: float, cosine_limit: float
) -> bool:
    return bool(
        stats["finite"]
        and stats["relative_l2"] <= relative_l2_limit
        and stats["cosine"] >= cosine_limit
    )


def _run_case(args: argparse.Namespace, case: str) -> dict[str, Any]:
    device = torch.device(args.device)
    qweight, scales, note = _load_case(
        args.model, case, args.tp_size, args.tp_rank, device
    )
    n, k = (int(dim) for dim in qweight.shape)
    gated_silu = case == "gate_up_fused"
    output_n = n // 2 if gated_silu else n
    raw = qweight.view(torch.uint8)
    codes = _qpn8_prepack(raw)
    group_scales = _qpn8_group_scales(scales, n, k)
    tm_weight, tm_scales, meta = sm70_ops.fp8_sm70_prepare(
        qweight, scales, 128, gated_silu
    )
    k_ld, q_ld = (int(value) for value in meta.tolist())
    reference_weight = _dequantized_weight(qweight, scales)
    cache_scrub = torch.empty(
        args.cache_scrub_mib * 1024 * 1024,
        dtype=torch.uint8,
        device=device,
    )
    torch.cuda.synchronize(device)

    case_result: dict[str, Any] = {
        "case": case,
        "note": note,
        "shape": {"n": n, "k": k, "output_n": output_n},
        "qweight_dtype": str(qweight.dtype),
        "scale_dtype": str(scales.dtype),
        "tm_meta": {"k_ld": k_ld, "q_ld": q_ld},
        "packed_bytes": int(codes.numel()),
        "group_scale_shape": list(group_scales.shape),
        "rows": [],
    }

    valid_configs = [
        (split_k, nacc, decoder == "fast", prefetch == "on")
        for split_k in args.split_k
        for nacc in args.nacc
        for decoder in args.decoder
        for prefetch in args.prefetch
        if (k // 16) % split_k == 0
        and not (prefetch == "on" and decoder != "fast")
        and not (gated_silu and split_k == 32)
    ]
    for m in args.m:
        torch.manual_seed(args.seed + m)
        inputs = torch.randn((m, k), device=device, dtype=torch.float16).mul_(0.1)
        reference_raw = inputs.float().matmul(reference_weight.t())
        if gated_silu:
            gate, up = reference_raw.chunk(2, dim=1)
            reference = (torch.nn.functional.silu(gate) * up).half()
        else:
            reference = reference_raw.half()

        tm_out = torch.empty((m, output_n), device=device, dtype=torch.float16)

        def launch_tm(
            tm_out: torch.Tensor = tm_out,
            inputs: torch.Tensor = inputs,
            tm_weight: torch.Tensor = tm_weight,
            tm_scales: torch.Tensor = tm_scales,
        ) -> None:
            sm70_ops.fp8_gemm_sm70_out(
                tm_out,
                inputs,
                tm_weight,
                tm_scales,
                128,
                k_ld,
                q_ld,
                gated_silu,
            )

        launch_tm()
        torch.cuda.synchronize(device)
        tm_quality = _error_stats(tm_out, reference)
        tm_bytes = n * k + 2 * (k // 128) * n + 2 * m * (output_n + k)
        for cache_state in args.cache_state:
            scrub = cache_scrub if cache_state == "cold" else None
            for mode, benchmark in (
                ("eager", _benchmark_eager),
                ("graph", _benchmark_graph),
            ):
                measured = (
                    benchmark(
                        launch_tm,
                        args.warmup,
                        args.iters,
                        args.trials,
                        scrub,
                    )
                    if mode == "eager"
                    else benchmark(
                        launch_tm,
                        tm_out,
                        args.warmup,
                        args.iters,
                        args.trials,
                        scrub,
                    )
                )
                case_result["rows"].append(
                    {
                        "backend": "turbomind_current",
                        "mode": mode,
                        "cache_state": cache_state,
                        "m": m,
                        "config": None,
                        "quality": tm_quality,
                        "quality_pass": _quality_pass(
                            tm_quality, args.relative_l2_limit, args.cosine_limit
                        ),
                        **measured,
                        **_derived_metrics(measured["timing"], m, n, k, tm_bytes),
                    }
                )

        for split_k, nacc, fast_decoder, prefetch_codes in valid_configs:
            qpn_out = torch.empty((m, output_n), device=device, dtype=torch.float16)

            def launch_qpn(
                split_k: int = split_k,
                nacc: int = nacc,
                fast_decoder: bool = fast_decoder,
                prefetch_codes: bool = prefetch_codes,
                qpn_out: torch.Tensor = qpn_out,
                inputs: torch.Tensor = inputs,
                codes: torch.Tensor = codes,
                group_scales: torch.Tensor = group_scales,
                gated_silu: bool = gated_silu,
            ) -> None:
                if gated_silu:
                    sm70_ops.fp8_qpn8_gated_pair_sm70_out(
                        qpn_out,
                        inputs,
                        codes,
                        group_scales,
                        split_k,
                        nacc,
                        fast_decoder,
                        prefetch_codes,
                    )
                else:
                    sm70_ops.fp8_qpn8_gemm_sm70_out(
                        qpn_out,
                        inputs,
                        codes,
                        group_scales,
                        split_k,
                        nacc,
                        fast_decoder,
                        prefetch_codes,
                    )

            launch_qpn()
            torch.cuda.synchronize(device)
            quality = _error_stats(qpn_out, reference)
            quality_pass = _quality_pass(
                quality, args.relative_l2_limit, args.cosine_limit
            )
            config = {
                "split_k": split_k,
                "nacc": nacc,
                "decoder": "fast" if fast_decoder else "scalar",
                "prefetch": prefetch_codes,
            }
            qpn_bytes = n * k + 2 * (k // 128) * (n // 32) + 2 * m * (n + k)
            if gated_silu:
                qpn_bytes -= 2 * m * (n - output_n)
            for cache_state in args.cache_state:
                scrub = cache_scrub if cache_state == "cold" else None
                for mode, benchmark in (
                    ("eager", _benchmark_eager),
                    ("graph", _benchmark_graph),
                ):
                    measured = (
                        benchmark(
                            launch_qpn,
                            args.warmup,
                            args.iters,
                            args.trials,
                            scrub,
                        )
                        if mode == "eager"
                        else benchmark(
                            launch_qpn,
                            qpn_out,
                            args.warmup,
                            args.iters,
                            args.trials,
                            scrub,
                        )
                    )
                    case_result["rows"].append(
                        {
                            "backend": "qpn8_experimental",
                            "mode": mode,
                            "cache_state": cache_state,
                            "m": m,
                            "config": config,
                            "quality": quality,
                            "quality_pass": quality_pass,
                            **measured,
                            **_derived_metrics(measured["timing"], m, n, k, qpn_bytes),
                        }
                    )

            del qpn_out

        del inputs, reference_raw, reference, tm_out

    del (
        reference_weight,
        tm_weight,
        tm_scales,
        codes,
        group_scales,
        qweight,
        scales,
        cache_scrub,
    )
    torch.cuda.empty_cache()
    gc.collect()
    return case_result


def _summarize(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for case in cases:
        rows: list[dict[str, Any]] = case["rows"]
        for m in sorted({row["m"] for row in rows}):
            for cache_state in sorted({row["cache_state"] for row in rows}):
                for mode in ("eager", "graph"):
                    current = next(
                        row
                        for row in rows
                        if row["m"] == m
                        and row["mode"] == mode
                        and row["cache_state"] == cache_state
                        and row["backend"] == "turbomind_current"
                    )
                    candidates = [
                        row
                        for row in rows
                        if row["m"] == m
                        and row["mode"] == mode
                        and row["cache_state"] == cache_state
                        and row["backend"] == "qpn8_experimental"
                        and row["quality_pass"]
                        and (row["replay_max_abs"] in (None, 0.0))
                    ]
                    best = min(candidates, key=lambda row: row["timing"]["median_us"])
                    current_us = current["timing"]["median_us"]
                    best_us = best["timing"]["median_us"]
                    summary.append(
                        {
                            "case": case["case"],
                            "shape": case["shape"],
                            "m": m,
                            "mode": mode,
                            "cache_state": cache_state,
                            "current_us": current_us,
                            "best_qpn8_us": best_us,
                            "speedup": current_us / best_us,
                            "best_config": best["config"],
                            "best_effective_gbps": best["effective_gbps"],
                            "best_useful_tflops": best["useful_tflops"],
                            "best_quality": best["quality"],
                        }
                    )
    return summary


def main() -> int:
    args = _parse_args()
    if args.qpn8_library is not None:
        torch.ops.load_library(str(args.qpn8_library.resolve()))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    if torch.cuda.get_device_capability(device) != (7, 0):
        raise RuntimeError("This benchmark requires an SM70/V100 device")
    if not hasattr(torch.ops._C, "fp8_qpn8_gemm_sm70_out"):
        raise RuntimeError("Missing _C::fp8_qpn8_gemm_sm70_out; build this source tree")
    if args.tp_size <= 0 or not 0 <= args.tp_rank < args.tp_size:
        raise ValueError("invalid TP size/rank")
    if any(m < 1 or m > 8 for m in args.m):
        raise ValueError("QPN8 operator supports M=1..8")
    if args.warmup < 1 or args.iters < 1 or args.trials < 1 or args.cache_scrub_mib < 1:
        raise ValueError("warmup, iters, trials, and cache scrub size must be positive")

    torch.cuda.set_device(device)
    cases = [_run_case(args, case) for case in args.cases]
    summary = _summarize(cases)
    payload = {
        "environment": {
            "model": str(args.model),
            "device": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "tp_size": args.tp_size,
            "tp_rank": args.tp_rank,
            "m": args.m,
            "warmup": args.warmup,
            "iters": args.iters,
            "trials": args.trials,
            "cache_state": args.cache_state,
            "cache_scrub_mib": args.cache_scrub_mib,
            "seed": args.seed,
        },
        "summary": summary,
        "cases": cases,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
