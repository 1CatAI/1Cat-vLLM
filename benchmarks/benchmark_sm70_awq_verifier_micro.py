# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Microbenchmark SM70 AWQ GEMMs used by the DDTree target verifier.

The default suite models one Qwen3.6-27B-AWQ TP rank verifier forward with
236 AWQ dense GEMM calls. It intentionally times only awq_gemm_sm70_out after
weights have been prepared and dispatch has been warmed, so the result can be
compared to Nsight's AWQ-kernel bucket.
"""

from __future__ import annotations

import argparse
import csv
import functools
import json
import os
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

DEFAULT_DDTREE_27B_CASES = (
    # label, checkpoint layer/base, rank-call count, TP layout
    (
        "mlp_gate_up",
        "model.language_model.layers.1.mlp",
        63,
        "gate_up_column",
    ),
    (
        "mlp_down",
        "model.language_model.layers.1.mlp.down_proj",
        63,
        "row",
    ),
    (
        "linear_attn_in_proj_qkvz",
        "model.language_model.layers.1.linear_attn",
        47,
        "qkvz_column",
    ),
    (
        "linear_attn_out_proj",
        "model.language_model.layers.1.linear_attn.out_proj",
        47,
        "row",
    ),
    (
        "full_attn_o_proj",
        "model.language_model.layers.3.self_attn.o_proj",
        16,
        "row",
    ),
)

DEFAULT_GATED_SILU_LAYER = "model.language_model.layers.1.mlp"


@dataclass(frozen=True)
class BenchCase:
    label: str
    layer: str
    count: int
    m: int
    layout: str = "full"


@dataclass
class PreparedBenchCase:
    case: BenchCase
    run: Any
    output: torch.Tensor
    k: int
    n: int
    k_ld: int
    q_ld: int
    backend: str


class _LoadedSm70Ops:
    @staticmethod
    def awq_sm70_prepare(
        qweight: torch.Tensor,
        scales: torch.Tensor,
        qzeros: torch.Tensor,
        group_size: int,
        interleave_gated_silu: bool = False,
    ) -> list[torch.Tensor]:
        return torch.ops._C.awq_sm70_prepare(
            qweight, scales, qzeros, group_size, interleave_gated_silu
        )

    @staticmethod
    def awq_gemm_sm70_out(
        out: torch.Tensor,
        input: torch.Tensor,
        weight: torch.Tensor,
        scales: torch.Tensor,
        group_size: int,
        k_ld: int,
        q_ld: int,
        gated_silu: bool = False,
    ) -> None:
        torch.ops._C.awq_gemm_sm70_out(
            out,
            input,
            weight,
            scales,
            group_size,
            k_ld,
            q_ld,
            gated_silu,
        )


def _import_sm70_ops(op_library: Path | None = None) -> Any:
    if op_library is not None:
        torch.ops.load_library(str(op_library.resolve()))
        return _LoadedSm70Ops()
    try:
        from vllm import _sm70_ops

        return _sm70_ops
    except ImportError:
        from vllm import _custom_ops

        return _custom_ops


@functools.cache
def _load_m5_batched_gemv_ops() -> Any:
    from torch.utils.cpp_extension import load

    source = Path(__file__).resolve().parent / "csrc" / "sm70_awq_m5_batched_gemv.cu"
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.0")
    load(
        name="sm70_awq_m5_micro_v2",
        sources=[str(source)],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-lineinfo"],
        is_python_module=False,
        verbose=os.getenv("VLLM_SM70_M5_EXTENSION_VERBOSE") == "1",
    )
    return torch.ops.sm70_awq_m5_micro


def _require_sm70(device: torch.device) -> None:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA device.")
    capability = torch.cuda.get_device_capability(device)
    if capability != (7, 0):
        raise RuntimeError(f"Expected SM70, got sm_{capability[0]}{capability[1]}.")


@functools.cache
def _weight_map(model_path: Path) -> dict[str, str]:
    index = json.loads((model_path / "model.safetensors.index.json").read_text())
    return index["weight_map"]


def _load_awq(
    model_path: Path,
    layer: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    weight_map = _weight_map(model_path)
    with safe_open(
        model_path / weight_map[f"{layer}.qweight"],
        framework="pt",
        device="cpu",
    ) as f:
        qweight = f.get_tensor(f"{layer}.qweight").to(device).contiguous()
        scales = f.get_tensor(f"{layer}.scales").to(device).contiguous()
        qzeros = f.get_tensor(f"{layer}.qzeros").to(device).contiguous()
    return qweight, scales, qzeros


def _slice_awq_output(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    start: int,
    end: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if start % 8 != 0 or end % 8 != 0:
        raise ValueError("AWQ output slices must be multiples of 8.")
    return (
        qweight[:, start // 8 : end // 8].contiguous(),
        scales[:, start:end].contiguous(),
        qzeros[:, start // 8 : end // 8].contiguous(),
    )


def _slice_awq_input(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    start: int,
    end: int,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if start % group_size != 0 or end % group_size != 0:
        raise ValueError("AWQ input slices must align to group_size.")
    return (
        qweight[start:end].contiguous(),
        scales[start // group_size : end // group_size].contiguous(),
        qzeros[start // group_size : end // group_size].contiguous(),
    )


def _load_tp_column_awq(
    model_path: Path,
    layer: str,
    tp_size: int,
    tp_rank: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    qweight, scales, qzeros = _load_awq(model_path, layer, device)
    n = int(qweight.shape[1] * 8)
    if n % tp_size != 0:
        raise ValueError(f"Projection output dim {n} is not divisible by TP {tp_size}.")
    n_rank = n // tp_size
    start = tp_rank * n_rank
    return _slice_awq_output(qweight, scales, qzeros, start, start + n_rank)


def _load_tp_row_awq(
    model_path: Path,
    layer: str,
    tp_size: int,
    tp_rank: int,
    group_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    qweight, scales, qzeros = _load_awq(model_path, layer, device)
    k = int(qweight.shape[0])
    if k % tp_size != 0:
        raise ValueError(f"Projection input dim {k} is not divisible by TP {tp_size}.")
    k_rank = k // tp_size
    start = tp_rank * k_rank
    return _slice_awq_input(
        qweight,
        scales,
        qzeros,
        start,
        start + k_rank,
        group_size,
    )


def _load_tp_rank_gate_up_awq(
    model_path: Path,
    layer_base: str,
    tp_size: int,
    tp_rank: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    gate_qweight, gate_scales, gate_qzeros = _load_awq(
        model_path, f"{layer_base}.gate_proj", device
    )
    up_qweight, up_scales, up_qzeros = _load_awq(
        model_path, f"{layer_base}.up_proj", device
    )
    if gate_qweight.shape != up_qweight.shape:
        raise ValueError("gate_proj/up_proj qweight shapes differ.")
    if gate_scales.shape != up_scales.shape or gate_qzeros.shape != up_qzeros.shape:
        raise ValueError("gate_proj/up_proj scale/zero shapes differ.")
    n_projection = int(gate_qweight.shape[1] * 8)
    if n_projection % tp_size != 0:
        raise ValueError(
            f"Projection output dim {n_projection} is not divisible by TP {tp_size}."
        )
    n_rank = n_projection // tp_size
    start = tp_rank * n_rank
    end = start + n_rank
    gate = _slice_awq_output(gate_qweight, gate_scales, gate_qzeros, start, end)
    up = _slice_awq_output(up_qweight, up_scales, up_qzeros, start, end)
    qweight = torch.cat([gate[0], up[0]], dim=1).contiguous()
    scales = torch.cat([gate[1], up[1]], dim=1).contiguous()
    qzeros = torch.cat([gate[2], up[2]], dim=1).contiguous()
    return qweight, scales, qzeros, n_projection, n_rank


def _load_tp_rank_qkvz_awq(
    model_path: Path,
    layer_base: str,
    tp_size: int,
    tp_rank: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    qkv = _load_tp_column_awq(
        model_path, f"{layer_base}.in_proj_qkv", tp_size, tp_rank, device
    )
    z = _load_tp_column_awq(
        model_path, f"{layer_base}.in_proj_z", tp_size, tp_rank, device
    )
    return (
        torch.cat([qkv[0], z[0]], dim=1).contiguous(),
        torch.cat([qkv[1], z[1]], dim=1).contiguous(),
        torch.cat([qkv[2], z[2]], dim=1).contiguous(),
    )


def _load_case_awq(
    model_path: Path,
    case: BenchCase,
    tp_size: int,
    tp_rank: int,
    group_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if case.layout == "gate_up_column":
        qweight, scales, qzeros, _, _ = _load_tp_rank_gate_up_awq(
            model_path, case.layer, tp_size, tp_rank, device
        )
        return qweight, scales, qzeros
    if case.layout == "qkvz_column":
        return _load_tp_rank_qkvz_awq(model_path, case.layer, tp_size, tp_rank, device)
    if case.layout == "row":
        return _load_tp_row_awq(
            model_path,
            case.layer,
            tp_size,
            tp_rank,
            group_size,
            device,
        )
    if case.layout == "full":
        return _load_awq(model_path, case.layer, device)
    raise ValueError(f"Unknown TP layout {case.layout!r} for {case.label}.")


def _make_input(m: int, k: int, device: torch.device) -> torch.Tensor:
    values = torch.arange(m * k, device=device, dtype=torch.int32)
    values = ((values % 1024).to(torch.float32) / 512.0) - 1.0
    return values.reshape(m, k).to(torch.float16)


def _time_cuda_call(
    fn: Any,
    device: torch.device,
    warmup: int,
    iters: int,
    batch_repeats: int = 1,
) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)

    times_ms: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(batch_repeats):
            fn()
        end.record()
        end.synchronize()
        times_ms.append(float(start.elapsed_time(end)) / batch_repeats)

    ordered = sorted(times_ms)
    p90_index = min(len(ordered) - 1, int(0.9 * (len(ordered) - 1)))
    return {
        "mean_us": statistics.fmean(times_ms) * 1000.0,
        "min_us": min(times_ms) * 1000.0,
        "p50_us": statistics.median(times_ms) * 1000.0,
        "p90_us": ordered[p90_index] * 1000.0,
        "max_us": max(times_ms) * 1000.0,
    }


def _parse_case(raw: str, default_m: int) -> BenchCase:
    parts = raw.split(":")
    if len(parts) not in (3, 4):
        raise ValueError(
            f"--case must be LABEL:LAYER:COUNT or LABEL:LAYER:COUNT:M, got {raw!r}"
        )
    label, layer, count = parts[:3]
    m = int(parts[3]) if len(parts) == 4 else default_m
    return BenchCase(label=label, layer=layer, count=int(count), m=m)


def _default_cases(default_m: int) -> list[BenchCase]:
    return [
        BenchCase(
            label=label,
            layer=layer,
            count=count,
            m=default_m,
            layout=layout,
        )
        for label, layer, count, layout in DEFAULT_DDTREE_27B_CASES
    ]


def _real_layer_cases(model_path: Path, default_m: int) -> list[BenchCase]:
    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    text_config = config.get("text_config", config)
    layer_types = text_config["layer_types"]
    if len(layer_types) != 64:
        raise ValueError(f"Expected 64 layer types, got {len(layer_types)}.")

    cases: list[BenchCase] = []
    for layer_idx in range(1, len(layer_types)):
        prefix = f"model.language_model.layers.{layer_idx}"
        if layer_types[layer_idx] == "linear_attention":
            cases.extend(
                [
                    BenchCase(
                        label=f"layer{layer_idx}.linear_attn_in_proj_qkvz",
                        layer=f"{prefix}.linear_attn",
                        count=1,
                        m=default_m,
                        layout="qkvz_column",
                    ),
                    BenchCase(
                        label=f"layer{layer_idx}.linear_attn_out_proj",
                        layer=f"{prefix}.linear_attn.out_proj",
                        count=1,
                        m=default_m,
                        layout="row",
                    ),
                ]
            )
        elif layer_types[layer_idx] == "full_attention":
            cases.append(
                BenchCase(
                    label=f"layer{layer_idx}.full_attn_o_proj",
                    layer=f"{prefix}.self_attn.o_proj",
                    count=1,
                    m=default_m,
                    layout="row",
                )
            )
        else:
            raise ValueError(f"Unknown layer type {layer_types[layer_idx]!r}.")
        cases.extend(
            [
                BenchCase(
                    label=f"layer{layer_idx}.mlp_gate_up",
                    layer=f"{prefix}.mlp",
                    count=1,
                    m=default_m,
                    layout="gate_up_column",
                ),
                BenchCase(
                    label=f"layer{layer_idx}.mlp_down",
                    layer=f"{prefix}.mlp.down_proj",
                    count=1,
                    m=default_m,
                    layout="row",
                ),
            ]
        )
    return cases


def _prepare_case(
    ops: Any,
    model: Path,
    case: BenchCase,
    group_size: int,
    device: torch.device,
    tp_size: int,
    tp_rank: int,
    m5_batched_gemv: str,
) -> PreparedBenchCase:
    qweight, scales, qzeros = _load_case_awq(
        model,
        case,
        tp_size,
        tp_rank,
        group_size,
        device,
    )
    k = int(qweight.shape[0])
    n = int(qweight.shape[1] * 8)
    x = _make_input(case.m, k, device)
    out = torch.empty((case.m, n), dtype=torch.float16, device=device)

    hybrid_down_fallback = m5_batched_gemv == "fp16-hybrid" and k == 8704 and n == 5120
    if m5_batched_gemv == "off" or hybrid_down_fallback:
        tm_weight, tm_scales, meta = ops.awq_sm70_prepare(
            qweight, scales, qzeros, group_size
        )
        k_ld = int(meta[0].item())
        q_ld = int(meta[1].item())
        backend = (
            "turbomind_awq_gemm_hybrid_down_fallback"
            if hybrid_down_fallback
            else "turbomind_awq_gemm"
        )

        def run() -> None:
            ops.awq_gemm_sm70_out(
                out,
                x,
                tm_weight,
                tm_scales,
                group_size,
                k_ld,
                q_ld,
                False,
            )

    else:
        if case.m != 5:
            raise ValueError("The TensorRT-style batched-GEMV requires exact M=5.")
        if group_size != 128:
            raise ValueError(
                "The first batched-GEMV prototype requires group-size 128."
            )
        m5_ops = _load_m5_batched_gemv_ops()
        prepared_weight, prepared_scales, zero_bias = m5_ops.prepare(
            qweight, scales, qzeros, group_size
        )
        candidate_mode = "fp16" if m5_batched_gemv == "fp16-hybrid" else m5_batched_gemv
        accum_mode = {
            "fp16": 0,
            "fp32": 1,
            "fp32-full": 2,
        }[candidate_mode]
        backend = f"tensorrt_style_m5_{candidate_mode}_accum"
        k_ld = 0
        q_ld = 0

        def run() -> None:
            m5_ops.out(
                out,
                x,
                prepared_weight,
                prepared_scales,
                zero_bias,
                group_size,
                accum_mode,
            )

    return PreparedBenchCase(
        case=case,
        run=run,
        output=out,
        k=k,
        n=n,
        k_ld=k_ld,
        q_ld=q_ld,
        backend=backend,
    )


def _run_case(
    prepared: PreparedBenchCase,
    group_size: int,
    device: torch.device,
    warmup: int,
    iters: int,
    tp_size: int,
    tp_rank: int,
    batch_repeats: int,
) -> dict[str, Any]:
    case = prepared.case
    run = prepared.run
    timing = _time_cuda_call(run, device, warmup, iters, batch_repeats)
    weighted_mean_ms = float(case.count) * timing["mean_us"] / 1000.0
    return {
        **_describe_case(prepared, group_size, tp_size, tp_rank),
        **timing,
        "weighted_mean_ms": weighted_mean_ms,
    }


def _describe_case(
    prepared: PreparedBenchCase,
    group_size: int,
    tp_size: int,
    tp_rank: int,
) -> dict[str, Any]:
    case = prepared.case
    return {
        "label": case.label,
        "layer": case.layer,
        "count": case.count,
        "layout": case.layout,
        "backend": prepared.backend,
        "tp_size": tp_size,
        "tp_rank": tp_rank,
        "m": case.m,
        "n": prepared.n,
        "k": prepared.k,
        "group_size": group_size,
        "k_ld": prepared.k_ld,
        "q_ld": prepared.q_ld,
        "desc_hint": (
            f"sm70_f16_u4k{group_size}_f16_tnt_fff_{case.m}x{prepared.n}x{prepared.k}_1"
            if prepared.backend.startswith("turbomind_awq_gemm")
            else f"sm70_trt_style_awq_{case.m}x{prepared.n}x{prepared.k}"
        ),
    }


def _run_representative_manifest(
    prepared_cases: list[PreparedBenchCase],
    device: torch.device,
    warmup: int,
    iters: int,
    tp_size: int,
    representative_weights: bool,
) -> dict[str, Any]:
    def run() -> None:
        for prepared in prepared_cases:
            for _ in range(prepared.case.count):
                prepared.run()

    timing_us = _time_cuda_call(run, device, warmup, iters)
    return {
        "manifest_version": "qwen3.6-27b-awq-tp2-m5-v2",
        "calls_per_rank": sum(item.case.count for item in prepared_cases),
        "cross_tp_calls": tp_size * sum(item.case.count for item in prepared_cases),
        "graph_captured": False,
        "kernel_backends": sorted({item.backend for item in prepared_cases}),
        "representative_weights": representative_weights,
        "note": (
            "Each shape reuses one real checkpoint layer."
            if representative_weights
            else "All 236 calls use their actual checkpoint layer weights."
        ),
        **{
            name.replace("_us", "_ms"): value / 1000.0
            for name, value in timing_us.items()
        },
    }


def _capture_outputs(
    prepared_cases: list[PreparedBenchCase], device: torch.device
) -> dict[str, torch.Tensor]:
    for prepared in prepared_cases:
        prepared.run()
    torch.cuda.synchronize(device)
    return {
        prepared.case.label: prepared.output.detach().cpu().clone()
        for prepared in prepared_cases
    }


def _compare_outputs(
    outputs: dict[str, torch.Tensor], references: dict[str, torch.Tensor]
) -> dict[str, dict[str, float | int]]:
    comparisons: dict[str, dict[str, float | int]] = {}
    for label, output in outputs.items():
        if label not in references:
            raise KeyError(f"Reference output is missing case {label!r}.")
        reference = references[label]
        if output.shape != reference.shape:
            raise ValueError(
                f"Output shape mismatch for {label}: "
                f"{output.shape} != {reference.shape}."
            )
        diff = (output.float() - reference.float()).abs()
        comparisons[label] = {
            "elements": output.numel(),
            "exact_elements": int((output == reference).sum().item()),
            "max_abs_diff": float(diff.max().item()),
            "mean_abs_diff": float(diff.mean().item()),
            "rms_abs_diff": float(diff.square().mean().sqrt().item()),
            "sign_mismatch_elements": int(
                (torch.signbit(output) != torch.signbit(reference)).sum().item()
            ),
        }
    return comparisons


def _run_gated_silu_candidate(
    ops: Any,
    model: Path,
    layer_base: str,
    count: int,
    m: int,
    group_size: int,
    tp_size: int,
    tp_rank: int,
    device: torch.device,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    qweight, scales, qzeros, n_projection, n_rank = _load_tp_rank_gate_up_awq(
        model_path=model,
        layer_base=layer_base,
        tp_size=tp_size,
        tp_rank=tp_rank,
        device=device,
    )
    tm_weight, tm_scales, meta = ops.awq_sm70_prepare(
        qweight, scales, qzeros, group_size, False
    )
    fused_weight, fused_scales, fused_meta = ops.awq_sm70_prepare(
        qweight, scales, qzeros, group_size, True
    )
    k_ld = int(meta[0].item())
    q_ld = int(meta[1].item())
    fused_k_ld = int(fused_meta[0].item())
    fused_q_ld = int(fused_meta[1].item())
    k = int(qweight.shape[0])
    n = int(qweight.shape[1] * 8)
    out_features = n // 2
    x = _make_input(m, k, device)
    gate_up = torch.empty((m, n), dtype=torch.float16, device=device)
    baseline = torch.empty((m, out_features), dtype=torch.float16, device=device)
    fused = torch.empty_like(baseline)

    def run_baseline() -> None:
        ops.awq_gemm_sm70_out(
            gate_up,
            x,
            tm_weight,
            tm_scales,
            group_size,
            k_ld,
            q_ld,
            False,
        )
        torch.ops._C.silu_and_mul(baseline, gate_up)

    def run_fused() -> None:
        ops.awq_gemm_sm70_out(
            fused,
            x,
            fused_weight,
            fused_scales,
            group_size,
            fused_k_ld,
            fused_q_ld,
            True,
        )

    run_baseline()
    run_fused()
    torch.cuda.synchronize(device)
    diff = (baseline - fused).float().abs()
    baseline_timing = _time_cuda_call(run_baseline, device, warmup, iters)
    fused_timing = _time_cuda_call(run_fused, device, warmup, iters)
    delta_us = baseline_timing["mean_us"] - fused_timing["mean_us"]
    return {
        "label": "mlp_gate_up_gated_silu_candidate",
        "layer_base": layer_base,
        "count": count,
        "m": m,
        "k": k,
        "n_projection": n_projection,
        "n_projection_per_tp_rank": n_rank,
        "n_merged_per_tp_rank": n,
        "out_features": out_features,
        "group_size": group_size,
        "tp_size": tp_size,
        "tp_rank": tp_rank,
        "baseline_gate_up_plus_silu": baseline_timing,
        "fused_gated_silu": fused_timing,
        "delta_mean_us": delta_us,
        "weighted_delta_ms": float(count) * delta_us / 1000.0,
        "baseline_weighted_mean_ms": float(count) * baseline_timing["mean_us"] / 1000.0,
        "fused_weighted_mean_ms": float(count) * fused_timing["mean_us"] / 1000.0,
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "label",
        "layer",
        "count",
        "layout",
        "backend",
        "tp_size",
        "tp_rank",
        "m",
        "n",
        "k",
        "group_size",
        "k_ld",
        "q_ld",
        "desc_hint",
        "mean_us",
        "min_us",
        "p50_us",
        "p90_us",
        "max_us",
        "weighted_mean_ms",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/home/ymzx/models/Qwen3.6-27B-AWQ"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--m", type=int, default=17)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument(
        "--batch-repeats",
        type=int,
        default=1,
        help=(
            "Queue this many operator calls between one CUDA event pair and "
            "divide elapsed time per call. Use more than one for kernel-only "
            "timing without per-sample event/synchronize distortion."
        ),
    )
    parser.add_argument(
        "--case",
        action="append",
        default=None,
        help="LABEL:LAYER:COUNT or LABEL:LAYER:COUNT:M. Repeatable.",
    )
    parser.add_argument(
        "--only-label",
        action="append",
        help="Run only matching labels from the default manifest. Repeatable.",
    )
    parser.add_argument(
        "--all-real-layers",
        action="store_true",
        help="Load all 236 real layer weights in runtime layer order.",
    )
    parser.add_argument(
        "--skip-case-timing",
        action="store_true",
        help="Skip per-case event timing and only run aggregate/output checks.",
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--csv-out", type=Path)
    parser.add_argument(
        "--op-library",
        type=Path,
        help=(
            "Load AWQ TurboMind ops from an explicit archived _C library instead "
            "of importing the worktree vllm._C."
        ),
    )
    parser.add_argument(
        "--aggregate-iters",
        type=int,
        default=0,
        help=(
            "Time the representative 236-call manifest this many times; 0 disables it."
        ),
    )
    parser.add_argument("--aggregate-warmup", type=int, default=5)
    parser.add_argument(
        "--outputs-out",
        type=Path,
        help="Save one output tensor per case for a separate-process A/B comparison.",
    )
    parser.add_argument(
        "--reference-outputs",
        type=Path,
        help="Compare case outputs against tensors saved by --outputs-out.",
    )
    parser.add_argument(
        "--gated-silu-candidate",
        action="store_true",
        help=(
            "Also benchmark the TP-rank merged gate_up_proj path as "
            "AWQ GEMM + silu_and_mul versus AWQ fused gated-SiLU epilogue."
        ),
    )
    parser.add_argument("--gated-silu-layer", default=DEFAULT_GATED_SILU_LAYER)
    parser.add_argument("--gated-silu-count", type=int, default=63)
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument(
        "--m5-batched-gemv",
        choices=("off", "fp16", "fp16-hybrid", "fp32", "fp32-full"),
        default="off",
        help=(
            "Use the benchmark-only TensorRT-style exact-M5 AWQ batched-GEMV. "
            "The default keeps the production TurboMind GEMM baseline."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.batch_repeats < 1:
        raise ValueError("--batch-repeats must be positive.")
    if args.aggregate_iters < 0 or args.aggregate_warmup < 0:
        raise ValueError("Aggregate iteration counts must be non-negative.")
    device = torch.device(args.device)
    _require_sm70(device)
    ops = _import_sm70_ops(args.op_library)
    if args.case and args.all_real_layers:
        raise ValueError("--case and --all-real-layers are mutually exclusive.")
    if args.all_real_layers:
        cases = _real_layer_cases(args.model, args.m)
    elif args.case:
        cases = [_parse_case(raw, args.m) for raw in args.case]
    else:
        cases = _default_cases(args.m)
    if args.only_label:
        labels = set(args.only_label)
        cases = [case for case in cases if case.label in labels]
        missing = labels - {case.label for case in cases}
        if missing:
            raise ValueError(f"Unknown --only-label values: {sorted(missing)}")

    prepared_cases = [
        _prepare_case(
            ops=ops,
            model=args.model,
            case=case,
            group_size=args.group_size,
            device=device,
            tp_size=args.tp_size,
            tp_rank=args.tp_rank,
            m5_batched_gemv=args.m5_batched_gemv,
        )
        for case in cases
    ]
    if args.skip_case_timing:
        rows = [
            _describe_case(prepared, args.group_size, args.tp_size, args.tp_rank)
            for prepared in prepared_cases
        ]
        total_ms = None
    else:
        rows = [
            _run_case(
                prepared=prepared,
                group_size=args.group_size,
                device=device,
                warmup=args.warmup,
                iters=args.iters,
                tp_size=args.tp_size,
                tp_rank=args.tp_rank,
                batch_repeats=args.batch_repeats,
            )
            for prepared in prepared_cases
        ]
        total_ms = sum(float(row["weighted_mean_ms"]) for row in rows)
    payload: dict[str, Any] = {
        "model": str(args.model),
        "device": args.device,
        "device_name": torch.cuda.get_device_name(device),
        "device_capability": list(torch.cuda.get_device_capability(device)),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "warmup": args.warmup,
        "iters": args.iters,
        "batch_repeats": args.batch_repeats,
        "m5_batched_gemv": args.m5_batched_gemv,
        "op_library": str(args.op_library) if args.op_library else None,
        "total_weighted_mean_ms": total_ms,
        "cases": rows,
    }
    if args.aggregate_iters:
        payload["representative_manifest"] = _run_representative_manifest(
            prepared_cases=prepared_cases,
            device=device,
            warmup=args.aggregate_warmup,
            iters=args.aggregate_iters,
            tp_size=args.tp_size,
            representative_weights=not args.all_real_layers,
        )
    outputs: dict[str, torch.Tensor] | None = None
    if args.outputs_out is not None or args.reference_outputs is not None:
        outputs = _capture_outputs(prepared_cases, device)
    if args.reference_outputs is not None:
        references = torch.load(
            args.reference_outputs, map_location="cpu", weights_only=True
        )
        payload["output_comparison"] = _compare_outputs(outputs or {}, references)
    if args.gated_silu_candidate:
        payload["gated_silu_candidate"] = _run_gated_silu_candidate(
            ops=ops,
            model=args.model,
            layer_base=args.gated_silu_layer,
            count=args.gated_silu_count,
            m=args.m,
            group_size=args.group_size,
            tp_size=args.tp_size,
            tp_rank=args.tp_rank,
            device=device,
            warmup=args.warmup,
            iters=args.iters,
        )
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    if args.csv_out is not None:
        _write_csv(args.csv_out, rows)
    if args.outputs_out is not None:
        args.outputs_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(outputs, args.outputs_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
