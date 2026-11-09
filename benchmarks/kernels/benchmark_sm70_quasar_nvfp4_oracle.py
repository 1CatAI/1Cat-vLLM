# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audit SM70 TurboMind against real QUASAR NVFP4 projection shards."""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open


@dataclass(frozen=True)
class Projection:
    name: str
    packed: torch.Tensor
    scales: torch.Tensor
    weight_global_divisor: float
    input_global_divisor: float


QPN2_CONFIGS = {
    # (K, physical N): (split K, independent accumulator chains)
    (1536, 5120): (8, 2),
    (4352, 5120): (16, 2),
    (5120, 3584): (16, 2),
    (5120, 4128): (16, 2),
    (5120, 8704): (8, 2),
}


class Checkpoint:
    def __init__(self, model: Path):
        self.model = model
        index_path = model / "model.safetensors.index.json"
        self.weight_map = json.loads(index_path.read_text())["weight_map"]

    def tensor(self, name: str) -> torch.Tensor:
        shard = self.model / self.weight_map[name]
        with safe_open(shard, framework="pt", device="cpu") as tensors:
            return tensors.get_tensor(name)


def _projection_keys(prefix: str) -> tuple[str, str, str, str]:
    return tuple(
        f"{prefix}.{suffix}"
        for suffix in (
            "weight_packed",
            "weight_scale",
            "weight_global_scale",
            "input_global_scale",
        )
    )  # type: ignore[return-value]


def _load_column_parallel(
    checkpoint: Checkpoint,
    name: str,
    prefixes: tuple[str, ...],
    tp_rank: int,
    tp_size: int,
) -> Projection:
    packed_parts: list[torch.Tensor] = []
    scale_parts: list[torch.Tensor] = []
    weight_divisors: list[float] = []
    input_divisors: list[float] = []
    for prefix in prefixes:
        packed_key, scale_key, weight_key, input_key = _projection_keys(prefix)
        packed = checkpoint.tensor(packed_key)
        scales = checkpoint.tensor(scale_key)
        if packed.shape[0] % tp_size:
            raise ValueError(f"{prefix}: output rows do not divide TP{tp_size}")
        output_sizes = (packed.shape[0],)
        if prefix.endswith(".in_proj_qkv"):
            config = json.loads((checkpoint.model / "config.json").read_text())
            config = config.get("text_config", config)
            key_dim = config["linear_num_key_heads"] * config["linear_key_head_dim"]
            value_dim = (
                config["linear_num_value_heads"] * config["linear_value_head_dim"]
            )
            output_sizes = (key_dim, key_dim, value_dim)
            if sum(output_sizes) != packed.shape[0]:
                raise ValueError(f"{prefix}: Q/K/V sizes do not match checkpoint")
        offset = 0
        for output_size in output_sizes:
            if output_size % tp_size:
                raise ValueError(f"{prefix}: logical projection does not divide TP")
            rows = output_size // tp_size
            start = offset + tp_rank * rows
            packed_parts.append(packed[start : start + rows].contiguous())
            scale_parts.append(scales[start : start + rows].contiguous())
            offset += output_size
        weight_divisors.append(float(checkpoint.tensor(weight_key).flatten()[0]))
        input_divisors.append(float(checkpoint.tensor(input_key).flatten()[0]))
    if len(set(weight_divisors)) != 1 or len(set(input_divisors)) != 1:
        raise ValueError(
            f"{name}: fused logical projections have different global scales"
        )
    return Projection(
        name=name,
        packed=torch.cat(packed_parts),
        scales=torch.cat(scale_parts),
        weight_global_divisor=weight_divisors[0],
        input_global_divisor=input_divisors[0],
    )


def _load_row_parallel(
    checkpoint: Checkpoint,
    name: str,
    prefix: str,
    tp_rank: int,
    tp_size: int,
) -> Projection:
    packed_key, scale_key, weight_key, input_key = _projection_keys(prefix)
    packed = checkpoint.tensor(packed_key)
    scales = checkpoint.tensor(scale_key)
    if packed.shape[1] % tp_size or scales.shape[1] % tp_size:
        raise ValueError(f"{prefix}: input columns do not divide TP{tp_size}")
    packed_cols = packed.shape[1] // tp_size
    scale_cols = scales.shape[1] // tp_size
    packed_start = tp_rank * packed_cols
    scale_start = tp_rank * scale_cols
    return Projection(
        name=name,
        packed=packed[:, packed_start : packed_start + packed_cols].contiguous(),
        scales=scales[:, scale_start : scale_start + scale_cols].contiguous(),
        weight_global_divisor=float(checkpoint.tensor(weight_key).flatten()[0]),
        input_global_divisor=float(checkpoint.tensor(input_key).flatten()[0]),
    )


def _load_projections(model: Path, tp_rank: int, tp_size: int) -> list[Projection]:
    checkpoint = Checkpoint(model)
    root = "model.language_model.layers"
    gdn = f"{root}.0.linear_attn"
    attn = f"{root}.3.self_attn"
    mlp = f"{root}.0.mlp"
    return [
        _load_column_parallel(
            checkpoint,
            "gdn_qkvzba",
            (
                f"{gdn}.in_proj_qkv",
                f"{gdn}.in_proj_z",
                f"{gdn}.in_proj_b",
                f"{gdn}.in_proj_a",
            ),
            tp_rank,
            tp_size,
        ),
        _load_row_parallel(checkpoint, "gdn_out", f"{gdn}.out_proj", tp_rank, tp_size),
        _load_column_parallel(
            checkpoint,
            "attention_qkv",
            (f"{attn}.q_proj", f"{attn}.k_proj", f"{attn}.v_proj"),
            tp_rank,
            tp_size,
        ),
        _load_row_parallel(
            checkpoint, "attention_out", f"{attn}.o_proj", tp_rank, tp_size
        ),
        _load_column_parallel(
            checkpoint,
            "mlp_gate_up",
            (f"{mlp}.gate_proj", f"{mlp}.up_proj"),
            tp_rank,
            tp_size,
        ),
        _load_row_parallel(
            checkpoint, "mlp_down", f"{mlp}.down_proj", tp_rank, tp_size
        ),
    ]


def _unpack_codes(packed: torch.Tensor) -> torch.Tensor:
    return (
        torch.stack((packed & 0xF, packed >> 4), dim=-1)
        .flatten(start_dim=-2)
        .t()
        .contiguous()
    )


def _dequantize_weight(projection: Projection, device: torch.device) -> torch.Tensor:
    packed = projection.packed.to(device)
    nibbles = torch.stack((packed & 0xF, packed >> 4), dim=-1).flatten(start_dim=-2)
    magnitudes = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float32,
        device=device,
    )
    weight = magnitudes[(nibbles & 7).long()]
    weight = torch.where((nibbles & 8) != 0, -weight, weight)
    effective_scales = (
        projection.scales.float().to(device) / projection.weight_global_divisor
    )
    return weight * effective_scales.repeat_interleave(16, dim=1)


def _quality(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, object]:
    actual_f = actual.float()
    expected_f = expected.float()
    delta = actual_f - expected_f
    denominator = torch.linalg.vector_norm(expected_f).clamp_min(1e-12)
    return {
        "finite": bool(torch.isfinite(actual).all()),
        "max_abs": float(delta.abs().max()),
        "mean_abs": float(delta.abs().mean()),
        "relative_l2": float(torch.linalg.vector_norm(delta) / denominator),
        "cosine": float(
            torch.nn.functional.cosine_similarity(actual_f, expected_f, dim=1)
            .mean()
            .item()
        ),
    }


def _graph_latency_us(call, warmup: int = 20, repeats: int = 100) -> float:
    for _ in range(3):
        call()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for _ in range(warmup):
        graph.replay()
    torch.accelerator.synchronize()
    samples = []
    for _ in range(5):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / repeats)
    return statistics.median(samples)


def _run_projection(
    projection: Projection,
    m: int,
    device: torch.device,
    generator: torch.Generator,
    sweep_qpn2: bool,
) -> dict[str, object]:
    from vllm import _sm70_ops

    qweight = _unpack_codes(projection.packed).to(device)
    effective_scales = (
        (projection.scales.t().float() / projection.weight_global_divisor)
        .half()
        .contiguous()
        .to(device)
    )
    x = (
        torch.randn(
            (m, qweight.shape[0]),
            dtype=torch.float16,
            device=device,
            generator=generator,
        )
        * 0.1
    )
    tm_weight, tm_scales, meta = _sm70_ops.nvfp4_sm70_prepare(
        qweight, effective_scales, 16, False
    )
    out = torch.empty((m, qweight.shape[1]), dtype=torch.float16, device=device)
    _sm70_ops.nvfp4_gemm_sm70_out(
        out,
        x,
        tm_weight,
        tm_scales,
        16,
        int(meta[0]),
        int(meta[1]),
        False,
    )
    weight = _dequantize_weight(projection, device)
    expected = x.float().matmul(weight.t())
    torch.accelerator.synchronize(device)
    result: dict[str, object] = {
        "name": projection.name,
        "m": m,
        "n": int(qweight.shape[1]),
        "k": int(qweight.shape[0]),
        "weight_global_divisor": projection.weight_global_divisor,
        "input_global_divisor": projection.input_global_divisor,
        "quality": _quality(out, expected),
    }
    production_out = out
    production_tm_weight = tm_weight
    production_tm_scales = tm_scales
    production_meta = meta
    if qweight.shape[1] % 16:
        padded_n = (qweight.shape[1] + 15) // 16 * 16
        padded_weight = torch.zeros(
            (qweight.shape[0], padded_n), dtype=qweight.dtype, device=device
        )
        padded_scales = torch.zeros(
            (effective_scales.shape[0], padded_n),
            dtype=effective_scales.dtype,
            device=device,
        )
        padded_weight[:, : qweight.shape[1]].copy_(qweight)
        padded_scales[:, : qweight.shape[1]].copy_(effective_scales)
        padded_tm_weight, padded_tm_scales, padded_meta = _sm70_ops.nvfp4_sm70_prepare(
            padded_weight, padded_scales, 16, False
        )
        padded_out = torch.empty((m, padded_n), dtype=torch.float16, device=device)
        _sm70_ops.nvfp4_gemm_sm70_out(
            padded_out,
            x,
            padded_tm_weight,
            padded_tm_scales,
            16,
            int(padded_meta[0]),
            int(padded_meta[1]),
            False,
        )
        torch.accelerator.synchronize(device)
        result["padded_n"] = padded_n
        result["padded_quality"] = _quality(padded_out[:, : qweight.shape[1]], expected)
        production_out = padded_out
        production_tm_weight = padded_tm_weight
        production_tm_scales = padded_tm_scales
        production_meta = padded_meta

    logical_n = projection.packed.shape[0]
    physical_n = (logical_n + 31) // 32 * 32
    qpn2_packed = projection.packed.to(device)
    qpn2_weight_scales = projection.scales.to(device)
    if physical_n != logical_n:
        physical_packed = torch.zeros(
            (physical_n, qpn2_packed.shape[1]), dtype=torch.uint8, device=device
        )
        physical_weight_scales = torch.zeros(
            (physical_n, qpn2_weight_scales.shape[1]),
            dtype=qpn2_weight_scales.dtype,
            device=device,
        )
        physical_packed[:logical_n].copy_(qpn2_packed)
        physical_weight_scales[:logical_n].copy_(qpn2_weight_scales)
        qpn2_packed = physical_packed
        qpn2_weight_scales = physical_weight_scales
    qpn2_codes, qpn2_scales = _sm70_ops.nvfp4_qpn2_prepare_sm70(
        qpn2_packed, qpn2_weight_scales
    )
    qpn2_out = torch.empty((m, physical_n), dtype=torch.float16, device=device)
    split_k, accumulator_chains = QPN2_CONFIGS[(x.shape[1], physical_n)]

    def run_qpn2() -> None:
        _sm70_ops.nvfp4_qpn2_gemm_sm70_out(
            qpn2_out,
            x,
            qpn2_codes,
            qpn2_scales,
            1.0 / projection.weight_global_divisor,
            split_k,
            accumulator_chains,
        )

    def run_turbomind() -> None:
        _sm70_ops.nvfp4_gemm_sm70_out(
            production_out,
            x,
            production_tm_weight,
            production_tm_scales,
            16,
            int(production_meta[0]),
            int(production_meta[1]),
            False,
        )

    result["qpn2_physical_n"] = physical_n
    turbomind_graph_us = _graph_latency_us(run_turbomind)
    result["turbomind_graph_us"] = turbomind_graph_us
    if m > 8:
        # Exercise the production large-M implementation independently of its
        # tuned M threshold. The branch is shape-invariant once selected, so a
        # small real-weight M keeps this oracle cheap while covering every
        # QUASAR attention/GDN/MLP projection.
        prefill_out = torch.empty_like(qpn2_out)

        def run_qpn2_prefill() -> None:
            _sm70_ops.nvfp4_qpn2_prefill_dispatch_sm70_out(
                prefill_out,
                x,
                qpn2_codes,
                qpn2_scales,
                1.0 / projection.weight_global_divisor,
                split_k,
                accumulator_chains,
                production_tm_weight,
                production_tm_scales,
                16,
                int(production_meta[0]),
                int(production_meta[1]),
                False,
                9,
            )

        run_qpn2_prefill()
        torch.accelerator.synchronize(device)
        result["qpn2_prefill_quality"] = _quality(prefill_out[:, :logical_n], expected)
        result["qpn2_prefill_vs_turbomind"] = _quality(
            prefill_out[:, :logical_n], production_out[:, :logical_n]
        )
        qpn2_prefill_graph_us = _graph_latency_us(run_qpn2_prefill)
        result["qpn2_prefill_graph_us"] = qpn2_prefill_graph_us
        result["qpn2_prefill_speedup"] = turbomind_graph_us / qpn2_prefill_graph_us
        torch.accelerator.empty_cache()
        return result

    run_qpn2()
    torch.accelerator.synchronize(device)
    result["qpn2_quality"] = _quality(qpn2_out[:, :logical_n], expected)
    result["qpn2_vs_turbomind"] = _quality(
        qpn2_out[:, :logical_n], production_out[:, :logical_n]
    )
    if projection.name == "mlp_gate_up":
        qpn2_gated_out = torch.empty(
            (m, physical_n // 2), dtype=torch.float16, device=device
        )
        _sm70_ops.nvfp4_qpn2_gated_sm70_out(
            qpn2_gated_out,
            x,
            qpn2_codes,
            qpn2_scales,
            1.0 / projection.weight_global_divisor,
            split_k,
            accumulator_chains,
        )
        torch.accelerator.synchronize(device)
        logical_half = logical_n // 2
        expected_gated = (
            torch.nn.functional.silu(expected[:, :logical_half])
            * (expected[:, logical_half:logical_n])
        )
        turbomind_gated = (
            torch.nn.functional.silu(production_out[:, :logical_half].float())
            * production_out[:, logical_half:logical_n].float()
        )
        result["qpn2_gated_quality"] = _quality(
            qpn2_gated_out[:, :logical_half], expected_gated
        )
        result["qpn2_gated_vs_turbomind"] = _quality(
            qpn2_gated_out[:, :logical_half], turbomind_gated
        )
    qpn2_graph_us = _graph_latency_us(run_qpn2)
    result["qpn2_graph_us"] = qpn2_graph_us
    result["qpn2_speedup"] = turbomind_graph_us / qpn2_graph_us
    if sweep_qpn2:
        candidates = []
        for candidate_split_k in (8, 16, 32):
            if (x.shape[1] // 16) % candidate_split_k:
                continue
            for candidate_accumulator_chains in (1, 2):

                def run_candidate(
                    split_k: int = candidate_split_k,
                    accumulator_chains: int = candidate_accumulator_chains,
                ) -> None:
                    _sm70_ops.nvfp4_qpn2_gemm_sm70_out(
                        qpn2_out,
                        x,
                        qpn2_codes,
                        qpn2_scales,
                        1.0 / projection.weight_global_divisor,
                        split_k,
                        accumulator_chains,
                    )

                run_candidate()
                torch.accelerator.synchronize(device)
                candidate = {
                    "split_k": candidate_split_k,
                    "accumulator_chains": candidate_accumulator_chains,
                    "graph_us": _graph_latency_us(run_candidate),
                    "quality": _quality(qpn2_out[:, :logical_n], expected),
                    "vs_turbomind": _quality(
                        qpn2_out[:, :logical_n], production_out[:, :logical_n]
                    ),
                }
                if projection.name == "mlp_gate_up" and candidate_split_k <= 16:
                    gated_out = torch.empty(
                        (m, physical_n // 2), dtype=torch.float16, device=device
                    )

                    def run_gated_candidate(
                        split_k: int = candidate_split_k,
                        accumulator_chains: int = candidate_accumulator_chains,
                        output: torch.Tensor = gated_out,
                    ) -> None:
                        _sm70_ops.nvfp4_qpn2_gated_sm70_out(
                            output,
                            x,
                            qpn2_codes,
                            qpn2_scales,
                            1.0 / projection.weight_global_divisor,
                            split_k,
                            accumulator_chains,
                        )

                    candidate["gated_graph_us"] = _graph_latency_us(run_gated_candidate)
                candidates.append(candidate)
        result["qpn2_sweep"] = candidates
    torch.accelerator.empty_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--m", type=int, nargs="+", default=(1, 8))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sweep-qpn2", action="store_true")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    device = torch.device(args.device)
    if torch.cuda.get_device_capability(device) != (7, 0):
        raise RuntimeError("This audit requires an exact SM70 GPU")
    generator = torch.Generator(device=device).manual_seed(20260902)
    projections = _load_projections(args.model, args.tp_rank, args.tp_size)
    rows = [
        _run_projection(projection, m, device, generator, args.sweep_qpn2)
        for projection in projections
        for m in args.m
    ]
    payload = {
        "model": str(args.model),
        "tp_rank": args.tp_rank,
        "tp_size": args.tp_size,
        "device": torch.cuda.get_device_name(device),
        "rows": rows,
    }
    encoded = json.dumps(payload, indent=2)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(encoded + "\n")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
