# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Race QPN2 against the current SM70 NVFP4 path on real Qwen3.8 weights.

This is a deliberately narrow verifier microbenchmark.  It loads the TP-local
gate/up and down projection shards from one native NVFP4 layer, prepares both
the current TurboMind layout and the QPN2 fragment layout, and measures a
CUDA-graph replay at verifier batch shapes through M=64. Weight loading and
preparation are outside the timed region.

QPN2 is compiled from an explicitly supplied source file so this benchmark can
evaluate a pinned external implementation before it is admitted to the vLLM
build.  The result records the source digest and numerical delta against the
current production operator.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import statistics
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from torch.utils.cpp_extension import load


@dataclass(frozen=True)
class Projection:
    name: str
    packed: torch.Tensor
    scales: torch.Tensor
    inverse_global_scale: float
    gated_silu: bool
    calls_per_round: int


QPN2_CONFIGS = {
    # (K, N): (split K, independent accumulator chains)
    (1536, 5120): (8, 2),
    (5120, 3584): (16, 2),
    (5120, 4128): (16, 2),
    (5120, 8704): (8, 2),
    (4352, 5120): (16, 2),
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_projection_shards(
    model: Path,
    layer_index: int,
    tp_rank: int,
    tp_size: int,
) -> tuple[Projection, Projection]:
    path = model / "model.safetensors"
    config = json.loads((model / "config.json").read_text())
    text_config = config.get("text_config", config)
    num_hidden_layers = int(text_config["num_hidden_layers"])
    prefix = f"model.language_model.layers.{layer_index}.mlp"
    intermediate_size = 17408
    hidden_size = 5120
    if intermediate_size % tp_size or hidden_size % tp_size:
        raise ValueError("model dimensions must divide the tensor-parallel size")
    intermediate_per_rank = intermediate_size // tp_size
    packed_k_per_rank = intermediate_per_rank // 2
    scale_k_per_rank = intermediate_per_rank // 16
    n_start = tp_rank * intermediate_per_rank
    n_end = n_start + intermediate_per_rank
    packed_k_start = tp_rank * packed_k_per_rank
    packed_k_end = packed_k_start + packed_k_per_rank
    scale_k_start = tp_rank * scale_k_per_rank
    scale_k_end = scale_k_start + scale_k_per_rank

    with safe_open(path, framework="pt", device="cpu") as tensors:
        gate_packed = tensors.get_slice(f"{prefix}.gate_proj.weight_packed")[
            n_start:n_end, :
        ]
        up_packed = tensors.get_slice(f"{prefix}.up_proj.weight_packed")[
            n_start:n_end, :
        ]
        gate_scales = tensors.get_slice(f"{prefix}.gate_proj.weight_scale")[
            n_start:n_end, :
        ]
        up_scales = tensors.get_slice(f"{prefix}.up_proj.weight_scale")[
            n_start:n_end, :
        ]
        gate_global = tensors.get_tensor(f"{prefix}.gate_proj.weight_global_scale")
        up_global = tensors.get_tensor(f"{prefix}.up_proj.weight_global_scale")

        down_packed = tensors.get_slice(f"{prefix}.down_proj.weight_packed")[
            :, packed_k_start:packed_k_end
        ]
        down_scales = tensors.get_slice(f"{prefix}.down_proj.weight_scale")[
            :, scale_k_start:scale_k_end
        ]
        down_global = tensors.get_tensor(f"{prefix}.down_proj.weight_global_scale")

    gate_up = Projection(
        name="gate_up_proj",
        packed=torch.cat((gate_packed, up_packed), dim=0).contiguous(),
        scales=torch.cat((gate_scales, up_scales), dim=0).contiguous(),
        inverse_global_scale=float(
            1.0 / torch.cat((gate_global, up_global)).max().float()
        ),
        gated_silu=True,
        calls_per_round=num_hidden_layers,
    )
    down = Projection(
        name="down_proj",
        packed=down_packed.contiguous(),
        scales=down_scales.contiguous(),
        inverse_global_scale=float(1.0 / down_global.max().float()),
        gated_silu=False,
        calls_per_round=num_hidden_layers,
    )
    return gate_up, down


def _synthetic_projections(model: Path) -> tuple[Projection, ...]:
    """Build deterministic tensors for target shapes absent from local weights."""
    config = json.loads((model / "config.json").read_text())
    text_config = config.get("text_config", config)
    layer_types = text_config["layer_types"]
    num_attention_layers = layer_types.count("full_attention")
    num_linear_attention_layers = layer_types.count("linear_attention")
    shapes = (
        ("attention_out_proj", 1536, 5120, len(layer_types)),
        ("attention_qkv_proj", 5120, 3584, num_attention_layers),
        ("gdn_in_proj_qkvz", 5120, 4128, num_linear_attention_layers),
    )
    projections = []
    for name, k, n, calls_per_round in shapes:
        packed = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8)
        scales = (torch.rand((n, k // 16)) * 0.5 + 0.25).to(torch.float8_e4m3fn)
        projections.append(
            Projection(
                name=name,
                packed=packed,
                scales=scales,
                inverse_global_scale=0.01,
                gated_silu=False,
                calls_per_round=calls_per_round,
            )
        )
    return tuple(projections)


def _unpack_nibbles(weight_packed: torch.Tensor) -> torch.Tensor:
    """Convert checkpoint [N,K/2] bytes to TurboMind's [K,N] codes."""
    return (
        torch.stack((weight_packed & 15, weight_packed >> 4), dim=-1)
        .flatten(start_dim=-2)
        .t()
        .contiguous()
    )


def _qpn2_prepack(
    codes: torch.Tensor, scales: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Byte-preserving permutation into v100-skinny's QPN2 layout."""
    n, k2 = codes.shape
    k = k2 * 2
    if n % 32 or k % 64:
        raise ValueError(f"QPN2 requires N%32=0 and K%64=0, got N={n}, K={k}")
    device = codes.device
    tiles, groups = n // 32, k // 16
    lane = torch.arange(32, device=device)
    column = ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) > 0).long() * 4
    k_order = torch.tensor(
        [0, 2, 4, 6, 1, 3, 5, 7, 8, 10, 12, 14, 9, 11, 13, 15],
        device=device,
    )
    nibbles = torch.stack((codes & 0xF, codes >> 4), dim=-1).view(n, k)
    group = torch.arange(groups, device=device)
    k_index = group.view(groups, 1) * 16 + k_order.view(1, 16)
    q_codes = torch.empty((tiles, groups, 32, 8), dtype=torch.uint8, device=device)
    q_scales = torch.empty((tiles, groups, 32), dtype=torch.uint8, device=device)
    # Bound the int64 gather intermediates for the gate/up matrix.
    tile_chunk = max(1, 36864 // groups)
    for tile_start in range(0, tiles, tile_chunk):
        tile_end = min(tile_start + tile_chunk, tiles)
        tile_count = tile_end - tile_start
        n_column = torch.arange(tile_start, tile_end, device=device).view(
            tile_count, 1
        ) * 32 + column.view(1, 32)
        nibble_block = nibbles[
            n_column.view(tile_count, 1, 32, 1).expand(tile_count, groups, 32, 16),
            k_index.view(1, groups, 1, 16).expand(tile_count, groups, 32, 16),
        ]
        q_codes[tile_start:tile_end] = nibble_block[..., 0::2] | (
            nibble_block[..., 1::2] << 4
        )
        q_scales[tile_start:tile_end] = scales[
            n_column.view(tile_count, 1, 32).expand(tile_count, groups, 32),
            group.view(1, groups, 1).expand(tile_count, groups, 32),
        ]
    return q_codes.view(-1).contiguous(), q_scales.view(-1).contiguous()


def _capture(call: Callable[[], Any]) -> tuple[torch.cuda.CUDAGraph, Any]:
    for _ in range(10):
        call()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = call()
    torch.accelerator.synchronize()
    return graph, output


def _time_graph(
    graph: torch.cuda.CUDAGraph, warmup: int, iterations: int, trials: int
) -> dict[str, float | list[float]]:
    for _ in range(warmup):
        graph.replay()
    torch.accelerator.synchronize()
    samples = []
    for _ in range(trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)) * 1000.0 / iterations)
    return {
        "samples_us": samples,
        "median_us": statistics.median(samples),
        "min_us": min(samples),
        "max_us": max(samples),
    }


def _quality(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, object]:
    actual_f = actual.float()
    expected_f = expected.float()
    delta = actual_f - expected_f
    denominator = torch.linalg.vector_norm(expected_f).clamp_min(1e-12)
    return {
        "finite": bool(torch.isfinite(actual).all()),
        "exact": bool(torch.equal(actual, expected)),
        "different_fraction": float((actual != expected).float().mean()),
        "max_abs": float(delta.abs().max()),
        "mean_abs": float(delta.abs().mean()),
        "relative_l2": float(torch.linalg.vector_norm(delta) / denominator),
        "cosine": float(
            torch.nn.functional.cosine_similarity(actual_f, expected_f, dim=1)
            .mean()
            .item()
        ),
    }


def _fp32_reference(
    projection: Projection, x: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """Evaluate the checkpoint's native E2M1 codes with FP32 accumulation."""
    packed = projection.packed.to(device)
    nibbles = torch.stack((packed & 0xF, packed >> 4), dim=-1).flatten(start_dim=-2)
    magnitudes = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float32,
        device=device,
    )
    weights = magnitudes[(nibbles & 7).long()]
    weights = torch.where((nibbles & 8) != 0, -weights, weights)
    weights.mul_(
        projection.scales.float()
        .to(device)
        .repeat_interleave(16, dim=1)
        .mul_(projection.inverse_global_scale)
    )
    raw = x.float().matmul(weights.t())
    if not projection.gated_silu:
        return raw
    gate, up = raw.chunk(2, dim=1)
    return torch.nn.functional.silu(gate).mul_(up)


def _run_projection(
    projection: Projection,
    extension: Any | None,
    production: bool,
    production_source_candidate: bool,
    m: int,
    device: torch.device,
    warmup: int,
    iterations: int,
    trials: int,
    split_k_override: int | None,
    accumulator_chains_override: int | None,
) -> dict[str, object]:
    from vllm import _sm70_ops as sm70_ops

    packed = projection.packed.to(device)
    qweight = _unpack_nibbles(projection.packed).to(device)
    effective_scales = (
        projection.scales.t()
        .float()
        .mul(projection.inverse_global_scale)
        .half()
        .contiguous()
        .to(device)
    )
    k, n = qweight.shape
    split_k, accumulator_chains = QPN2_CONFIGS[(k, n)]
    if split_k_override is not None:
        split_k = split_k_override
    if accumulator_chains_override is not None:
        accumulator_chains = accumulator_chains_override
    x = torch.randn((m, k), dtype=torch.float16, device=device) * 0.1

    tm_weight, tm_scales, meta = sm70_ops.nvfp4_sm70_prepare(
        qweight, effective_scales, 16, False
    )
    k_ld, q_ld = (int(value.item()) for value in meta[:2])
    tm_raw = torch.empty((m, n), dtype=torch.float16, device=device)
    final_n = n // 2 if projection.gated_silu else n
    tm_final = (
        torch.empty((m, final_n), dtype=torch.float16, device=device)
        if projection.gated_silu
        else tm_raw
    )

    def run_tm() -> torch.Tensor:
        sm70_ops.nvfp4_gemm_sm70_out(
            tm_raw, x, tm_weight, tm_scales, 16, k_ld, q_ld, False
        )
        if projection.gated_silu:
            torch.ops._C.silu_and_mul(tm_final, tm_raw)
        return tm_final

    qpn_final = torch.empty((m, final_n), dtype=torch.float16, device=device)
    run_qpn2_chunked = None
    if production:
        qpn_codes, qpn_scales = sm70_ops.nvfp4_qpn2_prepare_sm70(
            packed, projection.scales.to(device)
        )

        def run_qpn2() -> torch.Tensor:
            if projection.gated_silu:
                sm70_ops.nvfp4_qpn2_gated_sm70_out(
                    qpn_final,
                    x,
                    qpn_codes,
                    qpn_scales,
                    projection.inverse_global_scale,
                    split_k,
                    accumulator_chains,
                )
            else:
                sm70_ops.nvfp4_qpn2_gemm_sm70_out(
                    qpn_final,
                    x,
                    qpn_codes,
                    qpn_scales,
                    projection.inverse_global_scale,
                    split_k,
                    accumulator_chains,
                )
            return qpn_final

    elif production_source_candidate:
        qpn_codes, qpn_scales = torch.ops._qpn2_candidate.prepare(
            packed, projection.scales.to(device)
        )

        def run_candidate(
            candidate_out: torch.Tensor, candidate_input: torch.Tensor
        ) -> torch.Tensor:
            if projection.gated_silu:
                torch.ops._qpn2_candidate.gated(
                    candidate_out,
                    candidate_input,
                    qpn_codes,
                    qpn_scales,
                    projection.inverse_global_scale,
                    split_k,
                    accumulator_chains,
                )
            else:
                torch.ops._qpn2_candidate.gemm(
                    candidate_out,
                    candidate_input,
                    qpn_codes,
                    qpn_scales,
                    projection.inverse_global_scale,
                    split_k,
                    accumulator_chains,
                )
            return candidate_out

        def run_qpn2() -> torch.Tensor:
            return run_candidate(qpn_final, x)

        if m > 8:
            qpn_chunked = torch.empty_like(qpn_final)

            def run_qpn2_chunked() -> torch.Tensor:
                for row in range(0, m, 8):
                    rows = min(8, m - row)
                    run_candidate(
                        qpn_chunked.narrow(0, row, rows),
                        x.narrow(0, row, rows),
                    )
                return qpn_chunked

    else:
        if extension is None:
            raise AssertionError("external QPN2 extension was not loaded")
        scale_codes = projection.scales.view(torch.uint8).to(device)
        qpn_codes, qpn_scales = _qpn2_prepack(packed, scale_codes)

        def run_qpn2() -> torch.Tensor:
            qpn_raw = extension.gemm_qpn2(
                x,
                qpn_codes,
                qpn_scales,
                projection.inverse_global_scale,
                n,
                split_k,
                accumulator_chains,
            )
            if projection.gated_silu:
                torch.ops._C.silu_and_mul(qpn_final, qpn_raw)
                return qpn_final
            return qpn_raw

    tm_graph, tm_output = _capture(run_tm)
    qpn_graph, qpn_output = _capture(run_qpn2)
    tm_timing = _time_graph(tm_graph, warmup, iterations, trials)
    qpn_timing = _time_graph(qpn_graph, warmup, iterations, trials)
    chunked_timing = None
    qpn_chunked_output = None
    if run_qpn2_chunked is not None:
        chunked_graph, qpn_chunked_output = _capture(run_qpn2_chunked)
        chunked_timing = _time_graph(chunked_graph, warmup, iterations, trials)
    tm_graph.replay()
    qpn_graph.replay()
    torch.accelerator.synchronize()
    reference = _fp32_reference(projection, x, device)
    batch_invariance = None
    if qpn_chunked_output is not None:
        run_qpn2_chunked()
        torch.accelerator.synchronize()
        batch_invariance = _quality(qpn_output, qpn_chunked_output)
    saved_us = float(tm_timing["median_us"]) - float(qpn_timing["median_us"])
    native_saved_us = (
        float(chunked_timing["median_us"]) - float(qpn_timing["median_us"])
        if chunked_timing is not None
        else None
    )
    return {
        "name": projection.name,
        "m": m,
        "n": n,
        "k": k,
        "gated_silu": projection.gated_silu,
        "calls_per_round": projection.calls_per_round,
        "qpn2_config": {
            "split_k": split_k,
            "accumulator_chains": accumulator_chains,
        },
        "turbomind": tm_timing,
        "qpn2": qpn_timing,
        "qpn2_chunked_m8": chunked_timing,
        "speedup": float(tm_timing["median_us"]) / float(qpn_timing["median_us"]),
        "saved_ms_per_round": saved_us * projection.calls_per_round / 1000.0,
        "native_vs_chunked_speedup": (
            float(chunked_timing["median_us"]) / float(qpn_timing["median_us"])
            if chunked_timing is not None
            else None
        ),
        "native_saved_ms_per_round": (
            native_saved_us * projection.calls_per_round / 1000.0
            if native_saved_us is not None
            else None
        ),
        "quality_vs_turbomind": _quality(qpn_output, tm_output),
        "turbomind_quality_vs_fp32": _quality(tm_output, reference),
        "qpn2_quality_vs_fp32": _quality(qpn_output, reference),
        "qpn2_batch_invariance": batch_invariance,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=Path, default=Path("/home/ymzx/models/Qwen3.8-27B-NVFP4")
    )
    parser.add_argument("--layer", type=int, default=55)
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--m", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--split-k-override", type=int, choices=(8, 16, 32))
    parser.add_argument("--accumulator-chains-override", type=int, choices=(1, 2))
    parser.add_argument(
        "--synthetic-shapes",
        action="store_true",
        help="Race deterministic tensors for non-MLP Qwen3.8 projection shapes.",
    )
    parser.add_argument("--qpn2-source", type=Path)
    parser.add_argument(
        "--production-source-candidate",
        type=Path,
        help="Compile this repository's QPN2 source under a private namespace.",
    )
    parser.add_argument(
        "--production-library",
        type=Path,
        help="Load a complete candidate vllm._C containing production QPN2 ops.",
    )
    parser.add_argument("--extension-name", default="v100_skinny_qpn2_micro_v1")
    parser.add_argument("--json-out", type=Path, required=True)
    return parser.parse_args()


def _load_production_library(path: Path) -> None:
    stable_path = path.with_name("_C_stable_libtorch.abi3.so")
    if stable_path.exists():
        torch.ops.load_library(stable_path)
    spec = importlib.util.spec_from_file_location("vllm._C", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load production library: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["vllm._C"] = module
    spec.loader.exec_module(module)


def main() -> int:
    args = _parse_args()
    device = torch.device(args.device)
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(device) != (
        7,
        0,
    ):
        raise RuntimeError("benchmark requires an exact SM70 CUDA device")
    if not 1 <= args.m <= 64:
        raise ValueError("QPN2 benchmark supports M in [1, 64]")
    if args.production_library is not None:
        _load_production_library(args.production_library)
    if args.qpn2_source is not None and args.production_source_candidate is not None:
        raise ValueError("QPN2 source modes are mutually exclusive")
    production_source_candidate = args.production_source_candidate is not None
    extension = None
    if production_source_candidate:
        load(
            name=args.extension_name,
            sources=[str(args.production_source_candidate)],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "-lineinfo",
                "-gencode=arch=compute_70,code=sm_70",
                "-DVLLM_NVFP4_QPN2_STANDALONE",
                "-DVLLM_NVFP4_QPN2_BENCHMARK_CANDIDATE",
            ],
            verbose=False,
            is_python_module=False,
        )
    elif args.qpn2_source is not None:
        extension = load(
            name=args.extension_name,
            sources=[str(args.qpn2_source)],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "-lineinfo",
                "-gencode=arch=compute_70,code=sm_70",
            ],
            verbose=False,
        )
    elif args.production_library is None:
        raise ValueError(
            "one of --qpn2-source, --production-source-candidate, or "
            "--production-library is required"
        )
    production = not production_source_candidate and args.qpn2_source is None
    torch.manual_seed(20260824)
    projections = (
        _synthetic_projections(args.model)
        if args.synthetic_shapes
        else _load_projection_shards(args.model, args.layer, args.tp_rank, args.tp_size)
    )
    rows = [
        _run_projection(
            projection,
            extension,
            production,
            production_source_candidate,
            args.m,
            device,
            args.warmup,
            args.iterations,
            args.trials,
            args.split_k_override,
            args.accumulator_chains_override,
        )
        for projection in projections
    ]
    payload = {
        "device": torch.cuda.get_device_name(device),
        "capability": list(torch.cuda.get_device_capability(device)),
        "torch": torch.__version__,
        "model": str(args.model),
        "model_config_sha256": _sha256_file(args.model / "config.json"),
        "layer": args.layer,
        "tp_rank": args.tp_rank,
        "tp_size": args.tp_size,
        "synthetic_shapes": args.synthetic_shapes,
        "qpn2_source": str(args.qpn2_source) if args.qpn2_source else None,
        "qpn2_source_sha256": (
            _sha256_file(args.qpn2_source) if args.qpn2_source else None
        ),
        "production_source_candidate": (
            str(args.production_source_candidate)
            if args.production_source_candidate
            else None
        ),
        "production_source_candidate_sha256": (
            _sha256_file(args.production_source_candidate)
            if args.production_source_candidate
            else None
        ),
        "production_library": (
            str(args.production_library) if args.production_library else None
        ),
        "production_library_sha256": (
            _sha256_file(args.production_library) if args.production_library else None
        ),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "trials": args.trials,
        "split_k_override": args.split_k_override,
        "accumulator_chains_override": args.accumulator_chains_override,
        "rows": rows,
        "projected_total_saved_ms_per_round": sum(
            float(row["saved_ms_per_round"]) for row in rows
        ),
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
