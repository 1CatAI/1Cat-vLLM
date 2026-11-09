# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Screen row-major FP16 M=5 verifier GEMV on SM70.

The shape mix follows the Qwen3.8-Flash-Next TP4 target verifier.  This is a
microbenchmark only: a candidate must still pass full-model token, acceptance,
quality, and memory gates before it can be wired into runtime dispatch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from vllm.triton_utils import tl, triton

_M = 5

# Runtime-local (N, K) shapes and calls per target verification. Counts come
# from the 48-layer model structure; the accepted Nsight trace is the timing
# authority because graph overlap makes this weighted sum only a cost model.
_QWEN38_TP4_M5_SHAPES: tuple[tuple[str, int, int, int], ...] = (
    ("hc_down_inject", 336, 10240, 96),
    ("hc_up", 10240, 320, 96),
    ("gdn_qkvz", 4096, 2560, 36),
    ("gdn_ba", 24, 2560, 36),
    ("qsa_qkvg", 3584, 2560, 12),
    ("qsa_index_qk", 640, 2560, 12),
    ("attention_out", 2560, 1536, 48),
    ("shared_gate_up", 320, 2560, 48),
    ("shared_down", 2560, 160, 48),
    ("moe_router", 512, 2560, 48),
    ("ple_key", 10240, 2560, 1),
    ("ple_value", 2560, 2560, 1),
    ("lm_head", 62080, 2560, 1),
)

_CONFIGS: tuple[tuple[int, int], ...] = (
    (256, 4),
    (512, 4),
    (512, 8),
    (1024, 4),
    (1024, 8),
)

_QPN8_CONFIGS: dict[str, tuple[int, int, bool, int | None]] = {
    "hc_down_inject": (32, 1, False, 384),
    "hc_up": (4, 2, False, None),
    "gdn_qkvz": (16, 1, False, None),
    "qsa_qkvg": (16, 1, False, None),
    "attention_out": (12, 2, False, None),
}


@triton.jit
def _sm70_fp16_m5_batched_gemv_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Compute five output rows while loading each weight row once."""
    output_col = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    acc0 = 0.0
    acc1 = 0.0
    acc2 = 0.0
    acc3 = 0.0
    acc4 = 0.0
    for block_start in tl.static_range(0, K, BLOCK_K):
        k_offsets = block_start + offsets
        mask = k_offsets < K
        weight = tl.load(
            weight_ptr + output_col * K + k_offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        x0 = tl.load(x_ptr + k_offsets, mask=mask, other=0.0).to(tl.float32)
        x1 = tl.load(x_ptr + K + k_offsets, mask=mask, other=0.0).to(tl.float32)
        x2 = tl.load(x_ptr + 2 * K + k_offsets, mask=mask, other=0.0).to(tl.float32)
        x3 = tl.load(x_ptr + 3 * K + k_offsets, mask=mask, other=0.0).to(tl.float32)
        x4 = tl.load(x_ptr + 4 * K + k_offsets, mask=mask, other=0.0).to(tl.float32)
        acc0 += tl.sum(x0 * weight, axis=0)
        acc1 += tl.sum(x1 * weight, axis=0)
        acc2 += tl.sum(x2 * weight, axis=0)
        acc3 += tl.sum(x3 * weight, axis=0)
        acc4 += tl.sum(x4 * weight, axis=0)

    tl.store(out_ptr + output_col, acc0)
    tl.store(out_ptr + N + output_col, acc1)
    tl.store(out_ptr + 2 * N + output_col, acc2)
    tl.store(out_ptr + 3 * N + output_col, acc3)
    tl.store(out_ptr + 4 * N + output_col, acc4)


def _measure_graph_us(
    launch: Callable[[], None], *, warmups: int, repeats: int
) -> float:
    for _ in range(warmups):
        launch()
    torch.accelerator.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    for _ in range(warmups):
        graph.replay()
    torch.accelerator.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / repeats


def _output_metrics(output: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    error = output.float() - reference.float()
    reference_norm = reference.float().norm()
    return {
        "exact_equal": torch.equal(output, reference),
        "max_abs_error": error.abs().max().item(),
        "relative_l2_error": (error.norm() / reference_norm).item(),
        "cosine_similarity": torch.nn.functional.cosine_similarity(
            output.float().flatten(), reference.float().flatten(), dim=0
        ).item(),
        "output_sha256": hashlib.sha256(
            output.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
        ).hexdigest(),
    }


def _prepare_online_qpn8_weight(
    weight: torch.Tensor, padded_n: int
) -> tuple[torch.Tensor, torch.Tensor]:
    from vllm import _sm70_ops as sm70_ops

    n, k = weight.shape
    weight_for_quant = weight
    if padded_n != n:
        weight_for_quant = torch.cat(
            (weight, weight.new_zeros((padded_n - n, k))), dim=0
        )
    weight_f32 = weight_for_quant.float()
    channel_scales = weight_f32.abs().amax(dim=1, keepdim=True).div_(448.0)
    channel_scales = torch.where(
        channel_scales == 0,
        torch.ones_like(channel_scales),
        channel_scales,
    )
    qweight = (weight_f32 / channel_scales).to(torch.float8_e4m3fn)
    return sm70_ops.fp8_qpn8_prepare_sm70(
        qweight.contiguous(), channel_scales.contiguous()
    )


def _benchmark_shape(
    name: str,
    n: int,
    k: int,
    count: int,
    *,
    warmups: int,
    repeats: int,
    seed: int,
    include_triton: bool,
    include_lossy_qpn8: bool,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    x = torch.randn((_M, k), dtype=torch.float16, device="cuda")
    weight = torch.randn((n, k), dtype=torch.float16, device="cuda")
    reference = torch.empty((_M, n), dtype=torch.float16, device="cuda")
    output = torch.empty_like(reference)
    weight_t = weight.t()

    def torch_launch() -> None:
        torch.mm(x, weight_t, out=reference)

    baseline_us = _measure_graph_us(
        torch_launch,
        warmups=warmups,
        repeats=repeats,
    )
    torch_launch()
    torch.accelerator.synchronize()

    candidates: list[dict[str, Any]] = []
    for block_k, num_warps in _CONFIGS if include_triton else ():

        def candidate_launch(
            block_k: int = block_k,
            num_warps: int = num_warps,
        ) -> None:
            _sm70_fp16_m5_batched_gemv_kernel[(n,)](
                x,
                weight,
                output,
                N=n,
                K=k,
                BLOCK_K=block_k,
                num_warps=num_warps,
                num_stages=1,
            )

        try:
            latency_us = _measure_graph_us(
                candidate_launch,
                warmups=warmups,
                repeats=repeats,
            )
            candidate_launch()
            torch.accelerator.synchronize()
            row = {
                "kernel": "triton_m5_batched_gemv",
                "runtime_cost_included": True,
                "block_k": block_k,
                "num_warps": num_warps,
                "latency_us": latency_us,
                "speedup": baseline_us / latency_us,
                **_output_metrics(output, reference),
            }
        except Exception as err:
            row = {
                "kernel": "triton_m5_batched_gemv",
                "runtime_cost_included": True,
                "block_k": block_k,
                "num_warps": num_warps,
                "error": f"{type(err).__name__}: {err}",
            }
        candidates.append(row)

    for padded_m in (8, 16):
        if padded_m < _M:
            continue
        padded_x = torch.zeros((padded_m, k), dtype=x.dtype, device=x.device)
        padded_x[:_M].copy_(x)
        padded_output = torch.empty((padded_m, n), dtype=x.dtype, device=x.device)

        def padded_launch(
            padded_x: torch.Tensor = padded_x,
            padded_output: torch.Tensor = padded_output,
        ) -> None:
            torch.mm(padded_x, weight_t, out=padded_output)

        try:
            latency_us = _measure_graph_us(
                padded_launch,
                warmups=warmups,
                repeats=repeats,
            )
            padded_launch()
            torch.accelerator.synchronize()
            row = {
                "kernel": f"torch_mm_padded_m{padded_m}",
                "runtime_cost_included": False,
                "block_k": None,
                "num_warps": None,
                "latency_us": latency_us,
                "speedup": baseline_us / latency_us,
                **_output_metrics(padded_output[:_M], reference),
            }
        except Exception as err:
            row = {
                "kernel": f"torch_mm_padded_m{padded_m}",
                "runtime_cost_included": False,
                "block_k": None,
                "num_warps": None,
                "error": f"{type(err).__name__}: {err}",
            }
        candidates.append(row)

        def padded_copy_launch(
            padded_x: torch.Tensor = padded_x,
            padded_output: torch.Tensor = padded_output,
        ) -> None:
            padded_x[:_M].copy_(x)
            torch.mm(padded_x, weight_t, out=padded_output)

        try:
            latency_us = _measure_graph_us(
                padded_copy_launch,
                warmups=warmups,
                repeats=repeats,
            )
            padded_copy_launch()
            torch.accelerator.synchronize()
            row = {
                "kernel": f"torch_mm_padded_m{padded_m}_with_copy",
                "runtime_cost_included": True,
                "block_k": None,
                "num_warps": None,
                "latency_us": latency_us,
                "speedup": baseline_us / latency_us,
                **_output_metrics(padded_output[:_M], reference),
            }
        except Exception as err:
            row = {
                "kernel": f"torch_mm_padded_m{padded_m}_with_copy",
                "runtime_cost_included": True,
                "block_k": None,
                "num_warps": None,
                "error": f"{type(err).__name__}: {err}",
            }
        candidates.append(row)

    try:
        from vllm import _sm70_ops as sm70_ops

        prepared_weight, meta = sm70_ops.sm70_f16_prepare(weight)
        k_ld = int(meta[0].item())
        turbomind_output = torch.empty_like(reference)

        def turbomind_launch() -> None:
            sm70_ops.sm70_f16_gemm_out(
                turbomind_output,
                x,
                prepared_weight,
                k_ld,
                False,
            )

        latency_us = _measure_graph_us(
            turbomind_launch,
            warmups=warmups,
            repeats=repeats,
        )
        turbomind_launch()
        torch.accelerator.synchronize()
        row = {
            "kernel": "turbomind_f16_tensorcore",
            "runtime_cost_included": True,
            "persistent_weight_mib": (
                prepared_weight.numel() * prepared_weight.element_size() / 2**20
            ),
            "block_k": None,
            "num_warps": None,
            "latency_us": latency_us,
            "speedup": baseline_us / latency_us,
            **_output_metrics(turbomind_output, reference),
        }
    except Exception as err:
        row = {
            "kernel": "turbomind_f16_tensorcore",
            "runtime_cost_included": True,
            "block_k": None,
            "num_warps": None,
            "error": f"{type(err).__name__}: {err}",
        }
    candidates.append(row)

    qpn8_config = _QPN8_CONFIGS.get(name)
    if qpn8_config is not None and include_lossy_qpn8:
        split_k, nacc, prefetch, padded_n_override = qpn8_config
        padded_n = padded_n_override or n
        try:
            from vllm import _sm70_ops as sm70_ops

            codes, scales = _prepare_online_qpn8_weight(weight, padded_n)
            workspace = torch.empty(
                (4096 * 2560,), dtype=torch.float16, device=weight.device
            )
            qpn8_output = torch.empty(
                (_M, padded_n), dtype=torch.float16, device=weight.device
            )

            def qpn8_launch() -> None:
                sm70_ops.fp8_qpn8_dispatch_sm70_out(
                    qpn8_output,
                    workspace.data_ptr(),
                    x,
                    codes,
                    scales,
                    split_k,
                    nacc,
                    prefetch,
                    False,
                )

            latency_us = _measure_graph_us(
                qpn8_launch,
                warmups=warmups,
                repeats=repeats,
            )
            qpn8_launch()
            torch.accelerator.synchronize()
            row = {
                "kernel": "online_qpn8_dispatch",
                "runtime_cost_included": True,
                "persistent_weight_mib": (
                    (codes.numel() * codes.element_size())
                    + (scales.numel() * scales.element_size())
                )
                / 2**20,
                "replaced_fp16_weight_mib": weight.numel()
                * weight.element_size()
                / 2**20,
                "shared_workspace_mib": workspace.numel()
                * workspace.element_size()
                / 2**20,
                "config": {
                    "split_k": split_k,
                    "nacc": nacc,
                    "prefetch": prefetch,
                    "padded_n": padded_n,
                },
                "block_k": None,
                "num_warps": None,
                "latency_us": latency_us,
                "speedup": baseline_us / latency_us,
                **_output_metrics(qpn8_output[:, :n], reference),
            }
        except Exception as err:
            row = {
                "kernel": "online_qpn8_dispatch",
                "runtime_cost_included": True,
                "block_k": None,
                "num_warps": None,
                "error": f"{type(err).__name__}: {err}",
            }
        candidates.append(row)

    valid = [
        row
        for row in candidates
        if row.get("runtime_cost_included") and "latency_us" in row
    ]
    fastest = min(valid, key=lambda row: row["latency_us"]) if valid else None
    best = fastest if fastest and fastest["latency_us"] < baseline_us else None
    return {
        "name": name,
        "shape": [_M, n, k],
        "count_per_verifier": count,
        "weight_mib": weight.numel() * weight.element_size() / 2**20,
        "torch_latency_us": baseline_us,
        "torch_weight_gbps": weight.numel() * weight.element_size() / baseline_us / 1e3,
        "best": best,
        "candidates": candidates,
    }


def _benchmark_hc_pair(
    *,
    warmups: int,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    from vllm import _sm70_ops as sm70_ops
    from vllm.models.qwen4_exp.nvidia.ops.hc import hc_gate_mix, hc_silu

    torch.manual_seed(seed)
    x = torch.randn((_M, 10240), dtype=torch.float16, device="cuda")
    down_weight = torch.randn((336, 10240), dtype=torch.float16, device="cuda")
    up_weight = torch.randn((10240, 320), dtype=torch.float16, device="cuda")

    def torch_launch() -> tuple[torch.Tensor, torch.Tensor]:
        down = torch.nn.functional.linear(x, down_weight)
        lora = hc_silu(down[:, :320], 4)
        injection = down[:, 320:324]
        gate = torch.nn.functional.linear(lora, up_weight)
        return hc_gate_mix(x, gate, 4), injection

    torch_us = _measure_graph_us(
        torch_launch,
        warmups=warmups,
        repeats=repeats,
    )
    reference_block, reference_injection = torch_launch()
    torch.accelerator.synchronize()

    down_codes, down_scales = _prepare_online_qpn8_weight(down_weight, 384)
    up_codes, up_scales = _prepare_online_qpn8_weight(up_weight, 10240)
    workspace = torch.empty((4096 * 2560,), dtype=torch.float16, device="cuda")
    block_out = torch.empty((_M, 2560), dtype=torch.float16, device="cuda")
    injection_out = torch.empty((_M, 4), dtype=torch.float16, device="cuda")
    down_staging = torch.empty((_M, 384), dtype=torch.float16, device="cuda")
    lora_staging = torch.empty((_M, 320), dtype=torch.float16, device="cuda")
    gate_staging = torch.empty((_M, 10240), dtype=torch.float16, device="cuda")
    partials = torch.empty((32 * 384,), dtype=torch.float32, device="cuda")

    def qpn8_launch() -> None:
        sm70_ops.fp8_qpn8_hc_dispatch_sm70_out(
            block_out,
            injection_out,
            down_staging,
            lora_staging,
            gate_staging,
            partials,
            workspace.data_ptr(),
            x,
            down_codes,
            down_scales,
            up_codes,
            up_scales,
        )

    qpn8_us = _measure_graph_us(
        qpn8_launch,
        warmups=warmups,
        repeats=repeats,
    )
    qpn8_launch()
    torch.accelerator.synchronize()
    qpn8_weight_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in (down_codes, down_scales, up_codes, up_scales)
    )
    fp16_weight_bytes = sum(
        tensor.numel() * tensor.element_size() for tensor in (down_weight, up_weight)
    )
    return {
        "shape": {
            "m": _M,
            "down": [336, 10240],
            "up": [10240, 320],
        },
        "count_per_verifier": 96,
        "torch_latency_us": torch_us,
        "qpn8_latency_us": qpn8_us,
        "speedup": torch_us / qpn8_us,
        "saved_us_per_pair": torch_us - qpn8_us,
        "saved_us_per_verifier": (torch_us - qpn8_us) * 96,
        "persistent_qpn8_weight_mib": qpn8_weight_bytes / 2**20,
        "replaced_fp16_weight_mib": fp16_weight_bytes / 2**20,
        "shared_workspace_mib": workspace.numel() * workspace.element_size() / 2**20,
        "block_quality": _output_metrics(block_out, reference_block),
        "injection_quality": _output_metrics(injection_out, reference_injection),
    }


def main() -> None:
    global _M

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokens",
        type=int,
        choices=range(1, 17),
        default=_M,
        metavar="M",
        help="Decode/verifier batch width (1..16; default: 5).",
    )
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--include-triton",
        action="store_true",
        help="Include the already-rejected naive Triton M5 candidates.",
    )
    parser.add_argument(
        "--include-lossy-qpn8",
        action="store_true",
        help="Include lossy online QPN8 diagnostics; disabled by default.",
    )
    parser.add_argument(
        "--hc-pair-only",
        action="store_true",
        help="Benchmark the fused QPN8 HC down/SiLU/up/gate-mix path only.",
    )
    parser.add_argument(
        "--shapes",
        type=str,
        default="",
        help="Comma-separated shape names; empty runs the full verifier mix.",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    _M = args.tokens

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        raise RuntimeError("This benchmark requires an NVIDIA SM70 GPU.")
    selected = {item.strip() for item in args.shapes.split(",") if item.strip()}
    unknown = selected - {shape[0] for shape in _QWEN38_TP4_M5_SHAPES}
    if unknown:
        raise ValueError(f"Unknown shapes: {sorted(unknown)}")

    rows: list[dict[str, Any]] = []
    if not args.hc_pair_only:
        for index, shape in enumerate(_QWEN38_TP4_M5_SHAPES):
            name = shape[0]
            if selected and name not in selected:
                continue
            rows.append(
                _benchmark_shape(
                    *shape,
                    warmups=args.warmups,
                    repeats=args.repeats,
                    seed=args.seed + index,
                    include_triton=args.include_triton,
                    include_lossy_qpn8=args.include_lossy_qpn8,
                )
            )
            torch.accelerator.empty_cache()

    baseline_weighted_us = sum(
        row["count_per_verifier"] * row["torch_latency_us"] for row in rows
    )
    candidate_weighted_us = sum(
        row["count_per_verifier"]
        * (row["best"]["latency_us"] if row["best"] else row["torch_latency_us"])
        for row in rows
    )
    report = {
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "dtype": "float16",
        "m": _M,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "include_triton": args.include_triton,
        "include_lossy_qpn8": args.include_lossy_qpn8,
        "rows": rows,
        "hc_pair": (
            _benchmark_hc_pair(
                warmups=args.warmups,
                repeats=args.repeats,
                seed=args.seed,
            )
            if args.hc_pair_only
            else None
        ),
        "weighted_cost_model": {
            "torch_us": baseline_weighted_us,
            "best_candidate_us": candidate_weighted_us,
            "speedup": (
                baseline_weighted_us / candidate_weighted_us
                if candidate_weighted_us
                else None
            ),
            "saved_us": baseline_weighted_us - candidate_weighted_us,
            "note": (
                "Serial service-sum model; accepted Nsight graph timing "
                "remains authoritative."
            ),
        },
    }
    payload = json.dumps(report, indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
