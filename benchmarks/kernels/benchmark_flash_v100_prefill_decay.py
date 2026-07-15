# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Kernel-only Flash-V100 prefill decay benchmark.

This script bypasses model loading and exercises the same Flash-V100 Python/CUDA
ops used by the SM70 attention backend.  It is intended for fast iteration on
long-context prefill kernel changes.

Examples:
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
      python benchmarks/kernels/benchmark_flash_v100_prefill_decay.py \
        --mode chunked --prompt-lens 4096 16384 65536 --chunk-size 8096

    nsys profile -o /tmp/flash_v100_prefill_decay --trace=cuda,nvtx \
      python benchmarks/kernels/benchmark_flash_v100_prefill_decay.py \
        --mode paged-step --paged-steps 4096x65536 4096x131072 --nvtx
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Any

import torch

SOURCE_ROOT = Path(__file__).resolve().parents[2]
FLASH_V100_ROOT = SOURCE_ROOT / "flash-attention-v100"
sys.path.insert(0, str(SOURCE_ROOT))
sys.path.insert(0, str(FLASH_V100_ROOT))

from flash_attn_v100.flash_attn_interface import (  # noqa: E402
    flash_attn_func,
    flash_attn_prefill_paged,
)

try:
    from flash_attn_v100.flash_attn_interface import (  # noqa: E402
        flash_attn_prefill_paged_splitkv,
    )
except ImportError:  # pragma: no cover - old local builds.
    flash_attn_prefill_paged_splitkv = None


PROFILE_SHAPES = {
    # Qwen3.6-27B TP4 full-attention shape.
    "qwen36-27b-tp4": {"heads_q": 6, "heads_kv": 1, "head_dim": 256, "layers": 16},
    # Qwen3.6-35B-A3B TP1 local attention shape.
    "qwen35-tp1": {"heads_q": 16, "heads_kv": 2, "head_dim": 256, "layers": 40},
    # Qwen3.6-35B-A3B TP2 local attention shape.
    "qwen35-tp2": {"heads_q": 8, "heads_kv": 1, "head_dim": 256, "layers": 40},
    # Qwen3.6-35B-A3B TP4 local attention shape:
    # global heads=16, global kv_heads=2, head_dim=256.
    "qwen35-tp4": {"heads_q": 4, "heads_kv": 1, "head_dim": 256, "layers": 40},
    # Generic SM70 D=128 route for comparison.
    "h4-kv1-d128": {"heads_q": 4, "heads_kv": 1, "head_dim": 128, "layers": 1},
}


def _sync() -> None:
    torch.cuda.synchronize()


def _paged_block_m(
    head_dim: int,
    *,
    q_len: int,
    block_size: int,
    paged_split_kv: bool,
) -> int:
    if paged_split_kv:
        return 16
    if head_dim == 16:
        return 16
    if head_dim == 32:
        return 32
    if head_dim == 64:
        return 64
    if head_dim == 128:
        return 32
    if head_dim != 256:
        return 32

    # Match launcher_flash_attention_forward_paged's default-enabled D256
    # low-SMEM route so reported CTA counts describe the dispatched kernel.
    low_smem = (
        q_len > 1
        and block_size >= 16
        and block_size % 16 == 0
        and os.getenv("VLLM_FLASH_V100_PREFILL_D256_LOW_SMEM", "1") != "0"
    )
    bm32 = os.getenv("VLLM_FLASH_V100_PREFILL_D256_BM32", "0") != "0"
    bm32_phase = (
        os.getenv("VLLM_FLASH_V100_PREFILL_D256_BM32_PHASE", "1") != "0"
        and block_size == 784
        and q_len >= 32
    )
    if low_smem and (bm32 or bm32_phase):
        return 32
    return 16 if low_smem else 32


def _dynamic_reps(work_items: int) -> tuple[int, int]:
    if work_items <= 2048 * 8192:
        return 5, 20
    if work_items <= 4096 * 65536:
        return 2, 8
    if work_items <= 8192 * 131072:
        return 1, 4
    return 1, 2


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _time_cuda(
    fn,
    *,
    warmup: int | None,
    reps: int | None,
    work_items: int,
    nvtx_name: str | None = None,
) -> dict[str, float | int]:
    dyn_warmup, dyn_reps = _dynamic_reps(work_items)
    warmup = dyn_warmup if warmup is None else warmup
    reps = dyn_reps if reps is None else reps

    for _ in range(warmup):
        if nvtx_name is not None:
            torch.cuda.nvtx.range_push(f"warmup:{nvtx_name}")
        fn()
        if nvtx_name is not None:
            torch.cuda.nvtx.range_pop()
    _sync()

    times: list[float] = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        if nvtx_name is not None:
            torch.cuda.nvtx.range_push(nvtx_name)
        start.record()
        fn()
        end.record()
        if nvtx_name is not None:
            torch.cuda.nvtx.range_pop()
        _sync()
        times.append(start.elapsed_time(end))

    return {
        "median_ms": statistics.median(times),
        "mean_ms": statistics.mean(times),
        "min_ms": min(times),
        "p50_ms": _percentile(times, 0.50),
        "p90_ms": _percentile(times, 0.90),
        "p99_ms": _percentile(times, 0.99),
        "max_ms": max(times),
        "samples_ms": times,
        "warmup": warmup,
        "reps": reps,
    }


def _time_low_smem_layout_ab(
    fn,
    *,
    layout_env: str,
    warmup_pairs: int,
    rounds: int,
) -> dict[str, float | int]:
    """Alternates baseline/candidate to remove clock and thermal drift."""
    saved_layout = os.environ.get(layout_env)
    baseline_times: list[float] = []
    candidate_times: list[float] = []
    paired_deltas: list[float] = []

    def run_timed(enabled: bool) -> float:
        os.environ[layout_env] = "1" if enabled else "0"
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        return start.elapsed_time(end)

    try:
        for _ in range(warmup_pairs):
            os.environ[layout_env] = "0"
            fn()
            os.environ[layout_env] = "1"
            fn()
        _sync()

        for round_idx in range(rounds):
            order = (False, True) if round_idx % 2 == 0 else (True, False)
            pair: dict[bool, float] = {}
            for enabled in order:
                pair[enabled] = run_timed(enabled)
            baseline_times.append(pair[False])
            candidate_times.append(pair[True])
            paired_deltas.append(pair[True] - pair[False])
    finally:
        if saved_layout is None:
            os.environ.pop(layout_env, None)
        else:
            os.environ[layout_env] = saved_layout

    baseline_median = statistics.median(baseline_times)
    candidate_median = statistics.median(candidate_times)
    return {
        "rounds": rounds,
        "warmup_pairs": warmup_pairs,
        "baseline_median_ms": baseline_median,
        "candidate_median_ms": candidate_median,
        "candidate_vs_baseline_median_pct": (
            (candidate_median / baseline_median - 1.0) * 100.0
        ),
        "paired_delta_mean_ms": statistics.mean(paired_deltas),
        "paired_delta_median_ms": statistics.median(paired_deltas),
        "candidate_faster_pairs": sum(
            candidate < baseline
            for baseline, candidate in zip(baseline_times, candidate_times)
        ),
    }


def _effective_attention_flops(
    q_len: int,
    kv_len: int,
    heads_q: int,
    head_dim: int,
    *,
    causal: bool,
) -> float:
    if not causal:
        effective_keys = float(kv_len)
    elif q_len == kv_len:
        effective_keys = (q_len + 1) / 2.0
    else:
        prefix_len = max(kv_len - q_len, 0)
        effective_keys = prefix_len + (q_len + 1) / 2.0
    # QK and PV, each with multiply+add.
    return 4.0 * q_len * effective_keys * heads_q * head_dim


def _add_derived_metrics(
    result: dict[str, Any],
    *,
    heads_q: int,
    head_dim: int,
    causal: bool,
    layers: int,
) -> None:
    q_len = int(result["q_len"])
    kv_len = int(result["kv_len"])
    median_ms = float(result["median_ms"])
    flops = _effective_attention_flops(q_len, kv_len, heads_q, head_dim, causal=causal)
    result["effective_tflops"] = flops / (median_ms / 1000.0) / 1e12
    samples_ms = result.get("samples_ms")
    if isinstance(samples_ms, list) and samples_ms:
        launch_tflops = [
            flops / (sample_ms / 1000.0) / 1e12 for sample_ms in samples_ms
        ]
        result["launch_effective_tflops_samples"] = launch_tflops
        result["launch_effective_tflops_min"] = min(launch_tflops)
        result["launch_effective_tflops_p50"] = _percentile(launch_tflops, 0.50)
        result["launch_effective_tflops_p90"] = _percentile(launch_tflops, 0.90)
        result["launch_effective_tflops_p99"] = _percentile(launch_tflops, 0.99)
        result["launch_effective_tflops_max"] = max(launch_tflops)
    result["attention_layer_ms"] = median_ms
    result["attention_layers_scaled_s"] = median_ms * layers / 1000.0


def _make_paged_cache(
    kv_len: int,
    *,
    block_size: int,
    heads_kv: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    reverse_block_table: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_blocks = math.ceil(kv_len / block_size)
    k_cache = torch.randn(
        (num_blocks, block_size, heads_kv, head_dim),
        device=device,
        dtype=dtype,
    )
    v_cache = torch.randn_like(k_cache)
    block_table = torch.arange(num_blocks, device=device, dtype=torch.int32).view(
        1, num_blocks
    )
    if reverse_block_table:
        block_table = torch.flip(block_table, dims=(1,)).contiguous()
    seq_lens = torch.tensor([kv_len], device=device, dtype=torch.int32)
    return k_cache, v_cache, block_table, seq_lens


def bench_dense(
    length: int,
    *,
    heads_q: int,
    heads_kv: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    warmup: int | None,
    reps: int | None,
    layers: int,
    nvtx: bool,
) -> dict[str, Any]:
    q = torch.randn((1, length, heads_q, head_dim), device=device, dtype=dtype)
    k = torch.randn((1, length, heads_kv, head_dim), device=device, dtype=dtype)
    v = torch.randn((1, length, heads_kv, head_dim), device=device, dtype=dtype)
    holder: list[torch.Tensor | None] = [None]

    def run() -> None:
        holder[0] = flash_attn_func(q, k, v, causal=True)

    result: dict[str, Any] = {
        "mode": "dense",
        "q_len": length,
        "kv_len": length,
        "heads_q": heads_q,
        "heads_kv": heads_kv,
        "head_dim": head_dim,
        "grid_x": math.ceil(length / 32),
        "grid_z": heads_q,
        "ctas": math.ceil(length / 32) * heads_q,
    }
    result.update(
        _time_cuda(
            run,
            warmup=warmup,
            reps=reps,
            work_items=length * length,
            nvtx_name=f"dense_M{length}_N{length}" if nvtx else None,
        )
    )
    _add_derived_metrics(
        result,
        heads_q=heads_q,
        head_dim=head_dim,
        causal=True,
        layers=layers,
    )
    return result


def bench_paged_step(
    q_len: int,
    kv_len: int,
    *,
    heads_q: int,
    heads_kv: int,
    head_dim: int,
    block_size: int,
    dtype: torch.dtype,
    device: torch.device,
    warmup: int | None,
    reps: int | None,
    layers: int,
    nvtx: bool,
    paged_split_kv: bool,
    split_kv_tokens: int,
    check_reference: bool,
    check_low_smem_layout_reference: bool,
    layout_candidate_env: str | None,
    layout_ab_rounds: int,
    reverse_block_table: bool,
) -> dict[str, Any]:
    q = torch.randn((1, q_len, heads_q, head_dim), device=device, dtype=dtype)
    k_cache, v_cache, block_table, seq_lens = _make_paged_cache(
        kv_len,
        block_size=block_size,
        heads_kv=heads_kv,
        head_dim=head_dim,
        dtype=dtype,
        device=device,
        reverse_block_table=reverse_block_table,
    )
    holder: list[torch.Tensor | None] = [None]

    if paged_split_kv and flash_attn_prefill_paged_splitkv is None:
        raise RuntimeError("flash_attn_prefill_paged_splitkv is unavailable")

    def run() -> None:
        if paged_split_kv:
            holder[0] = flash_attn_prefill_paged_splitkv(
                q,
                k_cache,
                v_cache,
                block_table,
                seq_lens,
                causal=True,
                split_kv_tokens=split_kv_tokens,
                max_seq_len_hint=kv_len,
            )
            return
        holder[0] = flash_attn_prefill_paged(
            q,
            k_cache,
            v_cache,
            block_table,
            seq_lens,
            causal=True,
        )

    grid_y = math.ceil(kv_len / split_kv_tokens) if paged_split_kv else 1
    kernel_block_m = _paged_block_m(
        head_dim,
        q_len=q_len,
        block_size=block_size,
        paged_split_kv=paged_split_kv,
    )
    result: dict[str, Any] = {
        "mode": "paged-step-splitkv" if paged_split_kv else "paged-step",
        "q_len": q_len,
        "kv_len": kv_len,
        "heads_q": heads_q,
        "heads_kv": heads_kv,
        "head_dim": head_dim,
        "block_size": block_size,
        "cache_blocks": math.ceil(kv_len / block_size),
        "kernel_block_m": kernel_block_m,
        "grid_x": math.ceil(q_len / kernel_block_m),
        "grid_y": grid_y,
        "grid_z": heads_q,
        "ctas": math.ceil(q_len / kernel_block_m) * grid_y * heads_q,
        "split_kv_tokens": split_kv_tokens if paged_split_kv else None,
        "reverse_block_table": reverse_block_table,
    }
    result.update(
        _time_cuda(
            run,
            warmup=warmup,
            reps=reps,
            work_items=q_len * kv_len,
            nvtx_name=f"paged_M{q_len}_N{kv_len}" if nvtx else None,
        )
    )
    _add_derived_metrics(
        result,
        heads_q=heads_q,
        head_dim=head_dim,
        causal=True,
        layers=layers,
    )
    if check_reference:
        ref = flash_attn_prefill_paged(
            q,
            k_cache,
            v_cache,
            block_table,
            seq_lens,
            causal=True,
        )
        if holder[0] is None:
            run()
        _sync()
        diff = (holder[0] - ref).abs()
        result["reference_max_diff"] = float(diff.max().item())
        result["reference_mean_diff"] = float(diff.mean().item())
    if check_low_smem_layout_reference:
        if paged_split_kv:
            raise ValueError(
                "--check-low-smem-layout-reference does not support split-KV"
            )
        if layout_candidate_env is None:
            raise ValueError(
                "--check-low-smem-layout-reference requires --layout-candidate-env"
            )
        layout_env = layout_candidate_env
        saved_layout = os.environ.get(layout_env)
        try:
            os.environ[layout_env] = "0"
            baseline = flash_attn_prefill_paged(
                q,
                k_cache,
                v_cache,
                block_table,
                seq_lens,
                causal=True,
            )
            _sync()
            os.environ[layout_env] = "1"
            candidate = flash_attn_prefill_paged(
                q,
                k_cache,
                v_cache,
                block_table,
                seq_lens,
                causal=True,
            )
            _sync()
        finally:
            if saved_layout is None:
                os.environ.pop(layout_env, None)
            else:
                os.environ[layout_env] = saved_layout
        layout_diff = (candidate - baseline).abs()
        result["layout_reference_equal"] = bool(torch.equal(candidate, baseline))
        result["layout_reference_max_diff"] = float(layout_diff.max().item())
        result["layout_reference_mean_diff"] = float(layout_diff.mean().item())
    if layout_ab_rounds:
        if paged_split_kv:
            raise ValueError("--layout-ab-rounds does not support split-KV")
        if layout_candidate_env is None:
            raise ValueError("--layout-ab-rounds requires --layout-candidate-env")
        result["layout_ab"] = _time_low_smem_layout_ab(
            run,
            layout_env=layout_candidate_env,
            warmup_pairs=max(5, (warmup or 0) // 2),
            rounds=layout_ab_rounds,
        )
    return result


def bench_chunked_prompt(
    prompt_len: int,
    *,
    chunk_size: int,
    heads_q: int,
    heads_kv: int,
    head_dim: int,
    block_size: int,
    dtype: torch.dtype,
    device: torch.device,
    warmup: int | None,
    reps: int | None,
    layers: int,
    nvtx: bool,
    paged_split_kv: bool,
    split_kv_tokens: int,
    check_reference: bool,
    check_low_smem_layout_reference: bool,
    layout_candidate_env: str | None,
    layout_ab_rounds: int,
    reverse_block_table: bool,
) -> dict[str, Any]:
    chunk_results: list[dict[str, Any]] = []
    pos = 0
    chunk_idx = 0
    while pos < prompt_len:
        q_len = min(chunk_size, prompt_len - pos)
        kv_len = pos + q_len
        if pos == 0:
            chunk = bench_dense(
                q_len,
                heads_q=heads_q,
                heads_kv=heads_kv,
                head_dim=head_dim,
                dtype=dtype,
                device=device,
                warmup=warmup,
                reps=reps,
                layers=layers,
                nvtx=nvtx,
            )
            chunk["mode"] = "chunked-first-dense"
        else:
            chunk = bench_paged_step(
                q_len,
                kv_len,
                heads_q=heads_q,
                heads_kv=heads_kv,
                head_dim=head_dim,
                block_size=block_size,
                dtype=dtype,
                device=device,
                warmup=warmup,
                reps=reps,
                layers=layers,
                nvtx=nvtx,
                paged_split_kv=paged_split_kv,
                split_kv_tokens=split_kv_tokens,
                check_reference=check_reference,
                check_low_smem_layout_reference=check_low_smem_layout_reference,
                layout_candidate_env=layout_candidate_env,
                layout_ab_rounds=layout_ab_rounds,
                reverse_block_table=reverse_block_table,
            )
            chunk["mode"] = (
                "chunked-prefix-paged-splitkv"
                if paged_split_kv
                else "chunked-prefix-paged"
            )
        chunk["chunk_idx"] = chunk_idx
        chunk["prompt_len"] = prompt_len
        chunk["chunk_start"] = pos
        chunk_results.append(chunk)
        pos += q_len
        chunk_idx += 1
        torch.cuda.empty_cache()

    total_layer_ms = sum(float(chunk["median_ms"]) for chunk in chunk_results)
    total_flops = sum(
        _effective_attention_flops(
            int(chunk["q_len"]),
            int(chunk["kv_len"]),
            heads_q,
            head_dim,
            causal=True,
        )
        for chunk in chunk_results
    )
    return {
        "mode": "chunked-summary",
        "prompt_len": prompt_len,
        "chunk_size": chunk_size,
        "num_chunks": len(chunk_results),
        "heads_q": heads_q,
        "heads_kv": heads_kv,
        "head_dim": head_dim,
        "block_size": block_size,
        "attention_layer_ms": total_layer_ms,
        "attention_layer_tps": prompt_len / (total_layer_ms / 1000.0),
        "attention_layers_scaled_s": total_layer_ms * layers / 1000.0,
        "effective_tflops": total_flops / (total_layer_ms / 1000.0) / 1e12,
        "chunks": chunk_results,
    }


def _parse_paged_step(value: str) -> tuple[int, int]:
    try:
        q_len, kv_len = value.lower().split("x", 1)
        return int(q_len), int(kv_len)
    except Exception as exc:
        raise argparse.ArgumentTypeError(
            f"expected MxN, for example 4096x65536: {value}"
        ) from exc


def _dtype_from_string(value: str) -> torch.dtype:
    if value in ("fp16", "float16", "half"):
        return torch.float16
    if value in ("bf16", "bfloat16"):
        return torch.bfloat16
    raise argparse.ArgumentTypeError(f"unsupported dtype: {value}")


def _emit(result: dict[str, Any], *, jsonl: bool) -> None:
    if jsonl:
        print(json.dumps(result, sort_keys=True))
        return
    if result["mode"] == "chunked-summary":
        print(
            "chunked prompt={prompt_len} chunk={chunk_size} chunks={num_chunks} "
            "layer={attention_layer_ms:.3f} ms "
            "scaled_layers={attention_layers_scaled_s:.3f} s "
            "tps={attention_layer_tps:.1f} eff={effective_tflops:.2f} TF/s".format(
                **result
            )
        )
        for chunk in result["chunks"]:
            print(
                "  #{chunk_idx:02d} {mode} M={q_len} N={kv_len} "
                "ctas={ctas} median={median_ms:.3f} ms eff={effective_tflops:.2f} TF/s "
                "launch[min/p50/p90/max]={launch_effective_tflops_min:.2f}/"
                "{launch_effective_tflops_p50:.2f}/{launch_effective_tflops_p90:.2f}/"
                "{launch_effective_tflops_max:.2f}".format(**chunk)
            )
        return
    print(
        "{mode} M={q_len} N={kv_len} ctas={ctas} "
        "median={median_ms:.3f} ms scaled_layers={attention_layers_scaled_s:.3f} s "
        "eff={effective_tflops:.2f} TF/s "
        "launch[min/p50/p90/max]={launch_effective_tflops_min:.2f}/"
        "{launch_effective_tflops_p50:.2f}/{launch_effective_tflops_p90:.2f}/"
        "{launch_effective_tflops_max:.2f}".format(**result)
    )


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("dense", "paged-step", "chunked", "all"),
        default="chunked",
    )
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILE_SHAPES),
        default="qwen36-27b-tp4",
        help="Predefined local TP attention shape.",
    )
    parser.add_argument("--heads-q", type=int, default=None)
    parser.add_argument("--heads-kv", type=int, default=None)
    parser.add_argument("--head-dim", type=int, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument(
        "--prompt-lens",
        type=int,
        nargs="+",
        default=[4096, 16384, 65536],
        help="Prompt lengths for dense/chunked modes.",
    )
    parser.add_argument(
        "--dense-lens",
        type=int,
        nargs="+",
        default=None,
        help="Override dense lengths; defaults to --prompt-lens.",
    )
    parser.add_argument(
        "--paged-steps",
        type=_parse_paged_step,
        nargs="+",
        default=[(1024, 16384), (4096, 65536)],
        help="Paged prefix cases as MxN.",
    )
    parser.add_argument("--chunk-size", type=int, default=8096)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument(
        "--reverse-block-table",
        action="store_true",
        help="Reverse the logical-to-physical paged-cache block mapping.",
    )
    parser.add_argument("--dtype", type=_dtype_from_string, default=torch.float16)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--reps", type=int, default=None)
    parser.add_argument(
        "--paged-split-kv",
        action="store_true",
        help="Use the experimental split-KV paged prefill op for paged chunks.",
    )
    parser.add_argument("--split-kv-tokens", type=int, default=32768)
    parser.add_argument(
        "--check-reference",
        action="store_true",
        help="Compare the measured paged output against the legacy paged op.",
    )
    parser.add_argument(
        "--check-low-smem-layout-reference",
        action="store_true",
        help=(
            "Compare the low-SMEM candidate environment layout and baseline "
            "within one process."
        ),
    )
    parser.add_argument(
        "--layout-candidate-env",
        default=None,
        help="Environment variable set to 0/1 for a low-SMEM layout A/B.",
    )
    parser.add_argument(
        "--layout-ab-rounds",
        type=int,
        default=0,
        help="Alternate low-SMEM baseline/candidate timing for this many pairs.",
    )
    parser.add_argument("--jsonl", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument(
        "--nvtx",
        action="store_true",
        help="Annotate measured kernel calls for nsys traces.",
    )
    args = parser.parse_args()

    shape = PROFILE_SHAPES[args.profile]
    heads_q = args.heads_q or int(shape["heads_q"])
    heads_kv = args.heads_kv or int(shape["heads_kv"])
    head_dim = args.head_dim or int(shape["head_dim"])
    layers = args.layers or int(shape["layers"])
    if heads_q % heads_kv != 0:
        raise ValueError(f"heads_q={heads_q} must be divisible by heads_kv={heads_kv}")

    torch.manual_seed(args.seed)
    torch.cuda.set_device(args.device)
    device = torch.device(f"cuda:{args.device}")
    props = torch.cuda.get_device_properties(device)
    if props.major != 7 or props.minor != 0:
        raise RuntimeError(
            f"Flash-V100 kernels require sm70, got {props.name} "
            f"sm{props.major}{props.minor}"
        )

    header = {
        "python": {
            "executable": sys.executable,
            "version": sys.version.split()[0],
        },
        "flash_attn_v100_extension": getattr(
            sys.modules.get("flash_attn_v100_cuda"), "__file__", None
        ),
        "gpu": props.name,
        "capability": f"{props.major}.{props.minor}",
        "profile": args.profile,
        "heads_q": heads_q,
        "heads_kv": heads_kv,
        "head_dim": head_dim,
        "layers": layers,
        "chunk_size": args.chunk_size,
        "block_size": args.block_size,
        "dtype": str(args.dtype).replace("torch.", ""),
        "env": {
            "VLLM_FLASH_V100_PREFILL_D256_LOW_SMEM": os.getenv(
                "VLLM_FLASH_V100_PREFILL_D256_LOW_SMEM"
            ),
            "prefill_d256_low_smem_effective": (
                os.getenv("VLLM_FLASH_V100_PREFILL_D256_LOW_SMEM", "1") != "0"
            ),
            "VLLM_FLASH_V100_DENSE_D256_LOW_SMEM": os.getenv(
                "VLLM_FLASH_V100_DENSE_D256_LOW_SMEM"
            ),
            "VLLM_FLASH_V100_DENSE_D256_WMMA_QK": os.getenv(
                "VLLM_FLASH_V100_DENSE_D256_WMMA_QK"
            ),
            "VLLM_FLASH_V100_PREFILL_D256_SCALAR_QK": os.getenv(
                "VLLM_FLASH_V100_PREFILL_D256_SCALAR_QK"
            ),
            "VLLM_FLASH_V100_PREFILL_D256_BM32": os.getenv(
                "VLLM_FLASH_V100_PREFILL_D256_BM32"
            ),
            "prefill_d256_bm32_effective": (
                os.getenv("VLLM_FLASH_V100_PREFILL_D256_BM32", "0") != "0"
            ),
            "VLLM_FLASH_V100_PREFILL_D256_BM32_PHASE": os.getenv(
                "VLLM_FLASH_V100_PREFILL_D256_BM32_PHASE"
            ),
            "prefill_d256_bm32_phase_effective": (
                os.getenv("VLLM_FLASH_V100_PREFILL_D256_BM32_PHASE", "1") != "0"
            ),
            "VLLM_FLASH_V100_PREFILL_D256_BM32_ALL_P": os.getenv(
                "VLLM_FLASH_V100_PREFILL_D256_BM32_ALL_P"
            ),
            "prefill_d256_bm32_all_p_effective": (
                os.getenv("VLLM_FLASH_V100_PREFILL_D256_BM32_ALL_P", "1") != "0"
            ),
            "VLLM_FLASH_V100_PREFILL_D256_BM32_PAIR_SCRATCH": os.getenv(
                "VLLM_FLASH_V100_PREFILL_D256_BM32_PAIR_SCRATCH"
            ),
            "prefill_d256_bm32_pair_scratch_effective": (
                os.getenv("VLLM_FLASH_V100_PREFILL_D256_BM32_PAIR_SCRATCH", "1") != "0"
            ),
            "VLLM_FLASH_V100_PREFILL_D256_OUTPUT_STRIDE_268": os.getenv(
                "VLLM_FLASH_V100_PREFILL_D256_OUTPUT_STRIDE_268"
            ),
            "prefill_d256_output_stride_268_effective": (
                os.getenv(
                    "VLLM_FLASH_V100_PREFILL_D256_OUTPUT_STRIDE_268",
                    "1",
                )
                != "0"
            ),
            "VLLM_FLASH_V100_PREFILL_D256_SOFTWARE_PIPELINE": os.getenv(
                "VLLM_FLASH_V100_PREFILL_D256_SOFTWARE_PIPELINE"
            ),
            "VLLM_FLASH_V100_PREFILL_D256_SW_PIPELINE_QK": os.getenv(
                "VLLM_FLASH_V100_PREFILL_D256_SW_PIPELINE_QK"
            ),
            "prefill_d256_sw_pipeline_qk_effective": (
                os.getenv(
                    "VLLM_FLASH_V100_PREFILL_D256_SW_PIPELINE_QK",
                    "1",
                )
                != "0"
            ),
            "VLLM_FLASH_V100_PREFILL_D256_SW_PIPELINE_PV": os.getenv(
                "VLLM_FLASH_V100_PREFILL_D256_SW_PIPELINE_PV"
            ),
            "prefill_d256_sw_pipeline_pv_effective": (
                os.getenv(
                    "VLLM_FLASH_V100_PREFILL_D256_SW_PIPELINE_PV",
                    "1",
                )
                != "0"
            ),
            "VLLM_FLASH_V100_PREFILL_SCALAR_PV": os.getenv(
                "VLLM_FLASH_V100_PREFILL_SCALAR_PV"
            ),
            "VLLM_FLASH_V100_PREFILL_SPLIT_KV": os.getenv(
                "VLLM_FLASH_V100_PREFILL_SPLIT_KV"
            ),
            "VLLM_FLASH_V100_PREFILL_SPLIT_KV_TOKENS": os.getenv(
                "VLLM_FLASH_V100_PREFILL_SPLIT_KV_TOKENS"
            ),
        },
        "paged_split_kv": args.paged_split_kv,
        "split_kv_tokens": args.split_kv_tokens,
        "check_reference": args.check_reference,
        "check_low_smem_layout_reference": (args.check_low_smem_layout_reference),
        "layout_candidate_env": args.layout_candidate_env,
        "layout_ab_rounds": args.layout_ab_rounds,
        "reverse_block_table": args.reverse_block_table,
    }
    print(json.dumps({"run": header}, sort_keys=True))

    results: list[dict[str, Any]] = []
    if args.mode in ("dense", "all"):
        for length in args.dense_lens or args.prompt_lens:
            torch.cuda.empty_cache()
            result = bench_dense(
                length,
                heads_q=heads_q,
                heads_kv=heads_kv,
                head_dim=head_dim,
                dtype=args.dtype,
                device=device,
                warmup=args.warmup,
                reps=args.reps,
                layers=layers,
                nvtx=args.nvtx,
            )
            results.append(result)
            _emit(result, jsonl=args.jsonl)

    if args.mode in ("paged-step", "all"):
        for q_len, kv_len in args.paged_steps:
            torch.cuda.empty_cache()
            result = bench_paged_step(
                q_len,
                kv_len,
                heads_q=heads_q,
                heads_kv=heads_kv,
                head_dim=head_dim,
                block_size=args.block_size,
                dtype=args.dtype,
                device=device,
                warmup=args.warmup,
                reps=args.reps,
                layers=layers,
                nvtx=args.nvtx,
                paged_split_kv=args.paged_split_kv,
                split_kv_tokens=args.split_kv_tokens,
                check_reference=args.check_reference,
                check_low_smem_layout_reference=(args.check_low_smem_layout_reference),
                layout_candidate_env=args.layout_candidate_env,
                layout_ab_rounds=args.layout_ab_rounds,
                reverse_block_table=args.reverse_block_table,
            )
            results.append(result)
            _emit(result, jsonl=args.jsonl)

    if args.mode in ("chunked", "all"):
        for prompt_len in args.prompt_lens:
            torch.cuda.empty_cache()
            result = bench_chunked_prompt(
                prompt_len,
                chunk_size=args.chunk_size,
                heads_q=heads_q,
                heads_kv=heads_kv,
                head_dim=head_dim,
                block_size=args.block_size,
                dtype=args.dtype,
                device=device,
                warmup=args.warmup,
                reps=args.reps,
                layers=layers,
                nvtx=args.nvtx,
                paged_split_kv=args.paged_split_kv,
                split_kv_tokens=args.split_kv_tokens,
                check_reference=args.check_reference,
                check_low_smem_layout_reference=(args.check_low_smem_layout_reference),
                layout_candidate_env=args.layout_candidate_env,
                layout_ab_rounds=args.layout_ab_rounds,
                reverse_block_table=args.reverse_block_table,
            )
            results.append(result)
            _emit(result, jsonl=args.jsonl)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps({"run": header, "results": results}, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
