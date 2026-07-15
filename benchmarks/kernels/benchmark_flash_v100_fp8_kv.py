# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Flash-V100 FP16/FP8 paged-attention microbenchmark.

The default shape matches one TP2 rank of Qwen3.6-27B full attention:
Hq=12, Hkv=2, D=256. Variants deliberately separate KV dtype, page size,
and decode route so a full-model regression is not attributed to FP8 alone.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import torch

SOURCE_ROOT = Path(__file__).resolve().parents[2]
FLASH_V100_ROOT = SOURCE_ROOT / "flash-attention-v100"
sys.path.insert(0, str(SOURCE_ROOT))
sys.path.insert(0, str(FLASH_V100_ROOT))

from flash_attn_v100 import (  # noqa: E402
    flash_attn_decode_paged,
    flash_attn_decode_paged_xqa,
    flash_attn_prefill_paged,
    fp8_e5m2_paged_kv_to_fp16,
)

PREFILL_VARIANTS = (
    "fp16_b784",
    "fp16_b1568",
    "fp16_b1616",
    "fp8_b784",
    "fp8_b1568",
    "fp8_b1568_bridge",
    "fp8_b1616",
    "fp8_b1616_bridge",
)
DECODE_VARIANTS = (
    "fp16_b784_xqa",
    "fp16_b784_scalar",
    "fp16_b1568_scalar",
    "fp8_b784_xqa",
    "fp8_b784_scalar",
    "fp8_b1568_xqa",
    "fp8_b1568_scalar",
)


def _parse_variant(variant: str) -> tuple[str, int, str | None]:
    parts = variant.split("_")
    if len(parts) not in (2, 3):
        raise ValueError(f"Invalid variant: {variant}")
    cache_dtype = "fp8_e5m2" if parts[0] == "fp8" else "auto"
    block_size = int(parts[1].removeprefix("b"))
    route = parts[2] if len(parts) == 3 else None
    return cache_dtype, block_size, route


def _make_cache(
    context_len: int,
    block_size: int,
    cache_dtype: str,
    heads_kv: int,
    head_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_blocks = math.ceil(context_len / block_size)
    shape = (num_blocks, block_size, heads_kv, head_dim)
    key = torch.randn(shape, device=device, dtype=torch.float16)
    value = torch.randn_like(key)
    if cache_dtype == "fp8_e5m2":
        key = key.to(torch.float8_e5m2).view(torch.uint8)
        value = value.to(torch.float8_e5m2).view(torch.uint8)
    block_table = torch.arange(num_blocks, device=device, dtype=torch.int32).unsqueeze(
        0
    )
    seq_lens = torch.tensor([context_len], device=device, dtype=torch.int32)
    return key, value, block_table, seq_lens


def _measure(
    fn,
    *,
    warmup: int,
    reps: int,
    nvtx_name: str | None,
    cuda_profiler: bool,
) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    if cuda_profiler:
        torch.cuda.cudart().cudaProfilerStart()

    samples: list[float] = []
    try:
        for _ in range(reps):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            if nvtx_name is not None:
                torch.cuda.nvtx.range_push(nvtx_name)
            start.record()
            fn()
            end.record()
            end.synchronize()
            if nvtx_name is not None:
                torch.cuda.nvtx.range_pop()
            samples.append(start.elapsed_time(end))
    finally:
        if cuda_profiler:
            torch.cuda.cudart().cudaProfilerStop()

    return {
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.mean(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def _bench_variant(
    *,
    stage: str,
    variant: str,
    context_len: int,
    query_len: int,
    heads_q: int,
    heads_kv: int,
    head_dim: int,
    device: torch.device,
    warmup: int,
    reps: int,
    nvtx: bool,
    cuda_profiler: bool,
) -> dict[str, Any]:
    cache_dtype, block_size, route = _parse_variant(variant)
    key, value, block_table, seq_lens = _make_cache(
        context_len,
        block_size,
        cache_dtype,
        heads_kv,
        head_dim,
        device,
    )

    if stage == "prefill":
        query = torch.randn(
            (1, query_len, heads_q, head_dim),
            device=device,
            dtype=torch.float16,
        )
        bridge_timing = None
        attention_timing = None
        if route == "bridge":
            bridge_block_size = 784
            bridge_num_blocks = math.ceil(
                block_table.shape[1] * block_size / bridge_block_size
            )
            key_bridge = torch.empty(
                (bridge_num_blocks, bridge_block_size, heads_kv, head_dim),
                device=device,
                dtype=torch.float16,
            )
            value_bridge = torch.empty_like(key_bridge)
            bridge_block_table = torch.arange(
                bridge_num_blocks,
                device=device,
                dtype=torch.int32,
            ).unsqueeze(0)

            def run_bridge() -> tuple[torch.Tensor, torch.Tensor]:
                return fp8_e5m2_paged_kv_to_fp16(
                    key,
                    value,
                    block_table,
                    seq_lens,
                    key_bridge,
                    value_bridge,
                )

            def run_attention() -> torch.Tensor:
                return flash_attn_prefill_paged(
                    query,
                    key_bridge,
                    value_bridge,
                    bridge_block_table,
                    seq_lens,
                    kv_cache_dtype="auto",
                    causal=True,
                )

            def run() -> torch.Tensor:
                run_bridge()
                return run_attention()

            bridge_timing = _measure(
                run_bridge,
                warmup=warmup,
                reps=reps,
                nvtx_name=None,
                cuda_profiler=False,
            )
            attention_timing = _measure(
                run_attention,
                warmup=warmup,
                reps=reps,
                nvtx_name=None,
                cuda_profiler=False,
            )
        else:

            def run() -> torch.Tensor:
                return flash_attn_prefill_paged(
                    query,
                    key,
                    value,
                    block_table,
                    seq_lens,
                    kv_cache_dtype=cache_dtype,
                    causal=True,
                )

    else:
        query = torch.randn((1, heads_q, head_dim), device=device, dtype=torch.float16)
        if route == "xqa":

            def run() -> torch.Tensor:
                return flash_attn_decode_paged_xqa(
                    query,
                    key,
                    value,
                    block_table,
                    seq_lens,
                    kv_cache_dtype=cache_dtype,
                    max_seq_len_hint=context_len,
                )

        else:

            def run() -> torch.Tensor:
                return flash_attn_decode_paged(
                    query,
                    key,
                    value,
                    block_table,
                    seq_lens,
                    kv_cache_dtype=cache_dtype,
                    max_seq_len_hint=context_len,
                )

    name = f"{stage}_{variant}_N{context_len}"
    timing = _measure(
        run,
        warmup=warmup,
        reps=reps,
        nvtx_name=name if nvtx else None,
        cuda_profiler=cuda_profiler,
    )
    result = {
        "stage": stage,
        "variant": variant,
        "context_len": context_len,
        "query_len": query_len if stage == "prefill" else 1,
        "heads_q": heads_q,
        "heads_kv": heads_kv,
        "head_dim": head_dim,
        "cache_dtype": cache_dtype,
        "block_size": block_size,
        "route": route,
        **timing,
    }
    if stage == "prefill" and bridge_timing is not None:
        result["bridge_median_ms"] = bridge_timing["median_ms"]
        result["bridge_attention_median_ms"] = attention_timing["median_ms"]
    return result


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prefill", "decode", "all"), default="all")
    parser.add_argument("--variant", action="append", default=None)
    parser.add_argument(
        "--contexts", type=int, nargs="+", default=[16384, 65536, 131072]
    )
    parser.add_argument("--query-len", type=int, default=1568)
    parser.add_argument("--heads-q", type=int, default=12)
    parser.add_argument("--heads-kv", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--nvtx", action="store_true")
    parser.add_argument("--cuda-profiler", action="store_true")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    if args.heads_q % args.heads_kv:
        raise ValueError("--heads-q must be divisible by --heads-kv")
    torch.cuda.set_device(args.device)
    device = torch.device(f"cuda:{args.device}")
    props = torch.cuda.get_device_properties(device)
    if (props.major, props.minor) != (7, 0):
        raise RuntimeError(f"Flash-V100 requires sm70, got {props.name}")
    torch.manual_seed(args.seed)

    stages = ("prefill", "decode") if args.stage == "all" else (args.stage,)
    results: list[dict[str, Any]] = []
    for stage in stages:
        variants = args.variant or (
            PREFILL_VARIANTS if stage == "prefill" else DECODE_VARIANTS
        )
        allowed = PREFILL_VARIANTS if stage == "prefill" else DECODE_VARIANTS
        invalid = sorted(set(variants) - set(allowed))
        if invalid:
            raise ValueError(f"Invalid {stage} variants: {invalid}; allowed={allowed}")
        for context_len in args.contexts:
            for variant in variants:
                torch.cuda.empty_cache()
                result = _bench_variant(
                    stage=stage,
                    variant=variant,
                    context_len=context_len,
                    query_len=args.query_len,
                    heads_q=args.heads_q,
                    heads_kv=args.heads_kv,
                    head_dim=args.head_dim,
                    device=device,
                    warmup=args.warmup,
                    reps=args.reps,
                    nvtx=args.nvtx,
                    cuda_profiler=args.cuda_profiler,
                )
                results.append(result)
                print(json.dumps(result, sort_keys=True), flush=True)

    payload = {
        "run": {
            "gpu": props.name,
            "capability": f"{props.major}.{props.minor}",
            "seed": args.seed,
        },
        "results": results,
    }
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
