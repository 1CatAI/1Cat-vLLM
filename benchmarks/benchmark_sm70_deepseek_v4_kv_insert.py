# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark a block-parallel DeepSeek V4 FP8 KV-cache insert on SM70."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from vllm.models.deepseek_v4.common.ops.cache_utils import (
    quantize_and_insert_k_cache,
)
from vllm.models.deepseek_v4.common.ops.fp8_software import (
    fp32_to_fp8_e4m3fn_bits,
)
from vllm.triton_utils import tl, triton

_HEAD_DIM = 512
_FP8_DIM = 448
_BF16_DIM = 64
_QUANT_BLOCK = 64
_SCALE_DIM = 8
_TOKEN_DATA_SIZE = 576
_HEAD_BYTES = 584


@triton.jit
def _parallel_quantize_and_insert_k_kernel(
    k_ptr,
    slot_mapping_ptr,
    k_cache_ptr,
    num_tokens,
    input_dim: tl.constexpr,
    cache_block_size: tl.constexpr,
    block_stride: tl.constexpr,
    token_data_size: tl.constexpr,
    fp8_dim: tl.constexpr,
    scale_dim: tl.constexpr,
    quant_block: tl.constexpr,
    fp8_max: tl.constexpr,
):
    token_idx = tl.program_id(0)
    part_idx = tl.program_id(1)
    if token_idx >= num_tokens:
        return

    slot_idx = tl.load(slot_mapping_ptr + token_idx)
    if slot_idx == -1:
        return

    block_idx = slot_idx // cache_block_size
    pos_in_block = slot_idx % cache_block_size
    input_row = k_ptr + token_idx * input_dim
    cache_block = k_cache_ptr + block_idx.to(tl.int64) * block_stride
    token_data = cache_block + pos_in_block * token_data_size
    token_scales = (
        cache_block + cache_block_size * token_data_size + pos_in_block * scale_dim
    )

    offsets = tl.arange(0, quant_block)
    if part_idx < 7:
        input_offsets = part_idx * quant_block + offsets
        values = tl.load(input_row + input_offsets)
        block_max = tl.maximum(tl.max(tl.abs(values), axis=0), 1.0e-4)
        exponent = tl.ceil(tl.log2(block_max / fp8_max))
        scale = tl.exp2(exponent)
        scaled = tl.clamp(values / scale, -fp8_max, fp8_max)
        packed = fp32_to_fp8_e4m3fn_bits(scaled.to(tl.float32))
        tl.store(token_data + input_offsets, packed)
        encoded_scale = tl.maximum(tl.minimum(exponent + 127.0, 255.0), 0.0)
        tl.store(token_scales + part_idx, encoded_scale.to(tl.uint8))
    else:
        tl.store(token_scales + 7, tl.zeros((), dtype=tl.uint8))
        bf16_out = (token_data + fp8_dim).to(tl.pointer_type(tl.bfloat16))
        values = tl.load(input_row + fp8_dim + offsets)
        tl.store(bf16_out + offsets, values)


def _candidate(
    k: torch.Tensor,
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
    num_warps: int,
) -> None:
    cache_2d = cache.view(cache.shape[0], -1)
    _parallel_quantize_and_insert_k_kernel[(slot_mapping.shape[0], 8)](
        k,
        slot_mapping,
        cache_2d,
        slot_mapping.shape[0],
        input_dim=_HEAD_DIM,
        cache_block_size=block_size,
        block_stride=cache_2d.stride(0),
        token_data_size=_TOKEN_DATA_SIZE,
        fp8_dim=_FP8_DIM,
        scale_dim=_SCALE_DIM,
        quant_block=_QUANT_BLOCK,
        fp8_max=448.0,
        num_warps=num_warps,
    )


def _capture(fn) -> torch.cuda.CUDAGraph:
    for _ in range(4):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    graph.replay()
    torch.cuda.synchronize()
    return graph


def _time_graph(graph: torch.cuda.CUDAGraph, replays: int, repeats: int) -> list[float]:
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / replays)
    return samples


def _measure(args: argparse.Namespace, seed: int, num_warps: int) -> dict[str, object]:
    torch.manual_seed(seed)
    device = torch.device("cuda")
    values = torch.randn(
        (args.num_tokens, _HEAD_DIM), dtype=torch.float16, device=device
    )
    num_blocks = 3
    slot_mapping = torch.arange(args.num_tokens, dtype=torch.int64, device=device)
    slot_mapping += args.block_size + 3
    reference_cache = torch.zeros(
        (num_blocks, args.block_size, _HEAD_BYTES),
        dtype=torch.uint8,
        device=device,
    )
    candidate_cache = torch.zeros_like(reference_cache)

    def reference_call() -> None:
        quantize_and_insert_k_cache(
            values,
            reference_cache.view(num_blocks, -1),
            slot_mapping,
            block_size=args.block_size,
        )

    def candidate_call() -> None:
        _candidate(values, candidate_cache, slot_mapping, args.block_size, num_warps)

    reference_call()
    candidate_call()
    torch.cuda.synchronize()
    initial_equal = torch.equal(reference_cache, candidate_cache)
    differing_bytes = int(
        torch.count_nonzero(reference_cache != candidate_cache).item()
    )

    reference_graph = _capture(reference_call)
    candidate_graph = _capture(candidate_call)
    reference_samples = _time_graph(reference_graph, args.replays, args.repeats)
    candidate_samples = _time_graph(candidate_graph, args.replays, args.repeats)
    graph_equal = torch.equal(reference_cache, candidate_cache)

    reference_median = statistics.median(reference_samples)
    candidate_median = statistics.median(candidate_samples)
    result = {
        "seed": seed,
        "num_tokens": args.num_tokens,
        "block_size": args.block_size,
        "num_warps": num_warps,
        "reference_samples_ms": reference_samples,
        "candidate_samples_ms": candidate_samples,
        "reference_median_ms": reference_median,
        "candidate_median_ms": candidate_median,
        "speedup": reference_median / candidate_median,
        "projected_43_layer_saving_ms": (reference_median - candidate_median) * 43,
        "initial_bitwise_equal": initial_equal,
        "graph_bitwise_equal": graph_equal,
        "differing_bytes": differing_bytes,
    }
    if not initial_equal or not graph_equal:
        raise RuntimeError(f"KV insert parity failed: {result}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-tokens", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--warps", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--seeds", type=int, nargs="+", default=[103, 107, 109])
    parser.add_argument("--replays", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        raise RuntimeError("This benchmark requires an NVIDIA V100 (SM70).")

    results = [
        _measure(args, seed, num_warps)
        for seed in args.seeds
        for num_warps in args.warps
    ]
    payload = {
        "contract": {
            "model": "DeepSeek-V4-Flash",
            "tp": 8,
            "calls_per_token": 43,
            "dtype": "torch.float16",
            "fp8_format": "E4M3FN with UE8M0 block scales",
            "cuda_graph": True,
        },
        "results": results,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
