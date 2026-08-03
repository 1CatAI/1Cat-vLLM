# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Microbenchmark the DeepSeek V4 SM70 sparse decode attention kernel."""

import argparse
import json
import math
import statistics
from pathlib import Path

import torch

from vllm.models.deepseek_v4.common.ops.fp8_software import (
    fp8_e4m3fn_bits_to_fp32,
    fp8_e4m3fn_bits_to_fp32_bitcast,
)
from vllm.models.deepseek_v4.sm70.sparse_kernels import (
    sm70_sparse_attention_paged_fp8,
    sm70_sparse_attention_paged_fp8_splitk,
    sm70_sparse_attention_paged_fp8_splitk_qk_dsplit,
)
from vllm.triton_utils import tl, triton


@triton.jit
def _validate_fp8_decode_kernel(bits_ptr, arithmetic_ptr, bitcast_ptr):
    offsets = tl.arange(0, 256)
    bits = tl.load(bits_ptr + offsets)
    tl.store(arithmetic_ptr + offsets, fp8_e4m3fn_bits_to_fp32(bits))
    tl.store(bitcast_ptr + offsets, fp8_e4m3fn_bits_to_fp32_bitcast(bits))


def _make_cache(num_rows: int, block_size: int, device: torch.device) -> torch.Tensor:
    num_blocks = math.ceil(num_rows / block_size)
    cache = torch.zeros(
        (num_blocks, block_size * 584), dtype=torch.uint8, device=device
    )
    data = cache[:, : block_size * 576].view(num_blocks, block_size, 576)
    fp8 = torch.randn(
        (num_blocks, block_size, 448), dtype=torch.float32, device=device
    ).to(torch.float8_e4m3fn)
    data[:, :, :448].copy_(fp8.view(torch.uint8))
    rope = 0.125 * torch.randn(
        (num_blocks, block_size, 64), dtype=torch.bfloat16, device=device
    )
    data[:, :, 448:576].copy_(rope.view(torch.uint8).reshape_as(data[:, :, 448:576]))
    scales = cache[:, block_size * 576 :].view(num_blocks, block_size, 8)
    scales.fill_(124)
    return cache.view(num_blocks, block_size, 584)


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = min(round((len(ordered) - 1) * q), len(ordered) - 1)
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("c4", "c128", "swa"), default="c4")
    parser.add_argument(
        "--implementation",
        choices=("baseline", "splitk", "splitk-qk-dsplit"),
        default="baseline",
    )
    parser.add_argument("--extra-len", type=int)
    parser.add_argument("--extra-width", type=int)
    parser.add_argument("--main-len", type=int, default=128)
    parser.add_argument("--main-width", type=int)
    parser.add_argument("--num-tokens", type=int, default=1)
    parser.add_argument("--stage1-block-h", type=int, choices=(4, 8), default=8)
    parser.add_argument("--stage1-warps", type=int, choices=(2, 4, 8), default=4)
    parser.add_argument("--qk-block-d", type=int, choices=(32, 64, 128), default=64)
    parser.add_argument("--seed", type=int, default=4111)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--compare-baseline", action="store_true")
    parser.add_argument("--validate-fp8-decode", action="store_true")
    parser.add_argument("--save-output", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    capability = torch.cuda.get_device_capability()
    if capability != (7, 0):
        raise RuntimeError(f"Expected an SM70 device, got {capability}")

    if args.validate_fp8_decode:
        bits = torch.arange(256, dtype=torch.uint8, device="cuda")
        arithmetic = torch.empty(256, dtype=torch.float32, device="cuda")
        bitcast = torch.empty_like(arithmetic)
        _validate_fp8_decode_kernel[(1,)](bits, arithmetic, bitcast, num_warps=4)
        torch.cuda.synchronize()
        finite = torch.isfinite(arithmetic)
        finite_match = torch.equal(
            arithmetic[finite].view(torch.int32), bitcast[finite].view(torch.int32)
        )
        nan_match = torch.equal(torch.isnan(arithmetic), torch.isnan(bitcast))
        print(
            json.dumps(
                {
                    "finite_bitwise_match": finite_match,
                    "nan_mask_match": nan_match,
                    "finite_values": int(finite.sum().item()),
                },
                indent=2,
            )
        )
        return

    if args.compare_baseline and args.implementation == "baseline":
        raise ValueError("baseline comparison requires a split-K implementation")
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    num_heads = 8
    num_tokens = args.num_tokens
    main_len = args.main_len
    main_width = args.main_width if args.main_width is not None else main_len
    if main_width < main_len:
        raise ValueError("main width must cover main length")
    q = torch.randn((num_tokens, num_heads, 512), dtype=torch.float16, device=device)
    out = torch.empty_like(q)
    main_cache = _make_cache(main_len, 256, device)
    main_indices = torch.full(
        (num_tokens, main_width), -1, dtype=torch.int32, device=device
    )
    main_indices[:, :main_len] = torch.arange(
        main_len, dtype=torch.int32, device=device
    )[None, :]
    main_lengths = torch.full((num_tokens,), main_len, dtype=torch.int32, device=device)
    attn_sink = torch.randn((num_heads,), dtype=torch.float32, device=device)

    if args.case == "c4":
        extra_len = args.extra_len if args.extra_len is not None else 320
        extra_width = args.extra_width if args.extra_width is not None else 512
        extra_block_size = 64
    elif args.case == "c128":
        extra_len = args.extra_len if args.extra_len is not None else 10
        extra_width = args.extra_width if args.extra_width is not None else 32
        extra_block_size = 2
    else:
        extra_len = 0
        extra_width = 0
        extra_block_size = 1

    if extra_len:
        extra_cache = _make_cache(extra_len, extra_block_size, device)
        if extra_width < extra_len:
            raise ValueError("extra width must cover extra length")
        extra_indices = torch.full(
            (num_tokens, extra_width), -1, dtype=torch.int32, device=device
        )
        extra_indices[:, :extra_len] = torch.arange(
            extra_len, dtype=torch.int32, device=device
        )[None, :]
        extra_lengths = torch.full(
            (num_tokens,), extra_len, dtype=torch.int32, device=device
        )
    else:
        extra_cache = None
        extra_indices = None
        extra_lengths = None

    if args.implementation != "baseline":
        num_partials = math.ceil(main_indices.shape[-1] / 16)
        if extra_indices is not None:
            num_partials += math.ceil(extra_indices.shape[-1] / 16)
        partial_max = torch.empty(
            (q.shape[0], num_heads, num_partials),
            dtype=torch.float32,
            device=device,
        )
        partial_sum = torch.empty_like(partial_max)
        partial_acc = torch.empty(
            (q.shape[0], num_heads, num_partials, q.shape[-1]),
            dtype=torch.float32,
            device=device,
        )
        partial_probs = (
            torch.empty(
                (q.shape[0], num_heads, num_partials, 16),
                dtype=torch.float16,
                device=device,
            )
            if args.implementation == "splitk-qk-dsplit"
            else None
        )
        partial_qk = (
            torch.empty(
                (
                    q.shape[0],
                    num_heads,
                    num_partials,
                    q.shape[-1] // args.qk_block_d,
                    16,
                ),
                dtype=torch.float32,
                device=device,
            )
            if args.implementation == "splitk-qk-dsplit"
            else None
        )
    else:
        partial_max = None
        partial_sum = None
        partial_acc = None
        partial_probs = None
        partial_qk = None

    def run() -> None:
        if args.implementation != "baseline":
            assert partial_max is not None
            assert partial_sum is not None
            assert partial_acc is not None
            splitk_fn = {
                "splitk": sm70_sparse_attention_paged_fp8_splitk,
                "splitk-qk-dsplit": sm70_sparse_attention_paged_fp8_splitk_qk_dsplit,
            }[args.implementation]
            kwargs = dict(
                q=q,
                main_cache=main_cache,
                main_indices=main_indices,
                main_lengths=main_lengths,
                scale=512**-0.5,
                attn_sink=attn_sink,
                out=out,
                extra_cache=extra_cache,
                extra_indices=extra_indices,
                extra_lengths=extra_lengths,
                partial_max=partial_max,
                partial_sum=partial_sum,
                partial_acc=partial_acc,
                stage1_num_warps=args.stage1_warps,
                stage1_block_h=args.stage1_block_h,
            )
            if args.implementation == "splitk-qk-dsplit":
                assert partial_probs is not None
                kwargs["partial_probs"] = partial_probs
            if args.implementation == "splitk-qk-dsplit":
                assert partial_qk is not None
                kwargs["partial_qk"] = partial_qk
                kwargs["block_d"] = args.qk_block_d
            splitk_fn(**kwargs)
        else:
            sm70_sparse_attention_paged_fp8(
                q=q,
                main_cache=main_cache,
                main_indices=main_indices,
                main_lengths=main_lengths,
                scale=512**-0.5,
                attn_sink=attn_sink,
                out=out,
                extra_cache=extra_cache,
                extra_indices=extra_indices,
                extra_lengths=extra_lengths,
            )

    for _ in range(args.warmup):
        run()
    torch.cuda.synchronize()

    if args.cuda_graph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        run = graph.replay

    if args.profile:
        torch.cuda.cudart().cudaProfilerStart()
        run()
        torch.cuda.cudart().cudaProfilerStop()
        torch.cuda.synchronize()
        elapsed_ms = []
    else:
        elapsed_ms = []
        for _ in range(args.iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            run()
            end.record()
            end.synchronize()
            elapsed_ms.append(start.elapsed_time(end))

    torch.cuda.synchronize()
    comparison = None
    if args.compare_baseline:
        baseline_out = torch.empty_like(out)
        sm70_sparse_attention_paged_fp8(
            q=q,
            main_cache=main_cache,
            main_indices=main_indices,
            main_lengths=main_lengths,
            scale=512**-0.5,
            attn_sink=attn_sink,
            out=baseline_out,
            extra_cache=extra_cache,
            extra_indices=extra_indices,
            extra_lengths=extra_lengths,
        )
        run()
        torch.cuda.synchronize()
        difference = (baseline_out.float() - out.float()).abs()
        comparison = {
            "max_abs": difference.max().item(),
            "mean_abs": difference.mean().item(),
            "p99_abs": torch.quantile(difference.flatten(), 0.99).item(),
            "different_fp16": torch.count_nonzero(baseline_out != out).item(),
            "elements": out.numel(),
        }
    if args.save_output is not None:
        args.save_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(out.cpu(), args.save_output)

    result = {
        "case": args.case,
        "implementation": args.implementation,
        "seed": args.seed,
        "num_tokens": num_tokens,
        "stage1_block_h": args.stage1_block_h,
        "stage1_warps": args.stage1_warps,
        "qk_block_d": args.qk_block_d,
        "num_heads": num_heads,
        "main_len": main_len,
        "main_width": main_width,
        "extra_len": extra_len,
        "extra_width": extra_width,
        "main_block_size": 256,
        "extra_block_size": extra_block_size,
        "finite": bool(torch.isfinite(out).all().item()),
    }
    if comparison is not None:
        result["baseline_comparison"] = comparison
    if elapsed_ms:
        result.update(
            mean_ms=statistics.mean(elapsed_ms),
            p50_ms=_percentile(elapsed_ms, 0.50),
            p90_ms=_percentile(elapsed_ms, 0.90),
            p99_ms=_percentile(elapsed_ms, 0.99),
            min_ms=min(elapsed_ms),
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
