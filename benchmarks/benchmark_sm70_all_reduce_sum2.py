# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate SM70 custom all_reduce_sum2 against all_reduce(a + b).

Run with torchrun, for example:
  CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
    benchmarks/benchmark_sm70_all_reduce_sum2.py
"""

import argparse
import json
import os
from typing import Any

import torch
import torch.distributed as dist

from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce


def _dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    raise ValueError(f"unsupported dtype: {name}")


def _make_inputs(
    size: int,
    dtype: torch.dtype,
    rank: int,
    pattern: str,
) -> tuple[torch.Tensor, ...]:
    base = torch.arange(size, device="cuda", dtype=torch.float32)
    if pattern == "exact_int":
        # Small integer-valued tensors make bitwise equality meaningful for
        # fp16 too, but they do not exercise realistic rounding.
        a = ((base % 13) + rank).to(dtype)
        b = (((base * 3) % 17) - rank).to(dtype)
    elif pattern == "random_small":
        generator = torch.Generator(device="cuda")
        generator.manual_seed(20260602 + rank)
        a = (torch.rand(size, device="cuda", generator=generator) - 0.5).to(dtype)
        b = (torch.rand(size, device="cuda", generator=generator) - 0.5).to(dtype)
    else:
        raise ValueError(f"unsupported input pattern: {pattern}")
    return a, b


def _compare(
    out: torch.Tensor,
    ref: torch.Tensor,
    reference: str,
) -> dict[str, Any]:
    diff = (out.float() - ref.float()).abs()
    return {
        "reference": reference,
        "equal": bool(torch.equal(out, ref)),
        "max_diff": float(diff.max().item()),
        "mean_diff": float(diff.mean().item()),
        "out_checksum": float(out.float().sum().item()),
        "ref_checksum": float(ref.float().sum().item()),
    }


def _run_sum2_vs_custom_ar(
    ca: CustomAllreduce,
    a: torch.Tensor,
    b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    sum2_out = torch.empty_like(a)
    ref_out = torch.empty_like(a)

    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with ca.capture(), torch.cuda.graph(graph):
        local_sum = a + b
        ca.all_reduce(local_sum, out=ref_out, registered=True)
        ca.all_reduce_sum2(a, b, out=sum2_out)
    dist.barrier()
    graph.replay()
    torch.cuda.synchronize()
    return sum2_out, ref_out


def _check_one(
    ca: CustomAllreduce,
    size: int,
    dtype: torch.dtype,
    rank: int,
    pattern: str,
    references: list[str],
) -> dict[str, Any]:
    a, b = _make_inputs(size, dtype, rank, pattern)
    out, custom_ref = _run_sum2_vs_custom_ar(ca, a, b)
    comparisons: list[dict[str, Any]] = []
    if "custom_ar" in references:
        comparisons.append(_compare(out, custom_ref, "custom_ar"))
    if "nccl" in references:
        nccl_ref = a + b
        dist.all_reduce(nccl_ref, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        comparisons.append(_compare(out, nccl_ref, "nccl"))

    return {
        "size": size,
        "dtype": str(dtype).replace("torch.", ""),
        "pattern": pattern,
        "comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="1024,8192")
    parser.add_argument("--dtypes", default="float32,float16")
    parser.add_argument("--patterns", default="exact_int,random_small")
    parser.add_argument("--references", default="custom_ar,nccl")
    parser.add_argument(
        "--required-references",
        default="",
        help="Comma-separated references that must be bitwise equal. "
        "Defaults to all references.",
    )
    parser.add_argument("--max-size-bytes", type=int, default=8 * 1024 * 1024)
    args = parser.parse_args()

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
    sizes = [int(item) for item in args.sizes.split(",") if item]
    dtypes = [_dtype(item) for item in args.dtypes.split(",") if item]
    patterns = [item for item in args.patterns.split(",") if item]
    references = [item for item in args.references.split(",") if item]
    required_references = [
        item for item in args.required_references.split(",") if item
    ] or references
    results: list[dict[str, Any]] = []
    try:
        if ca.disabled:
            result = {
                "world_size": world_size,
                "rank": rank,
                "custom_allreduce_disabled": True,
            }
            if rank == 0:
                print(json.dumps(result, sort_keys=True))
            raise SystemExit(2)

        for dtype in dtypes:
            for size in sizes:
                for pattern in patterns:
                    results.append(
                        _check_one(ca, size, dtype, rank, pattern, references)
                    )
        ok = all(
            comparison["equal"]
            for item in results
            for comparison in item["comparisons"]
            if comparison["reference"] in required_references
        )
        result = {
            "world_size": world_size,
            "rank": rank,
            "custom_allreduce_disabled": False,
            "all_equal": ok,
            "required_references": required_references,
            "results": results,
        }
        if rank == 0:
            print(json.dumps(result, sort_keys=True))
        raise SystemExit(0 if ok else 1)
    finally:
        ca.close()
        dist.destroy_process_group(gloo_group)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
