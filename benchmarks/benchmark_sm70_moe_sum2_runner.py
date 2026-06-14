# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate the SM70 MoERunner shared+routed all_reduce_sum2 hook.

Run with torchrun, for example:
  VLLM_SM70_MOE_ADD_ALLREDUCE=1 CUDA_VISIBLE_DEVICES=0,1 \
    torchrun --nproc_per_node=2 benchmarks/benchmark_sm70_moe_sum2_runner.py
"""

import argparse
import json
import os
from types import SimpleNamespace
from typing import Any

import torch

from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.distributed.communication_op import tensor_model_parallel_all_reduce
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    graph_capture,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner


def _dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    raise ValueError(f"unsupported dtype: {name}")


def _make_inputs(
    tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    rank: int,
    pattern: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    base = torch.arange(tokens * hidden_size, device="cuda", dtype=torch.float32)
    base = base.reshape(tokens, hidden_size)
    if pattern == "exact_int":
        shared = ((base % 13) + rank).to(dtype)
        routed = (((base * 3) % 17) - rank).to(dtype)
    elif pattern == "random_small":
        generator = torch.Generator(device="cuda")
        generator.manual_seed(20260602 + rank)
        shared = (
            torch.rand(
                (tokens, hidden_size),
                device="cuda",
                generator=generator,
            )
            - 0.5
        ).to(dtype)
        routed = (
            torch.rand(
                (tokens, hidden_size),
                device="cuda",
                generator=generator,
            )
            - 0.5
        ).to(dtype)
    else:
        raise ValueError(f"unsupported input pattern: {pattern}")
    return shared, routed


def _make_runner(world_size: int) -> MoERunner:
    runner = object.__new__(MoERunner)
    runner.moe_config = SimpleNamespace(
        is_sequence_parallel=False,
        tp_size=world_size,
        ep_size=1,
        pcp_size=1,
    )
    runner._quant_method = SimpleNamespace(moe_kernel=None)
    runner.layer_name = "sm70_synthetic_moe_sum2"
    return runner


def _expected_route_hit(world_size: int, num_bytes: int) -> bool:
    del num_bytes
    return world_size in (2, 4, 6, 8)


def _check_one(
    runner: MoERunner,
    world_size: int,
    tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    rank: int,
    pattern: str,
) -> dict[str, Any]:
    shared, routed = _make_inputs(tokens, hidden_size, dtype, rank, pattern)
    expected_route_hit = _expected_route_hit(
        world_size,
        shared.numel() * shared.element_size(),
    )
    torch.cuda.synchronize()

    device = torch.device("cuda", torch.cuda.current_device())
    with graph_capture(device=device) as graph_capture_context:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=graph_capture_context.stream):
            hook_out = MoERunner._maybe_sm70_moe_sum2_allreduce(
                runner,
                shared,
                routed,
                hidden_size,
            )
            ref_out = tensor_model_parallel_all_reduce(shared + routed)
    graph.replay()
    torch.cuda.synchronize()

    if hook_out is None:
        return {
            "tokens": tokens,
            "hidden_size": hidden_size,
            "dtype": str(dtype).replace("torch.", ""),
            "pattern": pattern,
            "route_hit": False,
            "expected_route_hit": expected_route_hit,
            "equal": False,
            "max_diff": None,
            "mean_diff": None,
            "skipped_as_expected": not expected_route_hit,
        }

    diff = (hook_out.float() - ref_out.float()).abs()
    return {
        "tokens": tokens,
        "hidden_size": hidden_size,
        "dtype": str(dtype).replace("torch.", ""),
        "pattern": pattern,
        "route_hit": True,
        "expected_route_hit": expected_route_hit,
        "equal": bool(torch.equal(hook_out, ref_out)),
        "max_diff": float(diff.max().item()),
        "mean_diff": float(diff.mean().item()),
        "skipped_as_expected": False,
        "hook_checksum": float(hook_out.float().sum().item()),
        "ref_checksum": float(ref_out.float().sum().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", default="1,2,16")
    parser.add_argument("--hidden-size", type=int, default=8192)
    parser.add_argument("--dtypes", default="float32,float16")
    parser.add_argument("--patterns", default="exact_int,random_small")
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    torch.set_default_device(torch.device("cuda", local_rank))

    config = VllmConfig(
        parallel_config=ParallelConfig(tensor_parallel_size=world_size),
    )
    results: list[dict[str, Any]] = []
    with set_current_vllm_config(config):
        try:
            init_distributed_environment()
            initialize_model_parallel(tensor_model_parallel_size=world_size)
            runner = _make_runner(world_size)

            tokens_list = [int(item) for item in args.tokens.split(",") if item]
            dtypes = [_dtype(item) for item in args.dtypes.split(",") if item]
            patterns = [item for item in args.patterns.split(",") if item]
            for dtype in dtypes:
                for tokens in tokens_list:
                    for pattern in patterns:
                        results.append(
                            _check_one(
                                runner,
                                world_size,
                                tokens,
                                args.hidden_size,
                                dtype,
                                rank,
                                pattern,
                            )
                        )
            ok = all(
                (item["route_hit"] and item["equal"])
                if item["expected_route_hit"]
                else (not item["route_hit"])
                for item in results
            )
            result = {
                "world_size": world_size,
                "rank": rank,
                "all_equal": ok,
                "results": results,
            }
            if rank == 0:
                print(json.dumps(result, sort_keys=True))
            raise SystemExit(0 if ok else 1)
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()


if __name__ == "__main__":
    main()
