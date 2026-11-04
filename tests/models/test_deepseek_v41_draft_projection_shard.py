# SPDX-License-Identifier: Apache-2.0
"""Real TP shard/replicated parity, split-K mutation and drafter graph context.

Run under torchrun with 2/4/8 SM70 ranks and VLLM_V41_FP8_GEMV=1.
VLLM_V41K_LIB must name a built v41k extension. No replacement kernels.
"""

import os
from dataclasses import replace

import pytest
import torch

from vllm.models.deepseek_v41 import draft_projection
from vllm.models.deepseek_v41.fp8_linear import _gemv_ready, fp8_block32_linear


@pytest.fixture(scope="module")
def tp_group():
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world not in (2, 4, 8) or not torch.cuda.is_available():
        pytest.skip("torchrun with 2/4/8 SM70 GPUs required")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    from vllm.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
        tensor_model_parallel_all_gather,
    )

    init_distributed_environment(world, rank, "env://", local_rank, "nccl")
    initialize_model_parallel(tensor_model_parallel_size=world)
    try:
        ready = torch.tensor(
            int(torch.cuda.get_device_capability() == (7, 0) and _gemv_ready()),
            device="cuda",
        )
        torch.distributed.all_reduce(ready, op=torch.distributed.ReduceOp.MIN)
        if not ready.item():
            pytest.skip("every rank requires SM70 and a compatible v41k FP8 GEMV")
        yield rank, world, tensor_model_parallel_all_gather
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


def projection_inputs(m):
    """The same signed, finite FP8 inputs qualify parity and the mutant."""
    generator = torch.Generator(device="cuda").manual_seed(517)
    x = torch.randn(m, 15360, device="cuda", generator=generator).half() * 0.01
    raw = torch.randint(
        0, 256, (5120, 15360), device="cuda", dtype=torch.uint8, generator=generator
    )
    raw[(raw & 127) == 127] = 0
    weight = raw.view(torch.float8_e4m3fn)
    scales = torch.full((160, 480), 119, device="cuda", dtype=torch.uint8).view(
        torch.float8_e8m0fnu
    )
    return x, weight, scales


def collective_mismatch(actual, expected):
    mismatch = torch.tensor(
        int(not torch.equal(actual.view(torch.int16), expected.view(torch.int16))),
        device="cuda",
    )
    torch.distributed.all_reduce(mismatch, op=torch.distributed.ReduceOp.MAX)
    return bool(mismatch.item())


@pytest.mark.parametrize("m", range(1, 9))
def test_real_projection_shards_and_graph_replay(tp_group, m):
    from vllm.distributed import graph_capture
    from vllm.platforms import current_platform

    rank, world, gather = tp_group
    x, weight, scales = projection_inputs(m)
    assert draft_projection.projection_plan(5120, 15360, m, world, rank).split == 2

    def sharded():
        return draft_projection.sharded_projection(
            x, weight, scales, rank, world, gather
        )

    expected = fp8_block32_linear(x, weight, scales)
    actual = sharded()
    # Every rank reaches the same decision before anyone enters another collective.
    assert not collective_mismatch(actual, expected), "eager shard parity failed"
    sharded()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with (
        graph_capture(device=x.device) as context,
        torch.cuda.graph(
            graph, pool=current_platform.get_global_graph_pool(), stream=context.stream
        ),
    ):
        captured = sharded()
    x.mul_(0.5)
    expected = fp8_block32_linear(x, weight, scales)
    graph.replay()
    assert not collective_mismatch(captured, expected), "graph replay parity failed"


def test_wrong_local_split_is_detected(tp_group, monkeypatch):
    rank, world, gather = tp_group
    original_plan = draft_projection.projection_plan

    def local_split(n, k, m, size, local_rank):
        plan = original_plan(n, k, m, size, local_rank)
        local_ctas = ((n // size) + 31) // 32
        return replace(
            plan, split=max(1, min(k // 512, (240 + local_ctas - 1) // local_ctas))
        )

    monkeypatch.setattr(draft_projection, "projection_plan", local_split)
    detected = False
    for m in range(1, 9):
        x, weight, scales = projection_inputs(m)
        expected = fp8_block32_linear(x, weight, scales)
        mutated = draft_projection.sharded_projection(
            x, weight, scales, rank, world, gather
        )
        detected = collective_mismatch(mutated, expected) or detected
    assert detected, "INCONCLUSIVE: parity inputs did not detect the split-K mutation"
