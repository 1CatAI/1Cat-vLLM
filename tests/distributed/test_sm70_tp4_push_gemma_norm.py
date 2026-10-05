# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _check_push_norm(rank: int, rendezvous: str) -> None:
    from vllm.distributed.device_communicators.custom_all_reduce import (
        CustomAllreduce,
    )
    from vllm.model_executor.layers.layernorm import (
        _sm70_dflash2_gemma_fused_add_rms_norm,
    )

    torch.accelerator.set_device_index(rank)
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=4,
        init_method=rendezvous,
        timeout=timedelta(seconds=120),
    )
    ca = CustomAllreduce(
        dist.group.WORLD, torch.device("cuda", rank), max_size=1024 * 1024
    )
    assert not ca.disabled and ca.fully_connected and ca.sm70_tp4_push_buffer_ptrs
    codes = torch.empty(272 * 320 * 256, device="cuda", dtype=torch.uint8)
    try:
        for dtype in (torch.float16, torch.float32):
            x = torch.zeros(8, 5120, device="cuda", dtype=torch.float16)
            residual = torch.zeros_like(x, dtype=torch.float32)
            weight = torch.zeros(5120, device="cuda", dtype=dtype)
            ca.custom_all_reduce(x)
            ca.sm70_tp4_all_reduce_gemma_rms_norm(x, residual, weight, 1e-6)
            torch.accelerator.synchronize()
            dist.barrier()
            graphs, outputs = [], []
            for route in ("unfused", "fused", "prefetch"):
                graph = torch.cuda.CUDAGraph()
                with ca.capture(), torch.cuda.graph(graph):
                    for _ in range(140):
                        if route != "unfused":
                            output = ca.sm70_tp4_all_reduce_gemma_rms_norm(
                                x,
                                residual,
                                weight,
                                1e-6,
                                codes if route == "prefetch" else None,
                            )
                            # Separate packets must survive interleaved ordinary AR.
                            ca.custom_all_reduce(x)
                        else:
                            output = _sm70_dflash2_gemma_fused_add_rms_norm(
                                ca.custom_all_reduce(x), residual, weight, 1e-6
                            )
                graphs.append(graph)
                outputs.append(output)
            for cycle in range(8):
                torch.manual_seed(3104 + cycle * 4 + rank)
                x.normal_().mul_(0.125)
                torch.manual_seed(4104 + cycle)
                residual.normal_().mul_(0.125)
                weight.normal_().mul_(0.05)
                if cycle == 0:
                    x.zero_()
                    residual.zero_()
                dist.barrier()
                for graph in graphs:
                    if rank == cycle % 4:
                        torch.cuda._sleep(10000)
                    # Exercise many generation/epoch transitions after each
                    # live input update, including intentionally skewed ranks.
                    for _ in range(61):
                        graph.replay()
                torch.accelerator.synchronize()
                _, expected_residual = outputs[0]
                actual, actual_residual = outputs[1]
                hinted, hinted_residual = outputs[2]
                assert torch.equal(hinted.view(torch.uint8), actual.view(torch.uint8))
                assert torch.equal(hinted_residual, actual_residual)
                assert torch.equal(actual_residual, expected_residual)
                values = expected_residual.double()
                oracle = values * torch.rsqrt(
                    values.square().mean(-1, keepdim=True) + 1e-6
                )
                oracle = (oracle * (weight.double() + 1)).half()
                lo = torch.nextafter(oracle, torch.full_like(oracle, -float("inf")))
                hi = torch.nextafter(oracle, torch.full_like(oracle, float("inf")))
                assert bool(torch.all((actual >= lo) & (actual <= hi)))
                eager, eager_residual = ca.sm70_tp4_all_reduce_gemma_rms_norm(
                    x, residual, weight, 1e-6
                )
                torch.accelerator.synchronize()
                assert torch.equal(eager, actual)
                assert torch.equal(eager_residual, actual_residual)
            del graphs, outputs
    finally:
        ca.close()
        dist.destroy_process_group()


def test_sm70_tp4_push_norm_fp64_and_interleaved_replay(tmp_path):
    if torch.accelerator.device_count() != 4 or torch.cuda.get_device_capability() != (
        7,
        0,
    ):
        pytest.skip("requires an isolated full NVLink mesh of four V100 GPUs")
    mp.spawn(
        _check_push_norm,
        args=((tmp_path / "push-norm-rendezvous").as_uri(),),
        nprocs=4,
        join=True,
    )
