# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _check_large_push(rank: int, rendezvous: str) -> None:
    from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce

    torch.accelerator.set_device_index(rank)
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=4,
        init_method=rendezvous,
        timeout=timedelta(seconds=180),
    )
    communicator = CustomAllreduce(dist.group.WORLD, rank, max_size=1024 * 1024)
    assert not communicator.disabled and communicator.fully_connected
    try:
        # Include partial batches and alternating packet lengths: the same
        # persistent two-epoch buffer must not retain a previous graph's tail.
        shapes = [(rows, 5120) for rows in (8, 16, 32, 33, 40, 48, 56, 64, 65)]
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            inputs = [
                torch.empty(shape, dtype=dtype, device="cuda") for shape in shapes
            ]
            graphs, outputs, guards = [], [], []
            for enabled in (False, True):
                os.environ["VLLM_SM70_TP4_PUSH_ALLREDUCE_CONCURRENCY"] = str(
                    int(enabled)
                )
                storage = [
                    torch.full((x.numel() + 16,), 123.0, dtype=dtype, device="cuda")
                    for x in inputs
                ]
                out = [y[8:-8].view_as(x) for x, y in zip(inputs, storage)]
                graph = torch.cuda.CUDAGraph()
                torch.accelerator.synchronize()
                dist.barrier()
                with communicator.capture(), torch.cuda.graph(graph):
                    for _ in range(4):
                        for x, y in zip(inputs, out):
                            communicator.all_reduce(x, out=y, registered=True)
                graphs.append(graph)
                outputs.append(out)
                guards.append(storage)
            for cycle in range(12):
                torch.manual_seed(317 + rank + 11 * cycle)
                for x in inputs:
                    if cycle == 5:
                        x.fill_((65504.0, 0.0001, -65504.0, 0.03125)[rank])
                        x.flatten()[1::2].mul_(-1)
                    else:
                        x.normal_().mul_(0.03 * (cycle + 1))
                dist.barrier()
                for graph in graphs:
                    if rank == cycle % 4:
                        torch.cuda._sleep(10000)
                    graph.replay()
                torch.accelerator.synchronize()
                for actual, expected in zip(outputs[1], outputs[0]):
                    assert torch.equal(
                        actual.view(torch.uint8), expected.view(torch.uint8)
                    )
                for group in guards:
                    for storage in group:
                        assert bool(torch.all(storage[:8] == 123))
                        assert bool(torch.all(storage[-8:] == 123))
    finally:
        communicator.close()
        dist.destroy_process_group()


def test_large_push_preserves_order_tail_and_graph_replay(tmp_path, monkeypatch):
    if torch.accelerator.device_count() != 4 or torch.cuda.get_device_capability() != (
        7,
        0,
    ):
        pytest.skip("requires an isolated NVLink-connected group of four V100 GPUs")
    monkeypatch.setenv("VLLM_SM70_TP4_PUSH_ALLREDUCE", "1")
    for name in (
        "VLLM_CUSTOM_ALLREDUCE_ALGO",
        "VLLM_SM70_TP4_PUSH_ALLREDUCE_SMALL_MESSAGES",
    ):
        monkeypatch.delenv(name, raising=False)
    rendezvous = (tmp_path / "large-push-rendezvous").as_uri()
    mp.spawn(_check_large_push, args=(rendezvous,), nprocs=4, join=True)
