# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _check_medium_reduce(rank: int, rendezvous: str) -> None:
    from vllm import envs
    from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce

    torch.accelerator.set_device_index(rank)
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=4,
        init_method=rendezvous,
        timeout=timedelta(seconds=120),
    )
    communicator = CustomAllreduce(dist.group.WORLD, rank, max_size=1024 * 1024)
    assert not communicator.disabled and communicator.fully_connected
    try:
        for dtype in (torch.float16, torch.float32):
            for size in (393200, 393216, 491520, 524272, 524288, 655360):
                element_bytes = torch.tensor([], dtype=dtype).element_size()
                inputs = [
                    torch.empty(size // element_bytes, dtype=dtype, device="cuda")
                    for _ in range(4)
                ]
                outputs, guards, graphs = [], [], []
                # The diagnostic override retains the original sum order on
                # either side of the 512-KiB algorithm boundary.
                for reference in (True, False):
                    if reference:
                        os.environ["VLLM_CUSTOM_ALLREDUCE_ALGO"] = (
                            "1stage" if size < 524288 else "2stage"
                        )
                    else:
                        os.environ.pop("VLLM_CUSTOM_ALLREDUCE_ALGO", None)
                    envs.disable_envs_cache()
                    storage = [
                        torch.full((x.numel() + 16,), 123.0, dtype=dtype, device="cuda")
                        for x in inputs
                    ]
                    out = [x[8:-8] for x in storage]
                    graph = torch.cuda.CUDAGraph()
                    torch.accelerator.synchronize()
                    dist.barrier()
                    with communicator.capture(), torch.cuda.graph(graph):
                        for _ in range(8):
                            for x, y in zip(inputs, out):
                                communicator.all_reduce(x, out=y, registered=True)
                    outputs.append(out)
                    guards.append(storage)
                    graphs.append(graph)
                for cycle in range(4):
                    torch.manual_seed(371 + rank + cycle * 11)
                    for x in inputs:
                        if cycle == 2:
                            # A different rank-dependent accumulation order
                            # loses the small term in this cancellation case.
                            x.fill_((65504.0, 0.0001, -65504.0, 0.03125)[rank])
                            x[1::2].mul_(-1)
                        else:
                            x.normal_().mul_(0.03 * (cycle + 1))
                    dist.barrier()
                    for graph in graphs:
                        if rank == cycle:
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
                    # Also validate the uncaptured registered-buffer path.
                    os.environ["VLLM_CUSTOM_ALLREDUCE_ALGO"] = (
                        "1stage" if size < 524288 else "2stage"
                    )
                    expected = communicator.all_reduce(inputs[0], registered=False)
                    os.environ.pop("VLLM_CUSTOM_ALLREDUCE_ALGO", None)
                    actual = communicator.all_reduce(inputs[0], registered=False)
                    torch.accelerator.synchronize()
                    assert torch.equal(
                        actual.view(torch.uint8), expected.view(torch.uint8)
                    )
                del graphs, outputs, guards, inputs
    finally:
        communicator.close()
        dist.destroy_process_group()


def test_sm70_medium_all_reduce_preserves_order_and_replay(tmp_path, monkeypatch):
    if torch.accelerator.device_count() != 4 or torch.cuda.get_device_capability() != (
        7,
        0,
    ):
        pytest.skip("requires an isolated NVLink-connected group of four V100 GPUs")
    for name in (
        "VLLM_CUSTOM_ALLREDUCE_ALGO",
        "VLLM_CUSTOM_ALLREDUCE_BLOCK_LIMIT",
        "VLLM_SM70_TP4_M5_AR_THREADS",
        "VLLM_SM70_TP4_MTP_AR_BLOCK_TUNING",
    ):
        monkeypatch.delenv(name, raising=False)
    rendezvous = (tmp_path / "medium-reduce-rendezvous").as_uri()
    mp.spawn(_check_medium_reduce, args=(rendezvous,), nprocs=4, join=True)
