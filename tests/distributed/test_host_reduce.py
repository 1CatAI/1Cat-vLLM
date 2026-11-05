# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The host-memory two-rank all-reduce gives NCCL's bits, eagerly and in a graph."""

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.distributed.device_communicators.host_reduce import (
    HostReduceCommunicator,
)

SIZES = [8, 4096, 5120, 3 * 5120, 8 * 5120, 131072]


def _worker(rank: int, port: int, failures) -> None:
    os.environ["NCCL_P2P_DISABLE"] = "1"
    torch.accelerator.set_device_index(rank)
    device = torch.device("cuda", rank)
    init = f"tcp://127.0.0.1:{port}"
    dist.init_process_group("nccl", init_method=init, rank=rank, world_size=2)
    cpu_group = dist.new_group(backend="gloo")
    comm = HostReduceCommunicator(cpu_group, device, "tp:0")
    try:
        assert comm.status["enabled"], comm.status
        generator = torch.Generator(device=device).manual_seed(1234 + rank)
        for size in SIZES:
            for _ in range(3):
                x = torch.randn(size, generator=generator, device=device)
                x = x.half()
                expected = x.clone()
                dist.all_reduce(expected)
                got = comm.all_reduce(x)
                assert got is not None
                assert torch.equal(got, expected), size
        static = torch.randn(5120, generator=generator, device=device).half()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_out = comm.all_reduce(static)
        for _ in range(4):
            static.copy_(torch.randn(5120, device=device).half())
            expected = static.clone()
            dist.all_reduce(expected)
            graph.replay()
            torch.accelerator.synchronize()
            assert torch.equal(graph_out, expected)
        assert comm.all_reduce(torch.ones(3, device=device).half()) is None
        assert comm.all_reduce(torch.ones(8, device=device)) is None
    except Exception as exc:  # noqa: BLE001 - reported to the parent
        failures.put(f"rank {rank}: {exc!r}")
        raise
    finally:
        comm.close()
        dist.destroy_process_group()


@pytest.mark.skipif(
    torch.accelerator.device_count() < 2
    or not hasattr(torch.ops._C, "host_reduce_open"),
    reason="needs two GPUs and the native host all-reduce",
)
def test_host_reduce_matches_nccl():
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    ctx = mp.get_context("spawn")
    failures = ctx.Queue()
    mp.spawn(_worker, args=(port, failures), nprocs=2, join=True)
    assert failures.empty()
