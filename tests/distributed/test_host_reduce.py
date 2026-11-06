# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The host-memory two-rank all-reduce gives NCCL's bits, eagerly and in a graph,
and in its copy-engine mode for prefill-sized payloads."""

import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import vllm.distributed.device_communicators.host_reduce as host_reduce
from vllm.config.kernel import HostReduceConfig
from vllm.distributed.device_communicators.host_reduce import (
    HostReduceCommunicator,
)

SIZES = [8, 4096, 5120, 3 * 5120, 8 * 5120, 131072]
# Copy-engine mode: one piece, several pieces, 16 pieces, an uneven last piece.
COPY_SIZES = [5120 * 30, 5120 * 100, 5120 * 801, 5120 * 1600, 5120 * 2048]
COPY_MAX_BYTES = 5120 * 2048 * 2


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


def _copy_engine_worker(rank: int, port: int, failures) -> None:
    os.environ["NCCL_P2P_DISABLE"] = "1"
    torch.accelerator.set_device_index(rank)
    device = torch.device("cuda", rank)
    init = f"tcp://127.0.0.1:{port}"
    dist.init_process_group("nccl", init_method=init, rank=rank, world_size=2)
    cpu_group = dist.new_group(backend="gloo")
    policy = HostReduceConfig(copy_engine_max_bytes=COPY_MAX_BYTES)
    host_reduce.get_current_vllm_config_or_none = lambda: SimpleNamespace(
        kernel_config=SimpleNamespace(
            host_reduce=policy, collective_kernel_selections={}
        )
    )
    comm = HostReduceCommunicator(cpu_group, device, "tp:0")
    try:
        assert comm.status["copy_engine_max_bytes"] == COPY_MAX_BYTES
        generator = torch.Generator(device=device).manual_seed(4321 + rank)
        # Back to back without synchronizing, mixed with the kernel mode, on a
        # side stream while the default stream computes: slots are reused by
        # parity only after the peer acknowledged reading them.
        side = torch.cuda.Stream(device)
        inputs, outputs = [], []
        a = torch.randn(4096, 4096, device=device, dtype=torch.float16)
        for repeat in range(3):
            for size in COPY_SIZES + [5120]:
                x = torch.randn(size, generator=generator, device=device).half()
                inputs.append(x)
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    got = comm.all_reduce(x)
                    assert got is not None
                outputs.append(got)
                a = a @ a.T * 1e-3
        torch.accelerator.synchronize()
        for x, got in zip(inputs, outputs):
            expected = x.clone()
            dist.all_reduce(expected)
            assert torch.equal(got, expected), x.numel()
        too_large = torch.ones(COPY_MAX_BYTES // 2 + 8, device=device).half()
        assert comm.all_reduce(too_large) is None
        # A captured payload above max_bytes is rejected, so the caller's
        # next collective serves it; the capture itself must not fail.
        graph = torch.cuda.CUDAGraph()
        static = torch.ones(5120 * 64, device=device).half()
        with torch.cuda.graph(graph):
            assert comm.rejection_reason(static) == "copy_engine_mode_not_capturable"
            assert comm.all_reduce(static) is None
        assert comm.rejection_reason(static) is None
    except Exception as exc:  # noqa: BLE001 - reported to the parent
        failures.put(f"rank {rank}: {exc!r}")
        raise
    finally:
        comm.close()
        dist.destroy_process_group()


@pytest.mark.skipif(
    torch.accelerator.device_count() < 2
    or not hasattr(torch.ops._C, "host_reduce_copy_engine_out"),
    reason="needs two GPUs and the native copy-engine host all-reduce",
)
def test_host_reduce_copy_engine_matches_nccl():
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    ctx = mp.get_context("spawn")
    failures = ctx.Queue()
    mp.spawn(_copy_engine_worker, args=(port, failures), nprocs=2, join=True)
    assert failures.empty()


def test_copy_engine_limit_must_exceed_the_kernel_limit():
    # The copy-engine mode serves payloads above max_bytes only.
    HostReduceConfig(max_bytes=262144, copy_engine_max_bytes=0)
    HostReduceConfig(max_bytes=262144, copy_engine_max_bytes=262160)
    with pytest.raises(ValueError, match="copy_engine_max_bytes"):
        HostReduceConfig(max_bytes=262144, copy_engine_max_bytes=262144)
