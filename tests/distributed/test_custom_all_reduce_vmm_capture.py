# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capture custom all-reduce on inputs allocated before the capture window.

Under ``expandable_segments:True`` such inputs (for example resident MoE
output buffers) are CUDA VMM memory. The graph-buffer registration exports
every captured input with ``cudaIpcGetMemHandle``, which cannot export VMM
memory and exits the process ("invalid argument"). Two PCIe or NVLink GPUs
are enough: a TP2 custom all-reduce needs P2P access but no NVLink.
"""

import json
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

_WORLD_SIZE = 2
_NUMEL = 4096


def _values(rank: int, step: int, salt: int) -> torch.Tensor:
    # Small integers keep the FP16 sums exact.
    index = torch.arange(_NUMEL, device="cuda")
    return ((index * salt + 3 * rank + step) % 17).to(torch.float16)


def _capture_and_replay(comm, inp_a, inp_b, rank):
    inp_a.copy_(_values(rank, 0, 1))
    inp_b.copy_(_values(rank, 0, 2))
    # Eager warm-up through the staging buffer.
    warm = comm.custom_all_reduce(inp_a)
    torch.accelerator.synchronize()
    assert torch.equal(warm, sum(_values(r, 0, 1) for r in range(_WORLD_SIZE)))
    graph = torch.cuda.CUDAGraph()
    torch.accelerator.synchronize()
    dist.barrier()
    with comm.capture():
        with torch.cuda.graph(graph):
            out_a = comm.custom_all_reduce(inp_a)
            out_sum = comm.custom_all_reduce_sum2(inp_b, inp_a)
        # Read before registration resets them; absent without the guard.
        counts = (
            getattr(comm, "_ipc_guard_copies", None),
            getattr(comm, "_ipc_guard_fallbacks", None),
        )
    assert out_a is not None and out_sum is not None
    for step in (1, 2):
        # Replay must read the values the inputs hold at replay time.
        inp_a.copy_(_values(rank, step, 1))
        inp_b.copy_(_values(rank, step, 2))
        torch.accelerator.synchronize()
        dist.barrier()
        graph.replay()
        torch.accelerator.synchronize()
        expected_a = sum(_values(r, step, 1) for r in range(_WORLD_SIZE))
        expected_b = sum(_values(r, step, 2) for r in range(_WORLD_SIZE))
        assert torch.equal(out_a, expected_a)
        assert torch.equal(out_sum, expected_a + expected_b)
    return counts, graph


def _vmm_capture_worker(rank: int, rendezvous: str, result: str) -> None:
    from vllm.distributed.device_communicators import custom_all_reduce

    torch.accelerator.set_device_index(rank)
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=_WORLD_SIZE,
        init_method=rendezvous,
        timeout=timedelta(seconds=120),
    )
    comm = custom_all_reduce.CustomAllreduce(dist.group.WORLD, rank)
    graphs = []
    try:
        if comm.disabled:
            if rank == 0:
                Path(result).write_text(json.dumps({"disabled": True}))
            return
        # Allocated before the capture window: CUDA VMM memory.
        resident_a = torch.empty(_NUMEL, dtype=torch.float16, device="cuda")
        resident_b = torch.empty_like(resident_a)
        counts, graph = _capture_and_replay(comm, resident_a, resident_b, rank)
        graphs.append(graph)
        # One copy per captured input: all_reduce(a), all_reduce_sum2(b, a).
        assert counts == (3, 0), counts
        probe = custom_all_reduce._is_legacy_cuda_ipc_capable
        assert probe(resident_a.data_ptr()) is False

        # Resident buffers made by the helper need no copy.
        with custom_all_reduce.custom_ar_ipc_safe_allocation():
            safe_a = torch.empty(_NUMEL, dtype=torch.float16, device="cuda")
            safe_b = torch.empty_like(safe_a)
        assert probe(safe_a.data_ptr()) is True
        counts, graph = _capture_and_replay(comm, safe_a, safe_b, rank)
        graphs.append(graph)
        assert counts == (0, 0), counts
        if rank == 0:
            Path(result).write_text(json.dumps({"disabled": False}))
    finally:
        del graphs
        comm.close()
        dist.destroy_process_group()


def test_capture_of_inputs_allocated_in_expandable_segments(tmp_path, monkeypatch):
    if not torch.cuda.is_available() or torch.version.hip is not None:
        pytest.skip("requires CUDA")
    if torch.accelerator.device_count() < _WORLD_SIZE:
        pytest.skip("requires two GPUs")
    # Read by each spawned worker before its first CUDA allocation.
    monkeypatch.delenv("PYTORCH_ALLOC_CONF", raising=False)
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    rendezvous = (tmp_path / "rendezvous").as_uri()
    result = tmp_path / "result.json"
    mp.spawn(
        _vmm_capture_worker,
        args=(rendezvous, str(result)),
        nprocs=_WORLD_SIZE,
        join=True,
    )
    if json.loads(result.read_text())["disabled"]:
        pytest.skip("custom all-reduce is unavailable between these GPUs")
