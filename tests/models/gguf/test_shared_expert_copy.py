# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Four GPUs copy canonical original bytes with exclusive block writeback."""

import json
import multiprocessing as mp
import socket
import traceback

import numpy as np
import pytest
import torch

from vllm.model_executor.layers.quantization.gguf_shared_layout import (
    SharedExpertLayout,
)
from vllm.utils.shared_host import SharedHostArena

pytestmark = pytest.mark.skipif(
    torch.accelerator.device_count() != 4, reason="Expose exactly the four test GPUs"
)


def _exercise(host, native, rank, kind, role, expert_bytes, offsets):
    from vllm.model_executor.layers.quantization.gguf_shared_copy import (
        swap_original_experts,
    )

    source_offsets, victim_offsets, save_offsets, next_offsets = offsets
    copies = len(source_offsets)
    layout = SharedExpertLayout.for_type(5120, 2304, 4, rank, role, kind)
    shape = (layout.source_rows, layout.source_row_bytes)
    victims = [
        host[int(o) : int(o) + expert_bytes].numpy().reshape(shape)
        for o in victim_offsets
    ]
    local_shape = (copies, layout.device_rows, layout.device_row_bytes)
    nbytes = int(np.prod(local_shape))
    storage = torch.full(
        (nbytes + layout.guard_bytes,), 47, dtype=torch.uint8, device=rank
    )
    bank = storage[:nbytes].view(local_shape)
    bank.copy_(torch.from_numpy(np.stack([layout.pack(v) for v in victims])))
    mapped = native.mapped_view(host, rank)
    slots = torch.arange(copies, dtype=torch.int32, device=rank)
    loads = torch.from_numpy(source_offsets).to(rank)
    saves = torch.from_numpy(save_offsets).to(rank)
    count = torch.zeros(1, dtype=torch.int32, device=rank)

    def run():
        swap_original_experts(bank, mapped, slots, loads, saves, count, layout)

    run()
    torch.accelerator.synchronize(rank)
    np.testing.assert_array_equal(
        bank.cpu().numpy(), np.stack([layout.pack(v) for v in victims])
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    count.fill_(copies)
    graph.replay()
    torch.accelerator.synchronize(rank)
    for i, offset in enumerate(source_offsets):
        expert = host[int(offset) : int(offset) + expert_bytes].numpy().reshape(shape)
        np.testing.assert_array_equal(bank[i].cpu().numpy(), layout.pack(expert))
    saves.copy_(torch.from_numpy(next_offsets))
    for _ in range(3):
        graph.replay()
    torch.accelerator.synchronize(rank)
    count.zero_()
    graph.replay()
    torch.accelerator.synchronize(rank)
    assert bool((storage[nbytes:] == 47).all())


def _copy_rank(rank, connection, kind, role, expert_bytes, offsets):
    from vllm.model_executor.layers.quantization.gguf_expert_cache import _native

    native = arena = view = host = None
    registered = False
    try:
        torch.set_num_threads(1)
        torch.accelerator.set_device_index(rank)
        native = _native()
        arena = SharedHostArena.receive(connection)
        view = memoryview(arena.buffer)
        host = torch.frombuffer(view, dtype=torch.uint8)
        native.host_register(host.data_ptr(), host.numel())
        registered = True
        _exercise(host, native, rank, kind, role, expert_bytes, offsets)
        connection.send(json.dumps({"rank": rank, "passed": True}).encode())
    except BaseException:
        connection.send(
            json.dumps({"rank": rank, "error": traceback.format_exc()}).encode()
        )
        raise
    finally:
        if registered:
            assert native is not None and host is not None
            torch.accelerator.synchronize(rank)
            native.host_unregister(host.data_ptr())
        host = None
        if view is not None:
            view.release()
        if arena is not None:
            arena.close()
        connection.close()


@pytest.mark.parametrize("kind", [16, 17, 29])
@pytest.mark.parametrize("role", ["w1", "w3", "w2"])
@pytest.mark.parametrize("copies", [2, 44])
def test_four_card_raw_swap_reconstructs_original_and_preserves_guard(
    kind, role, copies
):
    first = SharedExpertLayout.for_type(5120, 2304, 4, 0, role, kind)
    shape = (first.source_rows, first.source_row_bytes)
    expert_bytes = int(np.prod(shape))
    random = np.random.default_rng(41)
    incoming = random.integers(0, 256, (copies, *shape), dtype=np.uint8)
    victims = random.integers(0, 256, (copies, *shape), dtype=np.uint8)
    page = 4096
    size = 4 * copies * expert_bytes + 2 * page
    source_offsets = page + np.arange(copies, dtype=np.int64) * expert_bytes
    victim_offsets = source_offsets + copies * expert_bytes
    save_offsets = source_offsets + 2 * copies * expert_bytes
    next_offsets = source_offsets + 3 * copies * expert_bytes
    context = mp.get_context("spawn")
    connections, processes = [], []
    with SharedHostArena.create(size) as arena:
        host = np.ndarray((arena.size,), dtype=np.uint8, buffer=arena.buffer)
        host.fill(47)
        for values, offsets in ((incoming, source_offsets), (victims, victim_offsets)):
            for i, offset in enumerate(offsets):
                host[int(offset) : int(offset) + expert_bytes] = values[i].ravel()
        expected = host.copy()
        for values, offsets in ((victims, save_offsets), (incoming, next_offsets)):
            for i, offset in enumerate(offsets):
                expected[int(offset) : int(offset) + expert_bytes] = values[i].ravel()
        try:
            for rank in range(4):
                parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
                parent.settimeout(90)
                process = context.Process(
                    target=_copy_rank,
                    args=(
                        rank,
                        child,
                        kind,
                        role,
                        expert_bytes,
                        (source_offsets, victim_offsets, save_offsets, next_offsets),
                    ),
                )
                process.start()
                child.close()
                connections.append(parent)
                processes.append(process)
                arena.send(parent)
            reports = [json.loads(conn.recv(65536)) for conn in connections]
            assert all(report.get("passed") for report in reports), reports
            for process in processes:
                process.join(15)
                assert process.exitcode == 0
            np.testing.assert_array_equal(host, expected)
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                process.join(5)
            for connection in connections:
                connection.close()
            del host
