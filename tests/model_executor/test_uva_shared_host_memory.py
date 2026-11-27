# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared, exactly sized pinned host copies of the UVA offloader."""

import gc
import mmap
import os
import time
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

import vllm.model_executor.offloader.uva as uva
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.offloader.uva import (
    UVAOffloader,
    _open_owner_arena,
    _register_pinned,
    _release_with_mapping,
    _share_key,
    _shared_host_copy,
)
from vllm.utils.shared_host import SharedHostArena
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="cudaHostRegister needs CUDA"
)

# Not a power of two, so a rounding allocator would show.
SHAPE = (3 * 256 + 7, 1024)
NBYTES = SHAPE[0] * SHAPE[1] * 2
# The arena is rounded up to whole pages and not further.
ARENA_BYTES = -(-NBYTES // mmap.PAGESIZE) * mmap.PAGESIZE


def _source(fill: float) -> torch.Tensor:
    return torch.full(SHAPE, fill, dtype=torch.float16)


def _mapped_bytes(pointer: int) -> int:
    """Size of the mapping that holds ``pointer``, 0 if none does."""
    with open("/proc/self/maps") as maps:
        for line in maps:
            start, end = (int(x, 16) for x in line.split()[0].split("-"))
            if start <= pointer < end:
                return end - start
    return 0


def _mapping_bytes(pointer: int) -> int:
    nbytes = _mapped_bytes(pointer)
    assert nbytes, "pointer not mapped"
    return nbytes


def _init_group(rank: int, world_size: int, store_path: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{store_path}",
        world_size=world_size,
        rank=rank,
    )


@pytest.fixture
def single_process_group(tmp_path):
    _init_group(0, 1, str(tmp_path / "store"))
    yield
    dist.destroy_process_group()


def _embedding(rows: int) -> VocabParallelEmbedding:
    # Only the type and the weight matter to the offloader; building a real
    # table needs the tensor-parallel group.
    module = VocabParallelEmbedding.__new__(VocabParallelEmbedding)
    nn.Module.__init__(module)
    module.weight = nn.Parameter(torch.ones(rows, 8, dtype=torch.float16))
    return module


@pytest.fixture
def parallel_ranks(monkeypatch):
    import vllm.config
    import vllm.distributed
    import vllm.distributed.parallel_state

    model_config = SimpleNamespace(model="org/model", revision="rev1")
    monkeypatch.setattr(
        vllm.config,
        "get_current_vllm_config",
        lambda: SimpleNamespace(model_config=model_config),
    )
    monkeypatch.setattr(vllm.distributed, "get_tensor_model_parallel_rank", lambda: 3)
    monkeypatch.setattr(
        vllm.distributed.parallel_state,
        "get_dp_group",
        lambda: SimpleNamespace(rank_in_group=2),
    )
    return model_config


def _offloader() -> UVAOffloader:
    offloader = UVAOffloader.__new__(UVAOffloader)
    offloader.pin_memory = True
    return offloader


def test_share_key_only_for_embedding_tables(
    single_process_group, parallel_ranks, monkeypatch
):
    table = _embedding(4)
    key = _share_key(table, "model.embed_tokens.weight", table.weight)
    assert key is not None
    assert "dp2" in key and "tp3" in key
    assert "model.embed_tokens.weight" in key
    assert "org/model@rev1" in key
    # The owner entry is a pid: another pid namespace must not match.
    assert os.readlink("/proc/self/ns/pid") in key
    real_readlink = os.readlink
    with monkeypatch.context() as patch:
        patch.setattr(
            uva.os,
            "readlink",
            lambda path: (
                "pid:[1]" if path == "/proc/self/ns/pid" else real_readlink(path)
            ),
        )
        assert _share_key(table, "model.embed_tokens.weight", table.weight) != key
    linear = nn.Linear(8, 4)
    assert _share_key(linear, "model.proj.weight", linear.weight) is None
    # The same table of another checkpoint must not match.
    parallel_ranks.model = "org/other"
    assert _share_key(table, "model.embed_tokens.weight", table.weight) != key


def test_share_key_needs_a_process_group(parallel_ranks):
    table = _embedding(4)
    assert _share_key(table, "model.embed_tokens.weight", table.weight) is None


def test_empty_tensor_is_never_shared(single_process_group, parallel_ranks):
    table = _embedding(0)
    assert _share_key(table, "model.embed_tokens.weight", table.weight) is None


def _forbid_arenas(monkeypatch) -> None:
    def refuse(*args, **kwargs):
        raise AssertionError("a private copy must not create a memfd arena")

    monkeypatch.setattr(SharedHostArena, "create", refuse)
    monkeypatch.setattr(
        uva, "current_platform", SimpleNamespace(is_cuda_alike=lambda: True)
    )
    # No CUDA host allocator on this machine; the stand-in keeps the data.
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self)


def test_private_and_empty_tensors_take_the_plain_path(
    single_process_group, parallel_ranks, monkeypatch
):
    _forbid_arenas(monkeypatch)
    linear = nn.Linear(8, 4)
    copied = _offloader()._host_copy(
        linear, "weight", linear.weight.data, parameter_prefix=""
    )
    assert torch.equal(copied, linear.weight.data)
    empty = _embedding(0)
    copied = _offloader()._host_copy(
        empty, "weight", empty.weight.data, parameter_prefix="model.embed_tokens"
    )
    assert copied.numel() == 0


def test_non_contiguous_table_is_refused(single_process_group):
    with pytest.raises(ValueError, match="non-contiguous"):
        _shared_host_copy(torch.ones(8, 8).t(), "test/non-contiguous")


def test_same_key_twice_in_one_process_is_an_error(single_process_group):
    arena, owner_pid = _shared_host_copy(_source(1), "test/twice")
    assert owner_pid == os.getpid()
    fds = len(os.listdir("/proc/self/fd"))
    with pytest.raises(RuntimeError, match="already shared by this process"):
        _shared_host_copy(_source(2), "test/twice")
    # The refused attempt gives its arena back.
    assert len(os.listdir("/proc/self/fd")) == fds
    arena.close()


def test_malformed_owner_entry_leaks_no_descriptor(single_process_group):
    read_end, write_end = os.pipe()
    try:
        before = len(os.listdir("/proc/self/fd"))
        # pid and fd name a real descriptor, the size is not a number.
        with pytest.raises(ValueError):
            _open_owner_arena(f"{os.getpid()}:{read_end}:notanumber:1:2")
        with pytest.raises(ValueError):
            _open_owner_arena(f"{os.getpid()}:{read_end}:1:2")
        assert len(os.listdir("/proc/self/fd")) == before
    finally:
        os.close(read_end)
        os.close(write_end)


def test_registration_follows_the_mapping_not_the_offloader():
    released = []
    arena = SharedHostArena.create(NBYTES)
    address = arena.address
    exporter = memoryview(arena.buffer)
    host = uva._host_view(exporter, _source(0))
    # cudaHostUnregister needs the pages still mapped when it runs. (A finalizer
    # on the mmap object runs after its munmap.)
    _release_with_mapping(exporter, lambda: released.append(_mapped_bytes(address)))
    # Dropping the arena object (as dropping an offloader that kept it would)
    # and every other reference leaves the tensor, which maps the memory.
    del arena, exporter
    gc.collect()
    assert not released
    derived = host[1:]
    del host
    gc.collect()
    assert not released
    del derived
    gc.collect()
    assert released == [ARENA_BYTES]
    # The mapping goes away right after the release.
    assert _mapped_bytes(address) == 0


@requires_cuda
def test_pinned_copy_is_exact_and_stays_pinned_without_the_arena_object():
    torch.zeros(1, device="cuda")
    arena = SharedHostArena.create(NBYTES)
    exporter = memoryview(arena.buffer)
    host = uva._host_view(exporter, _source(3))
    host.copy_(_source(3))
    assert arena.size == ARENA_BYTES
    _register_pinned(arena, exporter)
    assert host.is_pinned()
    assert _mapping_bytes(host.data_ptr()) == ARENA_BYTES
    device = get_accelerator_view_from_cpu_tensor(host)
    # Nothing but the tensor keeps the registration; the device view holds it.
    del arena, exporter, host
    gc.collect()
    assert device.is_cuda
    assert bool((device == 3).all())


def _share_worker(rank: int, store_path: str, results) -> None:
    _init_group(rank, 2, store_path)
    arena, owner_pid = _shared_host_copy(_source(rank + 1), "test/shared-table")
    host = uva._host_view(arena.buffer, _source(0))
    created = owner_pid == os.getpid()
    # Whichever process created the arena published its own values, and the
    # other one sees them; an attacher's writes are visible to the creator.
    if not created:
        host[-1, -1] = 9
    dist.barrier()
    results[rank] = (
        created,
        owner_pid,
        os.getpid(),
        arena.identity.inode,
        arena.size,
        float(host[0, 0]),
        float(host[-1, -1]),
    )
    dist.barrier()
    dist.destroy_process_group()


def test_table_is_shared_between_processes_without_cuda(tmp_path):
    manager = mp.get_context("spawn").Manager()
    results = manager.dict()
    mp.start_processes(
        _share_worker,
        args=(str(tmp_path / "store"), results),
        nprocs=2,
        start_method="spawn",
    )
    first, second = results[0], results[1]
    assert first[0] != second[0]
    creator_pid = first[2] if first[0] else second[2]
    assert first[1] == second[1] == creator_pid
    assert first[3] == second[3]
    assert first[4] == second[4] == ARENA_BYTES
    # Rank 0 and rank 1 started from different values; both read the creator's.
    creator_fill = float(1 if first[0] else 2)
    assert first[5] == second[5] == creator_fill
    assert first[6] == second[6] == 9.0


def _slow_creator_worker(rank: int, store_path: str, results) -> None:
    _init_group(rank, 2, store_path)
    key = "test/slow-creator"
    if rank == 0:
        original = torch.Tensor.copy_

        def slow_copy(self, *args, **kwargs):
            if self.numel() * self.element_size() == NBYTES:
                time.sleep(2)
            return original(self, *args, **kwargs)

        torch.Tensor.copy_ = slow_copy  # type: ignore[method-assign]
        _shared_host_copy(_source(7), key)
        torch.Tensor.copy_ = original  # type: ignore[method-assign]
    else:
        store = dist.distributed_c10d._get_default_store()
        store.wait([key], timedelta(seconds=60))
        # The key is visible: whatever it names must already hold the values.
        arena, owner_pid = _shared_host_copy(_source(0), key)
        host = uva._host_view(arena.buffer, _source(0))
        results[rank] = (float(host[0, 0]), float(host[-1, -1]), owner_pid)
    dist.barrier()
    dist.destroy_process_group()


def test_creator_fills_the_arena_before_publishing_it(tmp_path):
    manager = mp.get_context("spawn").Manager()
    results = manager.dict()
    mp.start_processes(
        _slow_creator_worker,
        args=(str(tmp_path / "store"), results),
        nprocs=2,
        start_method="spawn",
    )
    first_value, last_value, _ = results[1]
    assert first_value == last_value == 7.0


def _share_pinned_worker(rank: int, store_path: str, results) -> None:
    torch.accelerator.set_device_index(0)
    torch.zeros(1, device="cuda")
    _init_group(rank, 2, store_path)
    arena, owner_pid = _shared_host_copy(_source(5), "test/shared-pinned")
    exporter = memoryview(arena.buffer)
    host = uva._host_view(exporter, _source(5))
    _register_pinned(arena, exporter)
    device = get_accelerator_view_from_cpu_tensor(host)
    dist.barrier()
    results[rank] = (
        owner_pid == os.getpid(),
        host.is_pinned(),
        bool((device == 5).all()),
    )
    dist.barrier()
    dist.destroy_process_group()


@requires_cuda
def test_pinned_table_is_shared_between_processes(tmp_path):
    manager = mp.get_context("spawn").Manager()
    results = manager.dict()
    mp.start_processes(
        _share_pinned_worker,
        args=(str(tmp_path / "store"), results),
        nprocs=2,
        start_method="spawn",
    )
    assert results[0][0] != results[1][0]
    assert results[0][1:] == results[1][1:] == (True, True)
