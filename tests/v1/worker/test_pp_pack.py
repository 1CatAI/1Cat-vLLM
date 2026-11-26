# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One-message PP handover: layout, blocks, dispatch and the wire."""

import os
import socket
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.config import CUDAGraphMode
from vllm.config.kernel import KernelConfig
from vllm.distributed.parallel_state import GroupCoordinator
from vllm.sequence import IntermediateTensors
from vllm.v1.worker import gpu_worker
from vllm.v1.worker.gpu import pp_pack
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    ModelCudaGraphManager,
)
from vllm.v1.worker.gpu.pp_pack import (
    PACKED_KEY,
    SEGMENT_ALIGNMENT_BYTES,
    PackedHandoverBuffers,
    PackedHandoverLayout,
    PackedIntermediateTensors,
    handover_inputs,
    packs_handover,
    pp_pack_enabled,
    verify_handover_layouts,
)
from vllm.v1.worker.gpu_worker import Worker

DECODE = BatchExecutionDescriptor(
    cg_mode=CUDAGraphMode.FULL, num_tokens=5, num_reqs=1, uniform_token_count=5
)


def _stage_tensors(rows: int) -> dict[str, torch.Tensor]:
    """The GLM boundary: mixed dtypes, ranks and odd row sizes."""
    gen = torch.Generator().manual_seed(rows)
    return {
        "hidden_states": torch.randn(rows, 64, generator=gen).half(),
        "residual": torch.randn(rows, 4, 64, generator=gen).half(),
        "post": torch.randn(rows, 4, 1, generator=gen),
        "comb": torch.randn(rows, 4, 4, generator=gen),
        "ids": torch.randint(0, 100, (rows,), generator=gen, dtype=torch.int32),
        "flag": torch.randint(0, 2, (rows, 3), generator=gen).bool(),
    }


def _buffers(rows: int = 7) -> PackedHandoverBuffers:
    layout = PackedHandoverLayout.from_tensors(_stage_tensors(rows))
    return PackedHandoverBuffers(layout, torch.device("cpu"))


def test_layout_is_aligned_and_independent_of_dict_order() -> None:
    tensors = _stage_tensors(5)
    layout = PackedHandoverLayout.from_tensors(tensors)
    reordered = PackedHandoverLayout.from_tensors(dict(reversed(tensors.items())))
    assert layout.signature == reordered.signature
    for rows in (1, 5, 8):
        offsets = layout.offsets(rows)
        assert all(o % SEGMENT_ALIGNMENT_BYTES == 0 for o in offsets)
        assert layout.nbytes(rows) % SEGMENT_ALIGNMENT_BYTES == 0
        for segment, start, nxt in zip(
            layout.segments, offsets, [*offsets[1:], layout.nbytes(rows)]
        ):
            assert start + rows * segment.row_bytes <= nxt


def test_block_roundtrip_is_exact_for_mixed_dtypes() -> None:
    sender, receiver = _buffers(), _buffers()
    for rows in (1, 5, 7):
        source = _stage_tensors(rows)
        send = sender.block(rows)
        for name, value in source.items():
            send.tensors[name].copy_(value)
        recv = receiver.block(rows)
        recv.flat.copy_(send.flat)  # what the one NCCL message does
        for name, value in source.items():
            assert recv.tensors[name].dtype == value.dtype
            assert recv.tensors[name].shape == value.shape
            assert torch.equal(recv.tensors[name], value)


def test_views_alias_the_flat_block_without_overlap() -> None:
    block = _buffers().block(5)
    base = block.flat.data_ptr()
    spans = []
    for view in block.tensors.values():
        assert view.is_contiguous()
        start = view.data_ptr() - base
        assert start >= 0 and start + view.nbytes <= block.flat.numel()
        assert start % SEGMENT_ALIGNMENT_BYTES == 0
        spans.append((start, start + view.nbytes))
    spans.sort()
    assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))


def test_blocks_are_cached_per_row_count() -> None:
    buffers = _buffers()
    assert buffers.block(5) is buffers.block(5)
    assert buffers.block(5).flat.data_ptr() != buffers.block(10).flat.data_ptr()
    assert buffers.block(10).tensors["hidden_states"].shape[0] == 10


def test_layout_signature_follows_names_dtypes_and_row_shapes() -> None:
    base = _stage_tensors(5)
    same_rows_other_count = PackedHandoverLayout.from_tensors(_stage_tensors(9))
    assert PackedHandoverLayout.from_tensors(base).signature == (
        same_rows_other_count.signature
    )
    other_dtype = {**base, "hidden_states": base["hidden_states"].float()}
    other_shape = {**base, "residual": base["residual"][:, :2]}
    assert (
        PackedHandoverLayout.from_tensors(other_dtype).signature
        != PackedHandoverLayout.from_tensors(base).signature
    )
    assert (
        PackedHandoverLayout.from_tensors(other_shape).signature
        != PackedHandoverLayout.from_tensors(base).signature
    )


def test_packs_only_fixed_shape_full_decode_steps() -> None:
    assert packs_handover(DECODE)
    for desc in (
        BatchExecutionDescriptor(CUDAGraphMode.NONE, 5, 1, uniform_token_count=5),
        BatchExecutionDescriptor(CUDAGraphMode.PIECEWISE, 5, None),
        BatchExecutionDescriptor(CUDAGraphMode.FULL, 64, 64),
    ):
        assert not packs_handover(desc)


def test_graph_inputs_are_the_receive_block_only_for_packed_steps() -> None:
    standard = IntermediateTensors(_stage_tensors(16))
    buffers = _buffers()
    packed = handover_inputs(DECODE, standard, buffers)
    block = buffers.block(5)
    assert all(packed[k].data_ptr() == block.tensors[k].data_ptr() for k in packed)

    mixed = BatchExecutionDescriptor(CUDAGraphMode.FULL, 8, 8)
    plain = handover_inputs(mixed, standard, buffers)
    assert all(plain[k].data_ptr() == standard.tensors[k].data_ptr() for k in plain)
    assert all(v.shape[0] == 8 for v in plain.values())
    # Without a pool (feature off) even a decode step reads the standard buffers.
    off = handover_inputs(DECODE, standard, None)
    assert all(off[k].data_ptr() == standard.tensors[k].data_ptr() for k in off)


def _config(**overrides):
    parallel = dict(
        pipeline_parallel_size=5, tensor_parallel_size=1, data_parallel_size=1
    )
    parallel.update(overrides.pop("parallel", {}))
    use_v2_model_runner = overrides.pop("use_v2_model_runner", True)
    return SimpleNamespace(
        kernel_config=KernelConfig(**overrides),
        parallel_config=SimpleNamespace(**parallel),
        use_v2_model_runner=use_v2_model_runner,
    )


def test_admission_is_a_kernel_config_field_and_needs_pure_pp(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_V41_PP_RANK0", "0")
    assert not KernelConfig().pp_pack_handover
    assert pp_pack_enabled(_config(pp_pack_handover=True))
    assert not pp_pack_enabled(_config())
    # Off, nothing is checked: the field is the only switch.
    assert not pp_pack_enabled(
        _config(parallel={"tensor_parallel_size": 2}, use_v2_model_runner=False)
    )


@pytest.mark.parametrize(
    ("overrides", "pp_rank0", "reason"),
    [
        ({"parallel": {"pipeline_parallel_size": 1}}, "0", "pipeline parallelism"),
        ({"parallel": {"tensor_parallel_size": 2}}, "0", "tensor parallel size 1"),
        ({"parallel": {"data_parallel_size": 2}}, "0", "data parallel size 1"),
        ({"use_v2_model_runner": False}, "0", "the V2 model runner"),
        ({}, "1", "VLLM_V41_PP_RANK0 unset"),
    ],
)
def test_a_setting_the_handover_cannot_honour_is_rejected(
    monkeypatch, overrides, pp_rank0, reason
) -> None:
    monkeypatch.setenv("VLLM_V41_PP_RANK0", pp_rank0)
    with pytest.raises(ValueError, match=f"pp_pack_handover requires {reason}"):
        pp_pack_enabled(_config(pp_pack_handover=True, **overrides))


def test_worker_construction_rejects_a_setting_it_cannot_honour() -> None:
    """The check runs when the Worker is built, before any device or model."""
    from vllm.config import VllmConfig

    config = VllmConfig(kernel_config=KernelConfig(pp_pack_handover=True))
    with pytest.raises(ValueError, match="requires pipeline parallelism"):
        Worker(
            vllm_config=config,
            local_rank=0,
            rank=0,
            distributed_init_method="env://",
        )


def test_worker_receive_pool_follows_the_admission() -> None:
    pool = _buffers()
    worker = Worker.__new__(Worker)
    worker.model_runner = SimpleNamespace(pp_pack_recv=pool)
    worker._pp_pack_handover = False
    assert worker._pp_pack_recv is None
    worker._pp_pack_handover = True
    assert worker._pp_pack_recv is pool


def test_graph_manager_returns_the_send_block_for_packed_steps() -> None:
    manager = ModelCudaGraphManager.__new__(ModelCudaGraphManager)
    manager.is_last_pp_rank = False
    manager._pp_pack_enabled = True
    block_source = _buffers()
    manager.pp_pack_send = block_source
    manager.intermediate_tensors = IntermediateTensors(_stage_tensors(16))
    manager.graphs = {DECODE: SimpleNamespace(replay=lambda: None)}
    with patch("vllm.v1.worker.gpu.cudagraph_utils.get_offloader"):
        out = manager.run_fullgraph(DECODE)
        assert isinstance(out, PackedIntermediateTensors)
        assert out.packed.data_ptr() == block_source.block(5).flat.data_ptr()
        mixed = BatchExecutionDescriptor(CUDAGraphMode.FULL, 8, 8)
        manager.graphs[mixed] = SimpleNamespace(replay=lambda: None)
        plain = manager.run_fullgraph(mixed)
        assert not isinstance(plain, PackedIntermediateTensors)


class _Handle:
    waited = False

    def wait(self) -> None:
        self.waited = True


class _FakeGroup:
    def __init__(self) -> None:
        self.static_recvs: list[dict[str, torch.Tensor]] = []

    def irecv_tensor_dict_static(self, tensors):
        self.static_recvs.append(tensors)
        self.handle = _Handle()
        return [self.handle]


def _worker(pool: PackedHandoverBuffers):
    worker = SimpleNamespace(_pp_pack_recv=pool)
    worker._irecv_per_tensor_handover = lambda n, gather: ("per-tensor", n)
    return worker


def test_worker_receives_packed_steps_into_the_graph_input_block() -> None:
    pool, group = _buffers(), _FakeGroup()
    with patch.object(gpu_worker, "get_pp_group", return_value=group):
        received = Worker._irecv_pp_handover(
            _worker(pool), DECODE, num_tokens=5, all_gather_tensors={}
        )
    block = pool.block(5)
    assert group.static_recvs == [{PACKED_KEY: block.flat}]  # one message
    assert not group.handle.waited  # the wait is deferred to the graph input use
    views = received.tensors  # first use waits for the receive
    assert group.handle.waited
    assert views["residual"].data_ptr() == block.tensors["residual"].data_ptr()


def test_worker_keeps_the_per_tensor_path_for_variable_shapes() -> None:
    pool, group = _buffers(), _FakeGroup()
    prefill = BatchExecutionDescriptor(CUDAGraphMode.NONE, 300, 1)
    with patch.object(gpu_worker, "get_pp_group", return_value=group):
        out = Worker._irecv_pp_handover(
            _worker(pool), prefill, num_tokens=300, all_gather_tensors={}
        )
    assert out == ("per-tensor", 300)
    assert group.static_recvs == []


def _wire(rank: int, port: int, result_queue) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=2)
    group = GroupCoordinator.__new__(GroupCoordinator)
    group.world_size, group.rank_in_group, group.ranks = 2, rank, [0, 1]
    group.use_cpu_custom_send_recv = False
    group.device_group = group.cpu_group = dist.group.WORLD
    pool = _buffers()
    if rank == 0:
        send = pool.block(5)
        for name, value in _stage_tensors(5).items():
            send.tensors[name].copy_(value)
        handles = group.isend_tensor_dict_static({PACKED_KEY: send.flat})
        result_queue.put(("messages", len(handles)))
    else:
        recv = pool.block(5)
        for handle in group.irecv_tensor_dict_static({PACKED_KEY: recv.flat}):
            handle.wait()
        expected = _stage_tensors(5)
        result_queue.put(
            ("equal", all(torch.equal(recv.tensors[k], v) for k, v in expected.items()))
        )
    for handle in handles if rank == 0 else []:
        handle.wait()
    dist.barrier()
    dist.destroy_process_group()


def test_one_message_crosses_the_group_and_lands_in_the_views() -> None:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    procs = [ctx.Process(target=_wire, args=(r, port, queue)) for r in (0, 1)]
    for proc in procs:
        proc.start()
    results = dict(queue.get(timeout=120) for _ in procs)
    for proc in procs:
        proc.join(timeout=60)
        assert proc.exitcode == 0
    assert results == {"messages": 1, "equal": True}


def test_block_allocation_is_refused_while_capturing_a_graph() -> None:
    buffers = _buffers()
    buffers.block(5)
    with patch.object(pp_pack, "_is_capturing", return_value=True):
        assert buffers.block(5) is buffers.block(5)  # cached: no allocation
        with pytest.raises(AssertionError, match="while capturing"):
            buffers.block(10)
    assert buffers.captured_rows == (5,)


def _buffers_with(rows: int, dtype: torch.dtype | None, device="cpu"):
    tensors = _stage_tensors(rows)
    if dtype is not None:
        tensors["hidden_states"] = tensors["hidden_states"].to(dtype)
    buffers = PackedHandoverBuffers(
        PackedHandoverLayout.from_tensors(tensors), torch.device(device)
    )
    buffers.block(rows)
    return buffers


def _handshake(rank: int, port: int, case: str, result_queue) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=2)
    group = GroupCoordinator.__new__(GroupCoordinator)
    group.world_size, group.rank_in_group, group.ranks = 2, rank, [0, 1]
    group.cpu_group = dist.group.WORLD
    sender_dtype = torch.float32 if case == "dtype" else None
    sender = None if case == "sender_missing" else _buffers_with(5, sender_dtype)
    try:
        if rank == 0:
            verify_handover_layouts(group, None, sender)
            outcome = "sent"
        else:
            verify_handover_layouts(group, _buffers_with(5, None), None)
            outcome = "ok"
    except RuntimeError as error:
        outcome = f"raised: {error}"
    result_queue.put((rank, outcome))
    dist.barrier()
    dist.destroy_process_group()


def _run_handshake(case: str) -> dict[int, str]:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    procs = [
        ctx.Process(target=_handshake, args=(r, port, case, queue)) for r in (0, 1)
    ]
    for proc in procs:
        proc.start()
    results = dict(queue.get(timeout=120) for _ in procs)
    for proc in procs:
        proc.join(timeout=60)
        assert proc.exitcode == 0
    return results


def test_layout_handshake_passes_for_matching_neighbours() -> None:
    assert _run_handshake("match") == {0: "sent", 1: "ok"}


@pytest.mark.parametrize("case", ["dtype", "sender_missing"])
def test_layout_handshake_raises_on_the_receiver_for_a_mismatch(case: str) -> None:
    results = _run_handshake(case)
    assert results[0] == "sent"
    assert results[1].startswith("raised: PP packed handover layout mismatch")


if __name__ == "__main__":
    pytest.main([__file__])
