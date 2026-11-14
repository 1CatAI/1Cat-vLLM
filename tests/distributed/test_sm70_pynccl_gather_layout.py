# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.distributed.device_communicators.cuda_communicator import (
    _all_gather_existing_pynccl,
)


@pytest.mark.parametrize(
    "shape,dim", [((2, 3), 0), ((2, 3), -1), ((2, 3, 4), 1), ((0, 3), 0)]
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.int64])
def test_existing_communicator_preserves_concat_layout(shape, dim, dtype):
    value = torch.arange(torch.tensor(shape).prod().item()).reshape(shape).to(dtype)
    if value.ndim == 3:
        value = value.transpose(0, 2)
    chunks = [value + rank * 8 for rank in range(4)]

    class Communicator:
        def all_gather(self, output, input_):
            assert input_.is_contiguous()
            output.copy_(torch.stack(chunks))

    actual = _all_gather_existing_pynccl(value, Communicator(), 4, dim)
    torch.testing.assert_close(actual, torch.cat(chunks, dim), rtol=0, atol=0)


@pytest.mark.parametrize("shape,dim", [((), 0), ((2, 3), 2), ((2, 3), -3)])
def test_invalid_dimension_is_rejected_before_communication(shape, dim):
    class Communicator:
        def all_gather(self, *_):
            pytest.fail("Invalid geometry must not enter a collective")

    with pytest.raises(IndexError, match="Invalid all-gather dimension"):
        _all_gather_existing_pynccl(torch.empty(shape), Communicator(), 4, dim)


def _four_rank_worker(rank, init_url):
    from vllm.distributed.device_communicators import cuda_communicator as module
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

    torch.accelerator.set_device_index(rank)
    dist.init_process_group("gloo", init_method=init_url, rank=rank, world_size=4)
    try:
        communicator = PyNcclCommunicator(dist.group.WORLD, device=rank)
        assert not communicator.disabled
        transport = module.CudaCommunicator.__new__(module.CudaCommunicator)
        transport.pynccl_comm = communicator
        transport.world_size = 4
        transport._pynccl_strided_gather = module.current_platform.is_device_capability(
            70, device_id=rank
        )
        assert transport._pynccl_strided_gather
        module._PYNCCL_ALLGATHER = True
        for dtype in (torch.float16, torch.float32, torch.int64):
            for rows in (1, 5, 20, 128):
                base = (
                    torch.arange(rows * 64, device=f"cuda:{rank}").reshape(rows, 64)
                    % 64
                )
                value = (base.to(dtype) + rank * 8)[:, ::2]
                chunks = [(base.to(dtype) + owner * 8)[:, ::2] for owner in range(4)]
                for dim in (0, -1):
                    result = transport.all_gather(value, dim)
                    torch.testing.assert_close(
                        result, torch.cat(chunks, dim), rtol=0, atol=0
                    )
                    torch.accelerator.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        captured = transport.all_gather(value, dim)
                    value.add_(1)
                    graph.replay()
                    torch.accelerator.synchronize()
                    torch.testing.assert_close(
                        captured,
                        torch.cat([chunk + 1 for chunk in chunks], dim),
                        rtol=0,
                        atol=0,
                    )
                    value.sub_(1)
        torch.accelerator.synchronize()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    torch.accelerator.device_count() < 4, reason="requires four CUDA GPUs"
)
def test_four_rank_existing_pynccl_strided_gather_and_changed_input_graph_replay(
    tmp_path,
):
    mp.spawn(
        _four_rank_worker,
        args=(f"file://{tmp_path / 'gather.init'}",),
        nprocs=4,
        join=True,
    )


def test_existing_disable_switch_still_selects_the_base_communicator(monkeypatch):
    from types import SimpleNamespace

    from vllm.distributed.device_communicators import cuda_communicator as module

    transport = module.CudaCommunicator.__new__(module.CudaCommunicator)
    transport._pynccl_strided_gather = True
    transport.pynccl_comm = SimpleNamespace(disabled=False)
    # No custom all-reduce, so the all-reduce gather of 9fbc11c6f stays out.
    transport.ca_comm = None
    value, expected = torch.ones(2, 3), torch.empty(2, 12)
    calls: list[tuple[object, torch.Tensor, int]] = []

    def fallback(self, input_, dim):
        calls.append((self, input_, dim))
        return expected

    monkeypatch.setattr(module, "_PYNCCL_ALLGATHER", False)
    monkeypatch.setattr(module.DeviceCommunicatorBase, "all_gather", fallback)
    assert transport.all_gather(value, -1) is expected
    assert calls == [(transport, value, -1)]
