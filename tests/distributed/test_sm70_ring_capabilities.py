# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

from vllm.config.kernel import Sm70RingConfig
from vllm.distributed.device_communicators.sm70_ring import (
    Sm70RingCommunicator,
    find_peer_order,
)


def test_direct_ring_order_excludes_system_edges():
    edges = [
        [True, True, True, False],
        [True, True, False, True],
        [True, False, True, True],
        [False, True, True, True],
    ]
    order = find_peer_order(edges)
    assert order is not None
    assert all(edges[order[i]][order[i ^ bit]] for i in range(4) for bit in (1, 2))
    assert find_peer_order([[True] * 4 for _ in range(4)]) is None
    assert find_peer_order([[i == j for j in range(4)] for i in range(4)]) is None


@pytest.mark.parametrize(
    "bytes_,dtype,contiguous,reason",
    [
        (5120, torch.float16, True, None),
        (25600, torch.float16, True, None),
        (25602, torch.float16, True, "payload_outside_calibrated_byte_range"),
        (0, torch.float16, True, "payload_outside_calibrated_byte_range"),
        (5120, torch.float32, True, "fp16_input_required_for_exact_packet_encoding"),
        (5120, torch.float16, False, "requires_contiguous_local_cuda_input"),
    ],
)
def test_runtime_capability_guards(bytes_, dtype, contiguous, reason):
    comm = Sm70RingCommunicator.__new__(Sm70RingCommunicator)
    comm.policy = Sm70RingConfig()
    comm.device = torch.device("cuda:0")
    comm.status = {"enabled": True}
    width = 2 if dtype == torch.float16 else 4
    tensor = SimpleNamespace(
        dtype=dtype,
        device=comm.device,
        is_contiguous=lambda: contiguous,
        numel=lambda: bytes_ // width,
        element_size=lambda: width,
    )
    assert comm.rejection_reason(tensor) == reason


def test_policy_cannot_admit_uncalibrated_payload():
    assert Sm70RingConfig().enabled
    with pytest.raises(ValueError):
        Sm70RingConfig(max_bytes=25602)


@pytest.mark.parametrize("admitted", [True, False])
def test_dispatch_prefers_admitted_ring_and_preserves_fallback(monkeypatch, admitted):
    from vllm.distributed.device_communicators import cuda_communicator as cuda

    output = object()
    input_ = object()
    comm = cuda.CudaCommunicator.__new__(cuda.CudaCommunicator)
    comm.ring_comm = SimpleNamespace(all_reduce=lambda _: output if admitted else None)
    comm.pynccl_comm = SimpleNamespace(world_size=4)
    visited = []
    monkeypatch.setattr(cuda, "_trace_all_reduce_path", lambda *args: None)

    def symmetric_guard(*args):
        visited.append("fallback")
        return True

    monkeypatch.setattr(cuda, "should_nccl_symm_mem_allreduce", symmetric_guard)
    monkeypatch.setattr(
        torch.ops.vllm,
        "all_reduce_symmetric_with_copy",
        lambda _: output,
        raising=False,
    )
    assert comm.all_reduce(input_) is output
    assert visited == ([] if admitted else ["fallback"])
