# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

from vllm.config import KernelConfig
from vllm.platforms.interface import Platform
from vllm.v1.worker import gpu_worker


@pytest.mark.parametrize(
    "headroom,consumer,called",
    [(0, True, False), (512, False, False), (512, True, True)],
)
def test_budget_requires_request_and_actual_consumer(
    monkeypatch, headroom, consumer, called
):
    calls = []

    def apply_limit(size, device):
        calls.append((size, device))
        return 123

    platform = SimpleNamespace(limit_allocator_headroom=apply_limit)
    monkeypatch.setattr(gpu_worker, "current_platform", platform)
    config = KernelConfig(qsa_host_kv_allocator_headroom_mib=headroom)
    device = torch.device("cuda", 0)
    owner = SimpleNamespace(
        vllm_config=SimpleNamespace(kernel_config=config),
        device=device,
        compilation_config=SimpleNamespace(
            static_forward_context={
                "a": SimpleNamespace(host_kv=object() if consumer else None)
            }
        ),
    )
    assert gpu_worker.Worker._trim_qsa_warmup_cache(owner) is called
    assert calls == ([(512 * 1024**2, device)] if called else [])


def test_limit_is_resource_only_and_unsupported_platform_rejects():
    a = KernelConfig(qsa_host_kv=True)
    b = KernelConfig(qsa_host_kv=True, qsa_host_kv_allocator_headroom_mib=512)
    assert a.qsa_host_kv_allocator_headroom_mib == 0
    assert a.compute_hash() == b.compute_hash()
    with pytest.raises(NotImplementedError):
        Platform.limit_allocator_headroom(512 * 1024**2)
    for value in (-1, 4097):
        with pytest.raises(ValueError):
            KernelConfig(qsa_host_kv_allocator_headroom_mib=value)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.no_grad()
def test_cuda_limit_retains_graph_and_reclaims_old_cache():
    from vllm.platforms.cuda import CudaPlatform

    old_fraction = torch.cuda.get_per_process_memory_fraction()
    x = torch.ones(4096, device="cuda")
    state = torch.zeros_like(x)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        state.add_(x)
        result = state * 2
    torch.accelerator.synchronize()
    torch.accelerator.empty_cache()
    free, total = torch.accelerator.get_memory_info()
    non_torch = total - free - torch.accelerator.memory_reserved()
    budget = 192 * 1024**2
    try:
        limit = CudaPlatform.limit_allocator_headroom(total - non_torch - budget)
        assert abs(limit - budget) <= 1
        for value, size in ((2, 128), (5, 160), (-1, 96)):
            # Different live allocations prevent a test that only checks a setter.
            scratch = torch.empty(size * 1024**2, device="cuda", dtype=torch.uint8)
            scratch.fill_(value)
            del scratch
            x.fill_(value)
            before = state.clone()
            graph.replay()
            assert torch.equal(state, before + x)
            assert torch.equal(result, state * 2)
            assert torch.accelerator.memory_reserved() <= limit
        # A looser later request must preserve an existing tighter process cap.
        assert (
            CudaPlatform.limit_allocator_headroom(
                total - non_torch - budget - 32 * 1024**2
            )
            == limit
        )
        with pytest.raises(torch.OutOfMemoryError):
            torch.empty(256 * 1024**2, device="cuda", dtype=torch.uint8)
        before = state.clone()
        x.fill_(7)
        graph.replay()
        assert torch.equal(state, before + x)
        retained = torch.accelerator.memory_reserved()
        free, total = torch.accelerator.get_memory_info()
        non_torch = total - free - retained
        with pytest.raises(ValueError, match="cannot retain"):
            CudaPlatform.limit_allocator_headroom(total - non_torch + 1)
    finally:
        torch.cuda.set_per_process_memory_fraction(old_fraction)
