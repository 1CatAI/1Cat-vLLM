# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The KV budget must reflect the steady-state forward, not the first
(compiling) one: a cold torch.compile allocates scratch that has nothing to
do with serving, and on a cache miss it shrank the budget by GiBs."""

from types import SimpleNamespace

import pytest
import torch

import vllm.envs as envs
from vllm.config import CUDAGraphMode
from vllm.platforms import current_platform
from vllm.utils.mem_utils import MemorySnapshot
from vllm.v1.worker.gpu_worker import Worker

MiB = 1024 * 1024
pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="needs a CUDA device for memory stats"
)


class _FakeRunner:
    """First profile_run behaves like a cold compile: a large transient
    allocation plus a small one that stays; the second is the plain forward."""

    def __init__(self, device: torch.device, weights_bytes: int) -> None:
        self.model_memory_usage = weights_bytes
        self._device = device
        self.calls = 0
        self.graph_calls = 0
        self._kept: list[torch.Tensor] = []

    def profile_run(self) -> None:
        self.calls += 1
        if self.calls == 1:
            scratch = torch.empty(1024 * MiB, dtype=torch.uint8, device=self._device)
            self._kept.append(
                torch.empty(64 * MiB, dtype=torch.uint8, device=self._device)
            )
            del scratch
        else:
            activations = torch.empty(256 * MiB, dtype=torch.uint8, device=self._device)
            del activations
        torch.accelerator.synchronize(self._device)

    def profile_cudagraph_memory(self) -> int:
        self.graph_calls += 1
        return 32 * MiB


@pytest.mark.parametrize(
    "graph_mode,estimate_graphs,graph_bytes",
    [
        (CUDAGraphMode.NONE, True, 0),
        (CUDAGraphMode.FULL, False, 0),
        (CUDAGraphMode.FULL, True, 32 * MiB),
    ],
)
def test_kv_budget_ignores_cold_compile_scratch(
    monkeypatch, graph_mode, estimate_graphs, graph_bytes
) -> None:
    monkeypatch.setattr(
        envs, "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS", estimate_graphs
    )
    device = torch.device("cuda:0")
    torch.accelerator.empty_cache()
    worker = Worker.__new__(Worker)
    worker.use_v2_model_runner = False
    worker.device = device
    worker.init_snapshot = MemorySnapshot(device=device)
    weights_bytes = 512 * MiB
    weights = torch.empty(weights_bytes, dtype=torch.uint8, device=device)
    worker.requested_memory = int(worker.init_snapshot.free_memory * 0.9)
    worker.cache_config = SimpleNamespace(
        kv_cache_memory_bytes=None, gpu_memory_utilization=0.9
    )
    worker.vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mode=graph_mode)
    )
    runner = _FakeRunner(device, weights_bytes)
    worker.model_runner = runner

    available = worker.determine_available_memory()

    # Peak activation is the second (steady-state) forward, not the 1 GiB scratch.
    assert abs(worker.peak_activation_memory - 256 * MiB) <= 16 * MiB
    # What the warm-up left allocated is still charged to the budget. Other
    # processes on the device show up in non_torch_memory, so account for it.
    charged = (
        worker.requested_memory
        - available
        - worker.non_torch_memory
        - worker.peak_activation_memory
        - weights_bytes
        - graph_bytes
    )
    assert abs(charged - 64 * MiB) <= 32 * MiB
    assert runner.calls == 2
    assert runner.graph_calls == int(graph_bytes > 0)
    assert worker.cudagraph_memory_estimate == graph_bytes
    del weights


def test_explicit_kv_bytes_profiles_once_and_skips_memory_estimation() -> None:
    calls = []
    worker = Worker.__new__(Worker)
    worker.cache_config = SimpleNamespace(kv_cache_memory_bytes=128 * MiB)
    worker.init_snapshot = SimpleNamespace(free_memory=1024 * MiB)
    worker.model_runner = SimpleNamespace(profile_run=lambda: calls.append("profile"))
    assert worker.determine_available_memory() == 128 * MiB
    assert calls == ["profile"]
