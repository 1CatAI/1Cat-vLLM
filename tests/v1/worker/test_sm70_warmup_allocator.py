# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from vllm.config import CUDAGraphMode
from vllm.v1.worker import gpu_worker

MiB = 1024 * 1024


@pytest.mark.parametrize("cuda,sm70", [(True, True), (True, False), (False, False)])
@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_compile_allocator_restored_before_measuring_or_returning(
    monkeypatch, cuda, sm70, explicit, fail
):
    original = "roundup_power2_divisions:[256:1,512:2,>:4],max_split_size_mb:512"
    scoped = original.replace("max_split_size_mb:512", "max_split_size_mb:20")
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", original)
    monkeypatch.setattr(gpu_worker.current_platform, "is_cuda", lambda: cuda)
    monkeypatch.setattr(
        gpu_worker.current_platform, "is_device_capability", lambda _: sm70
    )
    monkeypatch.setattr(
        gpu_worker.envs, "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS", False
    )
    settings: list[str] = []
    monkeypatch.setattr(
        gpu_worker.torch._C, "_accelerator_setAllocatorSettings", settings.append
    )
    worker = gpu_worker.Worker.__new__(gpu_worker.Worker)
    worker.device = torch.device("cpu")
    worker.init_snapshot = SimpleNamespace(
        free_memory=1000 * MiB, torch_memory=10 * MiB
    )
    worker.requested_memory = 800 * MiB
    worker.cache_config = SimpleNamespace(
        kv_cache_memory_bytes=128 * MiB if explicit else None,
        gpu_memory_utilization=0.8,
    )
    worker.vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE)
    )
    worker.use_v2_model_runner = False
    calls = []

    def run():
        calls.append("run")
        if len(calls) == 1:
            assert settings == ([scoped] if cuda and sm70 else [])
            if fail:
                raise RuntimeError("compile failed")
        else:
            assert settings == ([scoped, original] if cuda and sm70 else [])

    worker.model_runner = SimpleNamespace(profile_run=run, model_memory_usage=100 * MiB)

    @contextmanager
    def profile(snapshot, weights_memory):
        assert settings == ([scoped, original] if cuda and sm70 else [])
        calls.append("measure")
        yield SimpleNamespace(
            before_profile=SimpleNamespace(torch_peak=20 * MiB, torch_memory=110 * MiB),
            after_profile=SimpleNamespace(free_memory=700 * MiB),
            non_torch_increase=50 * MiB,
            weights_memory=weights_memory,
        )

    monkeypatch.setattr(gpu_worker, "memory_profiling", profile)
    monkeypatch.setattr(
        gpu_worker.torch.accelerator,
        "memory_stats",
        lambda _: {"allocated_bytes.all.peak": 100 * MiB},
    )
    if fail:
        with pytest.raises(RuntimeError, match="compile failed"):
            worker.determine_available_memory()
        assert calls == ["run"]
    elif explicit:
        assert worker.determine_available_memory() == 128 * MiB
        assert calls == ["run"]
    else:
        assert worker.determine_available_memory() == 570 * MiB
        assert calls == ["run", "measure", "run"]
        assert worker.warmup_torch_memory == 0
        assert worker.peak_activation_memory == 80 * MiB
    assert settings == ([scoped, original] if cuda and sm70 else [])
    assert gpu_worker.os.environ["PYTORCH_CUDA_ALLOC_CONF"] == original
