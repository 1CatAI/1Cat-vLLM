# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU lease selection with a CPU-only NVML inventory, including a display card."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "h3_visibility", Path(__file__).resolve().parents[2] / "vllm/video/gpu.py"
)
assert spec is not None and spec.loader is not None
gpu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gpu)


@pytest.fixture
def inventory(monkeypatch, tmp_path):
    memory = [2] + [32] * 8
    busy: set[int] = set()
    nvml = SimpleNamespace(
        nvmlInit=lambda: None,
        nvmlShutdown=lambda: None,
        nvmlDeviceGetCount=lambda: len(memory),
        nvmlDeviceGetHandleByIndex=lambda i: i,
        nvmlDeviceGetHandleByUUID=lambda u: int(u.removeprefix("GPU-")),
        nvmlDeviceGetIndex=lambda i: i,
        nvmlDeviceGetUUID=lambda i: f"GPU-{i}",
        nvmlDeviceGetMemoryInfo=lambda i: SimpleNamespace(
            total=memory[i] * 1024**3, free=memory[i] * 1024**3
        ),
        nvmlDeviceGetComputeRunningProcesses=lambda i: (
            [SimpleNamespace(usedGpuMemory=1024**3)] if i in busy else []
        ),
    )
    monkeypatch.setitem(sys.modules, "pynvml", nvml)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(gpu, "LOCK_ROOT", tmp_path)
    return memory, busy


def test_display_gpu_does_not_hide_four_card_group(inventory):
    assert gpu._available_gpu_groups(4) == [(1, 2, 3, 4), (5, 6, 7, 8)]
    assert gpu.worker_device_mask((1, 2, 3, 4)) == "GPU-1,GPU-2,GPU-3,GPU-4"


@pytest.mark.parametrize("mask", ["5,6,7,8", "GPU-5,GPU-6,GPU-7,GPU-8"])
def test_explicit_group_never_falls_back_to_unselected_gpus(
    inventory, monkeypatch, mask
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
    assert gpu._available_gpu_groups(4) == [(5, 6, 7, 8)]
    inventory[1].add(5)
    assert gpu._available_gpu_groups(4) == []
    with pytest.raises(RuntimeError, match="unleased"):
        gpu.acquire_gpu_group(4)


@pytest.mark.parametrize("mask", ["", "-1", "1,2"])
def test_empty_or_insufficient_visibility_never_expands(inventory, monkeypatch, mask):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
    assert gpu._available_gpu_groups(4) == []


@pytest.mark.parametrize("mask", ["1,1,2,3", "1,2,3,99", "1,2,3,bad"])
def test_invalid_masks_are_refused(inventory, monkeypatch, mask):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
    with pytest.raises(ValueError):
        gpu._available_gpu_groups(4)


def test_selected_lock_is_respected_even_when_other_gpus_are_idle(
    inventory, monkeypatch
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,2,3,4")
    with gpu.GPUGroupLease((1, 2, 3, 4)), pytest.raises(RuntimeError, match="unleased"):
        gpu.acquire_gpu_group(4)
    with gpu.acquire_gpu_group(4) as lease:
        assert lease.gpu_ids == (1, 2, 3, 4)
