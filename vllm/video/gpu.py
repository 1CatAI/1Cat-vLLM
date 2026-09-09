# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Coordinate H3 with other cooperating V100 jobs before workers allocate memory."""

import fcntl
import os
from pathlib import Path

LOCK_ROOT = Path("/tmp")


def _available_gpu_groups(tp: int) -> list[tuple[int, ...]]:
    import pynvml as nvml

    if tp < 1:
        raise ValueError("H3 tensor parallel size must be positive")
    groups = []
    nvml.nvmlInit()
    try:
        count = nvml.nvmlDeviceGetCount()
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible is not None:
            if not visible.strip() or visible.strip() == "-1":
                return []
            indices = []
            for token in visible.split(","):
                token = token.strip()
                if token.startswith("GPU-"):
                    index = nvml.nvmlDeviceGetIndex(
                        nvml.nvmlDeviceGetHandleByUUID(token)
                    )
                elif token.isdecimal() and int(token) < count:
                    index = int(token)
                else:
                    raise ValueError(
                        "H3 GPU selection must contain NVML indices or GPU UUIDs"
                    )
                if index in indices:
                    raise ValueError("H3 GPU selection contains a duplicate device")
                indices.append(index)
        else:
            # A display GPU must not shift the four-card V100 group boundaries.
            indices = []
            for index in range(count):
                handle = nvml.nvmlDeviceGetHandleByIndex(index)
                if nvml.nvmlDeviceGetMemoryInfo(handle).total >= 30 * 1024**3:
                    indices.append(index)
        for start in range(0, len(indices), tp):
            group = tuple(indices[start : start + tp])
            if len(group) != tp:
                continue
            available = True
            for index in group:
                handle = nvml.nvmlDeviceGetHandleByIndex(index)
                processes = nvml.nvmlDeviceGetComputeRunningProcesses(handle)
                # Ignore a small desktop CUDA allocation; never displace jobs.
                if any(p.usedGpuMemory > 256 * 1024**2 for p in processes):
                    available = False
                if nvml.nvmlDeviceGetMemoryInfo(handle).free < 30 * 1024**3:
                    available = False
            if available:
                groups.append(group)
        return groups
    finally:
        nvml.nvmlShutdown()


def worker_device_mask(gpu_ids: tuple[int, ...]) -> str:
    """Use UUIDs when entering CUDA; NVML and CUDA ordinal order may differ."""
    import pynvml as nvml

    nvml.nvmlInit()
    try:
        uuids = [
            nvml.nvmlDeviceGetUUID(nvml.nvmlDeviceGetHandleByIndex(i)) for i in gpu_ids
        ]
        return ",".join(u.decode() if isinstance(u, bytes) else u for u in uuids)
    finally:
        nvml.nvmlShutdown()


def select_gpu_group(tp: int) -> tuple[int, ...]:
    """Read-only capacity probe; launchers should hold acquire_gpu_group instead."""
    groups = _available_gpu_groups(tp)
    if groups:
        return groups[0]
    raise RuntimeError("Neither configured GPU group has enough free memory")


class GPUGroupLease:
    def __init__(self, gpu_ids: tuple[int, ...]):
        self.gpu_ids = gpu_ids
        self._files = []
        names = [*(f"gpu{i}" for i in gpu_ids), "gpus" + "".join(map(str, gpu_ids))]
        try:
            for name in names:
                file = (LOCK_ROOT / f"1cat-vllm-v100-{name}.lock").open("a+")
                self._files.append(file)
                fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            self.close()
            raise

    def close(self):
        for file in self._files:
            file.close()
        self._files.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def acquire_gpu_group(tp: int) -> GPUGroupLease:
    for indices in _available_gpu_groups(tp):
        try:
            lease = GPUGroupLease(indices)
        except BlockingIOError:
            continue
        try:
            # Recheck after locking: a capacity probe alone races other loaders.
            if indices not in _available_gpu_groups(tp):
                lease.close()
                continue
            for file in lease._files:
                file.seek(0)
                file.truncate()
                file.write(f"pid={os.getpid()} task=native-h3 gpus={indices}\n")
                file.flush()
            return lease
        except BaseException:
            lease.close()
            raise
    raise RuntimeError("No free, unleased H3 GPU group; existing jobs remain active")
