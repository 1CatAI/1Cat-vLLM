# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact shared-host result transport for CUDA PLE consumers.

Only small result/flag buffers are registered. Table placement stays with the
existing storage planner. Consumers copy bytes on their own model stream;
the CPU producer publishes with system acquire/release atomics.
"""

import ctypes
import sys
import time
from dataclasses import dataclass, field
from functools import lru_cache

import torch
import torch.multiprocessing as torch_mp
from cuda.bindings import driver as cuda_driver


@lru_cache(maxsize=1)
def _atomics():
    if sys.platform != "linux":
        return None
    try:
        library = ctypes.CDLL("libatomic.so.1")
        load = library.__atomic_load_4
        load.argtypes = (ctypes.c_void_p, ctypes.c_int)
        load.restype = ctypes.c_uint32
        store = library.__atomic_store_4
        store.argtypes = (ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int)
        store.restype = None
        return library, load, store
    except (OSError, AttributeError):
        return None


def publish_host_flag(flag: torch.Tensor) -> None:
    """Release all result writes before making them visible to the GPU."""
    if flag.device.type != "cpu" or flag.dtype != torch.int32 or not flag.numel():
        raise ValueError("Host completion flag must be a nonempty CPU int32 tensor")
    functions = _atomics()
    if functions is None:
        raise RuntimeError("System 32-bit acquire/release atomics are unavailable")
    functions[2](flag.data_ptr(), 1, 3)  # GCC __ATOMIC_RELEASE.


def wait_host_resets(flags: list[torch.Tensor], timeout_s: float = 30) -> None:
    """Wait until every consumer has finished reading the previous result."""
    functions = _atomics()
    if functions is None:
        raise RuntimeError("System 32-bit acquire/release atomics are unavailable")
    if any(
        f.device.type != "cpu" or f.dtype != torch.int32 or not f.numel() for f in flags
    ):
        raise ValueError("Host completion flags must be nonempty CPU int32 tensors")
    pointers = [f.data_ptr() for f in flags]
    deadline = time.monotonic() + timeout_s
    while any(functions[1](p, 2) != 0 for p in pointers):  # __ATOMIC_ACQUIRE.
        if time.monotonic() >= deadline:
            raise TimeoutError("PLE consumers did not acknowledge the previous result")
        time.sleep(0)


def host_result_capability(device: torch.device) -> str | None:
    """Return a rejection reason; no model, TP count or exact-size predicate."""
    if device.type != "cuda" or device.index is None:
        return "cuda_device_required"
    if _atomics() is None:
        return "system_atomics_unavailable"
    if not hasattr(torch.ops._C, "get_cuda_view_from_cpu_tensor"):
        return "native_mapped_view_unavailable"
    for attribute in (
        cuda_driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY,
        # The v1 capability is deprecated and reports zero on CUDA 12 drivers.
        # The current mem-op capability covers cuStreamWaitValue32 as well.
        cuda_driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS,
    ):
        result = cuda_driver.cuDeviceGetAttribute(
            attribute, cuda_driver.CUdevice(device.index)
        )
        if result[0].value != 0 or not result[1]:
            return "mapped_stream_memory_ops_unavailable"
    return None


def _shared_zeros(shape, dtype):
    # Registration must survive the connector's file-descriptor serializer.
    strategy = torch_mp.get_sharing_strategy()
    torch_mp.set_sharing_strategy("file_descriptor")
    try:
        return torch.zeros(shape, dtype=dtype, device="cpu").share_memory_()
    finally:
        torch_mp.set_sharing_strategy(strategy)


@dataclass
class HostResultRegion:
    device: torch.device
    result: torch.Tensor
    flag: torch.Tensor
    cuda_flag: torch.Tensor | None = None
    registered: list[tuple[torch.Tensor, int, int]] = field(default_factory=list)

    @classmethod
    def create(cls, gpu_result: torch.Tensor) -> "HostResultRegion":
        if (
            not gpu_result.is_cuda
            or not gpu_result.is_contiguous()
            or gpu_result.ndim != 2
            or min(gpu_result.shape) <= 0
        ):
            raise ValueError("PLE result must be a nonempty contiguous CUDA matrix")
        device = gpu_result.device
        reason = host_result_capability(device)
        if reason is not None:
            raise RuntimeError(reason)
        region = cls(
            device,
            _shared_zeros(gpu_result.shape, gpu_result.dtype),
            _shared_zeros((16,), torch.int32),
        )
        try:
            with torch.accelerator.device_index(device.index):
                for tensor in (region.result, region.flag):
                    status = cuda_driver.cuMemHostRegister(
                        tensor.data_ptr(), tensor.numel() * tensor.element_size(), 3
                    )
                    if status[0].value != 0:
                        raise RuntimeError(f"PLE host registration failed: {status[0]}")
                    region.registered.append(
                        (
                            tensor,
                            tensor.data_ptr(),
                            tensor.numel() * tensor.element_size(),
                        )
                    )
                region.cuda_flag = torch.ops._C.get_cuda_view_from_cpu_tensor(
                    region.flag[:1]
                )
                if (
                    region.cuda_flag.device != device
                    or region.cuda_flag.data_ptr() != region.flag.data_ptr()
                ):
                    raise RuntimeError(
                        "PLE flag does not have a stable mapped CUDA view"
                    )
            return region
        except Exception:
            region.close()
            raise

    @property
    def pinned_bytes(self) -> int:
        return sum(size for _, _, size in self.registered)

    def validate_registration(self) -> None:
        if any(
            t.data_ptr() != pointer or not t.is_pinned()
            for t, pointer, _ in self.registered
        ) or (
            self.cuda_flag is None
            or not self.result.is_pinned()
            or self.cuda_flag.data_ptr() != self.flag.data_ptr()
        ):
            raise RuntimeError("PLE IPC serialization changed registered host storage")

    def close(self) -> None:
        if not self.registered:
            return
        with torch.accelerator.device_index(self.device.index):
            torch.accelerator.synchronize()
            for _, pointer, _ in reversed(self.registered):
                status = cuda_driver.cuMemHostUnregister(pointer)
                if status[0].value != 0:
                    raise RuntimeError(f"PLE host unregistration failed: {status[0]}")
        self.registered.clear()
