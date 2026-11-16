# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.distributed.device_communicators.cuda_wrapper import (
    CUDA_LIMIT_STACK_SIZE,
    CudaRTLibrary,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_lowering_the_stack_limit_returns_its_local_memory():
    # The worker restores the stack limit after a cold-compile warm-up; this
    # only helps if the driver hands the grown local memory back.
    torch.cuda.init()
    torch.accelerator.synchronize()
    cudart = CudaRTLibrary()
    original = cudart.cudaDeviceGetLimit(CUDA_LIMIT_STACK_SIZE)
    free_original, _ = torch.cuda.mem_get_info()
    try:
        cudart.cudaDeviceSetLimit(CUDA_LIMIT_STACK_SIZE, 10240)
        assert cudart.cudaDeviceGetLimit(CUDA_LIMIT_STACK_SIZE) >= 10240
        free_grown, _ = torch.cuda.mem_get_info()
    finally:
        cudart.cudaDeviceSetLimit(CUDA_LIMIT_STACK_SIZE, original)
    free_restored, _ = torch.cuda.mem_get_info()

    assert cudart.cudaDeviceGetLimit(CUDA_LIMIT_STACK_SIZE) == original
    grown = free_original - free_grown
    # At least 4 KiB more stack for every thread the device can hold.
    props = torch.cuda.get_device_properties(torch.accelerator.current_device_index())
    threads = props.multi_processor_count * props.max_threads_per_multi_processor
    assert grown >= threads * 4096
    # Restoring the limit gives back what growing it took (driver page slack).
    assert free_restored >= free_original - (64 << 20)
