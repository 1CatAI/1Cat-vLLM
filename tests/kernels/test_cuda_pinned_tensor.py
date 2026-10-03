# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resident PLE storage must own an exact allocation, outside Torch's cache."""

import gc

import pytest
import torch
import vllm._C  # noqa: F401


def test_empty_pinned_tensor_needs_no_allocation():
    reference = torch.empty(0, dtype=torch.float16)
    output = torch.ops._C.create_cuda_pinned_tensor(reference, [0, 160])
    assert output.shape == (0, 160)
    assert output.device.type == "cpu"
    assert output.dtype == reference.dtype


@pytest.mark.parametrize("shape", [[-1], [2**62, 8]])
def test_invalid_pinned_tensor_shape_rejected(shape):
    with pytest.raises(RuntimeError, match="nonnegative|overflow"):
        torch.ops._C.create_cuda_pinned_tensor(torch.empty(0), shape)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA host mapping")
@pytest.mark.parametrize("dtype", [torch.uint8, torch.float16, torch.float8_e4m3fn])
def test_exact_pinned_storage_mapping_and_lifetime(dtype):
    # 3 MiB would round to 4 MiB in Torch's caching host allocator. The native
    # allocation must leave its reservation counters unchanged.
    torch.empty(1, device="cuda")  # Initialize the host statistics interface.
    reference = torch.empty(0, dtype=dtype)
    count = 3 * 1024**2 // reference.element_size()
    before = torch.cuda.memory.host_memory_stats()["allocated_bytes.current"]
    host = torch.ops._C.create_cuda_pinned_tensor(reference, [count])
    after = torch.cuda.memory.host_memory_stats()["allocated_bytes.current"]
    assert before == after
    assert host.is_pinned()
    assert host.nbytes == 3 * 1024**2
    host.view(torch.uint8).fill_(17)
    view = torch.ops._C.get_cuda_view_from_cpu_tensor(host)
    assert view.device.type == "cuda"
    assert view.dtype == dtype
    assert torch.all(view.view(torch.uint8).cpu() == 17)
    host.view(torch.uint8).fill_(29)
    del host
    gc.collect()
    # The alias retains the allocation after its Python tensor goes away.
    assert torch.all(view.view(torch.uint8).cpu() == 29)
    del view
    gc.collect()
