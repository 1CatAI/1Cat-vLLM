# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""An unspecified device index resolves to the worker's own device.

``has_device_capability(80)`` and friends default to ``device_id=None``. On
CUDA that means: once the process has selected its device, answer for that
device; before that (engine core, API server), keep index 0 of the visibility
list as before. Explicit indices are untouched.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms.cuda import NonNvmlCudaPlatform, NvmlCudaPlatform
from vllm.platforms.interface import DeviceCapability, Platform

# A node that mixes card generations: Turing first, Volta second, Ampere last.
NODE = {
    0: DeviceCapability(major=7, minor=5),
    1: DeviceCapability(major=7, minor=0),
    2: DeviceCapability(major=8, minor=0),
}


@pytest.fixture
def cuda(monkeypatch) -> SimpleNamespace:
    """A fake CUDA runtime: whether a context exists, which ordinal is
    current, and which lookups (torch after init, NVML before) were made."""
    state = SimpleNamespace(initialized=False, current=0, torch=[], nvml=[])

    def torch_lookup(device_id: int) -> DeviceCapability:
        state.torch.append(device_id)
        return NODE[device_id]

    def nvml_lookup(device_id: int) -> DeviceCapability:
        state.nvml.append(device_id)
        return NODE[device_id]

    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: state.initialized)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: state.current)
    monkeypatch.setattr(NvmlCudaPlatform, "_torch_device_capability", torch_lookup)
    monkeypatch.setattr(NvmlCudaPlatform, "_nvml_device_capability", nvml_lookup)
    monkeypatch.setattr(NonNvmlCudaPlatform, "_device_capability", torch_lookup)
    return state


def test_without_a_cuda_context_the_default_stays_device_zero(cuda) -> None:
    cuda.initialized = False
    cuda.current = 2
    assert NvmlCudaPlatform.resolve_device_id(None) == 0
    assert NvmlCudaPlatform.get_device_capability() == NODE[0]
    assert cuda.nvml == [0] and cuda.torch == []


def test_with_a_cuda_context_the_default_is_the_current_device(cuda) -> None:
    cuda.initialized = True
    cuda.current = 1
    assert NvmlCudaPlatform.resolve_device_id(None) == 1
    assert NvmlCudaPlatform.get_device_capability() == NODE[1]


def test_after_initialization_torch_answers_and_nvml_is_left_alone(cuda) -> None:
    """NVML counts in PCI bus order, the CUDA runtime by default fastest
    first; once the process has a context, the torch ordinal is the truth."""
    cuda.initialized = True
    cuda.current = 2
    assert NvmlCudaPlatform.get_device_capability(1) == NODE[1]
    assert cuda.torch == [1] and cuda.nvml == []


def test_an_explicit_index_always_wins(cuda) -> None:
    cuda.initialized = True
    cuda.current = 1
    assert NvmlCudaPlatform.resolve_device_id(2) == 2
    assert NvmlCudaPlatform.get_device_capability(2) == NODE[2]
    assert NvmlCudaPlatform.has_device_capability(80, device_id=2)
    assert not NvmlCudaPlatform.has_device_capability(80, device_id=0)


@pytest.mark.parametrize(
    ("current", "has_80", "is_70", "family_70"),
    [(0, False, False, True), (1, False, True, True), (2, True, False, False)],
)
def test_every_query_follows_the_current_device(
    cuda, current: int, has_80: bool, is_70: bool, family_70: bool
) -> None:
    cuda.initialized = True
    cuda.current = current
    assert NvmlCudaPlatform.has_device_capability(80) is has_80
    assert NvmlCudaPlatform.is_device_capability(70) is is_70
    assert NvmlCudaPlatform.is_device_capability_family(70) is family_70


def test_the_cache_is_addressed_by_the_resolved_index(cuda) -> None:
    """The lookup behind the cache must see the concrete index, never None,
    so an unspecified device cannot freeze one process-wide answer."""
    cuda.initialized = True
    cuda.current = 1
    NvmlCudaPlatform.get_device_capability()
    cuda.current = 2
    NvmlCudaPlatform.get_device_capability()
    assert cuda.torch == [1, 2]


def test_non_nvml_platform_resolves_the_same_way(cuda) -> None:
    cuda.initialized = True
    cuda.current = 2
    assert NonNvmlCudaPlatform.get_device_capability() == NODE[2]
    assert NonNvmlCudaPlatform.has_device_capability(80)


def test_an_unusable_index_answers_none_after_initialization(monkeypatch) -> None:
    """The NVML path answers None for an index it cannot resolve; the torch
    path must keep that contract instead of raising torch's AssertionError."""

    def invalid(device_id: int) -> tuple[int, int]:
        raise AssertionError("Invalid device id")

    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", invalid)
    NvmlCudaPlatform._torch_device_capability.cache_clear()
    try:
        assert NvmlCudaPlatform.get_device_capability(7) is None
        assert not NvmlCudaPlatform.has_device_capability(80, device_id=7)
    finally:
        NvmlCudaPlatform._torch_device_capability.cache_clear()


def test_the_base_platform_keeps_device_zero() -> None:
    assert Platform.resolve_device_id(None) == 0
    assert Platform.resolve_device_id(3) == 3
    assert not Platform.has_device_capability(70)
