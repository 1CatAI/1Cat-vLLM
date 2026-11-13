# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capability checks run inside compiled model code (plain `vllm serve`)."""

import functools

import pytest
import torch

import vllm.platforms
from vllm.platforms.interface import DeviceCapability, Platform


class _CachedCapabilityPlatform(Platform):
    @classmethod
    @functools.lru_cache
    def get_device_capability(cls, device_id: int | None = None):
        # Like the CUDA platform: a C-implemented cache Dynamo cannot trace.
        return DeviceCapability(7, 0)


@pytest.mark.parametrize(
    "check,expected",
    [
        (lambda p: p.is_device_capability(70), True),
        (lambda p: p.is_device_capability((7, 5)), False),
        (lambda p: p.has_device_capability(60), True),
        (lambda p: p.is_device_capability_family(75), True),
    ],
)
def test_capability_checks_compile_in_fullgraph(monkeypatch, check, expected):
    monkeypatch.setattr(vllm.platforms, "current_platform", _CachedCapabilityPlatform)

    @torch.compile(fullgraph=True, backend="eager")
    def forward(x):
        return x + 1 if check(_CachedCapabilityPlatform) else x - 1

    assert forward(torch.zeros(1)).item() == (1 if expected else -1)
    # Eager callers keep the ordinary query.
    assert check(_CachedCapabilityPlatform) is expected
