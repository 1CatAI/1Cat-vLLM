# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

from vllm.models.deepseek_v4.common.ops import cache_utils


@pytest.mark.parametrize(
    "capability,software",
    [
        ((7, 0), True),
        ((7, 2), True),
        ((7, 5), True),
        ((8, 0), True),
        ((8, 6), True),
        ((8, 9), False),
        ((9, 0), False),
        ((10, 0), False),
    ],
)
def test_fp8_route_follows_device_capability(monkeypatch, capability, software):
    platform = SimpleNamespace(
        is_cuda=lambda: True,
        has_device_capability=lambda minimum: capability >= minimum,
    )
    monkeypatch.setattr(cache_utils, "current_platform", platform)
    assert cache_utils.needs_software_fp8() is software


def test_non_cuda_keeps_its_existing_route_without_querying_cuda(monkeypatch):
    def unexpected_query(*args):
        pytest.fail("CUDA capability must not be queried on another platform")

    monkeypatch.setattr(
        cache_utils,
        "current_platform",
        SimpleNamespace(is_cuda=lambda: False, has_device_capability=unexpected_query),
    )
    assert not cache_utils.needs_software_fp8()


@pytest.mark.parametrize("capability", [(7, 5), (8, 0), (8, 6)])
def test_pre_fp8_devices_do_not_enter_the_cutedsl_gather(monkeypatch, capability):
    monkeypatch.setattr(
        cache_utils,
        "current_platform",
        SimpleNamespace(
            is_cuda=lambda: True,
            has_device_capability=lambda minimum: capability >= minimum,
        ),
    )
    monkeypatch.setattr(cache_utils, "has_cutedsl", lambda: True)
    calls = []
    monkeypatch.setattr(
        cache_utils,
        "dequantize_and_gather_k_cache_triton",
        lambda *args: calls.append(args),
    )
    buffers = [torch.empty(1) for _ in range(5)]
    cache_utils.dequantize_and_gather_k_cache(*buffers, 16, 0)
    assert calls == [(*buffers, 16, 0)]


@pytest.mark.parametrize("capability", [(7, 5), (8, 0), (8, 6), (8, 9)])
def test_insert_launcher_passes_the_hardware_route_to_the_kernel(
    monkeypatch, capability
):
    monkeypatch.setattr(
        cache_utils,
        "current_platform",
        SimpleNamespace(
            is_cuda=lambda: True,
            has_device_capability=lambda minimum: capability >= minimum,
        ),
    )
    calls = []

    class Kernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                calls.append((grid, kwargs["use_software_fp8"]))

            return launch

    monkeypatch.setattr(cache_utils, "quantize_and_insert_k_kernel", Kernel())
    cache_utils.quantize_and_insert_k_cache(
        k=torch.empty(2, 512, dtype=torch.float16),
        slot_mapping=torch.tensor([0, 1]),
        k_cache=torch.empty(1, 16 * 584, dtype=torch.uint8),
        block_size=16,
    )
    assert calls == [((2,), capability < (8, 9))]
