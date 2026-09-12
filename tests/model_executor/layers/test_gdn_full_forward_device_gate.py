# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device gate for the Qwen GDN full-forward wrapper.

The wrapper runs the whole layer through one opaque custom op so that the
projections around the recurrent core keep a strict eager order. That is a
Volta tuning; on Turing it only costs the Inductor fusions around the gated
RMSNorm. The gate therefore has to answer for the device this worker builds
its layers on -- not for device 0, which is the same card for every rank on a
node that mixes Volta and Turing.
"""

import pytest
import torch

from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    _sm70_current_device_is_volta,
)
from vllm.platforms import current_platform


@pytest.fixture
def devices(monkeypatch):
    """Pin a per-device capability map and the index this worker is on."""

    def _apply(capabilities: dict[int, tuple[int, int]], current: int):
        monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
        monkeypatch.setattr(
            current_platform,
            "is_device_capability",
            lambda capability, device_id=0: capabilities.get(device_id) == capability,
        )
        monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: current)

    return _apply


def test_volta_worker_arms_the_wrapper(devices):
    devices({0: (7, 0)}, current=0)
    assert _sm70_current_device_is_volta()


def test_turing_worker_does_not_arm_the_wrapper(devices):
    devices({0: (7, 5)}, current=0)
    assert not _sm70_current_device_is_volta()


def test_ampere_worker_does_not_arm_the_wrapper(devices):
    devices({0: (8, 6)}, current=0)
    assert not _sm70_current_device_is_volta()


def test_mixed_node_answers_for_the_current_device(devices):
    """Turing at index 0, Volta at index 2: each rank must see its own card."""
    capabilities = {0: (7, 5), 1: (7, 5), 2: (7, 0), 3: (7, 0)}
    devices(capabilities, current=2)
    assert _sm70_current_device_is_volta()
    devices(capabilities, current=0)
    assert not _sm70_current_device_is_volta()


def test_non_cuda_platform_does_not_arm_the_wrapper(monkeypatch):
    monkeypatch.setattr(current_platform, "is_cuda", lambda: False)
    assert not _sm70_current_device_is_volta()
