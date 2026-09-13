# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The SM70 graph gates are pre-Ampere gates on the worker's own device.

Volta and Turing take the same graph tunings. The gate must ask the device
this process selected, not index 0 of the visibility list, which on a mixed
rig belongs to a different worker.
"""

import pytest

from vllm.platforms.interface import DeviceCapability
from vllm.v1.worker.gpu import cudagraph_utils as cg

VOLTA = DeviceCapability(7, 0)
TURING = DeviceCapability(7, 5)
AMPERE = DeviceCapability(8, 0)


def _fake_devices(monkeypatch, capabilities, current: int) -> list[int]:
    asked: list[int] = []

    def get_device_capability(device_id: int = 0) -> DeviceCapability:
        asked.append(device_id)
        return capabilities[device_id]

    monkeypatch.setattr(cg.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        cg.current_platform, "get_device_capability", get_device_capability
    )
    monkeypatch.setattr(cg.torch.accelerator, "current_device_index", lambda: current)
    return asked


@pytest.mark.parametrize(
    ("capability", "expected"),
    [(VOLTA, True), (TURING, True), (AMPERE, False)],
    ids=["volta", "turing", "ampere"],
)
def test_pre_ampere_by_capability(monkeypatch, capability, expected):
    _fake_devices(monkeypatch, [capability], current=0)
    assert cg._worker_device_is_pre_ampere() is expected


def test_gate_asks_the_workers_own_device(monkeypatch):
    # Index 0 is Ampere, this worker sits on the Turing card at index 1.
    asked = _fake_devices(monkeypatch, [AMPERE, TURING], current=1)
    assert cg._worker_device_is_pre_ampere() is True
    assert asked == [1]


def test_gate_is_off_without_cuda(monkeypatch):
    monkeypatch.setattr(cg.current_platform, "is_cuda", lambda: False)
    monkeypatch.setattr(
        cg.current_platform,
        "get_device_capability",
        lambda device_id=0: pytest.fail("no device query without CUDA"),
    )
    assert cg._worker_device_is_pre_ampere() is False
