# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The SM70 platform gate must answer for the worker's own device.

``is_exact_sm70_cuda_platform`` decides whether a worker takes the native
SM70 routes -- the TurboMind NVFP4/MXFP4 MoE paths, the SM70 W8A16-FP8 scheme
and four gates in ``modelopt``. On a node that mixes card generations, device 0
of the visibility list belongs to a different worker, so probing it answers the
wrong question in both directions.
"""

import pytest

from vllm.model_executor.layers.quantization import sm70_turbomind as sm70_tm

SM70 = (7, 0)
SM75 = (7, 5)


@pytest.fixture
def mixed_node(monkeypatch):
    """A node whose device 0 is Turing and whose device 1 is Volta."""
    capabilities = {0: SM75, 1: SM70}
    current = {"index": 0}

    def fake_is_device_capability(capability, device_id=0):
        return capabilities[device_id] == capability

    monkeypatch.setattr(sm70_tm.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        sm70_tm.current_platform,
        "is_device_capability",
        fake_is_device_capability,
    )
    monkeypatch.setattr(
        sm70_tm.torch.accelerator,
        "current_device_index",
        lambda: current["index"],
    )
    return current


def test_gate_is_true_on_the_volta_worker(mixed_node) -> None:
    mixed_node["index"] = 1
    assert sm70_tm.is_exact_sm70_cuda_platform()


def test_gate_is_false_on_the_turing_worker(mixed_node) -> None:
    mixed_node["index"] = 0
    assert not sm70_tm.is_exact_sm70_cuda_platform()


def test_gate_does_not_answer_for_device_zero(mixed_node) -> None:
    """The regression: device 0 is Turing, so a device-0 probe reports False
    for every rank -- including the Volta ones that own the SM70 kernels."""
    mixed_node["index"] = 1
    assert sm70_tm.is_exact_sm70_cuda_platform(), (
        "the Volta worker must not inherit device 0's Turing capability"
    )


def test_moe_route_selection_follows_the_gate(monkeypatch, mixed_node) -> None:
    """The MoE helpers are the reason this matters: they gate the native SM70
    kernels on this very function."""
    monkeypatch.setenv("VLLM_SM70_NVFP4_TURBOMIND", "1")
    monkeypatch.setenv("VLLM_SM70_MXFP4_TURBOMIND", "1")

    mixed_node["index"] = 1
    assert sm70_tm.should_use_nvfp4_moe_turbomind()
    assert sm70_tm.should_use_mxfp4_moe_turbomind()

    mixed_node["index"] = 0
    assert not sm70_tm.should_use_nvfp4_moe_turbomind()
    assert not sm70_tm.should_use_mxfp4_moe_turbomind()


def test_non_cuda_platform_is_never_sm70(monkeypatch) -> None:
    monkeypatch.setattr(sm70_tm.current_platform, "is_cuda", lambda: False)
    assert not sm70_tm.is_exact_sm70_cuda_platform()
