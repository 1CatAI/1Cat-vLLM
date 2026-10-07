# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FA2 ships one library per architecture; the loader follows the device.

The regular _vllm_fa2_C build covers the CUDA arch list, a Turing rig gets a
second file _vllm_fa2_C_sm75.abi3.so next to it. Both export the module
_vllm_fa2_C, so a process loads exactly one, on first use, for the
capability of the device the ops run on.
"""

import sys
from pathlib import Path

import pytest
import torch

from vllm.platforms.interface import DeviceCapability
from vllm.vllm_flash_attn import flash_attn_interface as fai


@pytest.fixture(autouse=True)
def _fresh_loader_state(monkeypatch):
    monkeypatch.setattr(fai, "_fa2_loaded_capability", None)
    monkeypatch.delitem(sys.modules, fai._FA2_MODULE, raising=False)


def test_turing_path_only_when_the_file_exists(monkeypatch, tmp_path: Path):
    turing = tmp_path / "_vllm_fa2_C_sm75.abi3.so"
    monkeypatch.setattr(fai, "_FA2_TURING_PATH", str(turing))
    assert fai._fa2_library_path((7, 5)) is None
    turing.write_bytes(b"")
    assert fai._fa2_library_path((7, 5)) == str(turing)


def test_other_capabilities_take_the_regular_build(monkeypatch, tmp_path: Path):
    regular = tmp_path / "_vllm_fa2_C.abi3.so"
    monkeypatch.setattr(
        fai, "_FA2_DEFAULT_SPEC", type("Spec", (), {"origin": str(regular)})()
    )
    assert fai._fa2_library_path((7, 0)) == str(regular)
    assert fai._fa2_library_path((8, 0)) == str(regular)
    monkeypatch.setattr(fai, "_FA2_DEFAULT_SPEC", None)
    assert fai._fa2_library_path((8, 0)) is None


def test_loader_picks_the_devices_library_once(monkeypatch, tmp_path: Path):
    # A Python file stands in for the extension: the loader only needs a
    # module spec it can execute.
    stub = tmp_path / "_vllm_fa2_C_sm75.py"
    stub.write_text("LOADED_FOR = 'sm75'\n")
    asked: list[int | None] = []

    def capability(device: int | None) -> DeviceCapability:
        asked.append(device)
        return DeviceCapability(7, 5)

    monkeypatch.setattr(fai.current_platform, "get_device_capability", capability)
    monkeypatch.setattr(
        fai, "_fa2_library_path", lambda cap: str(stub) if cap == (7, 5) else None
    )

    fai.load_fa2_library(torch.device("cuda:1"))
    assert sys.modules[fai._FA2_MODULE].LOADED_FOR == "sm75"
    assert fai._fa2_loaded_capability == (7, 5)
    # The second call does not ask the device again: one library per process.
    fai.load_fa2_library(torch.device("cuda:0"))
    assert asked == [1]


def test_ensure_loads_the_current_devices_library_once(monkeypatch):
    loaded: list[torch.device] = []
    monkeypatch.setattr(fai, "_fa2_loaded_capability", None)

    def forbidden():
        raise AssertionError("Capability probing must not initialize CUDA")

    monkeypatch.setattr(fai.torch.accelerator, "current_device_index", forbidden)

    def load(device: torch.device) -> None:
        loaded.append(device)
        fai._fa2_loaded_capability = (7, 0)

    monkeypatch.setattr(fai, "load_fa2_library", load)
    fai.ensure_fa2_library_loaded()
    fai.ensure_fa2_library_loaded()
    assert loaded == [torch.device("cuda")]


def test_loader_refuses_a_capability_without_library(monkeypatch):
    monkeypatch.setattr(
        fai.current_platform,
        "get_device_capability",
        lambda device: DeviceCapability(7, 5),
    )
    monkeypatch.setattr(fai, "_fa2_library_path", lambda cap: None)
    with pytest.raises(ImportError, match="7.5"):
        fai.load_fa2_library(torch.device("cuda:0"))
    assert fai._fa2_loaded_capability is None


def test_loader_reports_unknown_capability_without_creating_context(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Library selection must not create a CUDA context")

    monkeypatch.setattr(fai.torch.cuda, "_lazy_init", forbidden)
    monkeypatch.setattr(fai.current_platform, "get_device_capability", lambda _: None)
    with pytest.raises(ImportError, match="Cannot determine FA2 device capability"):
        fai.load_fa2_library(torch.device("cuda"))


@pytest.mark.parametrize("capability, supported", [(70, False), (75, True)])
def test_support_query_does_not_initialize_cuda(monkeypatch, capability, supported):
    def forbidden(*args, **kwargs):
        raise AssertionError("Support queries must not create a CUDA context")

    monkeypatch.setattr(fai, "FA2_AVAILABLE", True)
    monkeypatch.setattr(fai.torch.cuda, "_lazy_init", forbidden)
    monkeypatch.setattr(fai.torch.accelerator, "current_device_index", forbidden)
    monkeypatch.setattr(fai.current_platform, "resolve_device_id", lambda _: 2)
    monkeypatch.setattr(
        fai.current_platform,
        "get_device_capability",
        lambda device_id=None: DeviceCapability(capability // 10, capability % 10),
    )
    monkeypatch.setattr(
        fai.current_platform,
        "has_device_capability",
        lambda threshold, device: capability >= threshold,
    )
    monkeypatch.setattr(fai, "_fa2_library_path", lambda _: "installed")
    assert fai._is_fa2_supported()[0] is supported
