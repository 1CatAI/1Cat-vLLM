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
    asked: list[torch.device] = []

    def capability(device: torch.device) -> tuple[int, int]:
        asked.append(device)
        return (7, 5)

    monkeypatch.setattr(fai.torch.cuda, "get_device_capability", capability)
    monkeypatch.setattr(
        fai, "_fa2_library_path", lambda cap: str(stub) if cap == (7, 5) else None
    )

    fai.load_fa2_library(torch.device("cuda:1"))
    assert sys.modules[fai._FA2_MODULE].LOADED_FOR == "sm75"
    assert fai._fa2_loaded_capability == (7, 5)
    # The second call does not ask the device again: one library per process.
    fai.load_fa2_library(torch.device("cuda:0"))
    assert asked == [torch.device("cuda:1")]


def test_ensure_loads_the_current_devices_library_once(monkeypatch):
    loaded: list[torch.device] = []
    monkeypatch.setattr(fai, "_fa2_loaded_capability", None)
    monkeypatch.setattr(fai.torch.accelerator, "current_device_index", lambda: 2)

    def load(device: torch.device) -> None:
        loaded.append(device)
        fai._fa2_loaded_capability = (7, 0)

    monkeypatch.setattr(fai, "load_fa2_library", load)
    fai.ensure_fa2_library_loaded()
    fai.ensure_fa2_library_loaded()
    assert loaded == [torch.device("cuda", 2)]


def test_loader_refuses_a_capability_without_library(monkeypatch):
    monkeypatch.setattr(fai.torch.cuda, "get_device_capability", lambda device: (7, 5))
    monkeypatch.setattr(fai, "_fa2_library_path", lambda cap: None)
    with pytest.raises(ImportError, match="7.5"):
        fai.load_fa2_library(torch.device("cuda:0"))
    assert fai._fa2_loaded_capability is None
