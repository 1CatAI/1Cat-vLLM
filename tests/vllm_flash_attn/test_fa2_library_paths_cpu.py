# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check FA2 installation layouts without importing Torch or initializing CUDA."""

import ast
import os
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = (
    Path(__file__).resolve().parents[2] / "vllm/vllm_flash_attn/flash_attn_interface.py"
)


def load_path_selector(tmp_path, installed):
    """Execute the real discovery/selection code against an isolated layout."""
    for filename in installed:
        (tmp_path / filename).touch()
    regular = tmp_path / "_vllm_fa2_C.abi3.so"
    default_spec = SimpleNamespace(origin=str(regular)) if regular.exists() else None
    tree = ast.parse(SOURCE.read_text())
    start = next(
        i
        for i, node in enumerate(tree.body)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "_FA2_MODULE" for t in node.targets)
    )
    end = next(
        i
        for i, node in enumerate(tree.body)
        if isinstance(node, ast.FunctionDef) and node.name == "load_fa2_library"
    )
    provider = runpy.run_path(str(SOURCE.with_name("sm70_library.py")))
    namespace = {
        "has_qualified_fa2_library": provider["has_qualified_fa2_library"],
        "qualified_fa2_library_path": provider["qualified_fa2_library_path"],
        "__file__": str(tmp_path / "flash_attn_interface.py"),
        "__package__": "vllm.vllm_flash_attn",
        "os": os,
        "importlib": SimpleNamespace(
            util=SimpleNamespace(find_spec=lambda name: default_spec)
        ),
    }
    exec(
        compile(
            ast.Module(body=tree.body[start:end], type_ignores=[]), str(SOURCE), "exec"
        ),
        namespace,
    )
    return namespace


@pytest.mark.parametrize("capability", [(7, 0), (8, 0), (8, 6), (8, 9), (9, 0)])
def test_regular_install_falls_back_to_default(tmp_path, capability):
    state = load_path_selector(tmp_path, ["_vllm_fa2_C.abi3.so"])
    assert state["FA2_AVAILABLE"]
    assert state["FA2_UNAVAILABLE_REASON"] is None
    assert state["_fa2_library_path"](capability) == str(
        tmp_path / "_vllm_fa2_C.abi3.so"
    )


@pytest.mark.parametrize(
    ("capability", "filename"),
    [
        ((7, 0), "_vllm_fa2_C_sm70.abi3.so"),
        ((7, 5), "_vllm_fa2_C_sm75.abi3.so"),
        ((8, 0), "_vllm_fa2_C_sm80.abi3.so"),
        ((8, 6), "_vllm_fa2_C_sm80.abi3.so"),
        ((8, 9), "_vllm_fa2_C_sm80.abi3.so"),
    ],
)
@pytest.mark.parametrize("with_default", [False, True])
def test_qualified_library_is_available_and_preferred(
    tmp_path, capability, filename, with_default
):
    installed = [filename]
    if with_default:
        installed.append("_vllm_fa2_C.abi3.so")
    state = load_path_selector(tmp_path, installed)
    assert state["FA2_AVAILABLE"]
    assert state["FA2_UNAVAILABLE_REASON"] is None
    assert state["_fa2_library_path"](capability) == str(tmp_path / filename)


def test_mixed_production_layout_selects_each_workers_library(tmp_path):
    state = load_path_selector(
        tmp_path,
        [
            "_vllm_fa2_C.abi3.so",
            "_vllm_fa2_C_sm70.abi3.so",
            "_vllm_fa2_C_sm80.abi3.so",
        ],
    )
    select = state["_fa2_library_path"]
    assert select((7, 0)) == str(tmp_path / "_vllm_fa2_C_sm70.abi3.so")
    for capability in [(8, 6), (8, 9)]:
        assert select(capability) == str(tmp_path / "_vllm_fa2_C_sm80.abi3.so")


def test_missing_architecture_does_not_use_an_unrelated_library(tmp_path):
    state = load_path_selector(tmp_path, ["_vllm_fa2_C_sm70.abi3.so"])
    assert state["FA2_AVAILABLE"]
    assert state["_fa2_library_path"]((8, 0)) is None


@pytest.mark.parametrize("capability", [(8, 7), (9, 0), (10, 0), (12, 0)])
@pytest.mark.parametrize("with_default", [False, True])
def test_unqualified_architecture_keeps_ordinary_library(
    tmp_path, capability, with_default
):
    installed = ["_vllm_fa2_C_sm80.abi3.so"]
    if with_default:
        installed.append("_vllm_fa2_C.abi3.so")
    state = load_path_selector(tmp_path, installed)
    expected = str(tmp_path / "_vllm_fa2_C.abi3.so") if with_default else None
    assert state["_fa2_library_path"](capability) == expected


def test_turing_still_requires_its_dedicated_library(tmp_path):
    state = load_path_selector(tmp_path, ["_vllm_fa2_C.abi3.so"])
    assert state["_fa2_library_path"]((7, 5)) is None


def test_no_installed_library_is_unavailable(tmp_path):
    state = load_path_selector(tmp_path, [])
    assert not state["FA2_AVAILABLE"]
    assert (
        "no vllm.vllm_flash_attn._vllm_fa2_C library" in state["FA2_UNAVAILABLE_REASON"]
    )
    for capability in [(7, 0), (7, 5), (8, 0), (8, 9)]:
        assert state["_fa2_library_path"](capability) is None
