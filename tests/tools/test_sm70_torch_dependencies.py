# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Exercise setup's real dependency selection without importing build tooling."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from packaging.tags import Tag

ROOT = Path(__file__).resolve().parents[2]


def requirements(monkeypatch, interpreter, architecture, *, sm70=True, pin=False):
    monkeypatch.setenv("ONECAT_VLLM_PIN_TORCH_CU128", str(int(pin)))
    tree = ast.parse((ROOT / "setup.py").read_text())
    nodes: list[ast.stmt] = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "ONECAT_TORCH_CU128_URLS"
                for target in node.targets
            )
        )
        or (
            isinstance(node, ast.FunctionDef)
            and node.name in {"_onecat_torch_cu128_urls", "get_requirements"}
        )
    ]
    namespace: dict[str, Any] = {
        "ROOT_DIR": ROOT,
        "os": os,
        "sysconfig": SimpleNamespace(get_platform=lambda: f"linux-{architecture}"),
        "sys_tags": lambda: iter(
            [Tag(interpreter, interpreter, "linux_" + architecture)]
        ),
        "torch": SimpleNamespace(version=SimpleNamespace(cuda="12.8")),
        "_no_device": lambda: False,
        "_is_cuda": lambda: True,
        "_cuda_arch_contains": lambda *_: sm70,
    }
    exec(
        compile(ast.Module(nodes, type_ignores=[]), str(ROOT / "setup.py"), "exec"),
        namespace,
    )
    return namespace["get_requirements"]()


@pytest.mark.parametrize("interpreter", ["cp310", "cp311", "cp312", "cp313"])
@pytest.mark.parametrize("architecture", ["x86_64", "aarch64"])
@pytest.mark.parametrize("explicit_pin", [False, True])
def test_cu128_dependencies_match_build_interpreter_and_architecture(
    monkeypatch, interpreter, architecture, explicit_pin
):
    reqs = requirements(
        monkeypatch, interpreter, architecture, sm70=not explicit_pin, pin=explicit_pin
    )
    for package, version in (
        ("torch", "2.10.0"),
        ("torchaudio", "2.10.0"),
        ("torchvision", "0.25.0"),
    ):
        assert (
            f"{package} @ https://download.pytorch.org/whl/cu128/"
            f"{package}-{version}%2Bcu128-{interpreter}-{interpreter}-"
            f"manylinux_2_28_{architecture}.whl"
        ) in reqs


def test_non_sm70_dependencies_remain_index_resolved(monkeypatch):
    reqs = requirements(monkeypatch, "cp312", "x86_64", sm70=False)
    assert "torch==2.10.0" in reqs
    assert not any("download.pytorch.org" in req for req in reqs)


def test_pin_rejects_unsupported_architecture_instead_of_cross_install(monkeypatch):
    with pytest.raises(RuntimeError, match="CUDA 12.8.*platform"):
        requirements(monkeypatch, "cp312", "ppc64le")
