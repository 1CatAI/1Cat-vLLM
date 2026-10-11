# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise release metadata without importing or building the engine."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("sm70", [True, False])
def test_provider_is_exclusive_and_arch_specific(sm70):
    tree = ast.parse((ROOT / "setup.py").read_text())
    function = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "get_requirements"
    )
    namespace: dict[str, Any] = {
        "ROOT_DIR": ROOT,
        "os": os,
        "torch": SimpleNamespace(version=SimpleNamespace(cuda="12.8")),
        "_no_device": lambda: False,
        "_is_cuda": lambda: True,
        "_cuda_arch_contains": lambda major, minor: sm70 and (major, minor) == (7, 0),
        "_onecat_torch_cu128_urls": dict,
    }
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), "setup.py", "exec"),
        namespace,
    )
    requirements = [
        Requirement(s.split("#", 1)[0].strip()) for s in namespace["get_requirements"]()
    ]
    providers = [r for r in requirements if r.name == "flashinfer-python"]
    assert len(providers) == 1
    if sm70:
        version = (ROOT / "flashinfer-sm70/version.txt").read_text().strip()
        assert str(providers[0].specifier) == f"=={version}"
        assert not {
            "flashinfer-cubin",
            "nvidia-cutlass-dsl",
            "quack-kernels",
            "humming-kernels",
        } & {r.name for r in requirements}
        assert [
            str(r.specifier) for r in requirements if r.name == "apache-tvm-ffi"
        ] == ["==0.1.10"]
    else:
        assert str(providers[0].specifier) == "==0.6.11.post2"
        assert "flashinfer-cubin" in {r.name for r in requirements}
        assert "quack-kernels" in {r.name for r in requirements}
