# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import logging
import sysconfig
from pathlib import Path
from typing import Any

import pytest
import regex as re
from packaging.tags import cpython_tags, sys_tags
from setuptools import Distribution, Extension
from setuptools.command.bdist_wheel import bdist_wheel


def wheel_command(distribution):
    # Load only the wheel command, avoiding setup() and CUDA/network work.
    source = Path(__file__).resolve().parents[2] / "setup.py"
    tree = ast.parse(source.read_text())
    definitions: list[ast.stmt] = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "vllm_bdist_wheel"
    ]
    scope: dict[str, Any] = {
        "bdist_wheel": bdist_wheel,
        "re": re,
        "logger": logging.getLogger(__name__),
    }
    exec(
        compile(ast.Module(body=definitions, type_ignores=[]), str(source), "exec"),
        scope,
    )
    return scope.get("vllm_bdist_wheel", bdist_wheel)(distribution)


@pytest.mark.parametrize("limited_api", ["cp38", None])
@pytest.mark.parametrize("source", ["cmake", "precompiled", "package_only", "abi3"])
def test_wheel_tag_matches_native_abi(tmp_path, monkeypatch, source, limited_api):
    monkeypatch.chdir(tmp_path)
    extensions = [Extension("vllm._C", sources=[], py_limited_api=True)]
    package_data = {}
    if source == "cmake":
        extensions.append(
            Extension("vllm._sm70_sparse_attention_C", sources=[], py_limited_api=False)
        )
    elif source in ("precompiled", "package_only"):
        # Precompiled libraries can arrive as package data without SM70 targets
        # in ext_modules (e.g. TORCH_CUDA_ARCH_LIST is unset during repackaging).
        package_data["vllm"] = [
            "_sm70_sparse_attention_C" + sysconfig.get_config_var("EXT_SUFFIX")
        ]
        if source == "package_only":
            extensions = []
    distribution = Distribution(
        {
            "name": "wheel-abi-test",
            "version": "0.0.0",
            "ext_modules": extensions,
            "package_data": package_data,
        }
    )
    command = wheel_command(distribution)
    command.py_limited_api = limited_api
    command.ensure_finalized()
    tag = command.get_tag()
    if source == "abi3" and limited_api:
        assert tag[:2] == (limited_api, "abi3")
    else:
        current = next(sys_tags())
        assert tag[:2] == (current.interpreter, current.abi)
        # A CPython-specific wheel must not be accepted by another minor version.
        foreign = cpython_tags((3, 10), platforms=[tag[2]])
        if current.interpreter == "cp310":
            foreign = cpython_tags((3, 11), platforms=[tag[2]])
        assert "-".join(tag) not in {str(value) for value in foreign}
    # The selected tag and the installer-visible WHEEL metadata must agree.
    metadata = tmp_path / "wheel-abi-test.dist-info"
    metadata.mkdir()
    command.write_wheelfile(str(metadata))
    assert f"Tag: {'-'.join(tag)}\n" in (metadata / "WHEEL").read_text()


def test_pure_python_wheel_keeps_its_portable_tag(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    command = wheel_command(Distribution({"name": "pure-test", "version": "0.0.0"}))
    command.ensure_finalized()
    assert command.get_tag() == ("py3", "none", "any")
