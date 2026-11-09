# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import ast
import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path
from types import SimpleNamespace

import pytest
import regex as re
import torch.utils.cpp_extension

ROOT = Path(__file__).resolve().parents[2]


def setup_functions(tmp_path):
    path = ROOT / "setup.py"
    names = {
        "_volta_cuda_arch_list",
        "bundle_flash_qla_sm70",
        "bundle_flash_attn_v100",
    }
    nodes: list[ast.stmt] = [
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    qla = tmp_path / "qla"
    (qla / "csrc").mkdir(parents=True)
    (qla / "csrc/gdn_forward.cu").write_text("")
    fa = tmp_path / "fa"
    (fa / "flash_attn_v100").mkdir(parents=True)
    for name in ("flash_attn_v100_cuda", "paged_kv_utils"):
        (fa / f"{name}.abi3.so").write_bytes(b"test extension")
    namespace = {
        "os": os,
        "re": re,
        "sys": sys,
        "sysconfig": sysconfig,
        "subprocess": subprocess,
        "shutil": shutil,
        "Path": Path,
        "torch": torch,
        "FLASH_QLA_SM70_ROOT": qla,
        "FLASH_ATTN_V100_ROOT": fa,
        "FLASH_ATTN_V100_PACKAGE": fa / "flash_attn_v100",
        "remove_rpath": lambda path: None,
        "logger": SimpleNamespace(info=lambda *args: None),
    }
    exec(compile(ast.Module(nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


@pytest.mark.parametrize(
    "requested,override,expected",
    [
        (None, None, "7.0"),
        ("7.2", None, "7.2"),
        ("7.0;7.2;7.5", None, "7.0;7.2"),
        ("7.2+PTX;8.0", None, "7.2+PTX"),
        ("7.2", "7.0", "7.0"),
    ],
)
def test_flash_v100_bundle_respects_volta_targets(
    monkeypatch, tmp_path, requested, override, expected
):
    for name, value in [
        ("TORCH_CUDA_ARCH_LIST", requested),
        ("FLASH_ATTN_V100_CUDA_ARCH_LIST", override),
    ]:
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    calls = []
    monkeypatch.setattr(
        subprocess,
        "check_call",
        lambda *args, **kwargs: calls.append(kwargs["env"]["TORCH_CUDA_ARCH_LIST"]),
    )
    namespace = setup_functions(tmp_path)
    namespace["bundle_flash_attn_v100"](str(tmp_path / "wheel"))
    assert calls == [expected]


@pytest.mark.parametrize("raises", [False, True])
def test_flash_qla_targets_and_environment_restoration(monkeypatch, tmp_path, raises):
    requested = "7.2+PTX;8.0"
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", requested)
    calls = []
    artifact = tmp_path / "flash_qla_sm70_gdn_strided.so"
    artifact.write_bytes(b"test extension")

    def load(**kwargs):
        calls.append((os.environ["TORCH_CUDA_ARCH_LIST"], kwargs["extra_cuda_cflags"]))
        if raises:
            raise RuntimeError("build failed")
        return SimpleNamespace(__file__=str(artifact))

    monkeypatch.setattr(torch.utils.cpp_extension, "load", load)
    namespace = setup_functions(tmp_path)
    if raises:
        with pytest.raises(RuntimeError, match="build failed"):
            namespace["bundle_flash_qla_sm70"](
                str(tmp_path / "wheel"), str(tmp_path / "build")
            )
    else:
        namespace["bundle_flash_qla_sm70"](
            str(tmp_path / "wheel"), str(tmp_path / "build")
        )
    assert calls == [
        (
            "7.2+PTX",
            [
                "-O3",
                "-gencode=arch=compute_72,code=compute_72",
                "-gencode=arch=compute_72,code=sm_72",
            ],
        )
    ]
    assert os.environ["TORCH_CUDA_ARCH_LIST"] == requested


@pytest.mark.parametrize(
    "version,allowed", [("12.0", True), ("12.8", True), ("13.0", False)]
)
def test_cmake_preserves_sm72_only_on_supported_toolkits(tmp_path, version, allowed):
    source = (ROOT / "CMakeLists.txt").read_text()
    block = re.search(r"# Supported NVIDIA architectures\.[\s\S]*?\nendif\(\)", source)
    assert block is not None
    script = tmp_path / "targets.cmake"
    script.write_text(
        "cmake_minimum_required(VERSION 3.18)\n"
        f'set(CMAKE_CUDA_COMPILER_VERSION "{version}")\n'
        + block.group(0)
        + f'\ninclude("{ROOT / "cmake/utils.cmake"}")\n'
        + 'cuda_archs_loose_intersection(SELECTED "7.0" "7.2")\n'
        + 'if(NOT SELECTED STREQUAL "7.0")\n'
        + 'message(FATAL_ERROR "Volta binary compatibility lost")\nendif()\n'
        + 'if("7.2" IN_LIST CUDA_SUPPORTED_ARCHS)\nmessage("SM72_ALLOWED")\nendif()\n'
    )
    result = subprocess.run(
        ["cmake", "-P", str(script)], text=True, capture_output=True, check=True
    )
    assert ("SM72_ALLOWED" in result.stderr) is allowed
