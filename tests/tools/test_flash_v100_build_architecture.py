# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise extension declarations without initializing CUDA or running setup."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch.utils.cpp_extension


def extensions(monkeypatch):
    root = Path(__file__).resolve().parents[2] / "flash-attention-v100"
    path = root / "setup.py"
    tree = ast.parse(path.read_text())
    nodes: list[ast.stmt] = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_volta_gencode_flags", "get_ext_modules"}
    ]
    namespace: dict[str, Any] = {"os": os, "this_dir": root}
    monkeypatch.setattr(
        torch.utils.cpp_extension,
        "CUDAExtension",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    exec(compile(ast.Module(nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["get_ext_modules"]()


@pytest.mark.parametrize(
    "requested,expected",
    [
        (None, ["arch=compute_70,code=sm_70"]),
        ("7.0", ["arch=compute_70,code=sm_70"]),
        ("7.2", ["arch=compute_72,code=sm_72"]),
        ("7.0;7.2", ["arch=compute_70,code=sm_70", "arch=compute_72,code=sm_72"]),
        ("7.2 7.2", ["arch=compute_72,code=sm_72"]),
        (
            "7.2+PTX",
            ["arch=compute_72,code=sm_72", "arch=compute_72,code=compute_72"],
        ),
    ],
)
def test_both_extensions_target_the_requested_volta_architecture(
    monkeypatch, requested, expected
):
    if requested is None:
        monkeypatch.delenv("TORCH_CUDA_ARCH_LIST", raising=False)
    else:
        monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", requested)
    modules = extensions(monkeypatch)
    assert {module.name for module in modules} == {
        "flash_attn_v100_cuda",
        "paged_kv_utils",
    }
    for module in modules:
        flags = module.extra_compile_args["nvcc"]
        assert [flag for flag in flags if flag.startswith("arch=")] == expected


@pytest.mark.parametrize("requested", ["7.5", "8.0"])
def test_other_architectures_fail_with_a_clear_build_error(monkeypatch, requested):
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", requested)
    with pytest.raises(ValueError, match="7.0 and 7.2"):
        extensions(monkeypatch)
