# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib.metadata
import shutil
import subprocess
from pathlib import Path

import pytest
import regex as re
from packaging.version import Version

SM70_TILELANG_FIX_VERSION = Version("0.1.10")


def test_cuda_requirements_pin_sm70_tilelang_fix() -> None:
    requirements = Path(__file__).parents[2] / "requirements" / "cuda.txt"
    contents = requirements.read_text()

    assert "apache-tvm-ffi==0.1.10" in contents
    assert "tilelang==0.1.10" in contents


def test_tilelang_header_compiles_for_sm70(tmp_path: Path) -> None:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("requires nvcc")
    assert nvcc is not None

    compiler = subprocess.run(
        [nvcc, "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    match = re.search(r"release (\d+)\.", compiler.stdout)
    if match is None:
        pytest.fail(f"could not determine CUDA version from: {compiler.stdout}")
    assert match is not None
    if int(match.group(1)) >= 13:
        pytest.skip("CUDA 13 no longer supports SM70 offline compilation")

    try:
        tilelang = importlib.metadata.distribution("tilelang")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("requires the CUDA tilelang dependency")

    assert Version(tilelang.version) >= SM70_TILELANG_FIX_VERSION
    include_dir = Path(str(tilelang.locate_file("tilelang/src")))
    cutlass_include_dir = Path(
        str(tilelang.locate_file("tilelang/3rdparty/cutlass/include"))
    )
    assert (include_dir / "tl_templates/cuda/common.h").is_file()
    assert cutlass_include_dir.is_dir()

    source = tmp_path / "tilelang_sm70_header_smoke.cu"
    output = tmp_path / "tilelang_sm70_header_smoke.cubin"
    source.write_text(
        "#include <tl_templates/cuda/common.h>\n"
        'extern "C" __global__ void int_float_only(\n'
        "    const int* input, float* output) {\n"
        "  if (threadIdx.x == 0) output[0] = static_cast<float>(input[0]);\n"
        "}\n"
    )

    result = subprocess.run(
        [
            nvcc,
            "--cubin",
            "-O3",
            "-lineinfo",
            "-arch=sm_70",
            "-std=c++17",
            f"-I{include_dir}",
            f"-I{cutlass_include_dir}",
            "-o",
            str(output),
            str(source),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
