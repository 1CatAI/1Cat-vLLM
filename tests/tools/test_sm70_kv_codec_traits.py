# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compile reader parity on CPU hosts; no Torch/CUDA extension build needed."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest
import regex as re

pytestmark = pytest.mark.cpu_test
ROOT = Path(__file__).resolve().parents[2]
KERNEL = ROOT / "flash-attention-v100/kernel"
FIXTURES = ROOT / "tests/v1/attention/fixtures"


def _compile(nvcc, tmp_path, *extra):
    target = tmp_path / "reader.ptx"
    result = subprocess.run(
        [
            nvcc,
            "--ptx",
            "-std=c++17",
            "-arch=sm_70",
            "--use_fast_math",
            "-I",
            str(KERNEL),
            "-I",
            str(FIXTURES),
            *extra,
            str(FIXTURES / "sm70_kv_reader_probe.cu"),
            "-o",
            str(target),
        ],
        env={**os.environ, "TMPDIR": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stderr
    return target.read_text()


def _ptx_instructions(text):
    # Source locations/comments change with extraction; instructions must not.
    return re.sub(r"\s+", " ", re.sub(r"//[^\n]*|\.file[^\n]*", "", text)).strip()


def test_reader_scalar_packed_and_addressing_ptx_matches_legacy(tmp_path):
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("CUDA compiler needed for static SM70 reader parity")
    legacy = _compile(nvcc, tmp_path, "-DLEGACY_READER")
    current = _compile(nvcc, tmp_path)
    assert _ptx_instructions(current) == _ptx_instructions(legacy)


@pytest.mark.parametrize(
    "path",
    [
        KERNEL / "fp8_kv_utils.cuh",
        ROOT / "csrc/attention/sm70_grouped_long/kernel/fp8_kv_utils.cuh",
    ],
)
def test_legacy_headers_forward_to_one_traits_owner(path, tmp_path):
    text = path.read_text()
    assert "kv_codec_traits.cuh" in text
    assert "__device__" not in text
    assert "KV_CACHE_DTYPE_FP16 =" not in text
    nvcc = shutil.which("nvcc")
    if nvcc is not None:
        assert _ptx_instructions(_compile(nvcc, tmp_path, "-include", str(path))) == (
            _ptx_instructions(_compile(nvcc, tmp_path))
        )


def test_traits_are_packaged_in_both_source_distributions():
    from setuptools._distutils.filelist import FileList

    for project in (ROOT, ROOT / "flash-attention-v100"):
        files = FileList()
        files.findall(str(project))
        # FileList expects source paths relative to its distribution root.
        assert files.allfiles is not None
        files.allfiles = [str(Path(p).relative_to(project)) for p in files.allfiles]
        for line in (project / "MANIFEST.in").read_text().splitlines():
            if line.strip() and not line.startswith("#"):
                files.process_template_line(line)
        assert "flash-attention-v100/kernel/kv_codec_traits.cuh" in files.files or (
            "kernel/kv_codec_traits.cuh" in files.files
        )
        assert any(p.endswith("/kv_codec.cuh") for p in files.files)
