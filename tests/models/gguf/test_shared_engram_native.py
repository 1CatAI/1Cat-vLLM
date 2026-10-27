# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def reader_binary(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++17 compiler required")
    assert compiler is not None
    root = Path(__file__).resolve().parents[3]
    binary = tmp_path_factory.mktemp("engram-reader") / "reader"
    result = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pthread",
            "-I",
            str(root),
            str(Path(__file__).with_name("shared_engram_test.cpp")),
            "-o",
            str(binary),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode:
        pytest.fail(result.stderr, pytrace=False)
    return binary


@pytest.mark.parametrize(
    "case",
    [
        "rows_and_consumption",
        "repeated_requests",
        "short_read",
        "timeout",
        "diverged_rows",
        "out_of_bounds",
        "stop",
    ],
)
def test_single_engram_reader_contract(reader_binary, case):
    subprocess.run(
        [str(reader_binary), case], check=True, timeout=15, capture_output=True
    )
