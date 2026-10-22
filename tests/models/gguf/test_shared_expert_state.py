# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native lifetime fencing, independently of unavailable CUDA hardware."""

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def native_state_test(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler required for the native coordinator contract")
    assert compiler is not None
    root = Path(__file__).resolve().parents[3]
    binary = tmp_path_factory.mktemp("shared-expert") / "state-test"
    compiled = subprocess.run(
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
            str(Path(__file__).with_name("shared_expert_state_test.cpp")),
            "-o",
            str(binary),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if compiled.returncode:
        pytest.fail(compiled.stderr, pytrace=False)
    return binary


@pytest.mark.parametrize(
    "case",
    [
        "slow_rank_and_prefetch",
        "repeated_eviction",
        "invalid_sequences",
        "timeout_and_exit",
        "aliases_and_ranges",
        "concurrent_completion",
        "planner_eviction",
        "planner_leases",
        "coordinator_thread",
        "coordinator_failure",
    ],
)
def test_native_shared_swap_contract(native_state_test, case):
    subprocess.run(
        [str(native_state_test), case],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
