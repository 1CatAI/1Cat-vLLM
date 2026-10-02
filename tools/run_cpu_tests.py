# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run CPU tests without host NVML selecting a CUDA platform at import time."""

import os
import sys
from pathlib import Path

# Select the source tree even when invoked as tools/run_cpu_tests.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pytest  # noqa: E402

import vllm.platforms  # noqa: E402
from vllm.platforms.cpu import CpuPlatform  # noqa: E402

vllm.platforms._current_platform = CpuPlatform()

if __name__ == "__main__":
    raise SystemExit(pytest.main(sys.argv[1:]))
