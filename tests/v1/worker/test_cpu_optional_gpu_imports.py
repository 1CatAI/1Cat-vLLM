# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU workers must import without the optional GPU compiler."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "module",
    [
        "vllm.v1.worker.gpu.spec_decode.dflash2.lookup",
        "vllm.v1.worker.cpu_worker",
    ],
)
def test_cpu_import_without_triton(module):
    root = Path(__file__).resolve().parents[3]
    code = """
import sys
import torch
sys.modules['triton'] = None
import vllm.platforms
from vllm.platforms.cpu import CpuPlatform
vllm.platforms._current_platform = CpuPlatform()
import importlib
importlib.import_module(sys.argv[1])
from vllm.triton_utils import HAS_TRITON
from vllm.v1.worker.gpu.spec_decode.dflash2.lookup import _SCORE_STRIDE
assert not HAS_TRITON
assert _SCORE_STRIDE == 1 << 32
"""
    result = subprocess.run(
        [sys.executable, "-c", code, module],
        cwd=root,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(root)},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
