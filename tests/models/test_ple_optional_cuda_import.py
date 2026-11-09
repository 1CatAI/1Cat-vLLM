# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU model imports must not require CUDA's Python bindings."""

import os
import subprocess
import sys
from pathlib import Path


def test_ple_import_without_cuda_bindings():
    root = Path(__file__).resolve().parents[2]
    code = """
import importlib.abc
import sys
import torch
import vllm.platforms
from vllm.platforms.cpu import CpuPlatform
vllm.platforms._current_platform = CpuPlatform()
for name in list(sys.modules):
    if name == 'cuda' or name.startswith('cuda.'):
        del sys.modules[name]
class NoCuda(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'cuda' or fullname.startswith('cuda.'):
            raise ModuleNotFoundError('CUDA bindings unavailable', name=fullname)
sys.meta_path.insert(0, NoCuda())
from vllm.model_executor.layers.ple_offload_layer import PleOffloadLayer
assert issubclass(PleOffloadLayer, torch.nn.Module)
assert 'cuda.bindings.driver' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(root)},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
