# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU helpers can discover GDN layers without creating a CUDA context."""

import subprocess
import sys


def test_fla_import_uses_context_free_architecture_queries():
    script = """
import torch
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability

def forbidden(*args, **kwargs):
    raise AssertionError("FLA import must not create a CUDA context")

torch.cuda._lazy_init = forbidden
current_platform.get_device_capability = lambda device_id=None: DeviceCapability(7, 0)
current_platform.is_cuda = lambda: True
from vllm.model_executor.layers.fla.ops import (
    chunk_delta_h, chunk_o, chunk_scaled_dot_kkt, kda, utils,
)
assert utils.device_platform == "nvidia"
assert not utils.is_nvidia_hopper
assert all(module._is_sm70() for module in (
    chunk_delta_h, chunk_o, chunk_scaled_dot_kkt, kda,
))
assert not torch.cuda.is_initialized()
"""
    subprocess.run([sys.executable, "-c", script], check=True, timeout=60)
