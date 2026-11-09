# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright contributors to the 1Cat-vLLM project

"""Compile Volta device code with the CUDA runtime delivered by pip."""

import functools
import importlib.metadata
import logging
import shlex

_registered = False
logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def _cuda_headers():
    packages = (
        ("nvidia-cuda-runtime-cu12", "nvidia/cuda_runtime/include"),
        ("nvidia-cuda-nvcc-cu12", "nvidia/cuda_nvcc/include"),
        ("nvidia-cuda-cccl-cu12", "nvidia/cuda_cccl/include"),
    )
    headers = [
        str(importlib.metadata.distribution(name).locate_file(path))
        for name, path in packages
    ]
    logger.info("SM70 TileLang uses pip CUDA NVRTC; no local NVCC is required.")
    return headers


def register_sm70_runtime_compiler():
    """Preserve TileLang's native launcher while removing its NVCC dependency."""
    global _registered
    if _registered:
        return
    import tvm_ffi
    from tilelang import PassConfigKey
    from tilelang.contrib import nvcc
    from tilelang.env import CUTLASS_INCLUDE_DIR, TILELANG_TEMPLATE_PATH

    original = tvm_ffi.get_global_func("tilelang_callback_cuda_compile")

    @tvm_ffi.register_global_func("tilelang_callback_cuda_compile", override=True)
    def compile_cuda(code, target, pass_config=None):
        arch = nvcc.get_target_arch(nvcc.get_target_compute_version(target))
        if str(arch) not in ("70", "75"):
            return original(code, target, pass_config)
        from tilelang.contrib.nvrtc import compile_cuda as compile_nvrtc

        cfg = pass_config or {}
        options = [
            "-std=c++20",
            f"-I{TILELANG_TEMPLATE_PATH}",
            f"-I{CUTLASS_INCLUDE_DIR}",
            *(f"-I{path}" for path in _cuda_headers()),
        ]
        flags = cfg.get(PassConfigKey.TL_DEVICE_COMPILE_FLAGS, ())
        if isinstance(flags, str):
            flags = [flags]
        for flag in flags:
            options.extend(shlex.split(str(flag)))
        # Retain the math and register settings of the original NVCC callback.
        if cfg.get(PassConfigKey.TL_ENABLE_FAST_MATH, False):
            options.append("--use_fast_math")
        usage = cfg.get(PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL)
        if usage is not None:
            options.append(f"--ptxas-options=--register-usage-level={int(usage)}")
        return compile_nvrtc(
            code, target_format="cubin", arch=int(arch), options=options
        )

    _registered = True
