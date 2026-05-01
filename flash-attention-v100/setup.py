
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025, D.Skryabin

import os
from pathlib import Path
from packaging.version import parse
from setuptools import setup

this_dir = Path(__file__).parent.resolve()

def get_ext_modules():
    try:
        from torch.utils.cpp_extension import CUDAExtension
    except ImportError as e:
        raise RuntimeError(
            "torch is required to build flash_attn_v100. "
            "Please install torch >= 2.5 first (e.g., `pip install torch --index-url https://download.pytorch.org/whl/cu118`)."
        ) from e

    return [
        CUDAExtension(
            name="flash_attn_v100_cuda",
            sources=[
                "kernel/fused_mha_api.cpp",
                "kernel/fused_mha_forward.cu",
                "kernel/fused_mha_forward_paged.cu",
                "kernel/flash_decode_paged.cu",
                "kernel/fused_mha_backward.cu",
            ],
            include_dirs=[this_dir / "include", this_dir / "kernel"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "-gencode", "arch=compute_70,code=sm_70",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "--use_fast_math",
                ],
            },
        ),
        CUDAExtension(
            name="paged_kv_utils",
            sources=["kernel/paged_to_contiguous.cu"],
            include_dirs=[this_dir / "kernel"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "-gencode", "arch=compute_70,code=sm_70",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__",
                    "--use_fast_math",
                ],
            },
        )
    ]

def get_cmdclass():
    try:
        from torch.utils.cpp_extension import BuildExtension
    except ImportError as e:
        raise RuntimeError(
            "torch is required to build flash_attn_v100. "
            "Please install torch >= 2.5 first."
        ) from e

    class CustomBuildExtension(BuildExtension):
        def build_extensions(self):
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is required but not available.")
            if parse(torch.version.cuda) < parse("11.6"):
                raise RuntimeError(
                    f"CUDA version {torch.version.cuda} < 11.6 is not supported. "
                    "Please use CUDA ≥ 11.6 (e.g., PyTorch built with CUDA 11.8/12.x)."
                )
            super().build_extensions()

    return {"build_ext": CustomBuildExtension}

try:
    with open(this_dir / "README.md", encoding="utf-8") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = "Flash Attention implementation for Tesla V100"

setup(
    name="flash_attn_v100",
    version="26.02",
    packages=["flash_attn_v100"],
    ext_modules=get_ext_modules(),
    cmdclass=get_cmdclass(),
    python_requires=">=3.10",
    install_requires=["torch>=2.5", "einops", "packaging"],
    zip_safe=False,
    description="Flash Attention implementation under unsupported Tesla V100",
    license="BSD-3-Clause",
    author="D.Skryabin",
    author_email="tg @ai_bond007",
    url="https://github.com/ai-bond/flash-attention-v100",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
    ],
)
