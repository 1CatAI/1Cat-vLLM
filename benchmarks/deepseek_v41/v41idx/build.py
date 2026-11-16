# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build the V4.1 v41idx CUDA extension (SM70) into ./build/v41idx.so.

VLLM_V41_INDEX_TILED loads it from VLLM_V41_INDEX_TILED_LIB, defaulting to
benchmarks/deepseek_v41/v41idx/build/v41idx.so beside VLLM_V41K_LIB's v41k
directory. V41IDX_BUILD_DIR overrides the build directory. The nvcc flags are
v41k's (no fast-math: the kernel must round exactly like v41k's reference).
"""

import os

from torch.utils import cpp_extension as ce

here = os.path.dirname(os.path.abspath(__file__))
build = os.environ.get("V41IDX_BUILD_DIR") or os.path.join(here, "build")
os.makedirs(build, exist_ok=True)
ce.load(
    name="v41idx",
    sources=[os.path.join(here, "v41idx.cu")],
    extra_cuda_cflags=[
        "-O3",
        "-gencode",
        "arch=compute_70,code=sm_70",
        "--expt-relaxed-constexpr",
        "-std=c++20",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    ],
    extra_cflags=["-O3", "-std=c++20"],
    is_python_module=False,
    build_directory=build,
    verbose=False,
)
print("built", os.path.join(build, "v41idx.so"))
