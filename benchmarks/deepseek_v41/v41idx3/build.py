# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build the V4.1 v41idx3 CUDA extension (SM70) into ./build/v41idx3.so.

VLLM_V41_INDEX_TC loads it from VLLM_V41_INDEX_TC_LIB, defaulting to
benchmarks/deepseek_v41/v41idx3/build/v41idx3.so. V41IDX3_BUILD_DIR overrides
the build directory. The flags are the ones the library was measured with (no
fast-math: the fallback must round exactly like v41idx's chain).
"""

import os

from torch.utils import cpp_extension as ce

here = os.path.dirname(os.path.abspath(__file__))
build = os.environ.get("V41IDX3_BUILD_DIR") or os.path.join(here, "build")
os.makedirs(build, exist_ok=True)
ce.load(
    name="v41idx3",
    sources=[os.path.join(here, "v41idx3.cu")],
    extra_cuda_cflags=[
        "-O3",
        "-gencode",
        "arch=compute_70,code=sm_70",
        "--expt-relaxed-constexpr",
        "-std=c++17",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
    ],
    extra_cflags=["-O3", "-std=c++17"],
    is_python_module=False,
    build_directory=build,
    verbose=False,
)
print("built", os.path.join(build, "v41idx3.so"))
