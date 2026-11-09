# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build the V4.1 v41k CUDA extension (SM70) into ./build/v41k.so.

The runtime loads it from VLLM_V41K_LIB, defaulting to this build directory.
"""

import os

from torch.utils import cpp_extension as ce

here = os.path.dirname(os.path.abspath(__file__))
src = os.path.abspath(os.path.join(here, "..", "..", ".."))
ce.load(
    name="v41k",
    sources=[os.path.join(here, "v41k.cu")],
    extra_include_paths=[src + "/csrc", src + "/csrc/sm70_turbomind/lmdeploy"],
    extra_cuda_cflags=[
        "-O3",
        "-gencode",
        "arch=compute_70,code=sm_70",
        "--expt-relaxed-constexpr",
        "-std=c++20",
        "-DENABLE_SM70_TURBOMIND=1",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    ],
    extra_cflags=["-O3", "-std=c++20"],
    is_python_module=False,
    build_directory=os.path.join(here, "build"),
    verbose=False,
)
print("built", os.path.join(here, "build", "v41k.so"))
