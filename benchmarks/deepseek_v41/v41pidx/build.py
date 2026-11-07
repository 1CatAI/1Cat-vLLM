# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build the v41pidx CUDA extension (SM70) into BUILD_DIR/v41pidx.so.

A separate library from v41k.so (own torch.ops namespace ``v41pidx``), so it
never joins the v41k.so build lineage. The runtime loads it from
VLLM_V41_INDEX_FUSE_LIB. Build with the serving environment's python (same
torch ABI):

    V41PIDX_BUILD_DIR=/path/to/build python build.py
"""

import os

from torch.utils import cpp_extension as ce

here = os.path.dirname(os.path.abspath(__file__))
build = os.environ.get("V41PIDX_BUILD_DIR") or os.path.join(here, "build")
os.makedirs(build, exist_ok=True)
ce.load(
    name="v41pidx",
    sources=[os.path.join(here, "v41pidx.cu")],
    extra_cuda_cflags=[
        "-O3",
        "-gencode",
        "arch=compute_70,code=sm_70",
        "--expt-relaxed-constexpr",
        "-std=c++17",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        # No FMA contraction anywhere: every add/mul is spelled __fadd_rn /
        # __fmul_rn already; this keeps any helper honest too.
        "--fmad=false",
    ],
    extra_cflags=["-O3", "-std=c++17"],
    is_python_module=False,
    build_directory=build,
    verbose=False,
)
print("built", os.path.join(build, "v41pidx.so"))
