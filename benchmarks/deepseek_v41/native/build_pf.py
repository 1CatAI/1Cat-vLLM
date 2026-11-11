# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build v41sk variants with -DV41SK_PIPELINE=N into build/pf<N>/v41sk.so."""

import os
import sys

from torch.utils import cpp_extension as ce

src = os.path.expanduser("~/v41/src")
here = src + "/benchmarks/deepseek_v41/v41k"
for spec in sys.argv[1:]:
    pf, minw, co = (spec.split(":") + ["0", "-1"])[:3]
    d = f"{here}/build/pf{pf}w{minw}c{co}"
    os.makedirs(d, exist_ok=True)
    ce.load(
        name="v41sk",
        sources=[here + "/v41sk.cu"],
        extra_include_paths=[src + "/csrc"],
        extra_cuda_cflags=[
            "-O3",
            "-gencode",
            "arch=compute_70,code=sm_70",
            "--expt-relaxed-constexpr",
            "-std=c++20",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
            f"-DV41SK_PIPELINE={pf}",
            f"-DV41SK_MINW={minw}",
            f"-DV41SK_CARVEOUT={co}",
        ],
        extra_cflags=["-O3", "-std=c++20"],
        is_python_module=False,
        build_directory=d,
        verbose=False,
    )
    print("built", d)
