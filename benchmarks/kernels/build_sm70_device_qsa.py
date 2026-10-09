# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build only the isolated QSA research screen; not a model runtime module."""

import argparse
from pathlib import Path

from torch.utils.cpp_extension import load

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--probability-decomposition", action="store_true")
args = parser.parse_args()
source = (
    "sm70_device_qsa_probability_micro.cu"
    if args.probability_decomposition
    else "sm70_device_qsa_micro.cu"
)
load(
    name="round15_device_qsa",
    sources=[str(Path(__file__).resolve().parents[1] / "csrc" / source)],
    extra_cuda_cflags=[
        "-O3",
        "-std=c++20",
        "-Xptxas=-v",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
    ],
    is_python_module=False,
    verbose=True,
)
