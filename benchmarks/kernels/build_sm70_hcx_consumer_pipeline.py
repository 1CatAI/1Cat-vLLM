# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build the research-only HC-to-dense screen in an explicit cache directory."""

import argparse
import os
from pathlib import Path

from torch.utils.cpp_extension import load

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--cache-dir", type=Path, required=True)
args = parser.parse_args()
args.cache_dir.mkdir(parents=True, exist_ok=True)
os.environ["TORCH_CUDA_ARCH_LIST"] = "7.0"
root = Path(__file__).resolve().parents[2]
load(
    name="round14_hcx_consumer",
    sources=[str(root / "benchmarks/csrc/sm70_hcx_consumer_pipeline_micro.cu")],
    build_directory=str(args.cache_dir),
    extra_include_paths=[
        str(root / "csrc/sm70_turbomind/ops"),
        str(root / "csrc/sm70_turbomind/lmdeploy"),
    ],
    extra_cuda_cflags=[
        "-O3",
        "-std=c++20",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-Xptxas=-v",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    ],
    is_python_module=False,
    verbose=True,
)
