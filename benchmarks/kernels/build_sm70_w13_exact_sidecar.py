# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build the exact W13 scheduling screen without initializing CUDA."""

import argparse
from pathlib import Path

from torch.utils.cpp_extension import load

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--build-dir", type=Path, required=True)
args = parser.parse_args()
args.build_dir.mkdir(parents=True, exist_ok=True)
print(
    load(
        name="vllm_w13_exact_sm70",
        sources=[str(Path(__file__).with_name("sm70_w13_exact_sidecar.cu"))],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-lineinfo", "-Xptxas=-v"],
        build_directory=str(args.build_dir),
        is_python_module=False,
        verbose=True,
    ),
    flush=True,
)
