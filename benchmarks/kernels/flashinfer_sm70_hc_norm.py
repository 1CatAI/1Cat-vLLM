# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer-derived fused HC combine/Gemma-norm component, not dispatch."""

import os
import subprocess
from pathlib import Path

import torch

from benchmarks.kernels.flashinfer_sm70_gdn_conv import ROOT, SOURCE_SHA


def build():
    from torch.utils.cpp_extension import load

    source = Path(os.environ["FLASHINFER_SOURCE"])
    if (
        subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip()
        != SOURCE_SHA
    ):
        raise RuntimeError("Unrecognized FlashInfer revision")
    if os.environ.get("TORCH_CUDA_ARCH_LIST") != "7.0":
        raise RuntimeError("Set TORCH_CUDA_ARCH_LIST=7.0")
    return load(
        name="flashinfer_hc_norm_sm70_v1",
        sources=[str(ROOT / "benchmarks/csrc/sm70_flashinfer_hc_norm.cu")],
        extra_include_paths=[
            str(ROOT / "flashinfer-sm70/include"),
            str(source / "include"),
            str(source / "3rdparty/cccl/thrust"),
            str(source / "3rdparty/cccl/cub"),
            str(source / "3rdparty/cccl/libcudacxx/include"),
        ],
        extra_cuda_cflags=[
            "-O3",
            "-lineinfo",
            "--expt-relaxed-constexpr",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        ],
        is_python_module=False,
        verbose=True,
    )


class HCNorm:
    def __init__(self, residual, warps=4):
        self.combined = torch.empty_like(residual)
        self.normalized = torch.empty_like(residual)
        self.warps = warps

    def __call__(self, residual, block, injection, weight, eps=1e-6):
        torch.ops._C_flashinfer_hc_sm70.run(
            residual,
            block,
            injection,
            weight,
            self.combined,
            self.normalized,
            eps,
            self.warps,
        )
        return self.combined, self.normalized


if __name__ == "__main__":
    print(build())
