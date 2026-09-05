# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark-only instantiation of pinned FlashInfer CUDA QSA decode.

There is no model dispatch change. All mutable buffers belong to each instance;
construct/warm before CUDA Graph capture. The upstream Python SM75+ gate is not
disabled, and the previous Flash-V100 or Triton attention is never called here.
"""

import os
import subprocess
from pathlib import Path

import torch

UPSTREAM_SHA = "6c14bbd5ff34210404d5d4b5f6ff3b4b2527f59f"
CCCL_SHA = "16bd510c9b712e82b0ab6cbb630d8e29ba1f7116"
ROOT = Path(__file__).resolve().parents[2]


def build():
    from torch.utils.cpp_extension import load

    source = Path(
        os.environ.get(
            "FLASHINFER_SM70_QSA_SOURCE", ROOT / ".deps/flashinfer-6c14bbd5ff34"
        )
    ).resolve()
    sha = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if sha != UPSTREAM_SHA:
        raise RuntimeError(f"Expected FlashInfer {UPSTREAM_SHA}, got {sha}")
    if subprocess.check_output(
        ["git", "-C", str(source), "diff", "HEAD", "--", "include"], text=True
    ).strip():
        raise RuntimeError("Upstream CUDA headers must be unmodified")
    cccl = source / "3rdparty/cccl"
    if (
        subprocess.check_output(
            ["git", "-C", str(cccl), "rev-parse", "HEAD"], text=True
        ).strip()
        != CCCL_SHA
        or subprocess.check_output(
            ["git", "-C", str(cccl), "diff", "HEAD"], text=True
        ).strip()
    ):
        raise RuntimeError("Expected unmodified pinned CCCL submodule")
    if os.environ.get("TORCH_CUDA_ARCH_LIST") != "7.0":
        raise RuntimeError("Set TORCH_CUDA_ARCH_LIST=7.0 explicitly")
    return load(
        name="flashinfer_qsa_sm70_v1",
        sources=[str(ROOT / "benchmarks/csrc/sm70_flashinfer_qsa_decode.cu")],
        extra_include_paths=[
            str(ROOT / "flashinfer-sm70/include"),
            str(source / "include"),
            str(source / "3rdparty/cccl/cub"),
            str(source / "3rdparty/cccl/thrust"),
            str(source / "3rdparty/cccl/libcudacxx/include"),
        ],
        extra_cuda_cflags=[
            "-O3",
            "-lineinfo",
            "--expt-relaxed-constexpr",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
        ],
        is_python_module=False,
        verbose=True,
    )


class FlashInferQSA:
    """Persistent single-stream workspace; use distinct instances per stream."""

    def __init__(self, q, selection_width, splits):
        if q.ndim != 3 or q.shape[2] != 256 or q.dtype != torch.float16:
            raise ValueError("QSA prototype needs FP16 [rows, heads, 256] queries")
        if not 1 <= splits <= 64 or selection_width <= 0:
            raise ValueError("Require positive selection width and 1..64 splits")
        rows, heads, dim = q.shape
        if rows <= 0 or not 1 <= heads <= 32:
            raise ValueError("Require positive rows and 1..32 heads")
        width = ((selection_width + splits - 1) // splits) * splits
        self.splits = splits
        self.offsets = torch.empty((rows, width), device=q.device, dtype=torch.int64)
        self.metadata = torch.empty(
            rows + 2 + 2 * rows * splits, device=q.device, dtype=torch.int32
        )
        self.zero = torch.zeros(256, device=q.device, dtype=torch.float16)
        self.partial = torch.empty(
            (rows, splits, heads, dim), device=q.device, dtype=torch.float32
        )
        self.lse = torch.empty(
            (rows, splits, heads), device=q.device, dtype=torch.float32
        )
        self.output = torch.empty_like(q, memory_format=torch.contiguous_format)

    def __call__(self, q, k, v, indices, table, requests):
        torch.ops._C_flashinfer_qsa_sm70.run(
            q,
            k,
            v,
            indices,
            table,
            requests,
            self.offsets,
            self.metadata,
            self.zero,
            self.partial,
            self.lse,
            self.output,
            self.splits,
        )
        return self.output


if __name__ == "__main__":
    print(build())
