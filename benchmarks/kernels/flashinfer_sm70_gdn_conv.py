# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer-derived FP16 gate projection + conv + FP32 delta-rule prototype.

No production dispatch. A workspace is single-stream, state indices must be
unique live pool entries or negative padding. Geometry is specialized at JIT
time; no checkpoint name, max-seqs, KV dtype or TP degree is used as a gate.
"""

import os
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
SOURCE_SHA = "6c14bbd5ff34210404d5d4b5f6ff3b4b2527f59f"


def build(hidden=2560, q_heads=4, v_heads=12):
    from torch.utils.cpp_extension import load

    if os.environ.get("TORCH_CUDA_ARCH_LIST") != "7.0":
        raise RuntimeError("Set TORCH_CUDA_ARCH_LIST=7.0")
    return load(
        name=f"flashinfer_gdn_sm70_h{hidden}_q{q_heads}_v{v_heads}",
        sources=[str(ROOT / "benchmarks/csrc/sm70_flashinfer_gdn_conv.cu")],
        extra_include_paths=[str(ROOT / "flashinfer-sm70/include")],
        extra_cuda_cflags=[
            "-O3",
            "-lineinfo",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
            f"-DFI_GDN_HIDDEN={hidden}",
            f"-DFI_GDN_N_BA={2 * v_heads}",
            f"-DFI_GDN_QKV_DIM={(2 * q_heads + v_heads) * 128}",
            f"-DFI_GDN_H_Q={q_heads}",
            f"-DFI_GDN_HV={v_heads}",
            "-DFI_GDN_D=128",
            "-DFI_GDN_CONV_WIDTH=4",
            "-DFI_GDN_CONV_STATE_LEN=3",
        ],
        is_python_module=False,
        verbose=True,
    )


class FusedGDN:
    def __init__(self, rows, q_heads=4, v_heads=12, device="cuda"):
        if not 1 <= rows <= 64 or q_heads <= 0 or v_heads % q_heads:
            raise ValueError("Require 1..64 rows and integral GQA grouping")
        self.output = torch.empty(
            rows, v_heads, 128, device=device, dtype=torch.float16
        )
        self.conv_out = torch.empty(
            rows, (2 * q_heads + v_heads) * 128, device=device, dtype=torch.float16
        )
        self.partial = torch.empty(
            rows * 2 * v_heads * 160, device=device, dtype=torch.float32
        )

    def __call__(
        self,
        hidden,
        weights,
        qkv,
        conv_w,
        conv_bias,
        conv,
        A_log,
        dt_bias,
        state,
        indices,
    ):
        torch.ops._C_flashinfer_gdn_sm70.run(
            hidden,
            weights,
            qkv,
            conv_w,
            conv_bias,
            conv,
            A_log,
            dt_bias,
            state,
            indices,
            self.output,
            self.conv_out,
            self.partial,
        )
        return self.output


if __name__ == "__main__":
    print(build())
