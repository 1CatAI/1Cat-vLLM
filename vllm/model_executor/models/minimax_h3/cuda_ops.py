# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lazy build of native H3 SM70 extensions for source development."""

import os
from functools import lru_cache
from pathlib import Path

from vllm.model_executor.layers.sm70_diffusion import (
    _column_major_plan as _column_major_plan,
)
from vllm.model_executor.layers.sm70_diffusion import (
    fp16_gemm as fp16_gemm,
)
from vllm.model_executor.layers.sm70_diffusion import sm70_extension

w8a16_extension = sm70_extension


@lru_cache(maxsize=1)
def flashinfer_extension():
    try:
        from vllm import _h3_flashinfer_C

        return _h3_flashinfer_C
    except ImportError:
        pass
    from torch.utils.cpp_extension import load

    root = Path(__file__).resolve().parents[4]
    return load(
        name="onecat_h3_flashinfer_sm70",
        sources=[str(root / "flashinfer-sm70/csrc/h3_noncausal_sm70.cu")],
        extra_include_paths=[str(root / "flashinfer-sm70/include")],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_70,code=sm_70"],
        verbose=False,
    )


@lru_cache(maxsize=1)
def flashattn_extension():
    try:
        from vllm import _h3_flashattn_C

        return _h3_flashattn_C
    except ImportError:
        pass
    from torch.utils.cpp_extension import load

    root = Path(__file__).resolve().parents[4]
    cutlass_path = os.environ.get("VLLM_CUTLASS_SRC_DIR")
    if not cutlass_path:
        raise RuntimeError(
            "Build the H3 FlashAttention-V100 extension (_h3_flashattn_C), or "
            "set VLLM_CUTLASS_SRC_DIR to CUTLASS v4.4.2 for source development"
        )
    cutlass = Path(cutlass_path)
    return load(
        name="onecat_h3_flashattn_sm70",
        sources=[str(root / "flash-attention-v100/kernel/h3/forward.cu")],
        extra_include_paths=[
            str(cutlass / "include"),
            str(cutlass / "examples/41_fused_multi_head_attention"),
        ],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_70,code=sm_70"],
        verbose=False,
    )
