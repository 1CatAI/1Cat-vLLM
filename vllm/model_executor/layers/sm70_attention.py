# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit model-independent SM70 FP16 non-causal D128 attention.

The extension ABI retains historical H3 names. Inputs are BSHD; no model,
quantization, adapter or sampler identity participates in dispatch.
"""

import math
import os
from functools import lru_cache
from importlib import import_module
from pathlib import Path


@lru_cache(maxsize=1)
def flashinfer_extension():
    try:
        return import_module("vllm._h3_flashinfer_C")
    except ImportError:
        pass
    from torch.utils.cpp_extension import load

    root = Path(__file__).resolve().parents[3]
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
        return import_module("vllm._h3_flashattn_C")
    except ImportError:
        pass
    from torch.utils.cpp_extension import load

    root = Path(__file__).resolve().parents[3]
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


def noncausal_attention(q, k, v, *, scale, backend, query_tile=64, key_tile=0):
    """Run an explicitly selected SM70 implementation without changing precision.

    CUDA entrypoints validate device, FP16 dtype, D128 heads, index limits and
    shapes. FlashAttention supports unequal Q/K lengths and strided inputs;
    FlashInfer supports matching lengths and receives contiguous BSHD tensors.
    Padding/masking and model-specific sparse selection belong to callers.
    The default four-argument call remains compatible with existing wheels.
    """
    if not math.isfinite(scale) or not 0 < scale <= 3.4028234663852886e38:
        raise ValueError("Attention scale must be positive and finite in FP32")
    if backend == "FLASH_ATTN_V100":
        if query_tile not in (64, 128) or key_tile not in (0, 64, 128):
            raise ValueError("Unsupported SM70 attention tile")
        ops = flashattn_extension()
        if query_tile == 64 and key_tile == 0:
            return ops.forward(q, k, v, scale)
        return ops.forward(q, k, v, scale, key_tile, query_tile)
    if backend == "FLASHINFER_SM70":
        if query_tile != 64 or key_tile != 0:
            raise ValueError("Explicit attention tiles require FLASH_ATTN_V100")
        return flashinfer_extension().forward(
            q.contiguous(), k.contiguous(), v.contiguous(), scale
        )
    raise ValueError(f"Unsupported SM70 attention backend: {backend}")
