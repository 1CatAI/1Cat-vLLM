# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-independent SM70 FP16 GEMM with FP32 scaling and accumulation.

The extension ABI retains its historical H3 name. Dispatch depends on tensor
properties, not checkpoint names, quantization labels or model families.
"""

from functools import lru_cache
from importlib import import_module
from pathlib import Path

import torch


@lru_cache(maxsize=1)
def sm70_extension():
    try:
        return import_module("vllm._h3_w8a16_C")
    except ImportError:
        pass
    from torch.utils.cpp_extension import load

    root = Path(__file__).resolve().parents[3]
    source = root / "csrc/sm70_turbomind/ops/h3_w8a16.cu"
    if not source.is_file():
        raise RuntimeError("SM70 diffusion operators require the 1Cat source build")
    return load(
        name="onecat_h3_w8a16",
        sources=[str(source)],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_70,code=sm_70"],
        extra_ldflags=["-lcublas", "-lcublasLt"],
        verbose=False,
    )


@lru_cache(maxsize=32)
def _column_major_plan(device, m, n, k, output_fp32):
    return sm70_extension().ColumnMajorGemmPlan(device, m, n, k, output_fp32)


def fp16_gemm(input, weight, output_fp32=False):
    """Use a zero-workspace Volta plan for dense column-major H3 weights.

    Plan entries contain host descriptors only. Unaligned inputs, empty shapes
    and library versions without the validated algorithm use the original
    row-major GEMM. Warm up each shape before capturing a CUDA graph.
    """
    ops = sm70_extension()
    if weight.is_contiguous():
        return ops.gemm(input, weight, output_fp32)
    if (
        input.is_cuda
        and weight.device == input.device
        and input.dim() == weight.dim() == 2
        and input.is_contiguous()
        and input.dtype == weight.dtype == torch.float16
        and weight.stride() == (1, weight.shape[0])
        and input.shape[1] == weight.shape[1]
        and min(*input.shape, weight.shape[0]) > 0
        and input.data_ptr() % 16 == weight.data_ptr() % 16 == 0
    ):
        plan = _column_major_plan(
            input.device.index,
            input.shape[0],
            weight.shape[0],
            input.shape[1],
            bool(output_fp32),
        )
        if plan.supported:
            return plan.run(input, weight)
    return ops.gemm(input, weight.contiguous(), output_fp32)


def fp16_gemm_input(x):
    """Scale wide-range activations by exact powers of two before FP16 GEMM.

    Leave headroom for the 256-channel rotation's worst-case amplification.
    Row scaling is restored in FP32 after the projection.
    """
    flat = x.reshape(-1, x.shape[-1]).contiguous()
    if flat.dtype == torch.float16:
        return flat, None
    if flat.dtype != torch.float32:
        raise ValueError("SM70 GEMM activations must be FP16 or FP32")
    if flat.is_cuda:
        return sm70_extension().prepare_fp16(flat)
    maximum = flat.abs().amax(-1, keepdim=True)
    _, exponent = torch.frexp(maximum)
    scale = torch.ldexp(torch.ones_like(maximum), (exponent - 11).clamp_min(0))
    return (flat / scale).half(), scale


def fp16_linear_prepared(values, weight, scale=None, *, output_fp32=False):
    """Restore per-row scales in FP32 before any distributed reduction."""
    if values.dtype != torch.float16 or weight.dtype != torch.float16:
        raise ValueError("Prepared SM70 linear operands must be FP16")
    output_fp32 = output_fp32 or scale is not None
    flat = values.reshape(-1, values.shape[-1]).contiguous()
    if flat.is_cuda:
        output = fp16_gemm(flat, weight, output_fp32)
    else:
        output = torch.nn.functional.linear(flat.float(), weight.float())
        if not output_fp32:
            output = output.half()
    if scale is not None:
        output = output * scale
    return output.reshape(*values.shape[:-1], weight.shape[0])
