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


def supports_fused_scaled_add(output):
    """Old wheels and unsupported output layouts keep ordinary epilogues."""
    return (
        output.is_cuda
        and output.dtype in (torch.float16, torch.float32)
        and output.is_contiguous()
        and torch.cuda.get_device_capability(output.device) == (7, 0)
        and hasattr(sm70_extension(), "scaled_add_")
    )


def fp16_linear_add(x, weight, output, *, alpha, offset=0):
    """Add a scaled FP16 projection into a contiguous output slice in place.

    GEMM and row-scale restoration retain FP32 boundaries. FP16 output is
    rounded after this contribution, so callers combining overlapping deltas
    must retain an FP32 accumulation buffer until their last contribution.
    """
    if not output.is_contiguous():
        raise ValueError("SM70 projection addition requires contiguous output")
    values, scale = fp16_gemm_input(x)
    delta = fp16_gemm(values, weight, output_fp32=True)
    flat = output.view(-1, output.shape[-1])
    if supports_fused_scaled_add(output):
        sm70_extension().scaled_add_(flat, delta, scale, alpha, offset)
    else:
        if scale is not None:
            delta = delta * scale
        target = flat[:, offset : offset + weight.shape[0]]
        result = target.float().add(delta, alpha=alpha).to(output.dtype)
        target.copy_(result)
    return output
