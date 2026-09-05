# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from vllm.platforms import current_platform

current_platform.import_kernels()


def _maybe_load_fp8_qpn8_library() -> None:
    """Load an explicitly selected source-built QPN8 extension.

    Production builds register these operators in ``vllm._C``. This opt-in
    path lets source experiments add only the QPN8 operators to an otherwise
    compatible installed build, including in spawned TP workers.
    """
    library_path = os.getenv("VLLM_SM70_FP8_QPN8_LIBRARY")
    if library_path is None:
        return
    generic_override = os.getenv("VLLM_SM70_FP8_QPN8")
    specific_override = os.getenv("VLLM_SM70_FP8_QPN8_PP2_TP4")
    online_override = os.getenv("VLLM_SM70_QWEN4_EXP_ONLINE_QPN8")
    generic_enabled = generic_override == "1"
    specific_enabled = generic_override != "0" and specific_override == "1"
    online_enabled = online_override == "1"
    if generic_enabled or specific_enabled or online_enabled:
        torch.ops.load_library(library_path)


_maybe_load_fp8_qpn8_library()


def _nvfp4_qpn2_prefill_library_path() -> str | None:
    library_path = os.getenv("VLLM_SM70_NVFP4_QPN2_PREFILL_LIBRARY")
    if library_path is not None:
        return library_path
    bundled = sorted(
        Path(__file__).resolve().parent.glob("_sm70_nvfp4_qpn2_prefill_C*.so")
    )
    return str(bundled[-1]) if bundled else None


def has_deferred_nvfp4_qpn2_prefill_library() -> bool:
    """Return whether a separate large-M prefill fragment is configured."""
    return _nvfp4_qpn2_prefill_library_path() is not None


def load_deferred_nvfp4_qpn2_prefill_library() -> bool:
    """Load an isolated large-M QPN2-packed prefill fragment.

    Source-overlay deployments can retain a previously validated decode
    extension while adding the newer, large-M-only QPN2-packed prefill op.
    The op owns its temporary dense workspace, so registration before AOT
    prefill compilation does not retain the large buffer during decode graph
    capture.
    Production wheels normally link the op into ``vllm._C``; a bundled
    fragment is only used when the main extension does not provide it.
    """
    if hasattr(torch.ops._C, "nvfp4_qpn4_prefill_sm70_out"):
        return True

    library_path = _nvfp4_qpn2_prefill_library_path()
    if library_path is not None:
        torch.ops.load_library(library_path)
    return hasattr(torch.ops._C, "nvfp4_qpn4_prefill_sm70_out")


load_deferred_nvfp4_qpn2_prefill_library()


def _maybe_load_nvfp4_qpn_m1_library() -> None:
    """Load the narrow Qwen3.8 NVFP4 experiment in spawned TP workers."""
    library_path = os.getenv("VLLM_SM70_NVFP4_QPN_M1_LIBRARY")
    m1_enabled = os.getenv("VLLM_SM70_NVFP4_QWEN38_MOE_QPN_M1_DECODE", "1") != "0"
    batch_enabled = os.getenv("VLLM_SM70_NVFP4_QWEN38_MOE_QPN_BATCH_DECODE", "1") != "0"
    mtp5_enabled = os.getenv("VLLM_SM70_NVFP4_QWEN38_MOE_QPN_MTP5_DECODE", "0") != "0"
    route_enabled = m1_enabled or batch_enabled or mtp5_enabled
    if library_path is not None and route_enabled:
        torch.ops.load_library(library_path)


_maybe_load_nvfp4_qpn_m1_library()


def _maybe_load_glm53_fp16_gemv_library() -> None:
    """Load the exact GLM-5.3 decode GEMV during source-side validation."""
    if hasattr(torch.ops._C, "sm70_glm53_fp16_gemv_out"):
        return
    library_path = os.getenv("VLLM_SM70_GLM53_FP16_GEMV_LIBRARY")
    if library_path is not None:
        torch.ops.load_library(library_path)


_maybe_load_glm53_fp16_gemv_library()


def _maybe_load_sm70_sampler_library() -> None:
    """Load the sampler fragment only when the main ``vllm._C`` lacks it."""
    if hasattr(torch.ops._C, "sm70_sample_chunked_top20_philox_token_out"):
        return

    library_path = os.getenv("VLLM_SM70_SAMPLER_LIBRARY")
    if library_path is None:
        bundled = sorted(Path(__file__).resolve().parent.glob("_sm70_sampler_C*.so"))
        if bundled:
            library_path = str(bundled[-1])
    if library_path is not None:
        torch.ops.load_library(library_path)


_maybe_load_sm70_sampler_library()

if TYPE_CHECKING:

    def register_fake(fn):
        return lambda name: fn
else:
    try:
        from torch.library import register_fake
    except ImportError:
        from torch.library import impl_abstract as register_fake


def _op(name: str):
    if not hasattr(torch.ops._C, name):
        raise RuntimeError(
            f"SM70 TurboMind op _C::{name} is not available. "
            "Build vLLM with CUDA arch 7.0 to enable it."
        )
    return getattr(torch.ops._C, name)


def _qwen38_qpn8_op(name: str):
    """Prefer the task sidecar, then fall back to the production namespace."""
    sidecar = torch.ops._C_qwen38
    if hasattr(sidecar, name):
        return getattr(sidecar, name)
    return _op(name)


def has_fp8_qpn8_hc_dispatch() -> bool:
    return hasattr(torch.ops._C_qwen38, "fp8_qpn8_hc_dispatch_sm70_out") or hasattr(
        torch.ops._C, "fp8_qpn8_hc_dispatch_sm70_out"
    )


def has_nvfp4_qpn_m1_dispatch() -> bool:
    return hasattr(torch.ops._C_qwen38, "nvfp4_moe_qpn_m1_sm70_out") or hasattr(
        torch.ops._C, "nvfp4_moe_qpn_m1_sm70_out"
    )


def has_nvfp4_qpn_raw_scale_dispatch() -> bool:
    names = (
        "nvfp4_expand_raw_scales_sm70_out",
        "nvfp4_moe_qpn_raw_scale_sm70_out",
        "nvfp4_moe_qpn_raw_w13_swiglu_batch_sm70_out",
        "nvfp4_moe_qpn_raw_w2_reduce_sm70_out",
    )
    return all(
        hasattr(torch.ops._C_qwen38, name) or hasattr(torch.ops._C, name)
        for name in names
    )


def has_nvfp4_qwen38_w2_direct_reduce() -> bool:
    return hasattr(torch.ops._C_qwen38, "nvfp4_qwen38_w2_direct_reduce_out") or hasattr(
        torch.ops._C, "nvfp4_qwen38_w2_direct_reduce_out"
    )


def has_nvfp4_qwen38_w13_fused_swiglu() -> bool:
    return hasattr(torch.ops._C_qwen38, "nvfp4_qwen38_w13_fused_swiglu_out") or hasattr(
        torch.ops._C, "nvfp4_qwen38_w13_fused_swiglu_out"
    )


def has_qwen38_shared_gate_exact() -> bool:
    return hasattr(torch.ops._C_qwen38, "qwen38_shared_gate_exact_out") or hasattr(
        torch.ops._C, "qwen38_shared_gate_exact_out"
    )


def has_nvfp4_qpn_mtp5_dispatch() -> bool:
    """Reject extensions that only implement the legacy ten-route kernel."""
    return hasattr(torch.ops._C_qwen38, "nvfp4_moe_qpn_mtp5_sm70_out") or hasattr(
        torch.ops._C, "nvfp4_moe_qpn_mtp5_sm70_out"
    )


def has_nvfp4_qpn_w13_swiglu_batch_dispatch() -> bool:
    return hasattr(
        torch.ops._C_qwen38,
        "nvfp4_moe_qpn_w13_swiglu_batch_sm70_out",
    ) or hasattr(torch.ops._C, "nvfp4_moe_qpn_w13_swiglu_batch_sm70_out")


def has_nvfp4_grouped_decode_dispatch() -> bool:
    return all(
        hasattr(torch.ops._C, name)
        for name in ("nvfp4_grouped_w13_sm70_out", "nvfp4_grouped_w2_sm70_out")
    )


def nvfp4_grouped_w13_sm70_out(
    out: torch.Tensor,
    x: torch.Tensor,
    w: torch.Tensor,
    s: torch.Tensor,
    ids: torch.Tensor,
    rows: torch.Tensor,
    experts: torch.Tensor,
    sizes: torch.Tensor,
    total: torch.Tensor,
    split: int,
    interleaved: bool,
) -> None:
    torch.ops._C.nvfp4_grouped_w13_sm70_out(
        out, x, w, s, ids, rows, experts, sizes, total, split, interleaved
    )


def nvfp4_grouped_w2_sm70_out(
    out: torch.Tensor,
    routed: torch.Tensor,
    x: torch.Tensor,
    w: torch.Tensor,
    s: torch.Tensor,
    topk: torch.Tensor,
    rows: torch.Tensor,
    experts: torch.Tensor,
    sizes: torch.Tensor,
    total: torch.Tensor,
) -> None:
    torch.ops._C.nvfp4_grouped_w2_sm70_out(
        out, routed, x, w, s, topk, rows, experts, sizes, total
    )


if has_nvfp4_grouped_decode_dispatch():

    @register_fake("_C::nvfp4_grouped_w13_sm70_out")
    def _grouped_w13_fake(
        out, x, w, s, ids, rows, experts, sizes, total, split, interleaved
    ):
        return None

    @register_fake("_C::nvfp4_grouped_w2_sm70_out")
    def _grouped_w2_fake(out, routed, x, w, s, topk, rows, experts, sizes, total):
        return None


def has_nvfp4_qpn_w2_reduce_dispatch() -> bool:
    return hasattr(
        torch.ops._C_qwen38,
        "nvfp4_moe_qpn_w2_reduce_sm70_out",
    ) or hasattr(torch.ops._C, "nvfp4_moe_qpn_w2_reduce_sm70_out")


def silu_and_mul_interleaved(out: torch.Tensor, input: torch.Tensor) -> None:
    _op("silu_and_mul_interleaved")(out, input)


if hasattr(torch.ops._C, "silu_and_mul_interleaved"):

    @register_fake("_C::silu_and_mul_interleaved")
    def _silu_and_mul_interleaved_fake(out: torch.Tensor, input: torch.Tensor) -> None:
        del out, input
        return None


def awq_sm70_prepare(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    group_size: int,
    interleave_gated_silu: bool = False,
) -> list[torch.Tensor]:
    return _op("awq_sm70_prepare")(
        qweight, scales, qzeros, group_size, interleave_gated_silu
    )


if hasattr(torch.ops._C, "awq_sm70_prepare"):

    @register_fake("_C::awq_sm70_prepare")
    def _awq_sm70_prepare_fake(
        qweight: torch.Tensor,
        scales: torch.Tensor,
        qzeros: torch.Tensor,
        group_size: int,
        interleave_gated_silu: bool,
    ) -> list[torch.Tensor]:
        del qzeros, group_size, interleave_gated_silu
        n = qweight.size(1) * 8
        num_groups = scales.size(0)
        tm_weight = torch.empty_like(qweight)
        tm_scales = torch.empty(
            (num_groups, n),
            dtype=torch.int32,
            device=qweight.device,
        )
        meta = torch.empty((2,), dtype=torch.int64, device=qweight.device)
        return [tm_weight, tm_scales, meta]


def awq_sm70_prepare_compact(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    group_size: int,
    interleave_gated_silu: bool = False,
) -> list[torch.Tensor]:
    return _op("awq_sm70_prepare_compact")(
        qweight, scales, qzeros, group_size, interleave_gated_silu
    )


if hasattr(torch.ops._C, "awq_sm70_prepare_compact"):

    @register_fake("_C::awq_sm70_prepare_compact")
    def _awq_sm70_prepare_compact_fake(
        qweight: torch.Tensor,
        scales: torch.Tensor,
        qzeros: torch.Tensor,
        group_size: int,
        interleave_gated_silu: bool,
    ) -> list[torch.Tensor]:
        del qzeros, group_size, interleave_gated_silu
        n = qweight.size(1) * 8
        num_groups = scales.size(0)
        tm_weight = torch.empty_like(qweight)
        tm_scales = torch.empty(
            (num_groups, n, 3),
            dtype=torch.uint8,
            device=qweight.device,
        )
        meta = torch.empty((2,), dtype=torch.int64, device=qweight.device)
        return [tm_weight, tm_scales, meta]


def awq_sm70_dequantize_out(
    out: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
) -> None:
    _op("awq_sm70_dequantize_out")(out, qweight, scales, group_size)


if hasattr(torch.ops._C, "awq_sm70_dequantize_out"):

    @register_fake("_C::awq_sm70_dequantize_out")
    def _awq_sm70_dequantize_out_fake(
        out: torch.Tensor,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        group_size: int,
    ) -> None:
        del out, qweight, scales, group_size
        return None


def uint4_sm70_prepare(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    group_size: int,
    interleave_gated_silu: bool = False,
) -> list[torch.Tensor]:
    return _op("uint4_sm70_prepare")(
        qweight, scales, zeros, group_size, interleave_gated_silu
    )


if hasattr(torch.ops._C, "uint4_sm70_prepare"):

    @register_fake("_C::uint4_sm70_prepare")
    def _uint4_sm70_prepare_fake(
        qweight: torch.Tensor,
        scales: torch.Tensor,
        zeros: torch.Tensor,
        group_size: int,
        interleave_gated_silu: bool,
    ) -> list[torch.Tensor]:
        del zeros, group_size, interleave_gated_silu
        k = qweight.size(0)
        n = qweight.size(1)
        num_groups = scales.size(0)
        tm_weight = torch.empty(
            (k, n // 8),
            dtype=torch.int32,
            device=qweight.device,
        )
        tm_scales = torch.empty(
            (num_groups, n),
            dtype=torch.int32,
            device=qweight.device,
        )
        meta = torch.empty((2,), dtype=torch.int64, device=qweight.device)
        return [tm_weight, tm_scales, meta]


def fp8_sm70_prepare(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    interleave_gated_silu: bool = False,
) -> list[torch.Tensor]:
    return _op("fp8_sm70_prepare")(qweight, scales, group_size, interleave_gated_silu)


if hasattr(torch.ops._C, "fp8_sm70_prepare"):

    @register_fake("_C::fp8_sm70_prepare")
    def _fp8_sm70_prepare_fake(
        qweight: torch.Tensor,
        scales: torch.Tensor,
        group_size: int,
        interleave_gated_silu: bool,
    ) -> list[torch.Tensor]:
        del group_size, interleave_gated_silu
        n = qweight.size(0)
        k = qweight.size(1)
        num_groups = scales.size(1)
        tm_weight = torch.empty((k, n), dtype=torch.uint8, device=qweight.device)
        tm_scales = torch.empty(
            (num_groups, n),
            dtype=torch.float16,
            device=qweight.device,
        )
        meta = torch.empty((2,), dtype=torch.int64, device=qweight.device)
        return [tm_weight, tm_scales, meta]


def fp8_sm70_dequantize_out(
    out: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
) -> None:
    _op("fp8_sm70_dequantize_out")(out, qweight, scales, group_size)


if hasattr(torch.ops._C, "fp8_sm70_dequantize_out"):

    @register_fake("_C::fp8_sm70_dequantize_out")
    def _fp8_sm70_dequantize_out_fake(
        out: torch.Tensor,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        group_size: int,
    ) -> None:
        del out, qweight, scales, group_size
        return None


def mxfp4_sm70_prepare(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    interleave_gated_silu: bool = False,
) -> list[torch.Tensor]:
    return _op("mxfp4_sm70_prepare")(qweight, scales, group_size, interleave_gated_silu)


if hasattr(torch.ops._C, "mxfp4_sm70_prepare"):

    @register_fake("_C::mxfp4_sm70_prepare")
    def _mxfp4_sm70_prepare_fake(
        qweight: torch.Tensor,
        scales: torch.Tensor,
        group_size: int,
        interleave_gated_silu: bool,
    ) -> list[torch.Tensor]:
        del group_size, interleave_gated_silu
        k = qweight.size(0)
        n = qweight.size(1)
        num_groups = scales.size(0)
        tm_weight = torch.empty(
            (k, n // 8),
            dtype=torch.int32,
            device=qweight.device,
        )
        tm_scales = torch.empty(
            (num_groups, n),
            dtype=torch.uint8,
            device=qweight.device,
        )
        meta = torch.empty((2,), dtype=torch.int64, device=qweight.device)
        return [tm_weight, tm_scales, meta]


def nvfp4_sm70_prepare(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    interleave_gated_silu: bool = False,
) -> list[torch.Tensor]:
    return _op("nvfp4_sm70_prepare")(qweight, scales, group_size, interleave_gated_silu)


if hasattr(torch.ops._C, "nvfp4_sm70_prepare"):

    @register_fake("_C::nvfp4_sm70_prepare")
    def _nvfp4_sm70_prepare_fake(
        qweight: torch.Tensor,
        scales: torch.Tensor,
        group_size: int,
        interleave_gated_silu: bool,
    ) -> list[torch.Tensor]:
        del group_size, interleave_gated_silu
        k = qweight.size(0)
        n = qweight.size(1)
        num_groups = scales.size(0)
        tm_weight = torch.empty(
            (k, n // 8),
            dtype=torch.int32,
            device=qweight.device,
        )
        tm_scales = torch.empty(
            (num_groups, n),
            dtype=torch.float16,
            device=qweight.device,
        )
        meta = torch.empty((2,), dtype=torch.int64, device=qweight.device)
        return [tm_weight, tm_scales, meta]


def sm70_f16_prepare(weight: torch.Tensor) -> list[torch.Tensor]:
    return _op("sm70_f16_prepare")(weight)


if hasattr(torch.ops._C, "sm70_f16_prepare"):

    @register_fake("_C::sm70_f16_prepare")
    def _sm70_f16_prepare_fake(weight: torch.Tensor) -> list[torch.Tensor]:
        meta = torch.empty((1,), dtype=torch.int64, device=weight.device)
        return [torch.empty_like(weight), meta]


def sm70_glm53_tp8_cublaslt_out(
    out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
) -> None:
    _op("sm70_glm53_tp8_cublaslt_out")(out, input, weight)


if hasattr(torch.ops._C, "sm70_glm53_tp8_cublaslt_out"):

    @register_fake("_C::sm70_glm53_tp8_cublaslt_out")
    def _sm70_glm53_tp8_cublaslt_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        weight: torch.Tensor,
    ) -> None:
        return None


def awq_gemm_sm70(
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    k_ld: int,
    q_ld: int,
) -> torch.Tensor:
    return _op("awq_gemm_sm70")(input, qweight, scales, group_size, k_ld, q_ld)


if hasattr(torch.ops._C, "awq_gemm_sm70"):

    @register_fake("_C::awq_gemm_sm70")
    def _awq_gemm_sm70_fake(
        input: torch.Tensor,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        group_size: int,
        k_ld: int,
        q_ld: int,
    ) -> torch.Tensor:
        del scales, group_size, k_ld, q_ld
        return torch.empty(
            (input.size(0), qweight.size(1) * 8),
            dtype=input.dtype,
            device=input.device,
        )


def awq_gemm_sm70_out(
    out: torch.Tensor,
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    k_ld: int,
    q_ld: int,
    gated_silu: bool = False,
) -> None:
    _op("awq_gemm_sm70_out")(
        out, input, qweight, scales, group_size, k_ld, q_ld, gated_silu
    )


if hasattr(torch.ops._C, "awq_gemm_sm70_out"):

    @register_fake("_C::awq_gemm_sm70_out")
    def _awq_gemm_sm70_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        group_size: int,
        k_ld: int,
        q_ld: int,
        gated_silu: bool,
    ) -> None:
        return None


def awq_gemm_sm70_out_tile_reduce(
    out: torch.Tensor,
    staging: torch.Tensor,
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    k_ld: int,
    q_ld: int,
    fa_ptr: int,
    tile_numel: int,
    reducer_blocks: int,
    kernel_reducer_blocks: int,
    overlap: bool,
) -> None:
    _op("awq_gemm_sm70_out_tile_reduce")(
        out,
        staging,
        input,
        qweight,
        scales,
        group_size,
        k_ld,
        q_ld,
        fa_ptr,
        tile_numel,
        reducer_blocks,
        kernel_reducer_blocks,
        overlap,
    )


if hasattr(torch.ops._C, "awq_gemm_sm70_out_tile_reduce"):

    @register_fake("_C::awq_gemm_sm70_out_tile_reduce")
    def _awq_gemm_sm70_out_tile_reduce_fake(
        out: torch.Tensor,
        staging: torch.Tensor,
        input: torch.Tensor,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        group_size: int,
        k_ld: int,
        q_ld: int,
        fa_ptr: int,
        tile_numel: int,
        reducer_blocks: int,
        kernel_reducer_blocks: int,
        overlap: bool,
    ) -> None:
        return None


def fp8_gemm_sm70_out(
    out: torch.Tensor,
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    k_ld: int,
    q_ld: int,
    gated_silu: bool = False,
) -> None:
    _op("fp8_gemm_sm70_out")(
        out, input, qweight, scales, group_size, k_ld, q_ld, gated_silu
    )


if hasattr(torch.ops._C, "fp8_gemm_sm70_out"):

    @register_fake("_C::fp8_gemm_sm70_out")
    def _fp8_gemm_sm70_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        group_size: int,
        k_ld: int,
        q_ld: int,
        gated_silu: bool,
    ) -> None:
        return None


def fp8_gemm_sm70_prefill_prescaled_out(
    out: torch.Tensor,
    input: torch.Tensor,
    qweight: torch.Tensor,
    prescaled_factors: torch.Tensor,
    group_size: int,
    k_ld: int,
    q_ld: int,
) -> None:
    _op("fp8_gemm_sm70_prefill_prescaled_out")(
        out, input, qweight, prescaled_factors, group_size, k_ld, q_ld
    )


if hasattr(torch.ops._C, "fp8_gemm_sm70_prefill_prescaled_out"):

    @register_fake("_C::fp8_gemm_sm70_prefill_prescaled_out")
    def _fp8_gemm_sm70_prefill_prescaled_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        qweight: torch.Tensor,
        prescaled_factors: torch.Tensor,
        group_size: int,
        k_ld: int,
        q_ld: int,
    ) -> None:
        return None


def fp8_gemm_sm70_prescaled_m1_out(
    out: torch.Tensor,
    input: torch.Tensor,
    qweight: torch.Tensor,
    prescaled_factors: torch.Tensor,
    group_size: int,
    k_ld: int,
    q_ld: int,
) -> None:
    _op("fp8_gemm_sm70_prescaled_m1_out")(
        out, input, qweight, prescaled_factors, group_size, k_ld, q_ld
    )


if hasattr(torch.ops._C, "fp8_gemm_sm70_prescaled_m1_out"):

    @register_fake("_C::fp8_gemm_sm70_prescaled_m1_out")
    def _fp8_gemm_sm70_prescaled_m1_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        qweight: torch.Tensor,
        prescaled_factors: torch.Tensor,
        group_size: int,
        k_ld: int,
        q_ld: int,
    ) -> None:
        return None


def fp8_qpn8_prepare_sm70(
    qweight: torch.Tensor,
    scales: torch.Tensor,
) -> list[torch.Tensor]:
    """Pack checkpoint-native block/channel FP8 weights into QPN8 layout."""
    return _qwen38_qpn8_op("fp8_qpn8_prepare_sm70")(qweight, scales)


if hasattr(torch.ops._C, "fp8_qpn8_prepare_sm70"):

    @register_fake("_C::fp8_qpn8_prepare_sm70")
    def _fp8_qpn8_prepare_sm70_fake(
        qweight: torch.Tensor,
        scales: torch.Tensor,
    ) -> list[torch.Tensor]:
        n = qweight.size(0)
        k = qweight.size(1)
        codes = torch.empty((k, n), dtype=torch.uint8, device=qweight.device)
        scale_shape = (1, n) if scales.shape == (n, 1) else (k // 128, n // 32)
        group_scales = torch.empty(
            scale_shape, dtype=torch.float16, device=scales.device
        )
        return [codes, group_scales]


def fp8_qpn8_dequantize_sm70_out(
    out: torch.Tensor,
    codes: torch.Tensor,
    group_scales: torch.Tensor,
) -> None:
    """Materialize one QPN8 weight into a caller-owned FP16 workspace."""
    _qwen38_qpn8_op("fp8_qpn8_dequantize_sm70_out")(out, codes, group_scales)


if hasattr(torch.ops._C, "fp8_qpn8_dequantize_sm70_out"):

    @register_fake("_C::fp8_qpn8_dequantize_sm70_out")
    def _fp8_qpn8_dequantize_sm70_out_fake(
        out: torch.Tensor,
        codes: torch.Tensor,
        group_scales: torch.Tensor,
    ) -> None:
        return None


def fp8_qpn8_prefill_sm70_out(
    out: torch.Tensor,
    dense_weight_ptr: int,
    input: torch.Tensor,
    codes: torch.Tensor,
    group_scales: torch.Tensor,
    gated_silu: bool,
) -> None:
    """Dequantize QPN8 into bounded workspace and run a large-M FP16 GEMM."""
    _qwen38_qpn8_op("fp8_qpn8_prefill_sm70_out")(
        out,
        dense_weight_ptr,
        input,
        codes,
        group_scales,
        gated_silu,
    )


if hasattr(torch.ops._C, "fp8_qpn8_prefill_sm70_out"):

    @register_fake("_C::fp8_qpn8_prefill_sm70_out")
    def _fp8_qpn8_prefill_sm70_out_fake(
        out: torch.Tensor,
        dense_weight_ptr: int,
        input: torch.Tensor,
        codes: torch.Tensor,
        group_scales: torch.Tensor,
        gated_silu: bool,
    ) -> None:
        return None


def fp8_qpn8_dispatch_sm70_out(
    out: torch.Tensor,
    dense_weight_ptr: int,
    input: torch.Tensor,
    codes: torch.Tensor,
    group_scales: torch.Tensor,
    split_k: int,
    accumulator_chains: int,
    prefetch_codes: bool,
    gated_silu: bool,
) -> None:
    """Runtime-dispatch dynamic M without specializing a Python branch."""
    _qwen38_qpn8_op("fp8_qpn8_dispatch_sm70_out")(
        out,
        dense_weight_ptr,
        input,
        codes,
        group_scales,
        split_k,
        accumulator_chains,
        prefetch_codes,
        gated_silu,
    )


if hasattr(torch.ops._C, "fp8_qpn8_dispatch_sm70_out"):

    @register_fake("_C::fp8_qpn8_dispatch_sm70_out")
    def _fp8_qpn8_dispatch_sm70_out_fake(
        out: torch.Tensor,
        dense_weight_ptr: int,
        input: torch.Tensor,
        codes: torch.Tensor,
        group_scales: torch.Tensor,
        split_k: int,
        accumulator_chains: int,
        prefetch_codes: bool,
        gated_silu: bool,
    ) -> None:
        return None


def fp8_qpn8_gemm_sm70_out(
    out: torch.Tensor,
    input: torch.Tensor,
    codes: torch.Tensor,
    group_scales: torch.Tensor,
    split_k: int,
    accumulator_chains: int,
    fast_decoder: bool,
    prefetch_codes: bool = False,
) -> None:
    """Run the SM70 QPN8 FP8 GEMM into ``out``.

    The model- and shape-gated automatic route and operator benchmark share
    this entry point. ``codes`` and ``group_scales`` use the QPN8 layout.
    """
    _qwen38_qpn8_op("fp8_qpn8_gemm_sm70_out")(
        out,
        input,
        codes,
        group_scales,
        split_k,
        accumulator_chains,
        fast_decoder,
        prefetch_codes,
    )


if hasattr(torch.ops._C, "fp8_qpn8_gemm_sm70_out"):

    @register_fake("_C::fp8_qpn8_gemm_sm70_out")
    def _fp8_qpn8_gemm_sm70_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        codes: torch.Tensor,
        group_scales: torch.Tensor,
        split_k: int,
        accumulator_chains: int,
        fast_decoder: bool,
        prefetch_codes: bool,
    ) -> None:
        return None


def fp8_qpn8_gemm_ba_split_sm70_out(
    qkv_out: torch.Tensor,
    z_out: torch.Tensor,
    b_out: torch.Tensor,
    a_out: torch.Tensor,
    input: torch.Tensor,
    codes: torch.Tensor,
    group_scales: torch.Tensor,
    ba_weight: torch.Tensor,
) -> None:
    """Run the exact-shape GDN QKV/Z FP8 and b/a FP16 projections."""
    _qwen38_qpn8_op("fp8_qpn8_gemm_ba_split_sm70_out")(
        qkv_out,
        z_out,
        b_out,
        a_out,
        input,
        codes,
        group_scales,
        ba_weight,
    )


if hasattr(torch.ops._C, "fp8_qpn8_gemm_ba_split_sm70_out"):

    @register_fake("_C::fp8_qpn8_gemm_ba_split_sm70_out")
    def _fp8_qpn8_gemm_ba_split_sm70_out_fake(
        qkv_out: torch.Tensor,
        z_out: torch.Tensor,
        b_out: torch.Tensor,
        a_out: torch.Tensor,
        input: torch.Tensor,
        codes: torch.Tensor,
        group_scales: torch.Tensor,
        ba_weight: torch.Tensor,
    ) -> None:
        return None


def fp8_qpn8_dispatch_ba_split_sm70_out(
    qkv_out: torch.Tensor,
    z_out: torch.Tensor,
    b_out: torch.Tensor,
    a_out: torch.Tensor,
    qkvz_staging: torch.Tensor,
    ba_staging: torch.Tensor,
    dense_weight_ptr: int,
    input: torch.Tensor,
    codes: torch.Tensor,
    group_scales: torch.Tensor,
    ba_weight: torch.Tensor,
) -> None:
    """Dispatch exact M=1 fusion or the original large-M projection path."""
    _qwen38_qpn8_op("fp8_qpn8_dispatch_ba_split_sm70_out")(
        qkv_out,
        z_out,
        b_out,
        a_out,
        qkvz_staging,
        ba_staging,
        dense_weight_ptr,
        input,
        codes,
        group_scales,
        ba_weight,
    )


if hasattr(torch.ops._C, "fp8_qpn8_dispatch_ba_split_sm70_out"):

    @register_fake("_C::fp8_qpn8_dispatch_ba_split_sm70_out")
    def _fp8_qpn8_dispatch_ba_split_sm70_out_fake(
        qkv_out: torch.Tensor,
        z_out: torch.Tensor,
        b_out: torch.Tensor,
        a_out: torch.Tensor,
        qkvz_staging: torch.Tensor,
        ba_staging: torch.Tensor,
        dense_weight_ptr: int,
        input: torch.Tensor,
        codes: torch.Tensor,
        group_scales: torch.Tensor,
        ba_weight: torch.Tensor,
    ) -> None:
        return None


def fp8_qpn8_gated_pair_sm70_out(
    out: torch.Tensor,
    input: torch.Tensor,
    codes: torch.Tensor,
    group_scales: torch.Tensor,
    split_k: int,
    accumulator_chains: int,
    fast_decoder: bool,
    prefetch_codes: bool = False,
) -> None:
    """Run the single-kernel paired-tile QPN8 gated SiLU experiment."""
    _qwen38_qpn8_op("fp8_qpn8_gated_pair_sm70_out")(
        out,
        input,
        codes,
        group_scales,
        split_k,
        accumulator_chains,
        fast_decoder,
        prefetch_codes,
    )


def fp8_qpn8_hc_dispatch_sm70_out(
    block_out: torch.Tensor,
    injection_out: torch.Tensor,
    down_staging: torch.Tensor,
    lora_staging: torch.Tensor,
    gate_staging: torch.Tensor,
    partials: torch.Tensor,
    dense_weight_ptr: int,
    xn: torch.Tensor,
    down_codes: torch.Tensor,
    down_scales: torch.Tensor,
    up_codes: torch.Tensor,
    up_scales: torch.Tensor,
) -> None:
    """Dispatch the exact Qwen4Exp HC pair without a Python M branch."""
    _qwen38_qpn8_op("fp8_qpn8_hc_dispatch_sm70_out")(
        block_out,
        injection_out,
        down_staging,
        lora_staging,
        gate_staging,
        partials,
        dense_weight_ptr,
        xn,
        down_codes,
        down_scales,
        up_codes,
        up_scales,
    )


if hasattr(torch.ops._C, "fp8_qpn8_hc_dispatch_sm70_out"):

    @register_fake("_C::fp8_qpn8_hc_dispatch_sm70_out")
    def _fp8_qpn8_hc_dispatch_sm70_out_fake(
        block_out: torch.Tensor,
        injection_out: torch.Tensor,
        down_staging: torch.Tensor,
        lora_staging: torch.Tensor,
        gate_staging: torch.Tensor,
        partials: torch.Tensor,
        dense_weight_ptr: int,
        xn: torch.Tensor,
        down_codes: torch.Tensor,
        down_scales: torch.Tensor,
        up_codes: torch.Tensor,
        up_scales: torch.Tensor,
    ) -> None:
        return None


if hasattr(torch.ops._C_qwen38, "fp8_qpn8_hc_dispatch_sm70_out"):

    @register_fake("_C_qwen38::fp8_qpn8_hc_dispatch_sm70_out")
    def _fp8_qpn8_hc_dispatch_sm70_out_sidecar_fake(
        block_out: torch.Tensor,
        injection_out: torch.Tensor,
        down_staging: torch.Tensor,
        lora_staging: torch.Tensor,
        gate_staging: torch.Tensor,
        partials: torch.Tensor,
        dense_weight_ptr: int,
        xn: torch.Tensor,
        down_codes: torch.Tensor,
        down_scales: torch.Tensor,
        up_codes: torch.Tensor,
        up_scales: torch.Tensor,
    ) -> None:
        return None


if hasattr(torch.ops._C, "fp8_qpn8_gated_pair_sm70_out"):

    @register_fake("_C::fp8_qpn8_gated_pair_sm70_out")
    def _fp8_qpn8_gated_pair_sm70_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        codes: torch.Tensor,
        group_scales: torch.Tensor,
        split_k: int,
        accumulator_chains: int,
        fast_decoder: bool,
        prefetch_codes: bool,
    ) -> None:
        return None


def _register_qwen38_qpn8_out_fakes() -> None:
    """Teach Dynamo about out-variant operators from the task sidecar."""

    def fake_out(*args, **kwargs) -> None:
        del args, kwargs
        return None

    for name in (
        "fp8_qpn8_dequantize_sm70_out",
        "fp8_qpn8_prefill_sm70_out",
        "fp8_qpn8_dispatch_sm70_out",
        "fp8_qpn8_gemm_sm70_out",
        "fp8_qpn8_gemm_ba_split_sm70_out",
        "fp8_qpn8_dispatch_ba_split_sm70_out",
        "fp8_qpn8_gated_pair_sm70_out",
    ):
        if hasattr(torch.ops._C_qwen38, name):
            register_fake(f"_C_qwen38::{name}")(fake_out)


_register_qwen38_qpn8_out_fakes()


def nvfp4_qpn4_prepare_sm70(
    qweight: torch.Tensor,
    scales: torch.Tensor,
) -> list[torch.Tensor]:
    """Pack unpacked FP4 weights and FP16 group scales into QPN4 layout."""
    return _op("nvfp4_qpn4_prepare_sm70")(qweight, scales)


if hasattr(torch.ops._C, "nvfp4_qpn4_prepare_sm70"):

    @register_fake("_C::nvfp4_qpn4_prepare_sm70")
    def _nvfp4_qpn4_prepare_sm70_fake(
        qweight: torch.Tensor,
        scales: torch.Tensor,
    ) -> list[torch.Tensor]:
        k, n = qweight.shape
        codes = torch.empty((k, n // 2), dtype=torch.uint8, device=qweight.device)
        packed_scales = torch.empty_like(scales)
        return [codes, packed_scales]


def nvfp4_qpn4_prepare_scale_code_sm70(
    qweight: torch.Tensor,
    scale_codes: torch.Tensor,
) -> list[torch.Tensor]:
    """Pack unpacked FP4 weights and checkpoint E4M3 group-scale codes."""
    return _op("nvfp4_qpn4_prepare_scale_code_sm70")(qweight, scale_codes)


if hasattr(torch.ops._C, "nvfp4_qpn4_prepare_scale_code_sm70"):

    @register_fake("_C::nvfp4_qpn4_prepare_scale_code_sm70")
    def _nvfp4_qpn4_prepare_scale_code_sm70_fake(
        qweight: torch.Tensor,
        scale_codes: torch.Tensor,
    ) -> list[torch.Tensor]:
        k, n = qweight.shape
        codes = torch.empty((k, n // 2), dtype=torch.uint8, device=qweight.device)
        packed_scale_codes = torch.empty(
            (k // 16, n),
            dtype=torch.uint8,
            device=scale_codes.device,
        )
        return [codes, packed_scale_codes]


def nvfp4_qpn4_dequantize_sm70_out(
    out: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    global_scale: float,
    use_scale_code: bool,
) -> None:
    _op("nvfp4_qpn4_dequantize_sm70_out")(
        out, codes, scales, global_scale, use_scale_code
    )


if hasattr(torch.ops._C, "nvfp4_qpn4_dequantize_sm70_out"):

    @register_fake("_C::nvfp4_qpn4_dequantize_sm70_out")
    def _nvfp4_qpn4_dequantize_sm70_out_fake(
        out: torch.Tensor,
        codes: torch.Tensor,
        scales: torch.Tensor,
        global_scale: float,
        use_scale_code: bool,
    ) -> None:
        return None


def nvfp4_qpn4_prefill_sm70_out(
    out: torch.Tensor,
    dense_weight_ptr: int,
    input: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    global_scale: float,
    use_scale_code: bool,
    gated_silu: bool,
) -> None:
    _op("nvfp4_qpn4_prefill_sm70_out")(
        out,
        dense_weight_ptr,
        input,
        codes,
        scales,
        global_scale,
        use_scale_code,
        gated_silu,
    )


if hasattr(torch.ops._C, "nvfp4_qpn4_prefill_sm70_out"):

    @register_fake("_C::nvfp4_qpn4_prefill_sm70_out")
    def _nvfp4_qpn4_prefill_sm70_out_fake(
        out: torch.Tensor,
        dense_weight_ptr: int,
        input: torch.Tensor,
        codes: torch.Tensor,
        scales: torch.Tensor,
        global_scale: float,
        use_scale_code: bool,
        gated_silu: bool,
    ) -> None:
        return None


def nvfp4_qpn4_dispatch_sm70_out(
    out: torch.Tensor,
    dense_weight_ptr: int,
    input: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    global_scale: float,
    use_scale_code: bool,
    gated_silu: bool,
) -> None:
    """Dispatch exact M=1 decode or bounded-workspace large-M prefill."""
    _op("nvfp4_qpn4_dispatch_sm70_out")(
        out,
        dense_weight_ptr,
        input,
        codes,
        scales,
        global_scale,
        use_scale_code,
        gated_silu,
    )


if hasattr(torch.ops._C, "nvfp4_qpn4_dispatch_sm70_out"):

    @register_fake("_C::nvfp4_qpn4_dispatch_sm70_out")
    def _nvfp4_qpn4_dispatch_sm70_out_fake(
        out: torch.Tensor,
        dense_weight_ptr: int,
        input: torch.Tensor,
        codes: torch.Tensor,
        scales: torch.Tensor,
        global_scale: float,
        use_scale_code: bool,
        gated_silu: bool,
    ) -> None:
        return None


def nvfp4_qpn2_prepare_sm70(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
) -> list[torch.Tensor]:
    """Pack checkpoint-native NVFP4 tensors into the QPN2 layout."""
    return _op("nvfp4_qpn2_prepare_sm70")(weight_packed, weight_scale)


if hasattr(torch.ops._C, "nvfp4_qpn2_prepare_sm70"):

    @register_fake("_C::nvfp4_qpn2_prepare_sm70")
    def _nvfp4_qpn2_prepare_sm70_fake(
        weight_packed: torch.Tensor,
        weight_scale: torch.Tensor,
    ) -> list[torch.Tensor]:
        codes = torch.empty_like(weight_packed)
        scales = torch.empty(
            weight_scale.shape, dtype=torch.uint8, device=weight_scale.device
        )
        return [codes, scales]


def nvfp4_qpn2_gemm_sm70_out(
    out: torch.Tensor,
    input: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    global_scale: float,
    split_k: int,
    accumulator_chains: int,
) -> None:
    _op("nvfp4_qpn2_gemm_sm70_out")(
        out,
        input,
        codes,
        scales,
        global_scale,
        split_k,
        accumulator_chains,
    )


if hasattr(torch.ops._C, "nvfp4_qpn2_gemm_sm70_out"):

    @register_fake("_C::nvfp4_qpn2_gemm_sm70_out")
    def _nvfp4_qpn2_gemm_sm70_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        codes: torch.Tensor,
        scales: torch.Tensor,
        global_scale: float,
        split_k: int,
        accumulator_chains: int,
    ) -> None:
        return None


def nvfp4_qpn2_gated_sm70_out(
    out: torch.Tensor,
    input: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    global_scale: float,
    split_k: int,
    accumulator_chains: int,
) -> None:
    _op("nvfp4_qpn2_gated_sm70_out")(
        out,
        input,
        codes,
        scales,
        global_scale,
        split_k,
        accumulator_chains,
    )


if hasattr(torch.ops._C, "nvfp4_qpn2_gated_sm70_out"):

    @register_fake("_C::nvfp4_qpn2_gated_sm70_out")
    def _nvfp4_qpn2_gated_sm70_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        codes: torch.Tensor,
        scales: torch.Tensor,
        global_scale: float,
        split_k: int,
        accumulator_chains: int,
    ) -> None:
        return None


def nvfp4_qpn2_dispatch_sm70_out(
    out: torch.Tensor,
    input: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    global_scale: float,
    split_k: int,
    accumulator_chains: int,
    tm_weight: torch.Tensor,
    tm_scales: torch.Tensor,
    tm_group_size: int,
    tm_k_ld: int,
    tm_q_ld: int,
    gated_silu: bool,
) -> None:
    """Select QPN2 for M<=8 and TurboMind for larger dynamic M."""
    _op("nvfp4_qpn2_dispatch_sm70_out")(
        out,
        input,
        codes,
        scales,
        global_scale,
        split_k,
        accumulator_chains,
        tm_weight,
        tm_scales,
        tm_group_size,
        tm_k_ld,
        tm_q_ld,
        gated_silu,
    )


if hasattr(torch.ops._C, "nvfp4_qpn2_dispatch_sm70_out"):

    @register_fake("_C::nvfp4_qpn2_dispatch_sm70_out")
    def _nvfp4_qpn2_dispatch_sm70_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        codes: torch.Tensor,
        scales: torch.Tensor,
        global_scale: float,
        split_k: int,
        accumulator_chains: int,
        tm_weight: torch.Tensor,
        tm_scales: torch.Tensor,
        tm_group_size: int,
        tm_k_ld: int,
        tm_q_ld: int,
        gated_silu: bool,
    ) -> None:
        return None


def nvfp4_qpn2_prefill_dispatch_sm70_out(
    out: torch.Tensor,
    input: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    global_scale: float,
    split_k: int,
    accumulator_chains: int,
    tm_weight: torch.Tensor,
    tm_scales: torch.Tensor,
    tm_group_size: int,
    tm_k_ld: int,
    tm_q_ld: int,
    gated_silu: bool,
    min_prefill_m: int,
) -> None:
    """Keep QPN2 decode and QPN2-packed prefill behind one opaque op."""
    _op("nvfp4_qpn2_prefill_dispatch_sm70_out")(
        out,
        input,
        codes,
        scales,
        global_scale,
        split_k,
        accumulator_chains,
        tm_weight,
        tm_scales,
        tm_group_size,
        tm_k_ld,
        tm_q_ld,
        gated_silu,
        min_prefill_m,
    )


if hasattr(torch.ops._C, "nvfp4_qpn2_prefill_dispatch_sm70_out"):

    @register_fake("_C::nvfp4_qpn2_prefill_dispatch_sm70_out")
    def _nvfp4_qpn2_prefill_dispatch_sm70_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        codes: torch.Tensor,
        scales: torch.Tensor,
        global_scale: float,
        split_k: int,
        accumulator_chains: int,
        tm_weight: torch.Tensor,
        tm_scales: torch.Tensor,
        tm_group_size: int,
        tm_k_ld: int,
        tm_q_ld: int,
        gated_silu: bool,
        min_prefill_m: int,
    ) -> None:
        return None


def fp8_gemm_sm70_prefill_dispatch_out(
    out: torch.Tensor,
    dense_weight_ptr: int,
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    k_ld: int,
    q_ld: int,
    gated_silu: bool,
    min_prefill_m: int,
) -> None:
    _op("fp8_gemm_sm70_prefill_dispatch_out")(
        out,
        dense_weight_ptr,
        input,
        qweight,
        scales,
        group_size,
        k_ld,
        q_ld,
        gated_silu,
        min_prefill_m,
    )


if hasattr(torch.ops._C, "fp8_gemm_sm70_prefill_dispatch_out"):

    @register_fake("_C::fp8_gemm_sm70_prefill_dispatch_out")
    def _fp8_gemm_sm70_prefill_dispatch_out_fake(
        out: torch.Tensor,
        dense_weight_ptr: int,
        input: torch.Tensor,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        group_size: int,
        k_ld: int,
        q_ld: int,
        gated_silu: bool,
        min_prefill_m: int,
    ) -> None:
        del dense_weight_ptr
        return None


def mxfp4_gemm_sm70_out(
    out: torch.Tensor,
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    k_ld: int,
    q_ld: int,
    gated_silu: bool = False,
) -> None:
    _op("mxfp4_gemm_sm70_out")(
        out, input, qweight, scales, group_size, k_ld, q_ld, gated_silu
    )


if hasattr(torch.ops._C, "mxfp4_gemm_sm70_out"):

    @register_fake("_C::mxfp4_gemm_sm70_out")
    def _mxfp4_gemm_sm70_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        group_size: int,
        k_ld: int,
        q_ld: int,
        gated_silu: bool,
    ) -> None:
        return None


def mxfp4_moe_dense_stage_sm70_out(
    out: torch.Tensor,
    input: torch.Tensor,
    expert_offsets: torch.Tensor,
    dense_expert_ids: torch.Tensor,
    ptrs_w: torch.Tensor,
    ptrs_s: torch.Tensor,
    num_experts: int,
    k: int,
    n: int,
    group_size: int,
) -> None:
    _op("mxfp4_moe_dense_stage_sm70_out")(
        out,
        input,
        expert_offsets,
        dense_expert_ids,
        ptrs_w,
        ptrs_s,
        num_experts,
        k,
        n,
        group_size,
    )


if hasattr(torch.ops._C, "mxfp4_moe_dense_stage_sm70_out"):

    @register_fake("_C::mxfp4_moe_dense_stage_sm70_out")
    def _mxfp4_moe_dense_stage_sm70_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        expert_offsets: torch.Tensor,
        dense_expert_ids: torch.Tensor,
        ptrs_w: torch.Tensor,
        ptrs_s: torch.Tensor,
        num_experts: int,
        k: int,
        n: int,
        group_size: int,
    ) -> None:
        return None


def mxfp4_moe_qpn_m1_sm70_out(
    out: torch.Tensor,
    input: torch.Tensor,
    weights: torch.Tensor,
    scales: torch.Tensor,
    expert_ids: torch.Tensor,
    broadcast_input: bool,
) -> None:
    _op("mxfp4_moe_qpn_m1_sm70_out")(
        out, input, weights, scales, expert_ids, broadcast_input
    )


if hasattr(torch.ops._C, "mxfp4_moe_qpn_m1_sm70_out"):

    @register_fake("_C::mxfp4_moe_qpn_m1_sm70_out")
    def _mxfp4_moe_qpn_m1_sm70_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        weights: torch.Tensor,
        scales: torch.Tensor,
        expert_ids: torch.Tensor,
        broadcast_input: bool,
    ) -> None:
        return None


def nvfp4_moe_qpn_m1_sm70_out(
    out: torch.Tensor,
    input: torch.Tensor,
    weights: torch.Tensor,
    scales: torch.Tensor,
    expert_ids: torch.Tensor,
    broadcast_input: bool,
    split_k: int,
) -> None:
    _qwen38_qpn8_op("nvfp4_moe_qpn_m1_sm70_out")(
        out,
        input,
        weights,
        scales,
        expert_ids,
        broadcast_input,
        split_k,
    )


def nvfp4_moe_qpn_raw_scale_sm70_out(
    out: torch.Tensor,
    input: torch.Tensor,
    weights: torch.Tensor,
    scale_codes: torch.Tensor,
    global_scales: torch.Tensor,
    expert_ids: torch.Tensor,
    broadcast_input: bool,
    interleaved_w13: bool,
    split_k: int,
) -> None:
    _qwen38_qpn8_op("nvfp4_moe_qpn_raw_scale_sm70_out")(
        out,
        input,
        weights,
        scale_codes,
        global_scales,
        expert_ids,
        broadcast_input,
        interleaved_w13,
        split_k,
    )


def nvfp4_expand_raw_scales_sm70_out(
    out: torch.Tensor,
    scale_codes: torch.Tensor,
    global_scales: torch.Tensor,
    interleaved_w13: bool,
    fast_decode_rounding: bool = False,
) -> None:
    _qwen38_qpn8_op("nvfp4_expand_raw_scales_sm70_out")(
        out,
        scale_codes,
        global_scales,
        interleaved_w13,
        fast_decode_rounding,
    )


def nvfp4_moe_qpn_raw_w13_swiglu_batch_sm70_out(
    out: torch.Tensor,
    input: torch.Tensor,
    weights: torch.Tensor,
    scale_codes: torch.Tensor,
    global_scales: torch.Tensor,
    expert_ids: torch.Tensor,
    interleaved: bool,
) -> None:
    _qwen38_qpn8_op("nvfp4_moe_qpn_raw_w13_swiglu_batch_sm70_out")(
        out,
        input,
        weights,
        scale_codes,
        global_scales,
        expert_ids,
        interleaved,
    )


def nvfp4_moe_qpn_raw_w2_reduce_sm70_out(
    out: torch.Tensor,
    input: torch.Tensor,
    weights: torch.Tensor,
    scale_codes: torch.Tensor,
    global_scales: torch.Tensor,
    expert_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> None:
    _qwen38_qpn8_op("nvfp4_moe_qpn_raw_w2_reduce_sm70_out")(
        out,
        input,
        weights,
        scale_codes,
        global_scales,
        expert_ids,
        topk_weights,
    )


def nvfp4_moe_qpn_w13_swiglu_batch_sm70_out(
    out: torch.Tensor,
    input: torch.Tensor,
    weights: torch.Tensor,
    scales: torch.Tensor,
    expert_ids: torch.Tensor,
    interleaved: bool,
) -> None:
    _qwen38_qpn8_op("nvfp4_moe_qpn_w13_swiglu_batch_sm70_out")(
        out,
        input,
        weights,
        scales,
        expert_ids,
        interleaved,
    )


def nvfp4_moe_qpn_w2_reduce_sm70_out(
    out: torch.Tensor,
    input: torch.Tensor,
    weights: torch.Tensor,
    scales: torch.Tensor,
    expert_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> None:
    _qwen38_qpn8_op("nvfp4_moe_qpn_w2_reduce_sm70_out")(
        out,
        input,
        weights,
        scales,
        expert_ids,
        topk_weights,
    )


if hasattr(torch.ops._C, "nvfp4_moe_qpn_w2_reduce_sm70_out"):

    @register_fake("_C::nvfp4_moe_qpn_w2_reduce_sm70_out")
    def _nvfp4_moe_qpn_w2_reduce_sm70_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        weights: torch.Tensor,
        scales: torch.Tensor,
        expert_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> None:
        return None


if hasattr(torch.ops._C_qwen38, "nvfp4_moe_qpn_w2_reduce_sm70_out"):

    @register_fake("_C_qwen38::nvfp4_moe_qpn_w2_reduce_sm70_out")
    def _nvfp4_moe_qpn_w2_reduce_sm70_sidecar_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        weights: torch.Tensor,
        scales: torch.Tensor,
        expert_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> None:
        return None


if hasattr(torch.ops._C, "nvfp4_moe_qpn_w13_swiglu_batch_sm70_out"):

    @register_fake("_C::nvfp4_moe_qpn_w13_swiglu_batch_sm70_out")
    def _nvfp4_moe_qpn_w13_swiglu_batch_sm70_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        weights: torch.Tensor,
        scales: torch.Tensor,
        expert_ids: torch.Tensor,
        interleaved: bool,
    ) -> None:
        return None


if hasattr(torch.ops._C_qwen38, "nvfp4_moe_qpn_w13_swiglu_batch_sm70_out"):

    @register_fake("_C_qwen38::nvfp4_moe_qpn_w13_swiglu_batch_sm70_out")
    def _nvfp4_moe_qpn_w13_swiglu_batch_sm70_sidecar_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        weights: torch.Tensor,
        scales: torch.Tensor,
        expert_ids: torch.Tensor,
        interleaved: bool,
    ) -> None:
        return None


if hasattr(torch.ops._C, "nvfp4_moe_qpn_m1_sm70_out"):

    @register_fake("_C::nvfp4_moe_qpn_m1_sm70_out")
    def _nvfp4_moe_qpn_m1_sm70_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        weights: torch.Tensor,
        scales: torch.Tensor,
        expert_ids: torch.Tensor,
        broadcast_input: bool,
        split_k: int,
    ) -> None:
        return None


if hasattr(torch.ops._C_qwen38, "nvfp4_moe_qpn_m1_sm70_out"):

    @register_fake("_C_qwen38::nvfp4_moe_qpn_m1_sm70_out")
    def _nvfp4_moe_qpn_m1_sm70_out_sidecar_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        weights: torch.Tensor,
        scales: torch.Tensor,
        expert_ids: torch.Tensor,
        broadcast_input: bool,
        split_k: int,
    ) -> None:
        return None


if hasattr(torch.ops._C, "nvfp4_moe_qpn_raw_scale_sm70_out"):

    @register_fake("_C::nvfp4_moe_qpn_raw_scale_sm70_out")
    def _nvfp4_moe_qpn_raw_scale_sm70_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        weights: torch.Tensor,
        scale_codes: torch.Tensor,
        global_scales: torch.Tensor,
        expert_ids: torch.Tensor,
        broadcast_input: bool,
        interleaved_w13: bool,
        split_k: int,
    ) -> None:
        return None


def nvfp4_qwen38_w2_direct_reduce_out(
    out: torch.Tensor,
    input: torch.Tensor,
    weights: torch.Tensor,
    scales: torch.Tensor,
    expert_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> None:
    _qwen38_qpn8_op("nvfp4_qwen38_w2_direct_reduce_out")(
        out, input, weights, scales, expert_ids, topk_weights
    )


if hasattr(torch.ops._C, "nvfp4_qwen38_w2_direct_reduce_out"):

    @register_fake("_C::nvfp4_qwen38_w2_direct_reduce_out")
    def _nvfp4_qwen38_w2_direct_reduce_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        weights: torch.Tensor,
        scales: torch.Tensor,
        expert_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> None:
        return None


if hasattr(torch.ops._C_qwen38, "nvfp4_moe_qpn_raw_scale_sm70_out"):

    @register_fake("_C_qwen38::nvfp4_moe_qpn_raw_scale_sm70_out")
    def _nvfp4_moe_qpn_raw_scale_sm70_sidecar_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        weights: torch.Tensor,
        scale_codes: torch.Tensor,
        global_scales: torch.Tensor,
        expert_ids: torch.Tensor,
        broadcast_input: bool,
        interleaved_w13: bool,
        split_k: int,
    ) -> None:
        return None


if hasattr(torch.ops._C_qwen38, "nvfp4_qwen38_w2_direct_reduce_out"):

    @register_fake("_C_qwen38::nvfp4_qwen38_w2_direct_reduce_out")
    def _nvfp4_qwen38_w2_direct_reduce_out_sidecar_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        weights: torch.Tensor,
        scales: torch.Tensor,
        expert_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> None:
        return None


for _raw_namespace, _raw_prefix in (
    (torch.ops._C, "_C"),
    (torch.ops._C_qwen38, "_C_qwen38"),
):
    if hasattr(_raw_namespace, "nvfp4_expand_raw_scales_sm70_out"):
        register_fake(f"{_raw_prefix}::nvfp4_expand_raw_scales_sm70_out")(
            lambda out,
            scale_codes,
            global_scales,
            interleaved_w13,
            fast_decode_rounding: (None)
        )
    if hasattr(_raw_namespace, "nvfp4_moe_qpn_raw_w13_swiglu_batch_sm70_out"):
        register_fake(f"{_raw_prefix}::nvfp4_moe_qpn_raw_w13_swiglu_batch_sm70_out")(
            lambda out,
            input,
            weights,
            scale_codes,
            global_scales,
            expert_ids,
            interleaved: (None)
        )
    if hasattr(_raw_namespace, "nvfp4_moe_qpn_raw_w2_reduce_sm70_out"):
        register_fake(f"{_raw_prefix}::nvfp4_moe_qpn_raw_w2_reduce_sm70_out")(
            lambda out,
            input,
            weights,
            scale_codes,
            global_scales,
            expert_ids,
            topk_weights: (None)
        )


def nvfp4_qwen38_w13_fused_swiglu_out(
    out: torch.Tensor,
    input: torch.Tensor,
    weights: torch.Tensor,
    scales: torch.Tensor,
    expert_ids: torch.Tensor,
) -> None:
    _qwen38_qpn8_op("nvfp4_qwen38_w13_fused_swiglu_out")(
        out, input, weights, scales, expert_ids
    )


if hasattr(torch.ops._C, "nvfp4_qwen38_w13_fused_swiglu_out"):

    @register_fake("_C::nvfp4_qwen38_w13_fused_swiglu_out")
    def _nvfp4_qwen38_w13_fused_swiglu_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        weights: torch.Tensor,
        scales: torch.Tensor,
        expert_ids: torch.Tensor,
    ) -> None:
        return None


if hasattr(torch.ops._C_qwen38, "nvfp4_qwen38_w13_fused_swiglu_out"):

    @register_fake("_C_qwen38::nvfp4_qwen38_w13_fused_swiglu_out")
    def _nvfp4_qwen38_w13_fused_swiglu_out_sidecar_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        weights: torch.Tensor,
        scales: torch.Tensor,
        expert_ids: torch.Tensor,
    ) -> None:
        return None


def qwen38_shared_gate_exact_out(
    out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
) -> None:
    _qwen38_qpn8_op("qwen38_shared_gate_exact_out")(out, input, weight)


if hasattr(torch.ops._C, "qwen38_shared_gate_exact_out"):

    @register_fake("_C::qwen38_shared_gate_exact_out")
    def _qwen38_shared_gate_exact_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        weight: torch.Tensor,
    ) -> None:
        return None


if hasattr(torch.ops._C_qwen38, "qwen38_shared_gate_exact_out"):

    @register_fake("_C_qwen38::qwen38_shared_gate_exact_out")
    def _qwen38_shared_gate_exact_out_sidecar_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        weight: torch.Tensor,
    ) -> None:
        return None


def nvfp4_glm53_moe_q8_qpn_sm70_out(
    out: torch.Tensor,
    input: torch.Tensor,
    weights: torch.Tensor,
    scales: torch.Tensor,
    expert_ids: torch.Tensor,
    sorted_row_idx: torch.Tensor,
    w13: bool,
) -> None:
    _op("nvfp4_glm53_moe_q8_qpn_sm70_out")(
        out, input, weights, scales, expert_ids, sorted_row_idx, w13
    )


if hasattr(torch.ops._C, "nvfp4_glm53_moe_q8_qpn_sm70_out"):

    @register_fake("_C::nvfp4_glm53_moe_q8_qpn_sm70_out")
    def _nvfp4_glm53_moe_q8_qpn_sm70_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        weights: torch.Tensor,
        scales: torch.Tensor,
        expert_ids: torch.Tensor,
        sorted_row_idx: torch.Tensor,
        w13: bool,
    ) -> None:
        return None


def nvfp4_moe_qpn_mtp5_sm70_out(
    out: torch.Tensor,
    input: torch.Tensor,
    weights: torch.Tensor,
    scales: torch.Tensor,
    expert_ids: torch.Tensor,
    broadcast_input: bool,
    split_k: int,
) -> None:
    _qwen38_qpn8_op("nvfp4_moe_qpn_mtp5_sm70_out")(
        out,
        input,
        weights,
        scales,
        expert_ids,
        broadcast_input,
        split_k,
    )


if hasattr(torch.ops._C, "nvfp4_moe_qpn_mtp5_sm70_out"):

    @register_fake("_C::nvfp4_moe_qpn_mtp5_sm70_out")
    def _nvfp4_moe_qpn_mtp5_sm70_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        weights: torch.Tensor,
        scales: torch.Tensor,
        expert_ids: torch.Tensor,
        broadcast_input: bool,
        split_k: int,
    ) -> None:
        return None


if hasattr(torch.ops._C_qwen38, "nvfp4_moe_qpn_mtp5_sm70_out"):

    @register_fake("_C_qwen38::nvfp4_moe_qpn_mtp5_sm70_out")
    def _nvfp4_moe_qpn_mtp5_sm70_out_sidecar_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        weights: torch.Tensor,
        scales: torch.Tensor,
        expert_ids: torch.Tensor,
        broadcast_input: bool,
        split_k: int,
    ) -> None:
        return None


def nvfp4_moe_dense_stage_sm70_out(
    out: torch.Tensor,
    input: torch.Tensor,
    expert_offsets: torch.Tensor,
    dense_expert_ids: torch.Tensor,
    ptrs_w: torch.Tensor,
    ptrs_s: torch.Tensor,
    num_experts: int,
    k: int,
    n: int,
    group_size: int,
) -> None:
    _op("nvfp4_moe_dense_stage_sm70_out")(
        out,
        input,
        expert_offsets,
        dense_expert_ids,
        ptrs_w,
        ptrs_s,
        num_experts,
        k,
        n,
        group_size,
    )


if hasattr(torch.ops._C, "nvfp4_moe_dense_stage_sm70_out"):

    @register_fake("_C::nvfp4_moe_dense_stage_sm70_out")
    def _nvfp4_moe_dense_stage_sm70_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        expert_offsets: torch.Tensor,
        dense_expert_ids: torch.Tensor,
        ptrs_w: torch.Tensor,
        ptrs_s: torch.Tensor,
        num_experts: int,
        k: int,
        n: int,
        group_size: int,
    ) -> None:
        return None


def nvfp4_moe_indexed_dense_stage_sm70_out(
    out: torch.Tensor,
    input: torch.Tensor,
    input_row_indices: torch.Tensor,
    expert_offsets: torch.Tensor,
    dense_expert_ids: torch.Tensor,
    ptrs_w: torch.Tensor,
    ptrs_s: torch.Tensor,
    num_experts: int,
    k: int,
    n: int,
    group_size: int,
) -> None:
    _op("nvfp4_moe_indexed_dense_stage_sm70_out")(
        out,
        input,
        input_row_indices,
        expert_offsets,
        dense_expert_ids,
        ptrs_w,
        ptrs_s,
        num_experts,
        k,
        n,
        group_size,
    )


if hasattr(torch.ops._C, "nvfp4_moe_indexed_dense_stage_sm70_out"):

    @register_fake("_C::nvfp4_moe_indexed_dense_stage_sm70_out")
    def _nvfp4_moe_indexed_dense_stage_sm70_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        input_row_indices: torch.Tensor,
        expert_offsets: torch.Tensor,
        dense_expert_ids: torch.Tensor,
        ptrs_w: torch.Tensor,
        ptrs_s: torch.Tensor,
        num_experts: int,
        k: int,
        n: int,
        group_size: int,
    ) -> None:
        return None


def nvfp4_moe_indexed_fused_swiglu_sm70_out(
    out: torch.Tensor,
    input: torch.Tensor,
    input_row_indices: torch.Tensor,
    expert_offsets: torch.Tensor,
    dense_expert_ids: torch.Tensor,
    ptrs_w: torch.Tensor,
    ptrs_s: torch.Tensor,
    num_experts: int,
    k: int,
    n: int,
    group_size: int,
) -> None:
    _op("nvfp4_moe_indexed_fused_swiglu_sm70_out")(
        out,
        input,
        input_row_indices,
        expert_offsets,
        dense_expert_ids,
        ptrs_w,
        ptrs_s,
        num_experts,
        k,
        n,
        group_size,
    )


if hasattr(torch.ops._C, "nvfp4_moe_indexed_fused_swiglu_sm70_out"):

    @register_fake("_C::nvfp4_moe_indexed_fused_swiglu_sm70_out")
    def _nvfp4_moe_indexed_fused_swiglu_sm70_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        input_row_indices: torch.Tensor,
        expert_offsets: torch.Tensor,
        dense_expert_ids: torch.Tensor,
        ptrs_w: torch.Tensor,
        ptrs_s: torch.Tensor,
        num_experts: int,
        k: int,
        n: int,
        group_size: int,
    ) -> None:
        return None


def mxfp4_moe_single_token_prepare_w13_sm70_out(
    gate_up: torch.Tensor,
    compact_input: torch.Tensor,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    w13_ptrs_w: torch.Tensor,
    w13_ptrs_s: torch.Tensor,
    expert_offsets: torch.Tensor,
    inv_permuted_idx: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    w13_k: int,
    w13_n: int,
    group_size: int,
    hidden_logical_size: int,
) -> None:
    _op("mxfp4_moe_single_token_prepare_w13_sm70_out")(
        gate_up,
        compact_input,
        x,
        topk_ids,
        w13_ptrs_w,
        w13_ptrs_s,
        expert_offsets,
        inv_permuted_idx,
        sorted_expert_ids,
        w13_k,
        w13_n,
        group_size,
        hidden_logical_size,
    )


if hasattr(torch.ops._C, "mxfp4_moe_single_token_prepare_w13_sm70_out"):

    @register_fake("_C::mxfp4_moe_single_token_prepare_w13_sm70_out")
    def _mxfp4_moe_single_token_prepare_w13_sm70_out_fake(
        gate_up: torch.Tensor,
        compact_input: torch.Tensor,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        w13_ptrs_w: torch.Tensor,
        w13_ptrs_s: torch.Tensor,
        expert_offsets: torch.Tensor,
        inv_permuted_idx: torch.Tensor,
        sorted_expert_ids: torch.Tensor,
        w13_k: int,
        w13_n: int,
        group_size: int,
        hidden_logical_size: int,
    ) -> None:
        return None


def nvfp4_gemm_sm70_out(
    out: torch.Tensor,
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    k_ld: int,
    q_ld: int,
    gated_silu: bool = False,
) -> None:
    _op("nvfp4_gemm_sm70_out")(
        out, input, qweight, scales, group_size, k_ld, q_ld, gated_silu
    )


if hasattr(torch.ops._C, "nvfp4_gemm_sm70_out"):

    @register_fake("_C::nvfp4_gemm_sm70_out")
    def _nvfp4_gemm_sm70_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        group_size: int,
        k_ld: int,
        q_ld: int,
        gated_silu: bool,
    ) -> None:
        return None


def nvfp4_gemv_sm70_raw_out(
    out: torch.Tensor,
    input: torch.Tensor,
    qweight_packed: torch.Tensor,
    scales: torch.Tensor,
    partials: torch.Tensor,
    group_size: int,
    split_k: int,
) -> None:
    _op("nvfp4_gemv_sm70_raw_out")(
        out, input, qweight_packed, scales, partials, group_size, split_k
    )


if hasattr(torch.ops._C, "nvfp4_gemv_sm70_raw_out"):

    @register_fake("_C::nvfp4_gemv_sm70_raw_out")
    def _nvfp4_gemv_sm70_raw_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        qweight_packed: torch.Tensor,
        scales: torch.Tensor,
        partials: torch.Tensor,
        group_size: int,
        split_k: int,
    ) -> None:
        return None


def nvfp4_gemv_sm70_warp_out(
    out: torch.Tensor,
    input: torch.Tensor,
    qweight_packed: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
) -> None:
    _op("nvfp4_gemv_sm70_warp_out")(out, input, qweight_packed, scales, group_size)


if hasattr(torch.ops._C, "nvfp4_gemv_sm70_warp_out"):

    @register_fake("_C::nvfp4_gemv_sm70_warp_out")
    def _nvfp4_gemv_sm70_warp_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        qweight_packed: torch.Tensor,
        scales: torch.Tensor,
        group_size: int,
    ) -> None:
        return None


def nvfp4_gemv_sm70_h2_out(
    out: torch.Tensor,
    input: torch.Tensor,
    qweight_packed: torch.Tensor,
    scales: torch.Tensor,
    partials: torch.Tensor,
    group_size: int,
    split_k: int,
) -> None:
    _op("nvfp4_gemv_sm70_h2_out")(
        out, input, qweight_packed, scales, partials, group_size, split_k
    )


if hasattr(torch.ops._C, "nvfp4_gemv_sm70_h2_out"):

    @register_fake("_C::nvfp4_gemv_sm70_h2_out")
    def _nvfp4_gemv_sm70_h2_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        qweight_packed: torch.Tensor,
        scales: torch.Tensor,
        partials: torch.Tensor,
        group_size: int,
        split_k: int,
    ) -> None:
        return None


def fp8_gemm_sm70_out_auto(
    out: torch.Tensor,
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
) -> None:
    _op("fp8_gemm_sm70_out_auto")(out, input, qweight, scales)


def fp8_gemm_sm70_out_meta(
    out: torch.Tensor,
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    meta: torch.Tensor,
    gated_silu: bool = False,
) -> None:
    _op("fp8_gemm_sm70_out_meta")(out, input, qweight, scales, meta, gated_silu)


def sm70_f16_gemm(input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return _op("sm70_f16_gemm")(input, weight)


if hasattr(torch.ops._C, "sm70_f16_gemm"):

    @register_fake("_C::sm70_f16_gemm")
    def _sm70_f16_gemm_fake(
        input: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        return torch.empty(
            (input.size(0), weight.size(0)),
            dtype=input.dtype,
            device=input.device,
        )


def sm70_f16_gemm_out(
    out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    k_ld: int,
    gated_silu: bool = False,
) -> None:
    _op("sm70_f16_gemm_out")(out, input, weight, k_ld, gated_silu)


if hasattr(torch.ops._C, "sm70_f16_gemm_out"):

    @register_fake("_C::sm70_f16_gemm_out")
    def _sm70_f16_gemm_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        weight: torch.Tensor,
        k_ld: int,
        gated_silu: bool,
    ) -> None:
        return None


def sm70_glm_mhc_pre_norm_out(
    gemm_mul: torch.Tensor,
    gemm_sqrsum: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    comb_mix: torch.Tensor,
    layer_input: torch.Tensor,
    norm_weight: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult: float,
    sinkhorn_repeat: int,
    norm_eps: float,
) -> None:
    _op("sm70_glm_mhc_pre_norm_out")(
        gemm_mul,
        gemm_sqrsum,
        hc_scale,
        hc_base,
        residual,
        post_mix,
        comb_mix,
        layer_input,
        norm_weight,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult,
        sinkhorn_repeat,
        norm_eps,
    )


if hasattr(torch.ops._C, "sm70_glm_mhc_pre_norm_out"):

    @register_fake("_C::sm70_glm_mhc_pre_norm_out")
    def _sm70_glm_mhc_pre_norm_out_fake(
        gemm_mul: torch.Tensor,
        gemm_sqrsum: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        residual: torch.Tensor,
        post_mix: torch.Tensor,
        comb_mix: torch.Tensor,
        layer_input: torch.Tensor,
        norm_weight: torch.Tensor,
        rms_eps: float,
        hc_pre_eps: float,
        hc_sinkhorn_eps: float,
        hc_post_mult: float,
        sinkhorn_repeat: int,
        norm_eps: float,
    ) -> None:
        return None


def sm70_glm_mhc_post_dot_q8_out(
    residual_out: torch.Tensor,
    gemm_mul: torch.Tensor,
    gemm_sqrsum: torch.Tensor,
    comb_mix: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    tile_n: int,
) -> None:
    _op("sm70_glm_mhc_post_dot_q8_out")(
        residual_out,
        gemm_mul,
        gemm_sqrsum,
        comb_mix,
        residual,
        post_mix,
        x,
        weight,
        tile_n,
    )


if hasattr(torch.ops._C, "sm70_glm_mhc_post_dot_q8_out"):

    @register_fake("_C::sm70_glm_mhc_post_dot_q8_out")
    def _sm70_glm_mhc_post_dot_q8_out_fake(
        residual_out: torch.Tensor,
        gemm_mul: torch.Tensor,
        gemm_sqrsum: torch.Tensor,
        comb_mix: torch.Tensor,
        residual: torch.Tensor,
        post_mix: torch.Tensor,
        x: torch.Tensor,
        weight: torch.Tensor,
        tile_n: int,
    ) -> None:
        return None


def sm70_f16_indexed_rerank_out(
    out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    candidate_ids: torch.Tensor,
    selected_raw: torch.Tensor,
    selected_packed: torch.Tensor,
    expanded: torch.Tensor,
    partials: torch.Tensor,
    barriers: torch.Tensor,
    cta_n: int,
    split_k: int,
) -> None:
    _op("sm70_f16_indexed_rerank_out")(
        out,
        input,
        weight,
        candidate_ids,
        selected_raw,
        selected_packed,
        expanded,
        partials,
        barriers,
        cta_n,
        split_k,
    )


if hasattr(torch.ops._C, "sm70_f16_indexed_rerank_out"):

    @register_fake("_C::sm70_f16_indexed_rerank_out")
    def _sm70_f16_indexed_rerank_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        weight: torch.Tensor,
        candidate_ids: torch.Tensor,
        selected_raw: torch.Tensor,
        selected_packed: torch.Tensor,
        expanded: torch.Tensor,
        partials: torch.Tensor,
        barriers: torch.Tensor,
        cta_n: int,
        split_k: int,
    ) -> None:
        return None


def sm70_glm_kda_fg_b_out(
    f_out: torch.Tensor,
    g_out: torch.Tensor,
    f_input: torch.Tensor,
    g_input: torch.Tensor,
    f_weight: torch.Tensor,
    g_weight: torch.Tensor,
) -> None:
    _op("sm70_glm_kda_fg_b_out")(f_out, g_out, f_input, g_input, f_weight, g_weight)


if hasattr(torch.ops._C, "sm70_glm_kda_fg_b_out"):

    @register_fake("_C::sm70_glm_kda_fg_b_out")
    def _sm70_glm_kda_fg_b_out_fake(
        f_out: torch.Tensor,
        g_out: torch.Tensor,
        f_input: torch.Tensor,
        g_input: torch.Tensor,
        f_weight: torch.Tensor,
        g_weight: torch.Tensor,
    ) -> None:
        return None


def sm70_glm53_fp16_gemv_out(
    output: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
) -> None:
    _op("sm70_glm53_fp16_gemv_out")(output, input, weight)


if hasattr(torch.ops._C, "sm70_glm53_fp16_gemv_out"):

    @register_fake("_C::sm70_glm53_fp16_gemv_out")
    def _sm70_glm53_fp16_gemv_out_fake(
        output: torch.Tensor,
        input: torch.Tensor,
        weight: torch.Tensor,
    ) -> None:
        return None


def sm70_glm53_moe_permute_q8_out(
    input: torch.Tensor,
    topk_ids: torch.Tensor,
    permuted_input: torch.Tensor,
    sorted_row_idx: torch.Tensor,
    inv_permuted_idx: torch.Tensor,
    compact_offsets: torch.Tensor,
    active_expert_ids: torch.Tensor,
) -> None:
    _op("sm70_glm53_moe_permute_q8_out")(
        input,
        topk_ids,
        permuted_input,
        sorted_row_idx,
        inv_permuted_idx,
        compact_offsets,
        active_expert_ids,
    )


if hasattr(torch.ops._C, "sm70_glm53_moe_permute_q8_out"):

    @register_fake("_C::sm70_glm53_moe_permute_q8_out")
    def _sm70_glm53_moe_permute_q8_out_fake(
        input: torch.Tensor,
        topk_ids: torch.Tensor,
        permuted_input: torch.Tensor,
        sorted_row_idx: torch.Tensor,
        inv_permuted_idx: torch.Tensor,
        compact_offsets: torch.Tensor,
        active_expert_ids: torch.Tensor,
    ) -> None:
        return None


def sm70_f16_indexed_rerank_packed_out(
    out: torch.Tensor,
    input: torch.Tensor,
    packed_weight: torch.Tensor,
    candidate_ids: torch.Tensor,
    selected_packed: torch.Tensor,
    expanded: torch.Tensor,
    partials: torch.Tensor,
    barriers: torch.Tensor,
    cta_n: int,
    split_k: int,
) -> None:
    _op("sm70_f16_indexed_rerank_packed_out")(
        out,
        input,
        packed_weight,
        candidate_ids,
        selected_packed,
        expanded,
        partials,
        barriers,
        cta_n,
        split_k,
    )


if hasattr(torch.ops._C, "sm70_f16_indexed_rerank_packed_out"):

    @register_fake("_C::sm70_f16_indexed_rerank_packed_out")
    def _sm70_f16_indexed_rerank_packed_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        packed_weight: torch.Tensor,
        candidate_ids: torch.Tensor,
        selected_packed: torch.Tensor,
        expanded: torch.Tensor,
        partials: torch.Tensor,
        barriers: torch.Tensor,
        cta_n: int,
        split_k: int,
    ) -> None:
        return None


def sm70_f16_rerank_keys_out(
    keys: torch.Tensor,
    logits: torch.Tensor,
    candidate_ids: torch.Tensor,
) -> None:
    _op("sm70_f16_rerank_keys_out")(keys, logits, candidate_ids)


if hasattr(torch.ops._C, "sm70_f16_rerank_keys_out"):

    @register_fake("_C::sm70_f16_rerank_keys_out")
    def _sm70_f16_rerank_keys_out_fake(
        keys: torch.Tensor,
        logits: torch.Tensor,
        candidate_ids: torch.Tensor,
    ) -> None:
        return None


def sm70_f16_rerank_topk_out(
    values_out: torch.Tensor,
    ids_out: torch.Tensor,
    logits: torch.Tensor,
    candidate_ids: torch.Tensor,
    vocab_start_index: int,
) -> None:
    _op("sm70_f16_rerank_topk_out")(
        values_out,
        ids_out,
        logits,
        candidate_ids,
        vocab_start_index,
    )


if hasattr(torch.ops._C, "sm70_f16_rerank_topk_out"):

    @register_fake("_C::sm70_f16_rerank_topk_out")
    def _sm70_f16_rerank_topk_out_fake(
        values_out: torch.Tensor,
        ids_out: torch.Tensor,
        logits: torch.Tensor,
        candidate_ids: torch.Tensor,
        vocab_start_index: int,
    ) -> None:
        return None


def sm70_f16_lm_head_top1_out(
    values_out: torch.Tensor,
    indices_out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    k_ld: int,
    vocab_start_index: int,
    num_vocab_padding: int,
) -> None:
    _op("sm70_f16_lm_head_top1_out")(
        values_out,
        indices_out,
        input,
        weight,
        k_ld,
        vocab_start_index,
        num_vocab_padding,
    )


if hasattr(torch.ops._C, "sm70_f16_lm_head_top1_out"):

    @register_fake("_C::sm70_f16_lm_head_top1_out")
    def _sm70_f16_lm_head_top1_out_fake(
        values_out: torch.Tensor,
        indices_out: torch.Tensor,
        input: torch.Tensor,
        weight: torch.Tensor,
        k_ld: int,
        vocab_start_index: int,
        num_vocab_padding: int,
    ) -> None:
        return None


def sm70_f16_lm_head_top1_tc_out(
    values_out: torch.Tensor,
    indices_out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    k_ld: int,
    vocab_start_index: int,
    num_vocab_padding: int,
) -> None:
    _op("sm70_f16_lm_head_top1_tc_out")(
        values_out,
        indices_out,
        input,
        weight,
        k_ld,
        vocab_start_index,
        num_vocab_padding,
    )


if hasattr(torch.ops._C, "sm70_f16_lm_head_top1_tc_out"):

    @register_fake("_C::sm70_f16_lm_head_top1_tc_out")
    def _sm70_f16_lm_head_top1_tc_out_fake(
        values_out: torch.Tensor,
        indices_out: torch.Tensor,
        input: torch.Tensor,
        weight: torch.Tensor,
        k_ld: int,
        vocab_start_index: int,
        num_vocab_padding: int,
    ) -> None:
        return None


def sm70_f16_lm_head_top20_tc_out(
    values_out: torch.Tensor,
    indices_out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    k_ld: int,
    vocab_start_index: int,
    num_vocab_padding: int,
) -> None:
    _op("sm70_f16_lm_head_top20_tc_out")(
        values_out,
        indices_out,
        input,
        weight,
        k_ld,
        vocab_start_index,
        num_vocab_padding,
    )


if hasattr(torch.ops._C, "sm70_f16_lm_head_top20_tc_out"):

    @register_fake("_C::sm70_f16_lm_head_top20_tc_out")
    def _sm70_f16_lm_head_top20_tc_out_fake(
        values_out: torch.Tensor,
        indices_out: torch.Tensor,
        input: torch.Tensor,
        weight: torch.Tensor,
        k_ld: int,
        vocab_start_index: int,
        num_vocab_padding: int,
    ) -> None:
        return None


def sm70_merge_tail_top20_pack_out(
    pairs_out: torch.Tensor,
    base_values: torch.Tensor,
    base_indices: torch.Tensor,
    base_token_id_map: torch.Tensor,
    tail_logits: torch.Tensor,
    tail_token_ids: torch.Tensor,
    tail_row_start: int,
) -> None:
    _op("sm70_merge_tail_top20_pack_out")(
        pairs_out,
        base_values,
        base_indices,
        base_token_id_map,
        tail_logits,
        tail_token_ids,
        tail_row_start,
    )


if hasattr(torch.ops._C, "sm70_merge_tail_top20_pack_out"):

    @register_fake("_C::sm70_merge_tail_top20_pack_out")
    def _sm70_merge_tail_top20_pack_out_fake(
        pairs_out: torch.Tensor,
        base_values: torch.Tensor,
        base_indices: torch.Tensor,
        base_token_id_map: torch.Tensor,
        tail_logits: torch.Tensor,
        tail_token_ids: torch.Tensor,
        tail_row_start: int,
    ) -> None:
        return None


def sm70_sample_packed_top20_out(
    sampled_token_out: torch.Tensor,
    sparse_ids_out: torch.Tensor,
    sparse_probs_out: torch.Tensor,
    gathered_pairs: torch.Tensor,
    exponential: torch.Tensor,
    top_p: float,
) -> None:
    _op("sm70_sample_packed_top20_out")(
        sampled_token_out,
        sparse_ids_out,
        sparse_probs_out,
        gathered_pairs,
        exponential,
        top_p,
    )


if hasattr(torch.ops._C, "sm70_sample_packed_top20_out"):

    @register_fake("_C::sm70_sample_packed_top20_out")
    def _sm70_sample_packed_top20_out_fake(
        sampled_token_out: torch.Tensor,
        sparse_ids_out: torch.Tensor,
        sparse_probs_out: torch.Tensor,
        gathered_pairs: torch.Tensor,
        exponential: torch.Tensor,
        top_p: float,
    ) -> None:
        return None


def sm70_sample_sorted_top20_philox_out(
    sampled_token_out: torch.Tensor,
    sparse_ids_out: torch.Tensor,
    sparse_probs_out: torch.Tensor,
    top_values: torch.Tensor,
    top_indices: torch.Tensor,
    generator: torch.Generator | None,
    vocab_size: int,
    top_p: float,
) -> None:
    _op("sm70_sample_sorted_top20_philox_out")(
        sampled_token_out,
        sparse_ids_out,
        sparse_probs_out,
        top_values,
        top_indices,
        generator,
        vocab_size,
        top_p,
    )


if hasattr(torch.ops._C, "sm70_sample_sorted_top20_philox_out"):

    @register_fake("_C::sm70_sample_sorted_top20_philox_out")
    def _sm70_sample_sorted_top20_philox_out_fake(
        sampled_token_out: torch.Tensor,
        sparse_ids_out: torch.Tensor,
        sparse_probs_out: torch.Tensor,
        top_values: torch.Tensor,
        top_indices: torch.Tensor,
        generator: torch.Generator | None,
        vocab_size: int,
        top_p: float,
    ) -> None:
        return None


def sm70_sample_sorted_top20_philox_token_out(
    sampled_token_out: torch.Tensor,
    top_values: torch.Tensor,
    top_indices: torch.Tensor,
    generator: torch.Generator | None,
    vocab_size: int,
    top_p: float,
) -> None:
    _op("sm70_sample_sorted_top20_philox_token_out")(
        sampled_token_out,
        top_values,
        top_indices,
        generator,
        vocab_size,
        top_p,
    )


if hasattr(torch.ops._C, "sm70_sample_sorted_top20_philox_token_out"):

    @register_fake("_C::sm70_sample_sorted_top20_philox_token_out")
    def _sm70_sample_sorted_top20_philox_token_out_fake(
        sampled_token_out: torch.Tensor,
        top_values: torch.Tensor,
        top_indices: torch.Tensor,
        generator: torch.Generator | None,
        vocab_size: int,
        top_p: float,
    ) -> None:
        return None


def sm70_sample_chunked_top20_philox_token_out(
    sampled_token_out: torch.Tensor,
    global_values: torch.Tensor,
    local_indices: torch.Tensor,
    global_positions: torch.Tensor,
    generator: torch.Generator | None,
    vocab_size: int,
    top_p: float,
    chunk_size: int,
) -> None:
    _op("sm70_sample_chunked_top20_philox_token_out")(
        sampled_token_out,
        global_values,
        local_indices,
        global_positions,
        generator,
        vocab_size,
        top_p,
        chunk_size,
    )


if hasattr(torch.ops._C, "sm70_sample_chunked_top20_philox_token_out"):

    @register_fake("_C::sm70_sample_chunked_top20_philox_token_out")
    def _sm70_sample_chunked_top20_philox_token_out_fake(
        sampled_token_out: torch.Tensor,
        global_values: torch.Tensor,
        local_indices: torch.Tensor,
        global_positions: torch.Tensor,
        generator: torch.Generator | None,
        vocab_size: int,
        top_p: float,
        chunk_size: int,
    ) -> None:
        return None


def sm70_dynamic_draft_vocab_update_tail_out(
    lru_token_ids: torch.Tensor,
    local_tail_token_ids: torch.Tensor,
    source_row_indices: torch.Tensor,
    observed_output_ids: torch.Tensor,
    target_candidate_ids: torch.Tensor,
    base_token_mask: torch.Tensor,
    full_vocab_size: int,
    local_shard_start: int,
    local_shard_end: int,
) -> None:
    _op("sm70_dynamic_draft_vocab_update_tail_out")(
        lru_token_ids,
        local_tail_token_ids,
        source_row_indices,
        observed_output_ids,
        target_candidate_ids,
        base_token_mask,
        full_vocab_size,
        local_shard_start,
        local_shard_end,
    )


if hasattr(torch.ops._C, "sm70_dynamic_draft_vocab_update_tail_out"):

    @register_fake("_C::sm70_dynamic_draft_vocab_update_tail_out")
    def _sm70_dynamic_draft_vocab_update_tail_out_fake(
        lru_token_ids: torch.Tensor,
        local_tail_token_ids: torch.Tensor,
        source_row_indices: torch.Tensor,
        observed_output_ids: torch.Tensor,
        target_candidate_ids: torch.Tensor,
        base_token_mask: torch.Tensor,
        full_vocab_size: int,
        local_shard_start: int,
        local_shard_end: int,
    ) -> None:
        return None


def sm70_dynamic_draft_vocab_refresh_tail_weight_out(
    local_tail_weight: torch.Tensor,
    source_weight: torch.Tensor,
    source_row_indices: torch.Tensor,
) -> None:
    _op("sm70_dynamic_draft_vocab_refresh_tail_weight_out")(
        local_tail_weight,
        source_weight,
        source_row_indices,
    )


if hasattr(torch.ops._C, "sm70_dynamic_draft_vocab_refresh_tail_weight_out"):

    @register_fake("_C::sm70_dynamic_draft_vocab_refresh_tail_weight_out")
    def _sm70_dynamic_draft_vocab_refresh_tail_weight_out_fake(
        local_tail_weight: torch.Tensor,
        source_weight: torch.Tensor,
        source_row_indices: torch.Tensor,
    ) -> None:
        return None


def sm70_f16_gate_mul_out(
    out: torch.Tensor,
    input: torch.Tensor,
    gate_weight: torch.Tensor,
) -> None:
    _op("sm70_f16_gate_mul_out")(out, input, gate_weight)


if hasattr(torch.ops._C, "sm70_f16_gate_mul_out"):

    @register_fake("_C::sm70_f16_gate_mul_out")
    def _sm70_f16_gate_mul_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        gate_weight: torch.Tensor,
    ) -> None:
        return None


def sm70_gemm_import_cache(device_hint: torch.Tensor, path: str) -> int:
    return _op("sm70_gemm_import_cache")(device_hint, path)


def sm70_gemm_export_cache(device_hint: torch.Tensor, path: str) -> int:
    return _op("sm70_gemm_export_cache")(device_hint, path)


def awq_moe_build_strided_ptrs(
    tm_weights: torch.Tensor,
    tm_scales: torch.Tensor,
    k_ld: int,
    q_ld: int,
    num_experts: int,
) -> list[torch.Tensor]:
    return _op("awq_moe_build_strided_ptrs")(
        tm_weights, tm_scales, k_ld, q_ld, num_experts
    )


if hasattr(torch.ops._C, "awq_moe_build_strided_ptrs"):

    @register_fake("_C::awq_moe_build_strided_ptrs")
    def _awq_moe_build_strided_ptrs_fake(
        tm_weights: torch.Tensor,
        tm_scales: torch.Tensor,
        k_ld: int,
        q_ld: int,
        num_experts: int,
    ) -> list[torch.Tensor]:
        del tm_scales, k_ld, q_ld
        buf = num_experts * 16
        opts = dict(dtype=torch.uint8, device=tm_weights.device)
        return [torch.empty(buf, **opts), torch.empty(buf, **opts)]


def awq_moe_gemm_sm70_out(
    out: torch.Tensor,
    sorted_input: torch.Tensor,
    expert_offsets: torch.Tensor,
    strided_ptrs_w: torch.Tensor,
    strided_ptrs_s: torch.Tensor,
    num_experts: int,
    k: int,
    n: int,
    group_size: int,
    gated_silu: bool = False,
) -> None:
    _op("awq_moe_gemm_sm70_out")(
        out,
        sorted_input,
        expert_offsets,
        strided_ptrs_w,
        strided_ptrs_s,
        num_experts,
        k,
        n,
        group_size,
        gated_silu,
    )


def awq_moe_gemm_sm70_per_expert_dispatch_out(
    out: torch.Tensor,
    sorted_input: torch.Tensor,
    expert_offsets: torch.Tensor,
    strided_ptrs_w: torch.Tensor,
    strided_ptrs_s: torch.Tensor,
    num_experts: int,
    k: int,
    n: int,
    group_size: int,
    gated_silu: bool = False,
) -> None:
    _op("awq_moe_gemm_sm70_per_expert_dispatch_out")(
        out,
        sorted_input,
        expert_offsets,
        strided_ptrs_w,
        strided_ptrs_s,
        num_experts,
        k,
        n,
        group_size,
        gated_silu,
    )


if hasattr(torch.ops._C, "awq_moe_gemm_sm70_out"):

    @register_fake("_C::awq_moe_gemm_sm70_out")
    def _awq_moe_gemm_sm70_out_fake(
        out: torch.Tensor,
        sorted_input: torch.Tensor,
        expert_offsets: torch.Tensor,
        strided_ptrs_w: torch.Tensor,
        strided_ptrs_s: torch.Tensor,
        num_experts: int,
        k: int,
        n: int,
        group_size: int,
        gated_silu: bool,
    ) -> None:
        return None


if hasattr(torch.ops._C, "awq_moe_gemm_sm70_per_expert_dispatch_out"):

    @register_fake("_C::awq_moe_gemm_sm70_per_expert_dispatch_out")
    def _awq_moe_gemm_sm70_per_expert_dispatch_out_fake(
        out: torch.Tensor,
        sorted_input: torch.Tensor,
        expert_offsets: torch.Tensor,
        strided_ptrs_w: torch.Tensor,
        strided_ptrs_s: torch.Tensor,
        num_experts: int,
        k: int,
        n: int,
        group_size: int,
        gated_silu: bool,
    ) -> None:
        return None


def awq_moe_dense_stage_sm70_out(
    out: torch.Tensor,
    input: torch.Tensor,
    expert_offsets: torch.Tensor,
    dense_expert_ids: torch.Tensor,
    ptrs_w: torch.Tensor,
    ptrs_s: torch.Tensor,
    num_experts: int,
    k: int,
    n: int,
    group_size: int,
) -> None:
    _op("awq_moe_dense_stage_sm70_out")(
        out,
        input,
        expert_offsets,
        dense_expert_ids,
        ptrs_w,
        ptrs_s,
        num_experts,
        k,
        n,
        group_size,
    )


if hasattr(torch.ops._C, "awq_moe_dense_stage_sm70_out"):

    @register_fake("_C::awq_moe_dense_stage_sm70_out")
    def _awq_moe_dense_stage_sm70_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        expert_offsets: torch.Tensor,
        dense_expert_ids: torch.Tensor,
        ptrs_w: torch.Tensor,
        ptrs_s: torch.Tensor,
        num_experts: int,
        k: int,
        n: int,
        group_size: int,
    ) -> None:
        return None


def awq_moe_indexed_dense_w13_sm70_out(
    out: torch.Tensor,
    input: torch.Tensor,
    input_row_indices: torch.Tensor,
    expert_offsets: torch.Tensor,
    dense_expert_ids: torch.Tensor,
    ptrs_w: torch.Tensor,
    ptrs_s: torch.Tensor,
    num_experts: int,
    k: int,
    n: int,
    group_size: int,
) -> None:
    _op("awq_moe_indexed_dense_w13_sm70_out")(
        out,
        input,
        input_row_indices,
        expert_offsets,
        dense_expert_ids,
        ptrs_w,
        ptrs_s,
        num_experts,
        k,
        n,
        group_size,
    )


if hasattr(torch.ops._C, "awq_moe_indexed_dense_w13_sm70_out"):

    @register_fake("_C::awq_moe_indexed_dense_w13_sm70_out")
    def _awq_moe_indexed_dense_w13_sm70_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        input_row_indices: torch.Tensor,
        expert_offsets: torch.Tensor,
        dense_expert_ids: torch.Tensor,
        ptrs_w: torch.Tensor,
        ptrs_s: torch.Tensor,
        num_experts: int,
        k: int,
        n: int,
        group_size: int,
    ) -> None:
        return None


def awq_moe_active_dense_stage_sm70_out(
    out: torch.Tensor,
    input: torch.Tensor,
    permuted_experts_id: torch.Tensor,
    active_expert_offsets: torch.Tensor,
    active_expert_ids: torch.Tensor,
    ptrs_w: torch.Tensor,
    ptrs_s: torch.Tensor,
    total_slots: int,
    k: int,
    n: int,
    group_size: int,
) -> None:
    _op("awq_moe_active_dense_stage_sm70_out")(
        out,
        input,
        permuted_experts_id,
        active_expert_offsets,
        active_expert_ids,
        ptrs_w,
        ptrs_s,
        total_slots,
        k,
        n,
        group_size,
    )


if hasattr(torch.ops._C, "awq_moe_active_dense_stage_sm70_out"):

    @register_fake("_C::awq_moe_active_dense_stage_sm70_out")
    def _awq_moe_active_dense_stage_sm70_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        permuted_experts_id: torch.Tensor,
        active_expert_offsets: torch.Tensor,
        active_expert_ids: torch.Tensor,
        ptrs_w: torch.Tensor,
        ptrs_s: torch.Tensor,
        total_slots: int,
        k: int,
        n: int,
        group_size: int,
    ) -> None:
        return None


def awq_moe_single_token_dense_stage_sm70_out(
    out: torch.Tensor,
    input: torch.Tensor,
    expert_offsets: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    ptrs_w: torch.Tensor,
    ptrs_s: torch.Tensor,
    top_k: int,
    k: int,
    n: int,
    group_size: int,
) -> None:
    _op("awq_moe_single_token_dense_stage_sm70_out")(
        out,
        input,
        expert_offsets,
        sorted_expert_ids,
        ptrs_w,
        ptrs_s,
        top_k,
        k,
        n,
        group_size,
    )


if hasattr(torch.ops._C, "awq_moe_single_token_dense_stage_sm70_out"):

    @register_fake("_C::awq_moe_single_token_dense_stage_sm70_out")
    def _awq_moe_single_token_dense_stage_sm70_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        expert_offsets: torch.Tensor,
        sorted_expert_ids: torch.Tensor,
        ptrs_w: torch.Tensor,
        ptrs_s: torch.Tensor,
        top_k: int,
        k: int,
        n: int,
        group_size: int,
    ) -> None:
        return None


def awq_moe_single_token_indexed_dense_stage_sm70_out(
    out: torch.Tensor,
    input: torch.Tensor,
    expert_offsets: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    ptrs_w: torch.Tensor,
    ptrs_s: torch.Tensor,
    top_k: int,
    k: int,
    n: int,
    group_size: int,
) -> None:
    _op("awq_moe_single_token_indexed_dense_stage_sm70_out")(
        out,
        input,
        expert_offsets,
        sorted_expert_ids,
        ptrs_w,
        ptrs_s,
        top_k,
        k,
        n,
        group_size,
    )


if hasattr(torch.ops._C, "awq_moe_single_token_indexed_dense_stage_sm70_out"):

    @register_fake("_C::awq_moe_single_token_indexed_dense_stage_sm70_out")
    def _awq_moe_single_token_indexed_dense_stage_sm70_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        expert_offsets: torch.Tensor,
        sorted_expert_ids: torch.Tensor,
        ptrs_w: torch.Tensor,
        ptrs_s: torch.Tensor,
        top_k: int,
        k: int,
        n: int,
        group_size: int,
    ) -> None:
        return None


def awq_moe_single_token_dense_w13_sm70_out(
    gate_up: torch.Tensor,
    compact_input: torch.Tensor,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    w13_ptrs_w: torch.Tensor,
    w13_ptrs_s: torch.Tensor,
    expert_offsets: torch.Tensor,
    expert_offsets64: torch.Tensor,
    inv_permuted_idx: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    w13_k: int,
    w13_n: int,
    group_size: int,
    hidden_logical_size: int,
) -> None:
    _op("awq_moe_single_token_dense_w13_sm70_out")(
        gate_up,
        compact_input,
        x,
        topk_ids,
        w13_ptrs_w,
        w13_ptrs_s,
        expert_offsets,
        expert_offsets64,
        inv_permuted_idx,
        sorted_expert_ids,
        w13_k,
        w13_n,
        group_size,
        hidden_logical_size,
    )


if hasattr(torch.ops._C, "awq_moe_single_token_dense_w13_sm70_out"):

    @register_fake("_C::awq_moe_single_token_dense_w13_sm70_out")
    def _awq_moe_single_token_dense_w13_sm70_out_fake(
        gate_up: torch.Tensor,
        compact_input: torch.Tensor,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        w13_ptrs_w: torch.Tensor,
        w13_ptrs_s: torch.Tensor,
        expert_offsets: torch.Tensor,
        expert_offsets64: torch.Tensor,
        inv_permuted_idx: torch.Tensor,
        sorted_expert_ids: torch.Tensor,
        w13_k: int,
        w13_n: int,
        group_size: int,
        hidden_logical_size: int,
    ) -> None:
        return None


def awq_moe_single_token_indexed_dense_w13_sm70_out(
    gate_up: torch.Tensor,
    compact_input: torch.Tensor,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    w13_ptrs_w: torch.Tensor,
    w13_ptrs_s: torch.Tensor,
    expert_offsets: torch.Tensor,
    expert_offsets64: torch.Tensor,
    inv_permuted_idx: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    w13_k: int,
    w13_n: int,
    group_size: int,
    hidden_logical_size: int,
) -> None:
    _op("awq_moe_single_token_indexed_dense_w13_sm70_out")(
        gate_up,
        compact_input,
        x,
        topk_ids,
        w13_ptrs_w,
        w13_ptrs_s,
        expert_offsets,
        expert_offsets64,
        inv_permuted_idx,
        sorted_expert_ids,
        w13_k,
        w13_n,
        group_size,
        hidden_logical_size,
    )


if hasattr(torch.ops._C, "awq_moe_single_token_indexed_dense_w13_sm70_out"):

    @register_fake("_C::awq_moe_single_token_indexed_dense_w13_sm70_out")
    def _awq_moe_single_token_indexed_dense_w13_sm70_out_fake(
        gate_up: torch.Tensor,
        compact_input: torch.Tensor,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        w13_ptrs_w: torch.Tensor,
        w13_ptrs_s: torch.Tensor,
        expert_offsets: torch.Tensor,
        expert_offsets64: torch.Tensor,
        inv_permuted_idx: torch.Tensor,
        sorted_expert_ids: torch.Tensor,
        w13_k: int,
        w13_n: int,
        group_size: int,
        hidden_logical_size: int,
    ) -> None:
        return None


def awq_moe_single_token_compact_dense_w13_sm70_out(
    gate_up: torch.Tensor,
    compact_input: torch.Tensor,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    w13_ptrs_w: torch.Tensor,
    w13_ptrs_s: torch.Tensor,
    compact_w13_ptrs_w: torch.Tensor,
    compact_w13_ptrs_s: torch.Tensor,
    expert_offsets: torch.Tensor,
    expert_offsets64: torch.Tensor,
    inv_permuted_idx: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    w13_k: int,
    w13_n: int,
    group_size: int,
    hidden_logical_size: int,
) -> None:
    _op("awq_moe_single_token_compact_dense_w13_sm70_out")(
        gate_up,
        compact_input,
        x,
        topk_ids,
        w13_ptrs_w,
        w13_ptrs_s,
        compact_w13_ptrs_w,
        compact_w13_ptrs_s,
        expert_offsets,
        expert_offsets64,
        inv_permuted_idx,
        sorted_expert_ids,
        w13_k,
        w13_n,
        group_size,
        hidden_logical_size,
    )


if hasattr(torch.ops._C, "awq_moe_single_token_compact_dense_w13_sm70_out"):

    @register_fake("_C::awq_moe_single_token_compact_dense_w13_sm70_out")
    def _awq_moe_single_token_compact_dense_w13_sm70_out_fake(
        gate_up: torch.Tensor,
        compact_input: torch.Tensor,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        w13_ptrs_w: torch.Tensor,
        w13_ptrs_s: torch.Tensor,
        compact_w13_ptrs_w: torch.Tensor,
        compact_w13_ptrs_s: torch.Tensor,
        expert_offsets: torch.Tensor,
        expert_offsets64: torch.Tensor,
        inv_permuted_idx: torch.Tensor,
        sorted_expert_ids: torch.Tensor,
        w13_k: int,
        w13_n: int,
        group_size: int,
        hidden_logical_size: int,
    ) -> None:
        return None


def awq_moe_single_token_exact_layout_prepare(
    topk_ids: torch.Tensor,
    x: torch.Tensor,
    compact_input: torch.Tensor,
    expert_offsets: torch.Tensor,
    expert_offsets64: torch.Tensor,
    inv_permuted_idx: torch.Tensor,
    num_experts: int,
) -> None:
    _op("awq_moe_single_token_exact_layout_prepare")(
        topk_ids,
        x,
        compact_input,
        expert_offsets,
        expert_offsets64,
        inv_permuted_idx,
        num_experts,
    )


if hasattr(torch.ops._C, "awq_moe_single_token_exact_layout_prepare"):

    @register_fake("_C::awq_moe_single_token_exact_layout_prepare")
    def _awq_moe_single_token_exact_layout_prepare_fake(
        topk_ids: torch.Tensor,
        x: torch.Tensor,
        compact_input: torch.Tensor,
        expert_offsets: torch.Tensor,
        expert_offsets64: torch.Tensor,
        inv_permuted_idx: torch.Tensor,
        num_experts: int,
    ) -> None:
        return None


def awq_moe_single_token_weighted_reduce_out(
    sorted_output: torch.Tensor,
    topk_weights: torch.Tensor,
    inv_permuted_idx: torch.Tensor,
    out: torch.Tensor,
    top_k: int,
    hidden_logical_size: int,
) -> None:
    _op("awq_moe_single_token_weighted_reduce_out")(
        sorted_output,
        topk_weights,
        inv_permuted_idx,
        out,
        top_k,
        hidden_logical_size,
    )


if hasattr(torch.ops._C, "awq_moe_single_token_weighted_reduce_out"):

    @register_fake("_C::awq_moe_single_token_weighted_reduce_out")
    def _awq_moe_single_token_weighted_reduce_out_fake(
        sorted_output: torch.Tensor,
        topk_weights: torch.Tensor,
        inv_permuted_idx: torch.Tensor,
        out: torch.Tensor,
        top_k: int,
        hidden_logical_size: int,
    ) -> None:
        return None


def awq_moe_qpn_m1_sm70_out(
    out: torch.Tensor,
    intermediate: torch.Tensor,
    input: torch.Tensor,
    w13: torch.Tensor,
    s13: torch.Tensor,
    w2: torch.Tensor,
    s2: torch.Tensor,
    ids: torch.Tensor,
    topk: torch.Tensor,
) -> None:
    _op("awq_moe_qpn_m1_sm70_out")(
        out, intermediate, input, w13, s13, w2, s2, ids, topk
    )


if hasattr(torch.ops._C, "awq_moe_qpn_m1_sm70_out"):

    @register_fake("_C::awq_moe_qpn_m1_sm70_out")
    def _awq_moe_qpn_m1_sm70_out_fake(
        out: torch.Tensor,
        intermediate: torch.Tensor,
        input: torch.Tensor,
        w13: torch.Tensor,
        s13: torch.Tensor,
        w2: torch.Tensor,
        s2: torch.Tensor,
        ids: torch.Tensor,
        topk: torch.Tensor,
    ) -> None:
        return None


def awq_moe_single_token_sm70_out(
    out: torch.Tensor,
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    src_w13_ptrs_w_rows: torch.Tensor,
    src_w13_ptrs_s_rows: torch.Tensor,
    src_w2_ptrs_w_rows: torch.Tensor,
    src_w2_ptrs_s_rows: torch.Tensor,
    compact_input: torch.Tensor,
    intermediate: torch.Tensor,
    sorted_output: torch.Tensor,
    sorted_weights: torch.Tensor,
    dst_w13_ptrs_w_rows: torch.Tensor,
    dst_w13_ptrs_s_rows: torch.Tensor,
    dst_w2_ptrs_w_rows: torch.Tensor,
    dst_w2_ptrs_s_rows: torch.Tensor,
    expert_offsets: torch.Tensor,
    inv_permuted_idx: torch.Tensor,
    w13_k: int,
    w13_n: int,
    w2_k: int,
    w2_n: int,
    group_size: int,
    hidden_logical_size: int,
) -> None:
    _op("awq_moe_single_token_sm70_out")(
        out,
        x,
        topk_weights,
        topk_ids,
        src_w13_ptrs_w_rows,
        src_w13_ptrs_s_rows,
        src_w2_ptrs_w_rows,
        src_w2_ptrs_s_rows,
        compact_input,
        intermediate,
        sorted_output,
        sorted_weights,
        dst_w13_ptrs_w_rows,
        dst_w13_ptrs_s_rows,
        dst_w2_ptrs_w_rows,
        dst_w2_ptrs_s_rows,
        expert_offsets,
        inv_permuted_idx,
        w13_k,
        w13_n,
        w2_k,
        w2_n,
        group_size,
        hidden_logical_size,
    )


if hasattr(torch.ops._C, "awq_moe_single_token_sm70_out"):

    @register_fake("_C::awq_moe_single_token_sm70_out")
    def _awq_moe_single_token_sm70_out_fake(
        out: torch.Tensor,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        src_w13_ptrs_w_rows: torch.Tensor,
        src_w13_ptrs_s_rows: torch.Tensor,
        src_w2_ptrs_w_rows: torch.Tensor,
        src_w2_ptrs_s_rows: torch.Tensor,
        compact_input: torch.Tensor,
        intermediate: torch.Tensor,
        sorted_output: torch.Tensor,
        sorted_weights: torch.Tensor,
        dst_w13_ptrs_w_rows: torch.Tensor,
        dst_w13_ptrs_s_rows: torch.Tensor,
        dst_w2_ptrs_w_rows: torch.Tensor,
        dst_w2_ptrs_s_rows: torch.Tensor,
        expert_offsets: torch.Tensor,
        inv_permuted_idx: torch.Tensor,
        w13_k: int,
        w13_n: int,
        w2_k: int,
        w2_n: int,
        group_size: int,
        hidden_logical_size: int,
    ) -> None:
        del (
            out,
            x,
            topk_weights,
            topk_ids,
            src_w13_ptrs_w_rows,
            src_w13_ptrs_s_rows,
            src_w2_ptrs_w_rows,
            src_w2_ptrs_s_rows,
            compact_input,
            intermediate,
            sorted_output,
            dst_w13_ptrs_w_rows,
            dst_w13_ptrs_s_rows,
            dst_w2_ptrs_w_rows,
            dst_w2_ptrs_s_rows,
            expert_offsets,
            inv_permuted_idx,
            w13_k,
            w13_n,
            w2_k,
            w2_n,
            group_size,
            hidden_logical_size,
        )
        return None


def fp8_moe_gemm_sm70_out(
    out: torch.Tensor,
    sorted_input: torch.Tensor,
    expert_offsets: torch.Tensor,
    strided_ptrs_w: torch.Tensor,
    strided_ptrs_s: torch.Tensor,
    num_experts: int,
    k: int,
    n: int,
    group_size: int,
    gated_silu: bool = False,
) -> None:
    _op("fp8_moe_gemm_sm70_out")(
        out,
        sorted_input,
        expert_offsets,
        strided_ptrs_w,
        strided_ptrs_s,
        num_experts,
        k,
        n,
        group_size,
        gated_silu,
    )


def fp8_moe_gemm_sm70_per_expert_dispatch_out(
    out: torch.Tensor,
    sorted_input: torch.Tensor,
    expert_offsets: torch.Tensor,
    strided_ptrs_w: torch.Tensor,
    strided_ptrs_s: torch.Tensor,
    num_experts: int,
    k: int,
    n: int,
    group_size: int,
    gated_silu: bool = False,
) -> None:
    _op("fp8_moe_gemm_sm70_per_expert_dispatch_out")(
        out,
        sorted_input,
        expert_offsets,
        strided_ptrs_w,
        strided_ptrs_s,
        num_experts,
        k,
        n,
        group_size,
        gated_silu,
    )


if hasattr(torch.ops._C, "fp8_moe_gemm_sm70_out"):

    @register_fake("_C::fp8_moe_gemm_sm70_out")
    def _fp8_moe_gemm_sm70_out_fake(
        out: torch.Tensor,
        sorted_input: torch.Tensor,
        expert_offsets: torch.Tensor,
        strided_ptrs_w: torch.Tensor,
        strided_ptrs_s: torch.Tensor,
        num_experts: int,
        k: int,
        n: int,
        group_size: int,
        gated_silu: bool,
    ) -> None:
        return None


if hasattr(torch.ops._C, "fp8_moe_gemm_sm70_per_expert_dispatch_out"):

    @register_fake("_C::fp8_moe_gemm_sm70_per_expert_dispatch_out")
    def _fp8_moe_gemm_sm70_per_expert_dispatch_out_fake(
        out: torch.Tensor,
        sorted_input: torch.Tensor,
        expert_offsets: torch.Tensor,
        strided_ptrs_w: torch.Tensor,
        strided_ptrs_s: torch.Tensor,
        num_experts: int,
        k: int,
        n: int,
        group_size: int,
        gated_silu: bool,
    ) -> None:
        return None


def fp8_moe_dense_stage_sm70_out(
    out: torch.Tensor,
    input: torch.Tensor,
    expert_offsets: torch.Tensor,
    dense_expert_ids: torch.Tensor,
    ptrs_w: torch.Tensor,
    ptrs_s: torch.Tensor,
    num_experts: int,
    k: int,
    n: int,
    group_size: int,
) -> None:
    _op("fp8_moe_dense_stage_sm70_out")(
        out,
        input,
        expert_offsets,
        dense_expert_ids,
        ptrs_w,
        ptrs_s,
        num_experts,
        k,
        n,
        group_size,
    )


if hasattr(torch.ops._C, "fp8_moe_dense_stage_sm70_out"):

    @register_fake("_C::fp8_moe_dense_stage_sm70_out")
    def _fp8_moe_dense_stage_sm70_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        expert_offsets: torch.Tensor,
        dense_expert_ids: torch.Tensor,
        ptrs_w: torch.Tensor,
        ptrs_s: torch.Tensor,
        num_experts: int,
        k: int,
        n: int,
        group_size: int,
    ) -> None:
        return None


def fp8_moe_single_token_dense_stage_sm70_out(
    out: torch.Tensor,
    input: torch.Tensor,
    expert_offsets: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    ptrs_w: torch.Tensor,
    ptrs_s: torch.Tensor,
    top_k: int,
    k: int,
    n: int,
    group_size: int,
) -> None:
    _op("fp8_moe_single_token_dense_stage_sm70_out")(
        out,
        input,
        expert_offsets,
        sorted_expert_ids,
        ptrs_w,
        ptrs_s,
        top_k,
        k,
        n,
        group_size,
    )


if hasattr(torch.ops._C, "fp8_moe_single_token_dense_stage_sm70_out"):

    @register_fake("_C::fp8_moe_single_token_dense_stage_sm70_out")
    def _fp8_moe_single_token_dense_stage_sm70_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        expert_offsets: torch.Tensor,
        sorted_expert_ids: torch.Tensor,
        ptrs_w: torch.Tensor,
        ptrs_s: torch.Tensor,
        top_k: int,
        k: int,
        n: int,
        group_size: int,
    ) -> None:
        return None


def fp8_moe_single_token_indexed_dense_stage_sm70_out(
    out: torch.Tensor,
    input: torch.Tensor,
    expert_offsets: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    ptrs_w: torch.Tensor,
    ptrs_s: torch.Tensor,
    top_k: int,
    k: int,
    n: int,
    group_size: int,
) -> None:
    _op("fp8_moe_single_token_indexed_dense_stage_sm70_out")(
        out,
        input,
        expert_offsets,
        sorted_expert_ids,
        ptrs_w,
        ptrs_s,
        top_k,
        k,
        n,
        group_size,
    )


if hasattr(torch.ops._C, "fp8_moe_single_token_indexed_dense_stage_sm70_out"):

    @register_fake("_C::fp8_moe_single_token_indexed_dense_stage_sm70_out")
    def _fp8_moe_single_token_indexed_dense_stage_sm70_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        expert_offsets: torch.Tensor,
        sorted_expert_ids: torch.Tensor,
        ptrs_w: torch.Tensor,
        ptrs_s: torch.Tensor,
        top_k: int,
        k: int,
        n: int,
        group_size: int,
    ) -> None:
        return None


def fp8_moe_single_token_dense_w13_sm70_out(
    gate_up: torch.Tensor,
    compact_input: torch.Tensor,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    w13_ptrs_w: torch.Tensor,
    w13_ptrs_s: torch.Tensor,
    expert_offsets: torch.Tensor,
    expert_offsets64: torch.Tensor,
    inv_permuted_idx: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    w13_k: int,
    w13_n: int,
    group_size: int,
    hidden_logical_size: int,
) -> None:
    _op("fp8_moe_single_token_dense_w13_sm70_out")(
        gate_up,
        compact_input,
        x,
        topk_ids,
        w13_ptrs_w,
        w13_ptrs_s,
        expert_offsets,
        expert_offsets64,
        inv_permuted_idx,
        sorted_expert_ids,
        w13_k,
        w13_n,
        group_size,
        hidden_logical_size,
    )


if hasattr(torch.ops._C, "fp8_moe_single_token_dense_w13_sm70_out"):

    @register_fake("_C::fp8_moe_single_token_dense_w13_sm70_out")
    def _fp8_moe_single_token_dense_w13_sm70_out_fake(
        gate_up: torch.Tensor,
        compact_input: torch.Tensor,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        w13_ptrs_w: torch.Tensor,
        w13_ptrs_s: torch.Tensor,
        expert_offsets: torch.Tensor,
        expert_offsets64: torch.Tensor,
        inv_permuted_idx: torch.Tensor,
        sorted_expert_ids: torch.Tensor,
        w13_k: int,
        w13_n: int,
        group_size: int,
        hidden_logical_size: int,
    ) -> None:
        return None


def fp8_moe_single_token_indexed_dense_w13_sm70_out(
    gate_up: torch.Tensor,
    compact_input: torch.Tensor,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    w13_ptrs_w: torch.Tensor,
    w13_ptrs_s: torch.Tensor,
    expert_offsets: torch.Tensor,
    expert_offsets64: torch.Tensor,
    inv_permuted_idx: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    w13_k: int,
    w13_n: int,
    group_size: int,
    hidden_logical_size: int,
) -> None:
    _op("fp8_moe_single_token_indexed_dense_w13_sm70_out")(
        gate_up,
        compact_input,
        x,
        topk_ids,
        w13_ptrs_w,
        w13_ptrs_s,
        expert_offsets,
        expert_offsets64,
        inv_permuted_idx,
        sorted_expert_ids,
        w13_k,
        w13_n,
        group_size,
        hidden_logical_size,
    )


if hasattr(torch.ops._C, "fp8_moe_single_token_indexed_dense_w13_sm70_out"):

    @register_fake("_C::fp8_moe_single_token_indexed_dense_w13_sm70_out")
    def _fp8_moe_single_token_indexed_dense_w13_sm70_out_fake(
        gate_up: torch.Tensor,
        compact_input: torch.Tensor,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        w13_ptrs_w: torch.Tensor,
        w13_ptrs_s: torch.Tensor,
        expert_offsets: torch.Tensor,
        expert_offsets64: torch.Tensor,
        inv_permuted_idx: torch.Tensor,
        sorted_expert_ids: torch.Tensor,
        w13_k: int,
        w13_n: int,
        group_size: int,
        hidden_logical_size: int,
    ) -> None:
        return None


def fp8_moe_single_token_compact_dense_w13_sm70_out(
    gate_up: torch.Tensor,
    compact_input: torch.Tensor,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    w13_ptrs_w: torch.Tensor,
    w13_ptrs_s: torch.Tensor,
    compact_w13_ptrs_w: torch.Tensor,
    compact_w13_ptrs_s: torch.Tensor,
    expert_offsets: torch.Tensor,
    expert_offsets64: torch.Tensor,
    inv_permuted_idx: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    w13_k: int,
    w13_n: int,
    group_size: int,
    hidden_logical_size: int,
) -> None:
    _op("fp8_moe_single_token_compact_dense_w13_sm70_out")(
        gate_up,
        compact_input,
        x,
        topk_ids,
        w13_ptrs_w,
        w13_ptrs_s,
        compact_w13_ptrs_w,
        compact_w13_ptrs_s,
        expert_offsets,
        expert_offsets64,
        inv_permuted_idx,
        sorted_expert_ids,
        w13_k,
        w13_n,
        group_size,
        hidden_logical_size,
    )


if hasattr(torch.ops._C, "fp8_moe_single_token_compact_dense_w13_sm70_out"):

    @register_fake("_C::fp8_moe_single_token_compact_dense_w13_sm70_out")
    def _fp8_moe_single_token_compact_dense_w13_sm70_out_fake(
        gate_up: torch.Tensor,
        compact_input: torch.Tensor,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        w13_ptrs_w: torch.Tensor,
        w13_ptrs_s: torch.Tensor,
        compact_w13_ptrs_w: torch.Tensor,
        compact_w13_ptrs_s: torch.Tensor,
        expert_offsets: torch.Tensor,
        expert_offsets64: torch.Tensor,
        inv_permuted_idx: torch.Tensor,
        sorted_expert_ids: torch.Tensor,
        w13_k: int,
        w13_n: int,
        group_size: int,
        hidden_logical_size: int,
    ) -> None:
        return None


def fp8_moe_single_token_sm70_out(
    out: torch.Tensor,
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    src_w13_ptrs_w_rows: torch.Tensor,
    src_w13_ptrs_s_rows: torch.Tensor,
    src_w2_ptrs_w_rows: torch.Tensor,
    src_w2_ptrs_s_rows: torch.Tensor,
    compact_input: torch.Tensor,
    gate_up: torch.Tensor,
    intermediate: torch.Tensor,
    sorted_output: torch.Tensor,
    sorted_weights: torch.Tensor,
    dst_w13_ptrs_w_rows: torch.Tensor,
    dst_w13_ptrs_s_rows: torch.Tensor,
    dst_w2_ptrs_w_rows: torch.Tensor,
    dst_w2_ptrs_s_rows: torch.Tensor,
    expert_offsets: torch.Tensor,
    inv_permuted_idx: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    broadcast_input_indices: torch.Tensor,
    w2_raw_weight: torch.Tensor,
    w2_raw_scale_inv: torch.Tensor,
    w13_k: int,
    w13_n: int,
    w2_k: int,
    w2_n: int,
    group_size: int,
    hidden_logical_size: int,
    fused_gated_silu: bool,
    fused_weighted_reduce: bool,
    broadcast_input: bool,
    w2_direct_reduce: bool,
    indexed_expert_ptrs: bool,
    exact_per_route: bool,
) -> None:
    _op("fp8_moe_single_token_sm70_out")(
        out,
        x,
        topk_weights,
        topk_ids,
        src_w13_ptrs_w_rows,
        src_w13_ptrs_s_rows,
        src_w2_ptrs_w_rows,
        src_w2_ptrs_s_rows,
        compact_input,
        gate_up,
        intermediate,
        sorted_output,
        sorted_weights,
        dst_w13_ptrs_w_rows,
        dst_w13_ptrs_s_rows,
        dst_w2_ptrs_w_rows,
        dst_w2_ptrs_s_rows,
        expert_offsets,
        inv_permuted_idx,
        sorted_expert_ids,
        broadcast_input_indices,
        w2_raw_weight,
        w2_raw_scale_inv,
        w13_k,
        w13_n,
        w2_k,
        w2_n,
        group_size,
        hidden_logical_size,
        fused_gated_silu,
        fused_weighted_reduce,
        broadcast_input,
        w2_direct_reduce,
        indexed_expert_ptrs,
        exact_per_route,
    )


if hasattr(torch.ops._C, "fp8_moe_single_token_sm70_out"):

    @register_fake("_C::fp8_moe_single_token_sm70_out")
    def _fp8_moe_single_token_sm70_out_fake(
        out: torch.Tensor,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        src_w13_ptrs_w_rows: torch.Tensor,
        src_w13_ptrs_s_rows: torch.Tensor,
        src_w2_ptrs_w_rows: torch.Tensor,
        src_w2_ptrs_s_rows: torch.Tensor,
        compact_input: torch.Tensor,
        gate_up: torch.Tensor,
        intermediate: torch.Tensor,
        sorted_output: torch.Tensor,
        sorted_weights: torch.Tensor,
        dst_w13_ptrs_w_rows: torch.Tensor,
        dst_w13_ptrs_s_rows: torch.Tensor,
        dst_w2_ptrs_w_rows: torch.Tensor,
        dst_w2_ptrs_s_rows: torch.Tensor,
        expert_offsets: torch.Tensor,
        inv_permuted_idx: torch.Tensor,
        sorted_expert_ids: torch.Tensor,
        broadcast_input_indices: torch.Tensor,
        w2_raw_weight: torch.Tensor,
        w2_raw_scale_inv: torch.Tensor,
        w13_k: int,
        w13_n: int,
        w2_k: int,
        w2_n: int,
        group_size: int,
        hidden_logical_size: int,
        fused_gated_silu: bool,
        fused_weighted_reduce: bool,
        broadcast_input: bool,
        w2_direct_reduce: bool,
        indexed_expert_ptrs: bool,
        exact_per_route: bool,
    ) -> None:
        return None
