# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM70 linear bindings and their fake implementations."""

import torch

from .common import _op, _qwen38_qpn8_op, register_fake
from .policy import call_native

if hasattr(torch.ops._C, "gguf_affine_gemm_sm70_out"):

    @register_fake("_C::gguf_affine_gemm_sm70_out")
    def _gguf_affine_gemm_sm70_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        weight: torch.Tensor,
        stats: torch.Tensor,
        bits: int,
        k_ld: int,
        q_ld: int,
        group_size: int = 32,
        native_policy=(),
    ) -> None:
        return None


if hasattr(torch.ops._C, "gguf_affine_grouped_gemm_sm70_out"):

    @register_fake("_C::gguf_affine_grouped_gemm_sm70_out")
    def _gguf_affine_grouped_gemm_sm70_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        offsets: torch.Tensor,
        weight_ptrs: torch.Tensor,
        stats_ptrs: torch.Tensor,
        bits: int,
        num_experts: int,
        group_size: int = 32,
        native_policy=(),
    ) -> None:
        return None


if hasattr(torch.ops._C, "gguf_lut4_gemm_sm70_out"):

    @register_fake("_C::gguf_lut4_gemm_sm70_out")
    def _gguf_lut4_gemm_sm70_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        weight: torch.Tensor,
        stats: torch.Tensor,
        lut_id: int,
        k_ld: int,
        q_ld: int,
        group_size: int,
        native_policy=(),
    ) -> None:
        return None


if hasattr(torch.ops._C, "gguf_lut4_grouped_gemm_sm70_out"):

    @register_fake("_C::gguf_lut4_grouped_gemm_sm70_out")
    def _gguf_lut4_grouped_gemm_sm70_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        offsets: torch.Tensor,
        weight_ptrs: torch.Tensor,
        stats_ptrs: torch.Tensor,
        lut_id: int,
        num_experts: int,
        group_size: int,
        native_policy=(),
    ) -> None:
        return None


def awq_sm70_prepare(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    group_size: int,
    interleave_gated_silu: bool = False,
    native_policy: tuple[str, ...] = (),
) -> list[torch.Tensor]:
    return call_native(
        _op("awq_sm70_prepare"),
        native_policy,
        qweight,
        scales,
        qzeros,
        group_size,
        interleave_gated_silu,
    )


if hasattr(torch.ops._C, "awq_sm70_prepare"):

    @register_fake("_C::awq_sm70_prepare")
    def _awq_sm70_prepare_fake(
        qweight: torch.Tensor,
        scales: torch.Tensor,
        qzeros: torch.Tensor,
        group_size: int,
        interleave_gated_silu: bool,
        native_policy: tuple[str, ...] = (),
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
    native_policy: tuple[str, ...] = (),
) -> list[torch.Tensor]:
    return call_native(
        _op("awq_sm70_prepare_compact"),
        native_policy,
        qweight,
        scales,
        qzeros,
        group_size,
        interleave_gated_silu,
    )


if hasattr(torch.ops._C, "awq_sm70_prepare_compact"):

    @register_fake("_C::awq_sm70_prepare_compact")
    def _awq_sm70_prepare_compact_fake(
        qweight: torch.Tensor,
        scales: torch.Tensor,
        qzeros: torch.Tensor,
        group_size: int,
        interleave_gated_silu: bool,
        native_policy: tuple[str, ...] = (),
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
    native_policy: tuple[str, ...] = (),
) -> None:
    call_native(
        _op("awq_sm70_dequantize_out"), native_policy, out, qweight, scales, group_size
    )


if hasattr(torch.ops._C, "awq_sm70_dequantize_out"):

    @register_fake("_C::awq_sm70_dequantize_out")
    def _awq_sm70_dequantize_out_fake(
        out: torch.Tensor,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        group_size: int,
        native_policy: tuple[str, ...] = (),
    ) -> None:
        del out, qweight, scales, group_size
        return None


def uint4_sm70_prepare(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    group_size: int,
    interleave_gated_silu: bool = False,
    native_policy: tuple[str, ...] = (),
) -> list[torch.Tensor]:
    return call_native(
        _op("uint4_sm70_prepare"),
        native_policy,
        qweight,
        scales,
        zeros,
        group_size,
        interleave_gated_silu,
    )


if hasattr(torch.ops._C, "uint4_sm70_prepare"):

    @register_fake("_C::uint4_sm70_prepare")
    def _uint4_sm70_prepare_fake(
        qweight: torch.Tensor,
        scales: torch.Tensor,
        zeros: torch.Tensor,
        group_size: int,
        interleave_gated_silu: bool,
        native_policy: tuple[str, ...] = (),
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
    native_policy: tuple[str, ...] = (),
) -> list[torch.Tensor]:
    return call_native(
        _op("fp8_sm70_prepare"),
        native_policy,
        qweight,
        scales,
        group_size,
        interleave_gated_silu,
    )


if hasattr(torch.ops._C, "fp8_sm70_prepare"):

    @register_fake("_C::fp8_sm70_prepare")
    def _fp8_sm70_prepare_fake(
        qweight: torch.Tensor,
        scales: torch.Tensor,
        group_size: int,
        interleave_gated_silu: bool,
        native_policy: tuple[str, ...] = (),
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
    native_policy: tuple[str, ...] = (),
) -> None:
    call_native(
        _op("fp8_sm70_dequantize_out"), native_policy, out, qweight, scales, group_size
    )


if hasattr(torch.ops._C, "fp8_sm70_dequantize_out"):

    @register_fake("_C::fp8_sm70_dequantize_out")
    def _fp8_sm70_dequantize_out_fake(
        out: torch.Tensor,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        group_size: int,
        native_policy: tuple[str, ...] = (),
    ) -> None:
        del out, qweight, scales, group_size
        return None


def mxfp4_sm70_prepare(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    interleave_gated_silu: bool = False,
    native_policy: tuple[str, ...] = (),
) -> list[torch.Tensor]:
    return call_native(
        _op("mxfp4_sm70_prepare"),
        native_policy,
        qweight,
        scales,
        group_size,
        interleave_gated_silu,
    )


if hasattr(torch.ops._C, "mxfp4_sm70_prepare"):

    @register_fake("_C::mxfp4_sm70_prepare")
    def _mxfp4_sm70_prepare_fake(
        qweight: torch.Tensor,
        scales: torch.Tensor,
        group_size: int,
        interleave_gated_silu: bool,
        native_policy: tuple[str, ...] = (),
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
    native_policy: tuple[str, ...] = (),
) -> list[torch.Tensor]:
    return call_native(
        _op("nvfp4_sm70_prepare"),
        native_policy,
        qweight,
        scales,
        group_size,
        interleave_gated_silu,
    )


if hasattr(torch.ops._C, "nvfp4_sm70_prepare"):

    @register_fake("_C::nvfp4_sm70_prepare")
    def _nvfp4_sm70_prepare_fake(
        qweight: torch.Tensor,
        scales: torch.Tensor,
        group_size: int,
        interleave_gated_silu: bool,
        native_policy: tuple[str, ...] = (),
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


def sm70_f16_prepare(
    weight: torch.Tensor, native_policy: tuple[str, ...] = ()
) -> list[torch.Tensor]:
    return call_native(_op("sm70_f16_prepare"), native_policy, weight)


if hasattr(torch.ops._C, "sm70_f16_prepare"):

    @register_fake("_C::sm70_f16_prepare")
    def _sm70_f16_prepare_fake(
        weight: torch.Tensor, native_policy: tuple[str, ...] = ()
    ) -> list[torch.Tensor]:
        meta = torch.empty((1,), dtype=torch.int64, device=weight.device)
        return [torch.empty_like(weight), meta]


def sm70_glm53_tp8_cublaslt_out(
    out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    native_policy: tuple[str, ...] = (),
) -> None:
    call_native(_op("sm70_glm53_tp8_cublaslt_out"), native_policy, out, input, weight)


if hasattr(torch.ops._C, "sm70_glm53_tp8_cublaslt_out"):

    @register_fake("_C::sm70_glm53_tp8_cublaslt_out")
    def _sm70_glm53_tp8_cublaslt_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        weight: torch.Tensor,
        native_policy: tuple[str, ...] = (),
    ) -> None:
        return None


def awq_gemm_sm70(
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    k_ld: int,
    q_ld: int,
    native_policy: tuple[str, ...] = (),
) -> torch.Tensor:
    return call_native(
        _op("awq_gemm_sm70"),
        native_policy,
        input,
        qweight,
        scales,
        group_size,
        k_ld,
        q_ld,
    )


if hasattr(torch.ops._C, "awq_gemm_sm70"):

    @register_fake("_C::awq_gemm_sm70")
    def _awq_gemm_sm70_fake(
        input: torch.Tensor,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        group_size: int,
        k_ld: int,
        q_ld: int,
        native_policy: tuple[str, ...] = (),
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
    native_policy: tuple[str, ...] = (),
) -> None:
    call_native(
        _op("awq_gemm_sm70_out"),
        native_policy,
        out,
        input,
        qweight,
        scales,
        group_size,
        k_ld,
        q_ld,
        gated_silu,
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
        native_policy: tuple[str, ...] = (),
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
    native_policy: tuple[str, ...] = (),
) -> None:
    call_native(
        _op("awq_gemm_sm70_out_tile_reduce"),
        native_policy,
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
        native_policy: tuple[str, ...] = (),
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
    preserve_default_partition: bool = False,
    native_policy: tuple[str, ...] = (),
) -> None:
    if preserve_default_partition:
        call_native(
            _op("fp8_gemm_sm70_out"),
            native_policy,
            out,
            input,
            qweight,
            scales,
            group_size,
            k_ld,
            q_ld,
            gated_silu,
            True,
        )
        return
    call_native(
        _op("fp8_gemm_sm70_out"),
        native_policy,
        out,
        input,
        qweight,
        scales,
        group_size,
        k_ld,
        q_ld,
        gated_silu,
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
        preserve_default_partition: bool = False,
        native_policy: tuple[str, ...] = (),
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
    native_policy: tuple[str, ...] = (),
) -> None:
    call_native(
        _op("fp8_gemm_sm70_prefill_prescaled_out"),
        native_policy,
        out,
        input,
        qweight,
        prescaled_factors,
        group_size,
        k_ld,
        q_ld,
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
        native_policy: tuple[str, ...] = (),
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
    native_policy: tuple[str, ...] = (),
) -> None:
    call_native(
        _op("fp8_gemm_sm70_prescaled_m1_out"),
        native_policy,
        out,
        input,
        qweight,
        prescaled_factors,
        group_size,
        k_ld,
        q_ld,
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
        native_policy: tuple[str, ...] = (),
    ) -> None:
        return None


def fp8_qpn8_prepare_sm70(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    native_policy: tuple[str, ...] = (),
) -> list[torch.Tensor]:
    """Pack checkpoint-native block/channel FP8 weights into QPN8 layout."""
    return call_native(
        _qwen38_qpn8_op("fp8_qpn8_prepare_sm70"), native_policy, qweight, scales
    )


if hasattr(torch.ops._C, "fp8_qpn8_prepare_sm70"):

    @register_fake("_C::fp8_qpn8_prepare_sm70")
    def _fp8_qpn8_prepare_sm70_fake(
        qweight: torch.Tensor,
        scales: torch.Tensor,
        native_policy: tuple[str, ...] = (),
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
    native_policy: tuple[str, ...] = (),
) -> None:
    """Materialize one QPN8 weight into a caller-owned FP16 workspace."""
    call_native(
        _qwen38_qpn8_op("fp8_qpn8_dequantize_sm70_out"),
        native_policy,
        out,
        codes,
        group_scales,
    )


if hasattr(torch.ops._C, "fp8_qpn8_dequantize_sm70_out"):

    @register_fake("_C::fp8_qpn8_dequantize_sm70_out")
    def _fp8_qpn8_dequantize_sm70_out_fake(
        out: torch.Tensor,
        codes: torch.Tensor,
        group_scales: torch.Tensor,
        native_policy: tuple[str, ...] = (),
    ) -> None:
        return None


def fp8_qpn8_prefill_sm70_out(
    out: torch.Tensor,
    dense_weight_ptr: int,
    input: torch.Tensor,
    codes: torch.Tensor,
    group_scales: torch.Tensor,
    gated_silu: bool,
    native_policy: tuple[str, ...] = (),
) -> None:
    """Dequantize QPN8 into bounded workspace and run a large-M FP16 GEMM."""
    call_native(
        _qwen38_qpn8_op("fp8_qpn8_prefill_sm70_out"),
        native_policy,
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
        native_policy: tuple[str, ...] = (),
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
    native_policy: tuple[str, ...] = (),
) -> None:
    """Runtime-dispatch dynamic M without specializing a Python branch."""
    call_native(
        _qwen38_qpn8_op("fp8_qpn8_dispatch_sm70_out"),
        native_policy,
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
        native_policy: tuple[str, ...] = (),
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
    native_policy: tuple[str, ...] = (),
) -> None:
    """Run the SM70 QPN8 FP8 GEMM into ``out``.

    The model- and shape-gated automatic route and operator benchmark share
    this entry point. ``codes`` and ``group_scales`` use the QPN8 layout.
    """
    call_native(
        _qwen38_qpn8_op("fp8_qpn8_gemm_sm70_out"),
        native_policy,
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
        native_policy: tuple[str, ...] = (),
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
    native_policy: tuple[str, ...] = (),
) -> None:
    """Run the exact-shape GDN QKV/Z FP8 and b/a FP16 projections."""
    call_native(
        _qwen38_qpn8_op("fp8_qpn8_gemm_ba_split_sm70_out"),
        native_policy,
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
        native_policy: tuple[str, ...] = (),
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
    native_policy: tuple[str, ...] = (),
) -> None:
    """Dispatch exact M=1 fusion or the original large-M projection path."""
    call_native(
        _qwen38_qpn8_op("fp8_qpn8_dispatch_ba_split_sm70_out"),
        native_policy,
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
        native_policy: tuple[str, ...] = (),
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
    native_policy: tuple[str, ...] = (),
) -> None:
    """Run the single-kernel paired-tile QPN8 gated SiLU experiment."""
    call_native(
        _qwen38_qpn8_op("fp8_qpn8_gated_pair_sm70_out"),
        native_policy,
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
    native_policy: tuple[str, ...] = (),
) -> None:
    """Dispatch the exact Qwen4Exp HC pair without a Python M branch."""
    call_native(
        _qwen38_qpn8_op("fp8_qpn8_hc_dispatch_sm70_out"),
        native_policy,
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
        native_policy: tuple[str, ...] = (),
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
        native_policy=(),
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
        native_policy: tuple[str, ...] = (),
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
    native_policy: tuple[str, ...] = (),
) -> list[torch.Tensor]:
    """Pack unpacked FP4 weights and FP16 group scales into QPN4 layout."""
    return call_native(_op("nvfp4_qpn4_prepare_sm70"), native_policy, qweight, scales)


if hasattr(torch.ops._C, "nvfp4_qpn4_prepare_sm70"):

    @register_fake("_C::nvfp4_qpn4_prepare_sm70")
    def _nvfp4_qpn4_prepare_sm70_fake(
        qweight: torch.Tensor,
        scales: torch.Tensor,
        native_policy: tuple[str, ...] = (),
    ) -> list[torch.Tensor]:
        k, n = qweight.shape
        codes = torch.empty((k, n // 2), dtype=torch.uint8, device=qweight.device)
        packed_scales = torch.empty_like(scales)
        return [codes, packed_scales]


def nvfp4_qpn4_prepare_scale_code_sm70(
    qweight: torch.Tensor,
    scale_codes: torch.Tensor,
    native_policy: tuple[str, ...] = (),
) -> list[torch.Tensor]:
    """Pack unpacked FP4 weights and checkpoint E4M3 group-scale codes."""
    return call_native(
        _op("nvfp4_qpn4_prepare_scale_code_sm70"), native_policy, qweight, scale_codes
    )


if hasattr(torch.ops._C, "nvfp4_qpn4_prepare_scale_code_sm70"):

    @register_fake("_C::nvfp4_qpn4_prepare_scale_code_sm70")
    def _nvfp4_qpn4_prepare_scale_code_sm70_fake(
        qweight: torch.Tensor,
        scale_codes: torch.Tensor,
        native_policy: tuple[str, ...] = (),
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
    native_policy: tuple[str, ...] = (),
) -> None:
    call_native(
        _op("nvfp4_qpn4_dequantize_sm70_out"),
        native_policy,
        out,
        codes,
        scales,
        global_scale,
        use_scale_code,
    )


if hasattr(torch.ops._C, "nvfp4_qpn4_dequantize_sm70_out"):

    @register_fake("_C::nvfp4_qpn4_dequantize_sm70_out")
    def _nvfp4_qpn4_dequantize_sm70_out_fake(
        out: torch.Tensor,
        codes: torch.Tensor,
        scales: torch.Tensor,
        global_scale: float,
        use_scale_code: bool,
        native_policy: tuple[str, ...] = (),
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
    native_policy: tuple[str, ...] = (),
) -> None:
    call_native(
        _op("nvfp4_qpn4_prefill_sm70_out"),
        native_policy,
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
        native_policy: tuple[str, ...] = (),
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
    native_policy: tuple[str, ...] = (),
) -> None:
    """Dispatch exact M=1 decode or bounded-workspace large-M prefill."""
    call_native(
        _op("nvfp4_qpn4_dispatch_sm70_out"),
        native_policy,
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
        native_policy: tuple[str, ...] = (),
    ) -> None:
        return None


def nvfp4_qpn2_prepare_sm70(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    native_policy: tuple[str, ...] = (),
) -> list[torch.Tensor]:
    """Pack checkpoint-native NVFP4 tensors into the QPN2 layout."""
    return call_native(
        _op("nvfp4_qpn2_prepare_sm70"), native_policy, weight_packed, weight_scale
    )


if hasattr(torch.ops._C, "nvfp4_qpn2_prepare_sm70"):

    @register_fake("_C::nvfp4_qpn2_prepare_sm70")
    def _nvfp4_qpn2_prepare_sm70_fake(
        weight_packed: torch.Tensor,
        weight_scale: torch.Tensor,
        native_policy: tuple[str, ...] = (),
    ) -> list[torch.Tensor]:
        codes = torch.empty_like(weight_packed)
        scales = torch.empty(
            weight_scale.shape, dtype=torch.uint8, device=weight_scale.device
        )
        return [codes, scales]


def nvfp4_qpn2_bundle_sm70(
    codes: torch.Tensor, scales: torch.Tensor, native_policy: tuple[str, ...] = ()
) -> list[torch.Tensor]:
    """Return code/scale views sharing one interleaved native allocation."""
    return call_native(_op("nvfp4_qpn2_bundle_sm70"), native_policy, codes, scales)


if hasattr(torch.ops._C, "nvfp4_qpn2_bundle_sm70"):

    @register_fake("_C::nvfp4_qpn2_bundle_sm70")
    def _nvfp4_qpn2_bundle_sm70_fake(
        codes: torch.Tensor, scales: torch.Tensor, native_policy: tuple[str, ...] = ()
    ) -> list[torch.Tensor]:
        n, packed_k = codes.shape
        packed = codes.new_empty((n // 32, packed_k // 8, 288))
        return [packed[..., :256], packed[..., 256:]]


def nvfp4_qpn2_prepare_scales_sm70(
    weight_scale: torch.Tensor, native_policy: tuple[str, ...] = ()
) -> torch.Tensor:
    """Pack E4M3 scales, padding N to 32, without allocating weight codes."""
    return call_native(
        _op("nvfp4_qpn2_prepare_scales_sm70"), native_policy, weight_scale
    )


if hasattr(torch.ops._C, "nvfp4_qpn2_prepare_scales_sm70"):

    @register_fake("_C::nvfp4_qpn2_prepare_scales_sm70")
    def _nvfp4_qpn2_prepare_scales_sm70_fake(
        weight_scale: torch.Tensor,
        native_policy: tuple[str, ...] = (),
    ) -> torch.Tensor:
        n, groups = weight_scale.shape
        return torch.empty(
            ((n + 31) // 32 * 32, groups),
            device=weight_scale.device,
            dtype=torch.uint8,
        )


def nvfp4_qpn2_compact_tm_gemm_sm70_out(
    out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    scales: torch.Tensor,
    global_scale: float,
    k_ld: int,
    q_ld: int,
    gated_silu: bool = False,
    native_policy: tuple[str, ...] = (),
) -> None:
    """Restore temporary TurboMind scales from the retained QPN2 scale codes."""
    call_native(
        _op("nvfp4_qpn2_compact_tm_gemm_sm70_out"),
        native_policy,
        out,
        input,
        weight,
        scales,
        global_scale,
        k_ld,
        q_ld,
        gated_silu,
    )


if hasattr(torch.ops._C, "nvfp4_qpn2_compact_tm_gemm_sm70_out"):

    @register_fake("_C::nvfp4_qpn2_compact_tm_gemm_sm70_out")
    def _nvfp4_qpn2_compact_tm_gemm_sm70_out_fake(
        out,
        input,
        weight,
        scales,
        global_scale,
        k_ld,
        q_ld,
        gated_silu,
        native_policy: tuple[str, ...] = (),
    ) -> None:
        return None


def nvfp4_qpn2_tm_dispatch_sm70_out(
    out: torch.Tensor,
    input: torch.Tensor,
    tm_weight: torch.Tensor,
    scales: torch.Tensor,
    global_scale: float,
    split_k: int,
    accumulator_chains: int,
    tm_scales: torch.Tensor,
    tm_group_size: int,
    tm_k_ld: int,
    tm_q_ld: int,
    gated_silu: bool,
    min_prefill_m: int,
    prescaled_scales: bool = False,
    native_policy: tuple[str, ...] = (),
) -> None:
    """Use shared TurboMind codes for QPN2, TurboMind and dense prefill."""
    call_native(
        _op("nvfp4_qpn2_tm_dispatch_sm70_out"),
        native_policy,
        out,
        input,
        tm_weight,
        scales,
        global_scale,
        split_k,
        accumulator_chains,
        tm_scales,
        tm_group_size,
        tm_k_ld,
        tm_q_ld,
        gated_silu,
        min_prefill_m,
        *([True] if prescaled_scales else []),
    )


if hasattr(torch.ops._C, "nvfp4_qpn2_tm_dispatch_sm70_out"):

    @register_fake("_C::nvfp4_qpn2_tm_dispatch_sm70_out")
    def _nvfp4_qpn2_tm_dispatch_sm70_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        tm_weight: torch.Tensor,
        scales: torch.Tensor,
        global_scale: float,
        split_k: int,
        accumulator_chains: int,
        tm_scales: torch.Tensor,
        tm_group_size: int,
        tm_k_ld: int,
        tm_q_ld: int,
        gated_silu: bool,
        min_prefill_m: int,
        prescaled_scales: bool = False,
        native_policy: tuple[str, ...] = (),
    ) -> None:
        return None


def nvfp4_qpn2_gemm_sm70_out(
    out: torch.Tensor,
    input: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    global_scale: float,
    split_k: int,
    accumulator_chains: int,
    native_policy: tuple[str, ...] = (),
) -> None:
    call_native(
        _op("nvfp4_qpn2_gemm_sm70_out"),
        native_policy,
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
        native_policy: tuple[str, ...] = (),
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
    native_policy: tuple[str, ...] = (),
) -> None:
    call_native(
        _op("nvfp4_qpn2_gated_sm70_out"),
        native_policy,
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
        native_policy: tuple[str, ...] = (),
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
    native_policy: tuple[str, ...] = (),
) -> None:
    """Select QPN2 for M<=32 and TurboMind for larger dynamic M."""
    call_native(
        _op("nvfp4_qpn2_dispatch_sm70_out"),
        native_policy,
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
        native_policy: tuple[str, ...] = (),
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
    native_policy: tuple[str, ...] = (),
) -> None:
    """Keep QPN2 decode and QPN2-packed prefill behind one opaque op."""
    call_native(
        _op("nvfp4_qpn2_prefill_dispatch_sm70_out"),
        native_policy,
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
        native_policy: tuple[str, ...] = (),
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
    native_policy: tuple[str, ...] = (),
) -> None:
    call_native(
        _op("fp8_gemm_sm70_prefill_dispatch_out"),
        native_policy,
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
        native_policy: tuple[str, ...] = (),
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
    native_policy: tuple[str, ...] = (),
) -> None:
    call_native(
        _op("mxfp4_gemm_sm70_out"),
        native_policy,
        out,
        input,
        qweight,
        scales,
        group_size,
        k_ld,
        q_ld,
        gated_silu,
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
        native_policy: tuple[str, ...] = (),
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
    native_policy: tuple[str, ...] = (),
) -> None:
    call_native(
        _op("nvfp4_gemm_sm70_out"),
        native_policy,
        out,
        input,
        qweight,
        scales,
        group_size,
        k_ld,
        q_ld,
        gated_silu,
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
        native_policy: tuple[str, ...] = (),
    ) -> None:
        return None


def nvfp4_gemm_sm70_prescaled_out(
    out: torch.Tensor,
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    k_ld: int,
    q_ld: int,
    gated_silu: bool = False,
    native_policy: tuple[str, ...] = (),
) -> None:
    call_native(
        _op("nvfp4_gemm_sm70_prescaled_out"),
        native_policy,
        out,
        input,
        qweight,
        scales,
        group_size,
        k_ld,
        q_ld,
        gated_silu,
    )


if hasattr(torch.ops._C, "nvfp4_gemm_sm70_prescaled_out"):

    @register_fake("_C::nvfp4_gemm_sm70_prescaled_out")
    def _nvfp4_gemm_sm70_prescaled_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        group_size: int,
        k_ld: int,
        q_ld: int,
        gated_silu: bool,
        native_policy: tuple[str, ...] = (),
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
    native_policy: tuple[str, ...] = (),
) -> None:
    call_native(
        _op("nvfp4_gemv_sm70_raw_out"),
        native_policy,
        out,
        input,
        qweight_packed,
        scales,
        partials,
        group_size,
        split_k,
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
        native_policy: tuple[str, ...] = (),
    ) -> None:
        return None


def nvfp4_gemv_sm70_warp_out(
    out: torch.Tensor,
    input: torch.Tensor,
    qweight_packed: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    native_policy: tuple[str, ...] = (),
) -> None:
    call_native(
        _op("nvfp4_gemv_sm70_warp_out"),
        native_policy,
        out,
        input,
        qweight_packed,
        scales,
        group_size,
    )


if hasattr(torch.ops._C, "nvfp4_gemv_sm70_warp_out"):

    @register_fake("_C::nvfp4_gemv_sm70_warp_out")
    def _nvfp4_gemv_sm70_warp_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        qweight_packed: torch.Tensor,
        scales: torch.Tensor,
        group_size: int,
        native_policy: tuple[str, ...] = (),
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
    native_policy: tuple[str, ...] = (),
) -> None:
    call_native(
        _op("nvfp4_gemv_sm70_h2_out"),
        native_policy,
        out,
        input,
        qweight_packed,
        scales,
        partials,
        group_size,
        split_k,
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
        native_policy: tuple[str, ...] = (),
    ) -> None:
        return None


def fp8_gemm_sm70_out_auto(
    out: torch.Tensor,
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    native_policy: tuple[str, ...] = (),
) -> None:
    call_native(
        _op("fp8_gemm_sm70_out_auto"), native_policy, out, input, qweight, scales
    )


def fp8_gemm_sm70_out_meta(
    out: torch.Tensor,
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    meta: torch.Tensor,
    gated_silu: bool = False,
    native_policy: tuple[str, ...] = (),
) -> None:
    call_native(
        _op("fp8_gemm_sm70_out_meta"),
        native_policy,
        out,
        input,
        qweight,
        scales,
        meta,
        gated_silu,
    )


def sm70_f16_gemm(
    input: torch.Tensor, weight: torch.Tensor, native_policy: tuple[str, ...] = ()
) -> torch.Tensor:
    return call_native(_op("sm70_f16_gemm"), native_policy, input, weight)


if hasattr(torch.ops._C, "sm70_f16_gemm"):

    @register_fake("_C::sm70_f16_gemm")
    def _sm70_f16_gemm_fake(
        input: torch.Tensor,
        weight: torch.Tensor,
        native_policy: tuple[str, ...] = (),
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
    native_policy: tuple[str, ...] = (),
) -> None:
    call_native(
        _op("sm70_f16_gemm_out"), native_policy, out, input, weight, k_ld, gated_silu
    )


if hasattr(torch.ops._C, "sm70_f16_gemm_out"):

    @register_fake("_C::sm70_f16_gemm_out")
    def _sm70_f16_gemm_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        weight: torch.Tensor,
        k_ld: int,
        gated_silu: bool,
        native_policy: tuple[str, ...] = (),
    ) -> None:
        return None


def sm70_glm53_fp16_gemv_out(
    output: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    native_policy: tuple[str, ...] = (),
) -> None:
    call_native(_op("sm70_glm53_fp16_gemv_out"), native_policy, output, input, weight)


if hasattr(torch.ops._C, "sm70_glm53_fp16_gemv_out"):

    @register_fake("_C::sm70_glm53_fp16_gemv_out")
    def _sm70_glm53_fp16_gemv_out_fake(
        output: torch.Tensor,
        input: torch.Tensor,
        weight: torch.Tensor,
        native_policy: tuple[str, ...] = (),
    ) -> None:
        return None


def sm70_gemm_import_cache(
    device_hint: torch.Tensor, path: str, native_policy: tuple[str, ...] = ()
) -> int:
    return call_native(_op("sm70_gemm_import_cache"), native_policy, device_hint, path)


def sm70_gemm_export_cache(
    device_hint: torch.Tensor, path: str, native_policy: tuple[str, ...] = ()
) -> int:
    return call_native(_op("sm70_gemm_export_cache"), native_policy, device_hint, path)


if hasattr(torch.ops._C, "gguf_affine_dequantize_sm70_out"):

    @register_fake("_C::gguf_affine_dequantize_sm70_out")
    def _gguf_affine_dequantize_sm70_out_fake(
        out: torch.Tensor,
        weight: torch.Tensor,
        stats: torch.Tensor,
        bits: int,
        group_size: int,
        native_policy=(),
    ) -> None:
        return None


if hasattr(torch.ops._C, "gguf_affine_blas_sm70_out"):

    @register_fake("_C::gguf_affine_blas_sm70_out")
    def _gguf_affine_blas_sm70_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        weight: torch.Tensor,
        stats: torch.Tensor,
        bits: int,
        scratch: torch.Tensor,
        group_size: int,
        native_policy=(),
    ) -> None:
        return None


if hasattr(torch.ops._C, "gguf_lattice_gemm_sm70_out"):

    @register_fake("_C::gguf_lattice_gemm_sm70_out")
    def _gguf_lattice_gemm_sm70_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        weight: torch.Tensor,
        stats: torch.Tensor,
        source_type: int,
        k_ld: int,
        q_ld: int,
        group_size: int,
        native_policy=(),
    ) -> None:
        return None


if hasattr(torch.ops._C, "gguf_lattice_grouped_gemm_sm70_out"):

    @register_fake("_C::gguf_lattice_grouped_gemm_sm70_out")
    def _gguf_lattice_grouped_gemm_sm70_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        offsets: torch.Tensor,
        weight_ptrs: torch.Tensor,
        stats_ptrs: torch.Tensor,
        source_type: int,
        num_experts: int,
        group_size: int,
        native_policy=(),
    ) -> None:
        return None


if hasattr(torch.ops._C, "gguf_lattice_dequantize_sm70_out"):

    @register_fake("_C::gguf_lattice_dequantize_sm70_out")
    def _gguf_lattice_dequantize_sm70_out_fake(
        out: torch.Tensor,
        weight: torch.Tensor,
        stats: torch.Tensor,
        source_type: int,
        group_size: int,
        native_policy=(),
    ) -> None:
        return None


if hasattr(torch.ops._C, "gguf_lattice_grouped_vec_sm70_out"):

    @register_fake("_C::gguf_lattice_grouped_vec_sm70_out")
    def _gguf_lattice_grouped_vec_sm70_out_fake(
        out,
        input,
        offsets,
        weight_ptrs,
        stats_ptrs,
        source_type,
        num_experts,
        group_size,
        native_policy=(),
    ):
        return None


if hasattr(torch.ops._C, "gguf_lattice_blas_sm70_out"):

    @register_fake("_C::gguf_lattice_blas_sm70_out")
    def _gguf_lattice_blas_sm70_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        weight: torch.Tensor,
        stats: torch.Tensor,
        source_type: int,
        scratch: torch.Tensor,
        group_size: int,
        native_policy=(),
    ) -> None:
        return None


def gguf_affine_gemm_sm70_out(*args, native_policy=()):
    return call_native(_op("gguf_affine_gemm_sm70_out"), native_policy, *args)


def gguf_affine_grouped_gemm_sm70_out(*args, native_policy=()):
    return call_native(_op("gguf_affine_grouped_gemm_sm70_out"), native_policy, *args)


def gguf_lut4_gemm_sm70_out(*args, native_policy=()):
    return call_native(_op("gguf_lut4_gemm_sm70_out"), native_policy, *args)


def gguf_lut4_grouped_gemm_sm70_out(*args, native_policy=()):
    return call_native(_op("gguf_lut4_grouped_gemm_sm70_out"), native_policy, *args)


def gguf_lattice_gemm_sm70_out(*args, native_policy=()):
    return call_native(_op("gguf_lattice_gemm_sm70_out"), native_policy, *args)


def gguf_lattice_grouped_gemm_sm70_out(*args, native_policy=()):
    return call_native(_op("gguf_lattice_grouped_gemm_sm70_out"), native_policy, *args)
