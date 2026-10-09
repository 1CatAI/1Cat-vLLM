# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM70 moe bindings and their fake implementations."""

import torch

from .common import (
    _op,
    _qwen38_qpn8_op,
    has_nvfp4_grouped_batch_reduce_dispatch,
    has_nvfp4_grouped_decode_dispatch,
    register_fake,
)
from .policy import call_native, direct_native


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


def nvfp4_grouped_w2_batch_reduce_sm70_out(
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
    torch.ops._C.nvfp4_grouped_w2_batch_reduce_sm70_out(
        out, routed, x, w, s, topk, rows, experts, sizes, total
    )


if has_nvfp4_grouped_decode_dispatch():

    @register_fake("_C::nvfp4_grouped_w13_sm70_out")
    def _grouped_w13_fake(
        out,
        x,
        w,
        s,
        ids,
        rows,
        experts,
        sizes,
        total,
        split,
        interleaved,
        native_policy=(),
    ):
        return None

    @register_fake("_C::nvfp4_grouped_w2_sm70_out")
    def _grouped_w2_fake(
        out, routed, x, w, s, topk, rows, experts, sizes, total, native_policy=()
    ):
        return None


if has_nvfp4_grouped_batch_reduce_dispatch():

    @register_fake("_C::nvfp4_grouped_w2_batch_reduce_sm70_out")
    def _grouped_w2_batch_fake(
        out, routed, x, w, s, topk, rows, experts, sizes, total, native_policy=()
    ):
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
    native_policy: tuple[str, ...] = (),
) -> None:
    call_native(
        _op("mxfp4_moe_dense_stage_sm70_out"),
        native_policy,
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
        native_policy: tuple[str, ...] = (),
    ) -> None:
        return None


def mxfp4_moe_qpn_m1_sm70_out(
    out: torch.Tensor,
    input: torch.Tensor,
    weights: torch.Tensor,
    scales: torch.Tensor,
    expert_ids: torch.Tensor,
    broadcast_input: bool,
    native_policy: tuple[str, ...] = (),
) -> None:
    call_native(
        _op("mxfp4_moe_qpn_m1_sm70_out"),
        native_policy,
        out,
        input,
        weights,
        scales,
        expert_ids,
        broadcast_input,
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
        native_policy: tuple[str, ...] = (),
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
    native_policy: tuple[str, ...] = (),
) -> None:
    call_native(
        _qwen38_qpn8_op("nvfp4_moe_qpn_m1_sm70_out"),
        native_policy,
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
    native_policy: tuple[str, ...] = (),
) -> None:
    call_native(
        _qwen38_qpn8_op("nvfp4_moe_qpn_raw_scale_sm70_out"),
        native_policy,
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
    native_policy: tuple[str, ...] = (),
) -> None:
    call_native(
        _qwen38_qpn8_op("nvfp4_expand_raw_scales_sm70_out"),
        native_policy,
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
    native_policy: tuple[str, ...] = (),
) -> None:
    call_native(
        _qwen38_qpn8_op("nvfp4_moe_qpn_raw_w13_swiglu_batch_sm70_out"),
        native_policy,
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
    native_policy: tuple[str, ...] = (),
) -> None:
    call_native(
        _qwen38_qpn8_op("nvfp4_moe_qpn_raw_w2_reduce_sm70_out"),
        native_policy,
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
    native_policy: tuple[str, ...] = (),
) -> None:
    call_native(
        _qwen38_qpn8_op("nvfp4_moe_qpn_w13_swiglu_batch_sm70_out"),
        native_policy,
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
    native_policy: tuple[str, ...] = (),
) -> None:
    call_native(
        _qwen38_qpn8_op("nvfp4_moe_qpn_w2_reduce_sm70_out"),
        native_policy,
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
        native_policy: tuple[str, ...] = (),
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
        native_policy=(),
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
        native_policy: tuple[str, ...] = (),
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
        native_policy=(),
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
        native_policy: tuple[str, ...] = (),
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
        native_policy=(),
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
        native_policy: tuple[str, ...] = (),
    ) -> None:
        return None


def nvfp4_qwen38_w2_direct_reduce_out(
    out: torch.Tensor,
    input: torch.Tensor,
    weights: torch.Tensor,
    scales: torch.Tensor,
    expert_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    native_policy: tuple[str, ...] = (),
) -> None:
    call_native(
        _qwen38_qpn8_op("nvfp4_qwen38_w2_direct_reduce_out"),
        native_policy,
        out,
        input,
        weights,
        scales,
        expert_ids,
        topk_weights,
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
        native_policy: tuple[str, ...] = (),
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
        native_policy=(),
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
        native_policy=(),
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
    native_policy: tuple[str, ...] = (),
) -> None:
    call_native(
        _qwen38_qpn8_op("nvfp4_qwen38_w13_fused_swiglu_out"),
        native_policy,
        out,
        input,
        weights,
        scales,
        expert_ids,
    )


if hasattr(torch.ops._C, "nvfp4_qwen38_w13_fused_swiglu_out"):

    @register_fake("_C::nvfp4_qwen38_w13_fused_swiglu_out")
    def _nvfp4_qwen38_w13_fused_swiglu_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        weights: torch.Tensor,
        scales: torch.Tensor,
        expert_ids: torch.Tensor,
        native_policy: tuple[str, ...] = (),
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
        native_policy=(),
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
    native_policy: tuple[str, ...] = (),
) -> None:
    call_native(
        _op("nvfp4_glm53_moe_q8_qpn_sm70_out"),
        native_policy,
        out,
        input,
        weights,
        scales,
        expert_ids,
        sorted_row_idx,
        w13,
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
        native_policy: tuple[str, ...] = (),
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
    native_policy: tuple[str, ...] = (),
) -> None:
    call_native(
        _qwen38_qpn8_op("nvfp4_moe_qpn_mtp5_sm70_out"),
        native_policy,
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
        native_policy: tuple[str, ...] = (),
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
        native_policy=(),
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
    native_policy: tuple[str, ...] = (),
) -> None:
    call_native(
        _op("nvfp4_moe_dense_stage_sm70_out"),
        native_policy,
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
        native_policy: tuple[str, ...] = (),
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
    native_policy: tuple[str, ...] = (),
) -> None:
    call_native(
        _op("nvfp4_moe_indexed_dense_stage_sm70_out"),
        native_policy,
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
        native_policy: tuple[str, ...] = (),
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
    native_policy: tuple[str, ...] = (),
) -> None:
    call_native(
        _op("nvfp4_moe_indexed_fused_swiglu_sm70_out"),
        native_policy,
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
        native_policy: tuple[str, ...] = (),
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
    native_policy: tuple[str, ...] = (),
) -> None:
    call_native(
        _op("mxfp4_moe_single_token_prepare_w13_sm70_out"),
        native_policy,
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
        native_policy: tuple[str, ...] = (),
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
    native_policy: tuple[str, ...] = (),
) -> None:
    call_native(
        _op("sm70_glm53_moe_permute_q8_out"),
        native_policy,
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
        native_policy: tuple[str, ...] = (),
    ) -> None:
        return None


def awq_moe_build_strided_ptrs(
    tm_weights: torch.Tensor,
    tm_scales: torch.Tensor,
    k_ld: int,
    q_ld: int,
    num_experts: int,
    native_policy: tuple[str, ...] = (),
) -> list[torch.Tensor]:
    return call_native(
        _op("awq_moe_build_strided_ptrs"),
        native_policy,
        tm_weights,
        tm_scales,
        k_ld,
        q_ld,
        num_experts,
    )


if hasattr(torch.ops._C, "awq_moe_build_strided_ptrs"):

    @register_fake("_C::awq_moe_build_strided_ptrs")
    def _awq_moe_build_strided_ptrs_fake(
        tm_weights: torch.Tensor,
        tm_scales: torch.Tensor,
        k_ld: int,
        q_ld: int,
        num_experts: int,
        native_policy: tuple[str, ...] = (),
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
    native_policy: tuple[str, ...] = (),
) -> None:
    call_native(
        _op("awq_moe_gemm_sm70_out"),
        native_policy,
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
    native_policy: tuple[str, ...] = (),
) -> None:
    call_native(
        _op("awq_moe_gemm_sm70_per_expert_dispatch_out"),
        native_policy,
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
        native_policy: tuple[str, ...] = (),
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
        native_policy: tuple[str, ...] = (),
    ) -> None:
        return None


@direct_native("_C")
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
    native_policy: tuple[str, ...] = (),
) -> None:
    call_native(
        _op("awq_moe_dense_stage_sm70_out"),
        native_policy,
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
        native_policy: tuple[str, ...] = (),
    ) -> None:
        return None


@direct_native("_C")
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
    native_policy: tuple[str, ...] = (),
) -> None:
    call_native(
        _op("awq_moe_indexed_dense_w13_sm70_out"),
        native_policy,
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
        native_policy: tuple[str, ...] = (),
    ) -> None:
        return None


@direct_native("_C")
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
    native_policy: tuple[str, ...] = (),
) -> None:
    call_native(
        _op("awq_moe_active_dense_stage_sm70_out"),
        native_policy,
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
        native_policy: tuple[str, ...] = (),
    ) -> None:
        return None


@direct_native("_C")
def awq_moe_chunked_w2_sm70_out(
    out: torch.Tensor,
    chunk_output: torch.Tensor,
    input: torch.Tensor,
    expert_offsets: torch.Tensor,
    permuted_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    chunk_expert_offsets: torch.Tensor,
    chunk_range_begin: torch.Tensor,
    chunk_range_end: torch.Tensor,
    chunk_a_indices: torch.Tensor,
    chunk_inv_permuted_idx: torch.Tensor,
    ptrs_w: torch.Tensor,
    ptrs_s: torch.Tensor,
    num_tokens: int,
    top_k: int,
    num_experts: int,
    k: int,
    n: int,
    hidden_logical_size: int,
    group_size: int,
    chunk_tokens: int,
    native_policy: tuple[str, ...] = (),
) -> None:
    call_native(
        _op("awq_moe_chunked_w2_sm70_out"),
        native_policy,
        out,
        chunk_output,
        input,
        expert_offsets,
        permuted_idx,
        topk_weights,
        chunk_expert_offsets,
        chunk_range_begin,
        chunk_range_end,
        chunk_a_indices,
        chunk_inv_permuted_idx,
        ptrs_w,
        ptrs_s,
        num_tokens,
        top_k,
        num_experts,
        k,
        n,
        hidden_logical_size,
        group_size,
        chunk_tokens,
    )


if hasattr(torch.ops._C, "awq_moe_chunked_w2_sm70_out"):

    @register_fake("_C::awq_moe_chunked_w2_sm70_out")
    def _awq_moe_chunked_w2_sm70_out_fake(
        out: torch.Tensor,
        chunk_output: torch.Tensor,
        input: torch.Tensor,
        expert_offsets: torch.Tensor,
        permuted_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        chunk_expert_offsets: torch.Tensor,
        chunk_range_begin: torch.Tensor,
        chunk_range_end: torch.Tensor,
        chunk_a_indices: torch.Tensor,
        chunk_inv_permuted_idx: torch.Tensor,
        ptrs_w: torch.Tensor,
        ptrs_s: torch.Tensor,
        num_tokens: int,
        top_k: int,
        num_experts: int,
        k: int,
        n: int,
        hidden_logical_size: int,
        group_size: int,
        chunk_tokens: int,
        native_policy: tuple[str, ...] = (),
    ) -> None:
        return None


@direct_native("_C")
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
    native_policy: tuple[str, ...] = (),
) -> None:
    call_native(
        _op("awq_moe_single_token_dense_stage_sm70_out"),
        native_policy,
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
        native_policy: tuple[str, ...] = (),
    ) -> None:
        return None


@direct_native("_C")
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
    native_policy: tuple[str, ...] = (),
) -> None:
    call_native(
        _op("awq_moe_single_token_indexed_dense_stage_sm70_out"),
        native_policy,
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
        native_policy: tuple[str, ...] = (),
    ) -> None:
        return None


@direct_native("_C")
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
    native_policy: tuple[str, ...] = (),
) -> None:
    call_native(
        _op("awq_moe_single_token_dense_w13_sm70_out"),
        native_policy,
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
        native_policy: tuple[str, ...] = (),
    ) -> None:
        return None


@direct_native("_C")
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
    native_policy: tuple[str, ...] = (),
) -> None:
    call_native(
        _op("awq_moe_single_token_indexed_dense_w13_sm70_out"),
        native_policy,
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
        native_policy: tuple[str, ...] = (),
    ) -> None:
        return None


@direct_native("_C")
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
    native_policy: tuple[str, ...] = (),
) -> None:
    call_native(
        _op("awq_moe_single_token_compact_dense_w13_sm70_out"),
        native_policy,
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
        native_policy: tuple[str, ...] = (),
    ) -> None:
        return None


@direct_native("_C")
def awq_moe_single_token_exact_layout_prepare(
    topk_ids: torch.Tensor,
    x: torch.Tensor,
    compact_input: torch.Tensor,
    expert_offsets: torch.Tensor,
    expert_offsets64: torch.Tensor,
    inv_permuted_idx: torch.Tensor,
    num_experts: int,
    native_policy: tuple[str, ...] = (),
) -> None:
    call_native(
        _op("awq_moe_single_token_exact_layout_prepare"),
        native_policy,
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
        native_policy: tuple[str, ...] = (),
    ) -> None:
        return None


@direct_native("_C")
def awq_moe_single_token_weighted_reduce_out(
    sorted_output: torch.Tensor,
    topk_weights: torch.Tensor,
    inv_permuted_idx: torch.Tensor,
    out: torch.Tensor,
    top_k: int,
    hidden_logical_size: int,
    native_policy: tuple[str, ...] = (),
) -> None:
    call_native(
        _op("awq_moe_single_token_weighted_reduce_out"),
        native_policy,
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
        native_policy: tuple[str, ...] = (),
    ) -> None:
        return None


@direct_native("_C")
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
    native_policy: tuple[str, ...] = (),
) -> None:
    call_native(
        _op("awq_moe_qpn_m1_sm70_out"),
        native_policy,
        out,
        intermediate,
        input,
        w13,
        s13,
        w2,
        s2,
        ids,
        topk,
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
        native_policy: tuple[str, ...] = (),
    ) -> None:
        return None


@direct_native("_C")
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
    native_policy: tuple[str, ...] = (),
) -> None:
    call_native(
        _op("awq_moe_single_token_sm70_out"),
        native_policy,
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
        native_policy: tuple[str, ...] = (),
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
    native_policy: tuple[str, ...] = (),
) -> None:
    call_native(
        _op("fp8_moe_gemm_sm70_out"),
        native_policy,
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
    native_policy: tuple[str, ...] = (),
) -> None:
    call_native(
        _op("fp8_moe_gemm_sm70_per_expert_dispatch_out"),
        native_policy,
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
        native_policy: tuple[str, ...] = (),
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
        native_policy: tuple[str, ...] = (),
    ) -> None:
        return None


@direct_native("_C")
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
    native_policy: tuple[str, ...] = (),
) -> None:
    call_native(
        _op("fp8_moe_dense_stage_sm70_out"),
        native_policy,
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
        native_policy: tuple[str, ...] = (),
    ) -> None:
        return None


@direct_native("_C")
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
    native_policy: tuple[str, ...] = (),
) -> None:
    call_native(
        _op("fp8_moe_single_token_dense_stage_sm70_out"),
        native_policy,
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
        native_policy: tuple[str, ...] = (),
    ) -> None:
        return None


@direct_native("_C")
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
    native_policy: tuple[str, ...] = (),
) -> None:
    call_native(
        _op("fp8_moe_single_token_indexed_dense_stage_sm70_out"),
        native_policy,
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
        native_policy: tuple[str, ...] = (),
    ) -> None:
        return None


@direct_native("_C")
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
    native_policy: tuple[str, ...] = (),
) -> None:
    call_native(
        _op("fp8_moe_single_token_dense_w13_sm70_out"),
        native_policy,
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
        native_policy: tuple[str, ...] = (),
    ) -> None:
        return None


@direct_native("_C")
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
    native_policy: tuple[str, ...] = (),
) -> None:
    call_native(
        _op("fp8_moe_single_token_indexed_dense_w13_sm70_out"),
        native_policy,
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
        native_policy: tuple[str, ...] = (),
    ) -> None:
        return None


@direct_native("_C")
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
    native_policy: tuple[str, ...] = (),
) -> None:
    call_native(
        _op("fp8_moe_single_token_compact_dense_w13_sm70_out"),
        native_policy,
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
        native_policy: tuple[str, ...] = (),
    ) -> None:
        return None


@direct_native("_C")
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
    native_policy: tuple[str, ...] = (),
) -> None:
    call_native(
        _op("fp8_moe_single_token_sm70_out"),
        native_policy,
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
        native_policy: tuple[str, ...] = (),
    ) -> None:
        return None
