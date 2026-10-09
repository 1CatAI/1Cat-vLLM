# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit native arguments and initialization-only binding capture."""

from functools import partial

import torch

GGUF_OPERATORS = (
    "gguf_affine_gemm_sm70_out",
    "gguf_affine_grouped_gemm_sm70_out",
    "gguf_lut4_gemm_sm70_out",
    "gguf_lut4_grouped_gemm_sm70_out",
    "gguf_lattice_gemm_sm70_out",
    "gguf_lattice_grouped_gemm_sm70_out",
)

CONFIGURED_OPERATORS = GGUF_OPERATORS + (
    "awq_gemm_sm70",
    "awq_gemm_sm70_out",
    "awq_gemm_sm70_out_tile_reduce",
    "awq_moe_active_dense_stage_sm70_out",
    "awq_moe_build_strided_ptrs",
    "awq_moe_chunked_w2_sm70_out",
    "awq_moe_dense_stage_sm70_out",
    "awq_moe_gemm_sm70_out",
    "awq_moe_gemm_sm70_per_expert_dispatch_out",
    "awq_moe_indexed_dense_w13_sm70_out",
    "awq_moe_qpn_m1_sm70_out",
    "awq_moe_single_token_compact_dense_w13_sm70_out",
    "awq_moe_single_token_dense_stage_sm70_out",
    "awq_moe_single_token_dense_w13_sm70_out",
    "awq_moe_single_token_exact_layout_prepare",
    "awq_moe_single_token_indexed_dense_stage_sm70_out",
    "awq_moe_single_token_indexed_dense_w13_sm70_out",
    "awq_moe_single_token_sm70_out",
    "awq_moe_single_token_weighted_reduce_out",
    "awq_sm70_dequantize_out",
    "awq_sm70_prepare",
    "awq_sm70_prepare_compact",
    "fp8_gemm_sm70_out",
    "fp8_gemm_sm70_out_auto",
    "fp8_gemm_sm70_out_meta",
    "fp8_gemm_sm70_prefill_dispatch_out",
    "fp8_gemm_sm70_prefill_prescaled_out",
    "fp8_gemm_sm70_prescaled_m1_out",
    "fp8_moe_dense_stage_sm70_out",
    "fp8_moe_gemm_sm70_out",
    "fp8_moe_gemm_sm70_per_expert_dispatch_out",
    "fp8_moe_single_token_compact_dense_w13_sm70_out",
    "fp8_moe_single_token_dense_stage_sm70_out",
    "fp8_moe_single_token_dense_w13_sm70_out",
    "fp8_moe_single_token_indexed_dense_stage_sm70_out",
    "fp8_moe_single_token_indexed_dense_w13_sm70_out",
    "fp8_moe_single_token_sm70_out",
    "fp8_qpn8_dequantize_sm70_out",
    "fp8_qpn8_dispatch_ba_split_sm70_out",
    "fp8_qpn8_dispatch_sm70_out",
    "fp8_qpn8_gated_pair_sm70_out",
    "fp8_qpn8_gemm_ba_split_sm70_out",
    "fp8_qpn8_gemm_sm70_out",
    "fp8_qpn8_hc_dispatch_sm70_out",
    "fp8_qpn8_prefill_sm70_out",
    "fp8_qpn8_prepare_sm70",
    "fp8_sm70_dequantize_out",
    "fp8_sm70_prepare",
    "mxfp4_gemm_sm70_out",
    "mxfp4_moe_dense_stage_sm70_out",
    "mxfp4_moe_qpn_m1_sm70_out",
    "mxfp4_moe_single_token_prepare_w13_sm70_out",
    "mxfp4_sm70_prepare",
    "nvfp4_expand_raw_scales_sm70_out",
    "nvfp4_gemm_sm70_out",
    "nvfp4_gemm_sm70_prescaled_out",
    "nvfp4_gemv_sm70_h2_out",
    "nvfp4_gemv_sm70_raw_out",
    "nvfp4_gemv_sm70_warp_out",
    "nvfp4_glm53_moe_q8_qpn_sm70_out",
    "nvfp4_moe_dense_stage_sm70_out",
    "nvfp4_moe_indexed_dense_stage_sm70_out",
    "nvfp4_moe_indexed_fused_swiglu_sm70_out",
    "nvfp4_moe_qpn_m1_sm70_out",
    "nvfp4_moe_qpn_mtp5_sm70_out",
    "nvfp4_moe_qpn_raw_scale_sm70_out",
    "nvfp4_moe_qpn_raw_w13_swiglu_batch_sm70_out",
    "nvfp4_moe_qpn_raw_w2_reduce_sm70_out",
    "nvfp4_moe_qpn_w13_swiglu_batch_sm70_out",
    "nvfp4_moe_qpn_w2_reduce_sm70_out",
    "nvfp4_qpn2_bundle_sm70",
    "nvfp4_qpn2_compact_tm_gemm_sm70_out",
    "nvfp4_qpn2_dispatch_sm70_out",
    "nvfp4_qpn2_gated_sm70_out",
    "nvfp4_qpn2_gemm_sm70_out",
    "nvfp4_qpn2_prefill_dispatch_sm70_out",
    "nvfp4_qpn2_prepare_scales_sm70",
    "nvfp4_qpn2_prepare_sm70",
    "nvfp4_qpn2_tm_dispatch_sm70_out",
    "nvfp4_qpn4_dequantize_sm70_out",
    "nvfp4_qpn4_dispatch_sm70_out",
    "nvfp4_qpn4_prefill_sm70_out",
    "nvfp4_qpn4_prepare_scale_code_sm70",
    "nvfp4_qpn4_prepare_sm70",
    "nvfp4_qwen38_w13_fused_swiglu_out",
    "nvfp4_qwen38_w2_direct_reduce_out",
    "nvfp4_sm70_prepare",
    "sm70_f16_gemm",
    "sm70_f16_gemm_out",
    "sm70_f16_prepare",
    "sm70_gemm_export_cache",
    "sm70_gemm_import_cache",
    "sm70_glm53_fp16_gemv_out",
    "sm70_glm53_moe_permute_q8_out",
    "sm70_glm53_tp8_cublaslt_out",
    "uint4_sm70_prepare",
)

ROUTING_OPERATORS = (
    "moe_permute",
    "moe_permute_with_scratch",
    "moe_permute_metadata_with_scratch",
    "moe_unpermute",
)


def direct_native(namespace):
    """Mark a positional, argument-preserving compatibility wrapper.

    Prepared owners may bind its native callable once. Keep the function itself
    in the marker so instrumentation using functools.wraps is never bypassed.
    """

    def decorate(operation):
        operation._sm70_direct = (operation, namespace)
        return operation

    return decorate


def call_native(operation, native_policy, *args, **kwargs):
    # Historical research fragments have their own unchanged ABI. They may be
    # used with matching legacy values; initialization rejects typed conflicts.
    qualified_name = getattr(operation, "_qualified_op_name", "")
    if qualified_name.startswith("_C_qwen38::"):
        return operation(*args, **kwargs)
    if native_policy:
        return operation(
            *args,
            **kwargs,
            native_policy=native_policy[0]
            if len(native_policy) == 1
            else native_policy,
        )
    return operation(*args, **kwargs)


def native_policy_abi_available() -> bool:
    from vllm.config.sm70_native import NATIVE_FIELDS

    return all(
        hasattr(namespace, "sm70_native_policy_abi")
        and namespace.sm70_native_policy_abi() == len(NATIVE_FIELDS)
        for namespace in (torch.ops._C, torch.ops._moe_C)
    )


def call_routing(name, native_policy, *args, **kwargs):
    return call_native(getattr(torch.ops._moe_C, name), native_policy, *args, **kwargs)


@direct_native("_moe_C")
def moe_permute(*args, native_policy=()):
    return call_routing("moe_permute", native_policy, *args)


@direct_native("_moe_C")
def moe_permute_with_scratch(*args, native_policy=()):
    return call_routing("moe_permute_with_scratch", native_policy, *args)


@direct_native("_moe_C")
def moe_permute_metadata_with_scratch(*args, native_policy=()):
    return call_routing("moe_permute_metadata_with_scratch", native_policy, *args)


@direct_native("_moe_C")
def moe_unpermute(*args, native_policy=()):
    return call_routing("moe_unpermute", native_policy, *args)


class NativeBindings:
    """Bind once per prepared owner; token execution reads no configuration."""

    def __init__(self, values=()):
        available = native_policy_abi_available()
        sidecar = torch.ops._C_qwen38
        legacy_sidecar = any(hasattr(sidecar, name) for name in CONFIGURED_OPERATORS)
        if values and (not available or legacy_sidecar):
            import os

            from vllm.config.sm70_native import NATIVE_FIELDS, UNSET

            conflicts = [
                alias
                for (_, alias, _, _), value in zip(NATIVE_FIELDS, values)
                if value != UNSET and value != os.getenv(alias, UNSET)
            ]
            if conflicts:
                raise RuntimeError(
                    "Explicit SM70 native policy requires the policy-argument ABI: "
                    + ", ".join(conflicts)
                )
        self.values = values if available else ()
        self.arguments = self.values
        namespaces = (torch.ops._C, torch.ops._moe_C)
        if self.values and all(
            hasattr(namespace, "sm70_prepare_native_policy_token")
            for namespace in namespaces
        ):
            token = "sm70:1:" + "".join(
                f"{len(value.encode('utf-8'))}:{value}" for value in self.values
            )
            for namespace in namespaces:
                namespace.sm70_prepare_native_policy_token(token)
            self.arguments = (token,)
        if self.values:
            from vllm import _sm70_ops

            for name in CONFIGURED_OPERATORS + ROUTING_OPERATORS:
                operation = getattr(_sm70_ops, name, None)
                if operation is not None:
                    arguments: str | tuple[str, ...] = self.arguments
                    direct = getattr(operation, "_sm70_direct", None)
                    if direct and direct[0] is operation and len(arguments) == 1:
                        namespace = getattr(torch.ops, direct[1])
                        if hasattr(namespace, name):
                            operation = getattr(namespace, name)
                            arguments = arguments[0]
                    setattr(self, name, partial(operation, native_policy=arguments))

    def __getattr__(self, name):
        from vllm import _sm70_ops

        return getattr(_sm70_ops, name)


def register_policy_op(name, schema, operation, fake):
    """Use an explicit schema because Torch inference lacks optional str lists."""
    from vllm.platforms import current_platform
    from vllm.utils.torch_utils import vllm_lib

    vllm_lib.define(name + schema)
    vllm_lib.impl(name, operation, dispatch_key=current_platform.dispatch_key)
    vllm_lib._register_fake(name, fake)


def call_gguf_native(operation, native_policy, *args):
    """GGUF vector/BLAS kernels retain their independent native signatures."""
    name = getattr(operation, "_qualified_op_name", "").split("::")[-1]
    return call_native(
        operation, native_policy if name in GGUF_OPERATORS else (), *args
    )
