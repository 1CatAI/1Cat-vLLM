# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM70 FP8 MoE method backed by TurboMind batched GEMM kernels."""

import os

import torch
from torch.nn import Parameter

from vllm import _sm70_ops as sm70_ops
from vllm import envs
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoEMethodBase,
    FusedMoeWeightScaleSupported,
    RoutedExperts,
    SharedExperts,
)
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.quantization.sm70_moe_router import (
    Sm70MoeStageRoute,
    select_sm70_quantized_moe_route,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    process_fp8_input_tensor_strategy_moe,
    process_fp8_weight_tensor_strategy_moe,
)
from vllm.model_executor.utils import set_weight_attrs

logger = init_logger(__name__)

_DEFAULT_PERSISTENT_MAX_TOKENS = 32


def _log_runtime_route_once(message: str, *args) -> None:
    if torch.compiler.is_compiling():
        return
    logger.info_once(message, *args)


def _single_token_weighted_reduce_enabled() -> bool:
    if not (
        envs.VLLM_SM70_MOE_SINGLE_TOKEN_FASTPATH
        or envs.VLLM_SM70_MOE_SINGLE_TOKEN_UNPERMUTE_FASTPATH
    ):
        return False
    return hasattr(torch.ops._C, "awq_moe_single_token_weighted_reduce_out")


def _single_token_indexed_w13_enabled() -> bool:
    if not (
        envs.VLLM_SM70_MOE_SINGLE_TOKEN_INDEXED_STAGE_FASTPATH
        or envs.VLLM_SM70_MOE_SINGLE_TOKEN_INDEXED_W13_FASTPATH
    ):
        return False
    return hasattr(torch.ops._C, "fp8_moe_single_token_indexed_dense_w13_sm70_out")


def _single_token_compact_w13_enabled() -> bool:
    if not envs.VLLM_SM70_MOE_SINGLE_TOKEN_COMPACT_W13_FASTPATH:
        return False
    return hasattr(torch.ops._C, "fp8_moe_single_token_compact_dense_w13_sm70_out")


def _legacy_single_token_compact_enabled() -> bool:
    if not envs.VLLM_SM70_FP8_MOE_LEGACY_SINGLE_TOKEN_COMPACT:
        return False
    return hasattr(torch.ops._C, "fp8_moe_single_token_sm70_out")


def _permute_with_scratch_enabled() -> bool:
    return envs.VLLM_SM70_FP8_MOE_PERMUTE_WITH_SCRATCH and hasattr(
        torch.ops._moe_C, "moe_permute_with_scratch"
    )


def _single_token_indexed_w2_enabled() -> bool:
    if not (
        envs.VLLM_SM70_FP8_MOE_SINGLE_TOKEN_INDEXED_W2_FASTPATH
        or envs.VLLM_SM70_MOE_SINGLE_TOKEN_INDEXED_STAGE_FASTPATH
        or envs.VLLM_SM70_MOE_SINGLE_TOKEN_INDEXED_W2_FASTPATH
    ):
        return False
    return hasattr(torch.ops._C, "fp8_moe_single_token_indexed_dense_stage_sm70_out")


class Fp8SM70MoEMethod(FusedMoEMethodBase):
    """SM70 FP8 MoE path backed by TurboMind kernels.

    The default production lane is native batched FP8 MoE. Extra decode
    shortcuts are wired here but remain independently gated until their FP8
    numeric and model-level quality evidence is accepted.
    """

    def __init__(self, quant_config, layer: RoutedExperts):
        super().__init__(layer.moe_config)
        self.quant_config = quant_config
        self.weight_block_size = quant_config.weight_block_size
        self.block_quant = self.weight_block_size is not None
        self.weight_scale_name = (
            "weight_scale_inv" if self.block_quant else "weight_scale"
        )
        if not self.block_quant:
            raise ValueError("Fp8SM70MoEMethod requires block-wise FP8 weights.")
        if tuple(self.weight_block_size) != (128, 128):
            raise ValueError(
                "Fp8SM70MoEMethod only supports FP8 block size [128, 128]."
            )
        if self.moe.has_bias:
            raise NotImplementedError("SM70 FP8 MoE does not support bias yet.")
        self.group_size = 128
        self.use_batched_gemm = envs.VLLM_SM70_FP8_MOE_BATCHED_GEMM
        self.use_batched_w13_per_expert_dispatch = (
            envs.VLLM_SM70_FP8_MOE_BATCHED_W13_PER_EXPERT_DISPATCH
        )
        self.use_batched_w2_per_expert_dispatch = (
            envs.VLLM_SM70_FP8_MOE_BATCHED_W2_PER_EXPERT_DISPATCH
        )
        self.use_permute_with_scratch = _permute_with_scratch_enabled()
        self.compact_compare_reference = bool(
            int(os.getenv("VLLM_SM70_FP8_MOE_LEGACY_SINGLE_TOKEN_COMPACT_COMPARE", "0"))
        )
        self.compact_exact_layout = bool(
            int(
                os.getenv(
                    "VLLM_SM70_FP8_MOE_LEGACY_SINGLE_TOKEN_COMPACT_EXACT_LAYOUT", "1"
                )
            )
        )
        self.compact_native_unpermute = bool(
            int(
                os.getenv(
                    "VLLM_SM70_FP8_MOE_LEGACY_SINGLE_TOKEN_COMPACT_NATIVE_UNPERMUTE",
                    "0",
                )
            )
        )
        self.compact_decomposed = bool(
            int(
                os.getenv(
                    "VLLM_SM70_FP8_MOE_LEGACY_SINGLE_TOKEN_COMPACT_DECOMPOSED", "0"
                )
            )
        )
        self.compact_compare_max_reports = int(
            os.getenv(
                "VLLM_SM70_FP8_MOE_LEGACY_SINGLE_TOKEN_COMPACT_COMPARE_REPORTS", "16"
            )
        )
        self._compact_compare_reports = 0
        if envs.VLLM_SM70_FP8_MOE_COMPACT_STRICT_COMPARE_FAIL:
            logger.warning_once(
                "VLLM_SM70_FP8_MOE_COMPACT_STRICT_COMPARE_FAIL is a no-op "
                "in latest vLLM. The old FP8 compact/router/gated MoE family "
                "is paused for quality risk; latest SM70 FP8 MoE uses the "
                "safe dense-stage/active-expert route unless an accepted "
                "replacement passes the numeric and model-level gates."
            )

    def create_weights(
        self,
        layer: RoutedExperts,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        layer.num_experts = num_experts
        layer.orig_dtype = params_dtype
        layer.weight_block_size = self.weight_block_size

        params_dtype = torch.float8_e4m3fn
        block_n, block_k = self.weight_block_size

        w13_weight = Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w13_scale = Parameter(
            torch.ones(
                num_experts,
                2 * ((intermediate_size_per_partition + block_n - 1) // block_n),
                (hidden_size + block_k - 1) // block_k,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        w2_scale = Parameter(
            torch.ones(
                num_experts,
                (hidden_size + block_n - 1) // block_n,
                (intermediate_size_per_partition + block_k - 1) // block_k,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale_inv", w13_scale)
        layer.register_parameter("w2_weight_scale_inv", w2_scale)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.BLOCK.value}
        )
        set_weight_attrs(w13_scale, extra_weight_attrs)
        set_weight_attrs(w2_scale, extra_weight_attrs)

        layer.w13_input_scale = None
        layer.w2_input_scale = None

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        w13 = layer.w13_weight
        w2 = layer.w2_weight
        w13_scale = layer.w13_weight_scale_inv.float()
        w2_scale = layer.w2_weight_scale_inv.float()

        if self.quant_config.activation_scheme == "static":
            w13_input_scale, w2_input_scale = process_fp8_input_tensor_strategy_moe(
                layer.w13_input_scale, layer.w2_input_scale
            )
            layer.w13_input_scale = w13_input_scale
            layer.w2_input_scale = w2_input_scale

        if not self.block_quant:
            shard_size = layer.intermediate_size_per_partition
            w13, w13_scale = process_fp8_weight_tensor_strategy_moe(
                w13, w13_scale, shard_size, layer.local_num_experts
            )

        num_experts = int(w13.shape[0])
        w13_tm_weights, w13_tm_scales, w13_meta = [], [], []
        w2_tm_weights, w2_tm_scales, w2_meta = [], [], []
        for expert_id in range(num_experts):
            r13 = sm70_ops.fp8_sm70_prepare(
                w13[expert_id],
                w13_scale[expert_id],
                self.group_size,
            )
            w13_tm_weights.append(r13[0])
            w13_tm_scales.append(r13[1])
            w13_meta.append(r13[2])

            r2 = sm70_ops.fp8_sm70_prepare(
                w2[expert_id],
                w2_scale[expert_id],
                self.group_size,
            )
            w2_tm_weights.append(r2[0])
            w2_tm_scales.append(r2[1])
            w2_meta.append(r2[2])

        layer.w13_tm_weight = Parameter(
            torch.stack(w13_tm_weights), requires_grad=False
        )
        layer.w13_tm_scales = Parameter(torch.stack(w13_tm_scales), requires_grad=False)
        layer.w13_tm_meta = Parameter(torch.stack(w13_meta), requires_grad=False)
        layer.w2_tm_weight = Parameter(torch.stack(w2_tm_weights), requires_grad=False)
        layer.w2_tm_scales = Parameter(torch.stack(w2_tm_scales), requires_grad=False)
        layer.w2_tm_meta = Parameter(torch.stack(w2_meta), requires_grad=False)

        w13_k_ld, w13_q_ld = int(w13_meta[0][0].item()), int(w13_meta[0][1].item())
        w2_k_ld, w2_q_ld = int(w2_meta[0][0].item()), int(w2_meta[0][1].item())
        w13_ptrs = sm70_ops.awq_moe_build_strided_ptrs(
            layer.w13_tm_weight,
            layer.w13_tm_scales,
            w13_k_ld,
            w13_q_ld,
            num_experts,
        )
        w2_ptrs = sm70_ops.awq_moe_build_strided_ptrs(
            layer.w2_tm_weight,
            layer.w2_tm_scales,
            w2_k_ld,
            w2_q_ld,
            num_experts,
        )
        layer.w13_strided_ptrs_w = Parameter(w13_ptrs[0], requires_grad=False)
        layer.w13_strided_ptrs_s = Parameter(w13_ptrs[1], requires_grad=False)
        layer.w2_strided_ptrs_w = Parameter(w2_ptrs[0], requires_grad=False)
        layer.w2_strided_ptrs_s = Parameter(w2_ptrs[1], requires_grad=False)
        ptr_row_bytes = int(layer.w13_strided_ptrs_w.numel() // num_experts)
        layer.sm70_ptr_row_bytes = ptr_row_bytes
        layer.w13_strided_ptrs_w_rows = layer.w13_strided_ptrs_w.view(
            num_experts, ptr_row_bytes
        )
        layer.w13_strided_ptrs_s_rows = layer.w13_strided_ptrs_s.view(
            num_experts, ptr_row_bytes
        )
        layer.w2_strided_ptrs_w_rows = layer.w2_strided_ptrs_w.view(
            num_experts, ptr_row_bytes
        )
        layer.w2_strided_ptrs_s_rows = layer.w2_strided_ptrs_s.view(
            num_experts, ptr_row_bytes
        )

        layer.sm70_num_experts = num_experts
        layer.sm70_hidden_logical_size = int(w2.shape[1])
        layer.sm70_w13_k_dim = int(layer.w13_tm_weight.shape[1])
        layer.sm70_w13_n_dim = int(layer.w13_tm_weight.shape[2])
        layer.sm70_w2_k_dim = int(layer.w2_tm_weight.shape[1])
        layer.sm70_w2_n_dim = int(layer.w2_tm_weight.shape[2])
        layer.sm70_intermediate_size = layer.sm70_w2_k_dim
        layer.sm70_fp8_moe_batched_gemm = self.use_batched_gemm
        layer.sm70_fp8_moe_batched_w13_per_expert_dispatch = (
            self.use_batched_w13_per_expert_dispatch
        )
        layer.sm70_fp8_moe_batched_w2_per_expert_dispatch = (
            self.use_batched_w2_per_expert_dispatch
        )
        layer.sm70_fp8_moe_permute_with_scratch = self.use_permute_with_scratch

        self._allocate_buffers(layer)
        del layer.w13_weight, layer.w2_weight
        del layer.w13_weight_scale_inv, layer.w2_weight_scale_inv
        logger.info_once(
            "SM70 FP8 MoE TurboMind %s path enabled (%d experts).",
            "batched" if self.use_batched_gemm else "per-expert dense",
            num_experts,
        )

    @eager_break_during_capture
    def _dense_moe_gemm_out(
        self,
        out: torch.Tensor,
        sorted_input: torch.Tensor,
        expert_offsets64: torch.Tensor,
        tm_weight: torch.Tensor,
        tm_scales: torch.Tensor,
        meta: torch.Tensor,
    ) -> None:
        offsets = expert_offsets64.detach().cpu().tolist()
        for expert_id in range(len(offsets) - 1):
            start = offsets[expert_id]
            end = offsets[expert_id + 1]
            if start == end:
                continue
            sm70_ops.fp8_gemm_sm70_out_meta(
                out[start:end],
                sorted_input[start:end],
                tm_weight[expert_id],
                tm_scales[expert_id],
                meta[expert_id],
            )

    def _allocate_buffers(self, layer: RoutedExperts) -> None:
        device = layer.w13_tm_weight.device
        top_k = self.moe.experts_per_token
        persistent_tokens = _DEFAULT_PERSISTENT_MAX_TOKENS
        max_slots = persistent_tokens * top_k
        hidden_size = layer.sm70_hidden_logical_size
        num_experts = layer.sm70_num_experts
        layer._fp8_buf_max_tokens = persistent_tokens
        layer._fp8_buf_max_slots = max_slots
        layer._fp8_buf_top_k = top_k
        layer._fp8_buf_output = torch.empty(
            persistent_tokens, hidden_size, dtype=torch.float16, device=device
        )
        layer._fp8_buf_permuted_input = torch.empty(
            max_slots, hidden_size, dtype=torch.float16, device=device
        )
        layer._fp8_buf_intermediate = torch.empty(
            max_slots, layer.sm70_intermediate_size, dtype=torch.float16, device=device
        )
        layer._fp8_buf_gate_up = torch.empty(
            max_slots, layer.sm70_w13_n_dim, dtype=torch.float16, device=device
        )
        layer._fp8_buf_sorted_output = torch.empty(
            max_slots, hidden_size, dtype=torch.float16, device=device
        )
        layer._fp8_buf_expert_offsets = torch.empty(
            num_experts + 1, dtype=torch.int32, device=device
        )
        layer._fp8_buf_expert_offsets64 = torch.empty(
            num_experts + 1, dtype=torch.int64, device=device
        )
        layer._fp8_buf_inv_permuted_idx = torch.empty(
            persistent_tokens, top_k, dtype=torch.int32, device=device
        )
        layer._fp8_buf_topk_ids = torch.empty(
            persistent_tokens, top_k, dtype=torch.int32, device=device
        )
        layer._fp8_buf_token_expert_indices = torch.arange(
            max_slots, dtype=torch.int32, device=device
        ).view(persistent_tokens, top_k)
        layer._fp8_buf_permuted_idx = torch.empty(
            max_slots, dtype=torch.int32, device=device
        )
        layer._fp8_buf_sorted_expert_ids = torch.empty(
            max_slots, dtype=torch.int32, device=device
        )
        if self.use_permute_with_scratch:
            sort_workspace_size = torch.ops._moe_C.moe_permute_sort_workspace_size(
                max_slots, layer.global_num_experts
            )
        else:
            sort_workspace_size = 0
        layer._fp8_buf_sort_workspace = torch.empty(
            sort_workspace_size, dtype=torch.int8, device=device
        )
        layer._fp8_buf_permuted_experts_id = torch.empty(
            max_slots, dtype=torch.int32, device=device
        )
        layer._fp8_buf_sorted_row_idx = torch.empty(
            max_slots, dtype=torch.int32, device=device
        )
        layer._fp8_buf_topk_ids_for_sort = torch.empty(
            max_slots, dtype=torch.int32, device=device
        )
        layer._fp8_buf_active_expert_offsets = torch.arange(
            max_slots + 1, dtype=torch.int32, device=device
        )
        layer._fp8_buf_sorted_weights = torch.empty(
            top_k, dtype=torch.float32, device=device
        )
        layer._fp8_buf_broadcast_input_indices = torch.empty(
            top_k, dtype=torch.int32, device=device
        )
        layer._fp8_buf_dense_expert_ids = torch.arange(
            num_experts, dtype=torch.int32, device=device
        )
        ptr_row_bytes = int(layer.sm70_ptr_row_bytes)
        layer._fp8_buf_compact_w13_ptrs_w = torch.empty(
            top_k * ptr_row_bytes, dtype=torch.uint8, device=device
        )
        layer._fp8_buf_compact_w13_ptrs_s = torch.empty(
            top_k * ptr_row_bytes, dtype=torch.uint8, device=device
        )
        layer._fp8_buf_legacy_w13_ptrs_w = torch.empty(
            top_k, ptr_row_bytes, dtype=torch.uint8, device=device
        )
        layer._fp8_buf_legacy_w13_ptrs_s = torch.empty(
            top_k, ptr_row_bytes, dtype=torch.uint8, device=device
        )
        layer._fp8_buf_legacy_w2_ptrs_w = torch.empty(
            top_k, ptr_row_bytes, dtype=torch.uint8, device=device
        )
        layer._fp8_buf_legacy_w2_ptrs_s = torch.empty(
            top_k, ptr_row_bytes, dtype=torch.uint8, device=device
        )
        layer._fp8_buf_empty_weight = torch.empty(
            0, dtype=torch.float8_e4m3fn, device=device
        )
        layer._fp8_buf_empty_scale = torch.empty(0, dtype=torch.float32, device=device)

    def _apply_batched_reference_for_compare(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids_i32: torch.Tensor,
        buffers: dict[str, torch.Tensor],
        top_k: int,
    ) -> dict[str, torch.Tensor]:
        num_tokens = x.shape[0]
        total_slots = num_tokens * top_k
        ref_permuted_input = torch.empty_like(buffers["permuted_input"])
        ref_gate_up = torch.empty_like(buffers["gate_up"])
        ref_intermediate = torch.empty_like(buffers["intermediate"])
        ref_sorted_output = torch.empty_like(buffers["sorted_output"])
        ref_output = torch.empty_like(buffers["output"])
        ref_output.zero_()
        ref_expert_offsets = torch.empty_like(buffers["expert_offsets"])
        ref_expert_offsets64 = torch.empty_like(buffers["expert_offsets64"])
        ref_inv_permuted_idx = torch.empty_like(buffers["inv_permuted_idx"])
        ref_permuted_idx = torch.empty_like(buffers["permuted_idx"])

        if layer.sm70_fp8_moe_permute_with_scratch:
            ref_permuted_idx.fill_(total_slots)
            torch.ops._moe_C.moe_permute_with_scratch(
                x,
                topk_ids_i32,
                buffers["token_expert_indices"],
                layer.expert_map,
                layer.global_num_experts,
                layer.local_num_experts,
                top_k,
                ref_permuted_input,
                ref_expert_offsets64,
                ref_inv_permuted_idx,
                ref_permuted_idx,
                torch.empty_like(buffers["sort_workspace"]),
                torch.empty_like(buffers["permuted_experts_id"]),
                torch.empty_like(buffers["sorted_row_idx"]),
                torch.empty_like(buffers["topk_ids_for_sort"]),
            )
        else:
            torch.ops._moe_C.moe_permute(
                x,
                topk_ids_i32,
                buffers["token_expert_indices"],
                layer.expert_map,
                layer.global_num_experts,
                layer.local_num_experts,
                top_k,
                ref_permuted_input,
                ref_expert_offsets64,
                ref_inv_permuted_idx,
                ref_permuted_idx,
            )
        ref_expert_offsets.copy_(ref_expert_offsets64, non_blocking=True)

        route_plan = select_sm70_quantized_moe_route(
            batched_enabled=layer.sm70_fp8_moe_batched_gemm,
            num_tokens=num_tokens,
            total_slots=total_slots,
            w13_per_expert_dispatch=(
                layer.sm70_fp8_moe_batched_w13_per_expert_dispatch
            ),
            w2_per_expert_dispatch=(layer.sm70_fp8_moe_batched_w2_per_expert_dispatch),
        )
        if route_plan.w13 == Sm70MoeStageRoute.PER_EXPERT_DISPATCH:
            sm70_ops.fp8_moe_gemm_sm70_per_expert_dispatch_out(
                ref_gate_up,
                ref_permuted_input,
                ref_expert_offsets,
                layer.w13_strided_ptrs_w,
                layer.w13_strided_ptrs_s,
                layer.sm70_num_experts,
                layer.sm70_w13_k_dim,
                layer.sm70_w13_n_dim,
                self.group_size,
                False,
            )
        elif route_plan.w13 == Sm70MoeStageRoute.BATCHED:
            sm70_ops.fp8_moe_gemm_sm70_out(
                ref_gate_up,
                ref_permuted_input,
                ref_expert_offsets,
                layer.w13_strided_ptrs_w,
                layer.w13_strided_ptrs_s,
                layer.sm70_num_experts,
                layer.sm70_w13_k_dim,
                layer.sm70_w13_n_dim,
                self.group_size,
                False,
            )
        else:
            sm70_ops.fp8_moe_dense_stage_sm70_out(
                ref_gate_up,
                ref_permuted_input,
                ref_expert_offsets,
                layer._fp8_buf_dense_expert_ids,
                layer.w13_strided_ptrs_w,
                layer.w13_strided_ptrs_s,
                layer.sm70_num_experts,
                layer.sm70_w13_k_dim,
                layer.sm70_w13_n_dim,
                self.group_size,
            )
        torch.ops._C.silu_and_mul(ref_intermediate, ref_gate_up)
        if route_plan.w2 == Sm70MoeStageRoute.PER_EXPERT_DISPATCH:
            sm70_ops.fp8_moe_gemm_sm70_per_expert_dispatch_out(
                ref_sorted_output,
                ref_intermediate,
                ref_expert_offsets,
                layer.w2_strided_ptrs_w,
                layer.w2_strided_ptrs_s,
                layer.sm70_num_experts,
                layer.sm70_w2_k_dim,
                layer.sm70_w2_n_dim,
                self.group_size,
                False,
            )
        elif route_plan.w2 == Sm70MoeStageRoute.BATCHED:
            sm70_ops.fp8_moe_gemm_sm70_out(
                ref_sorted_output,
                ref_intermediate,
                ref_expert_offsets,
                layer.w2_strided_ptrs_w,
                layer.w2_strided_ptrs_s,
                layer.sm70_num_experts,
                layer.sm70_w2_k_dim,
                layer.sm70_w2_n_dim,
                self.group_size,
                False,
            )
        else:
            sm70_ops.fp8_moe_dense_stage_sm70_out(
                ref_sorted_output,
                ref_intermediate,
                ref_expert_offsets,
                layer._fp8_buf_dense_expert_ids,
                layer.w2_strided_ptrs_w,
                layer.w2_strided_ptrs_s,
                layer.sm70_num_experts,
                layer.sm70_w2_k_dim,
                layer.sm70_w2_n_dim,
                self.group_size,
            )
        torch.ops._moe_C.moe_unpermute(
            ref_sorted_output,
            topk_weights,
            ref_inv_permuted_idx,
            ref_expert_offsets64,
            top_k,
            ref_output,
        )
        return {
            "permuted_input": ref_permuted_input,
            "expert_offsets": ref_expert_offsets,
            "expert_offsets64": ref_expert_offsets64,
            "inv_permuted_idx": ref_inv_permuted_idx,
            "gate_up": ref_gate_up,
            "intermediate": ref_intermediate,
            "sorted_output": ref_sorted_output,
            "output": ref_output,
        }

    def _apply_compact_reference_for_compare(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids_i32: torch.Tensor,
        buffers: dict[str, torch.Tensor],
        top_k: int,
    ) -> dict[str, torch.Tensor]:
        ref_permuted_input = torch.empty_like(buffers["permuted_input"])
        ref_gate_up = torch.empty_like(buffers["gate_up"])
        ref_intermediate = torch.empty_like(buffers["intermediate"])
        ref_sorted_output = torch.empty_like(buffers["sorted_output"])
        ref_output = torch.empty_like(buffers["output"])
        ref_output.zero_()
        ref_expert_offsets = torch.empty_like(buffers["expert_offsets"])
        ref_expert_offsets64 = torch.empty_like(buffers["expert_offsets64"])
        ref_inv_permuted_idx = torch.empty_like(buffers["inv_permuted_idx"])

        sm70_ops.awq_moe_single_token_exact_layout_prepare(
            topk_ids_i32,
            x,
            ref_permuted_input,
            ref_expert_offsets,
            ref_expert_offsets64,
            ref_inv_permuted_idx,
            layer.sm70_num_experts,
        )
        sm70_ops.fp8_moe_gemm_sm70_out(
            ref_gate_up,
            ref_permuted_input,
            ref_expert_offsets,
            layer.w13_strided_ptrs_w,
            layer.w13_strided_ptrs_s,
            layer.sm70_num_experts,
            layer.sm70_w13_k_dim,
            layer.sm70_w13_n_dim,
            self.group_size,
            False,
        )
        torch.ops._C.silu_and_mul(ref_intermediate, ref_gate_up)
        sm70_ops.fp8_moe_gemm_sm70_out(
            ref_sorted_output,
            ref_intermediate,
            ref_expert_offsets,
            layer.w2_strided_ptrs_w,
            layer.w2_strided_ptrs_s,
            layer.sm70_num_experts,
            layer.sm70_w2_k_dim,
            layer.sm70_w2_n_dim,
            self.group_size,
            False,
        )
        torch.ops._moe_C.moe_unpermute(
            ref_sorted_output,
            topk_weights,
            ref_inv_permuted_idx,
            ref_expert_offsets64,
            top_k,
            ref_output,
        )
        return {
            "permuted_input": ref_permuted_input,
            "expert_offsets": ref_expert_offsets,
            "expert_offsets64": ref_expert_offsets64,
            "inv_permuted_idx": ref_inv_permuted_idx,
            "gate_up": ref_gate_up,
            "intermediate": ref_intermediate,
            "sorted_output": ref_sorted_output,
            "output": ref_output,
        }

    def _maybe_report_compare(
        self,
        layer: RoutedExperts,
        prefix: str,
        reference_tensors: dict[str, torch.Tensor],
        actual_tensors: dict[str, torch.Tensor],
        topk_ids_i32: torch.Tensor,
    ) -> None:
        if self._compact_compare_reports >= self.compact_compare_max_reports:
            return

        def _max_diff(name: str) -> float:
            actual = actual_tensors[name]
            expected = reference_tensors[name]
            return float((actual - expected).abs().max().item())

        logger.warning(
            "SM70 FP8 %s compare: layer=%s report=%d "
            "perm=%g off_eq=%s off64_eq=%s inv_eq=%s "
            "w13=%g silu=%g w2=%g out=%g topk_ids=%s",
            prefix,
            getattr(layer, "layer_name", "<unknown>"),
            self._compact_compare_reports,
            _max_diff("permuted_input"),
            torch.equal(
                actual_tensors["expert_offsets"],
                reference_tensors["expert_offsets"],
            ),
            torch.equal(
                actual_tensors["expert_offsets64"],
                reference_tensors["expert_offsets64"],
            ),
            torch.equal(
                actual_tensors["inv_permuted_idx"],
                reference_tensors["inv_permuted_idx"],
            ),
            _max_diff("gate_up"),
            _max_diff("intermediate"),
            _max_diff("sorted_output"),
            _max_diff("output"),
            topk_ids_i32.detach().cpu().view(-1).tolist(),
        )
        self._compact_compare_reports += 1

    def _get_buffers(
        self, layer: RoutedExperts, total_slots: int, num_tokens: int
    ) -> dict[str, torch.Tensor]:
        if (
            total_slots <= layer._fp8_buf_max_slots
            and num_tokens <= layer._fp8_buf_max_tokens
        ):
            return {
                "output": layer._fp8_buf_output[:num_tokens],
                "permuted_input": layer._fp8_buf_permuted_input[:total_slots],
                "intermediate": layer._fp8_buf_intermediate[:total_slots],
                "gate_up": layer._fp8_buf_gate_up[:total_slots],
                "sorted_output": layer._fp8_buf_sorted_output[:total_slots],
                "expert_offsets": layer._fp8_buf_expert_offsets,
                "expert_offsets64": layer._fp8_buf_expert_offsets64,
                "inv_permuted_idx": layer._fp8_buf_inv_permuted_idx[:num_tokens],
                "topk_ids": layer._fp8_buf_topk_ids[:num_tokens],
                "token_expert_indices": layer._fp8_buf_token_expert_indices[
                    :num_tokens
                ],
                "permuted_idx": layer._fp8_buf_permuted_idx[:total_slots],
                "sorted_expert_ids": layer._fp8_buf_sorted_expert_ids[:total_slots],
                "sort_workspace": layer._fp8_buf_sort_workspace,
                "permuted_experts_id": layer._fp8_buf_permuted_experts_id[:total_slots],
                "sorted_row_idx": layer._fp8_buf_sorted_row_idx[:total_slots],
                "topk_ids_for_sort": layer._fp8_buf_topk_ids_for_sort[:total_slots],
                "active_expert_offsets": (
                    layer._fp8_buf_active_expert_offsets[: total_slots + 1]
                ),
                "sorted_weights": layer._fp8_buf_sorted_weights,
                "broadcast_input_indices": layer._fp8_buf_broadcast_input_indices,
                "compact_w13_ptrs_w": layer._fp8_buf_compact_w13_ptrs_w,
                "compact_w13_ptrs_s": layer._fp8_buf_compact_w13_ptrs_s,
                "legacy_w13_ptrs_w": layer._fp8_buf_legacy_w13_ptrs_w,
                "legacy_w13_ptrs_s": layer._fp8_buf_legacy_w13_ptrs_s,
                "legacy_w2_ptrs_w": layer._fp8_buf_legacy_w2_ptrs_w,
                "legacy_w2_ptrs_s": layer._fp8_buf_legacy_w2_ptrs_s,
                "empty_weight": layer._fp8_buf_empty_weight,
                "empty_scale": layer._fp8_buf_empty_scale,
            }

        device = layer._fp8_buf_output.device
        top_k = layer._fp8_buf_top_k
        hidden_size = layer.sm70_hidden_logical_size
        if self.use_permute_with_scratch:
            sort_workspace_size = torch.ops._moe_C.moe_permute_sort_workspace_size(
                total_slots, layer.global_num_experts
            )
            sort_workspace = torch.empty(
                sort_workspace_size, dtype=torch.int8, device=device
            )
            active_expert_offsets = torch.arange(
                total_slots + 1, dtype=torch.int32, device=device
            )
        else:
            sort_workspace = layer._fp8_buf_sort_workspace
            active_expert_offsets = layer._fp8_buf_active_expert_offsets[
                : total_slots + 1
            ]
        return {
            "output": torch.empty(
                num_tokens, hidden_size, dtype=torch.float16, device=device
            ),
            "permuted_input": torch.empty(
                total_slots, hidden_size, dtype=torch.float16, device=device
            ),
            "intermediate": torch.empty(
                total_slots,
                layer.sm70_intermediate_size,
                dtype=torch.float16,
                device=device,
            ),
            "gate_up": torch.empty(
                total_slots,
                layer.sm70_w13_n_dim,
                dtype=torch.float16,
                device=device,
            ),
            "sorted_output": torch.empty(
                total_slots, hidden_size, dtype=torch.float16, device=device
            ),
            "expert_offsets": torch.empty(
                layer.sm70_num_experts + 1, dtype=torch.int32, device=device
            ),
            "expert_offsets64": torch.empty(
                layer.sm70_num_experts + 1, dtype=torch.int64, device=device
            ),
            "inv_permuted_idx": torch.empty(
                num_tokens, top_k, dtype=torch.int32, device=device
            ),
            "topk_ids": torch.empty(
                num_tokens, top_k, dtype=torch.int32, device=device
            ),
            "token_expert_indices": torch.arange(
                total_slots, dtype=torch.int32, device=device
            ).view(num_tokens, top_k),
            "permuted_idx": torch.empty(total_slots, dtype=torch.int32, device=device),
            "sorted_expert_ids": torch.empty(
                total_slots, dtype=torch.int32, device=device
            ),
            "sort_workspace": sort_workspace,
            "permuted_experts_id": torch.empty(
                total_slots, dtype=torch.int32, device=device
            ),
            "sorted_row_idx": torch.empty(
                total_slots, dtype=torch.int32, device=device
            ),
            "topk_ids_for_sort": torch.empty(
                total_slots, dtype=torch.int32, device=device
            ),
            "active_expert_offsets": active_expert_offsets,
            "sorted_weights": layer._fp8_buf_sorted_weights,
            "broadcast_input_indices": layer._fp8_buf_broadcast_input_indices,
            "compact_w13_ptrs_w": layer._fp8_buf_compact_w13_ptrs_w,
            "compact_w13_ptrs_s": layer._fp8_buf_compact_w13_ptrs_s,
            "legacy_w13_ptrs_w": layer._fp8_buf_legacy_w13_ptrs_w,
            "legacy_w13_ptrs_s": layer._fp8_buf_legacy_w13_ptrs_s,
            "legacy_w2_ptrs_w": layer._fp8_buf_legacy_w2_ptrs_w,
            "legacy_w2_ptrs_s": layer._fp8_buf_legacy_w2_ptrs_s,
            "empty_weight": layer._fp8_buf_empty_weight,
            "empty_scale": layer._fp8_buf_empty_scale,
        }

    @property
    def supports_eplb(self) -> bool:
        return False

    def _apply_legacy_single_token_compact(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids_i32: torch.Tensor,
        buffers: dict[str, torch.Tensor],
        top_k: int,
        output: torch.Tensor,
    ) -> torch.Tensor:
        exact_layout = self.compact_exact_layout
        _log_runtime_route_once(
            "SM70 FP8 MoE legacy single-token %s compact path enabled "
            "(top_k=%d, experts=%d).",
            "exact-layout" if exact_layout else "top-k descriptor",
            top_k,
            layer.sm70_num_experts,
        )
        reference_tensors = None
        if self.compact_compare_reference:
            reference_tensors = self._apply_batched_reference_for_compare(
                layer, x, topk_weights, topk_ids_i32, buffers, top_k
            )
        if self.compact_decomposed:
            _log_runtime_route_once(
                "SM70 FP8 MoE legacy single-token exact-layout decomposed "
                "compact path enabled (top_k=%d, experts=%d).",
                top_k,
                layer.sm70_num_experts,
            )
            sm70_ops.awq_moe_single_token_exact_layout_prepare(
                topk_ids_i32,
                x,
                buffers["permuted_input"],
                buffers["expert_offsets"],
                buffers["expert_offsets64"],
                buffers["inv_permuted_idx"],
                layer.sm70_num_experts,
            )
            sm70_ops.fp8_moe_gemm_sm70_out(
                buffers["gate_up"],
                buffers["permuted_input"],
                buffers["expert_offsets"],
                layer.w13_strided_ptrs_w,
                layer.w13_strided_ptrs_s,
                layer.sm70_num_experts,
                layer.sm70_w13_k_dim,
                layer.sm70_w13_n_dim,
                self.group_size,
                False,
            )
            torch.ops._C.silu_and_mul(buffers["intermediate"], buffers["gate_up"])
            sm70_ops.fp8_moe_gemm_sm70_out(
                buffers["sorted_output"],
                buffers["intermediate"],
                buffers["expert_offsets"],
                layer.w2_strided_ptrs_w,
                layer.w2_strided_ptrs_s,
                layer.sm70_num_experts,
                layer.sm70_w2_k_dim,
                layer.sm70_w2_n_dim,
                self.group_size,
                False,
            )
            output.zero_()
            torch.ops._moe_C.moe_unpermute(
                buffers["sorted_output"],
                topk_weights,
                buffers["inv_permuted_idx"],
                buffers["expert_offsets64"],
                top_k,
                output,
            )
            return output
        sm70_ops.fp8_moe_single_token_sm70_out(
            output,
            x,
            topk_weights,
            topk_ids_i32,
            layer.w13_strided_ptrs_w_rows,
            layer.w13_strided_ptrs_s_rows,
            layer.w2_strided_ptrs_w_rows,
            layer.w2_strided_ptrs_s_rows,
            buffers["permuted_input"],
            buffers["gate_up"],
            buffers["intermediate"],
            buffers["sorted_output"],
            buffers["sorted_weights"],
            buffers["legacy_w13_ptrs_w"],
            buffers["legacy_w13_ptrs_s"],
            buffers["legacy_w2_ptrs_w"],
            buffers["legacy_w2_ptrs_s"],
            buffers["expert_offsets"]
            if exact_layout
            else buffers["active_expert_offsets"],
            buffers["inv_permuted_idx"],
            buffers["sorted_expert_ids"],
            buffers["broadcast_input_indices"],
            buffers["empty_weight"],
            buffers["empty_scale"],
            layer.sm70_w13_k_dim,
            layer.sm70_w13_n_dim,
            layer.sm70_w2_k_dim,
            layer.sm70_w2_n_dim,
            self.group_size,
            layer.sm70_hidden_logical_size,
            False,
            False,
            False,
            False,
            False,
            exact_layout,
        )
        if self.compact_native_unpermute:
            output.zero_()
            torch.ops._moe_C.moe_unpermute(
                buffers["sorted_output"],
                topk_weights,
                buffers["inv_permuted_idx"],
                None,
                top_k,
                output,
            )
        if reference_tensors is not None:
            actual_expert_offsets = (
                buffers["expert_offsets"]
                if exact_layout
                else buffers["active_expert_offsets"]
            )
            self._maybe_report_compare(
                layer,
                "compact",
                reference_tensors,
                {
                    "permuted_input": buffers["permuted_input"],
                    "expert_offsets": actual_expert_offsets,
                    "expert_offsets64": actual_expert_offsets.to(dtype=torch.int64),
                    "inv_permuted_idx": buffers["inv_permuted_idx"],
                    "gate_up": buffers["gate_up"],
                    "intermediate": buffers["intermediate"],
                    "sorted_output": buffers["sorted_output"],
                    "output": output,
                },
                topk_ids_i32,
            )
        return output

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        del shared_experts, shared_experts_input
        if layer.apply_router_weight_on_input:
            raise NotImplementedError(
                "SM70 FP8 MoE does not support apply_router_weight_on_input yet."
            )

        num_tokens = x.shape[0]
        top_k = topk_ids.shape[1]
        total_slots = num_tokens * top_k
        buffers = self._get_buffers(layer, total_slots, num_tokens)
        output = buffers["output"]
        output.zero_()
        if total_slots == 0:
            return output

        topk_ids_i32 = buffers["topk_ids"]
        topk_ids_i32.copy_(topk_ids, non_blocking=True)
        if (
            num_tokens == 1
            and layer.sm70_fp8_moe_batched_gemm
            and _legacy_single_token_compact_enabled()
        ):
            return self._apply_legacy_single_token_compact(
                layer, x, topk_weights, topk_ids_i32, buffers, top_k, output
            )
        if num_tokens == 1 and not layer.sm70_fp8_moe_batched_gemm:
            use_compact_w13 = _single_token_compact_w13_enabled()
            use_indexed_w13 = (
                not use_compact_w13 and _single_token_indexed_w13_enabled()
            )
            use_indexed_w2 = _single_token_indexed_w2_enabled()
            _log_runtime_route_once(
                "SM70 FP8 MoE single-token active-expert dense path enabled "
                "(top_k=%d, experts=%d).",
                top_k,
                layer.sm70_num_experts,
            )
            if use_indexed_w13 or use_indexed_w2:
                _log_runtime_route_once(
                    "SM70 FP8 MoE single-token indexed dense-stage path "
                    "enabled (top_k=%d, w13=%s, w2=%s).",
                    top_k,
                    use_indexed_w13,
                    use_indexed_w2,
                )
            if use_compact_w13:
                _log_runtime_route_once(
                    "SM70 FP8 MoE single-token compact grouped W13 path "
                    "enabled (top_k=%d).",
                    top_k,
                )
                sm70_ops.fp8_moe_single_token_compact_dense_w13_sm70_out(
                    buffers["gate_up"],
                    buffers["permuted_input"],
                    x,
                    topk_ids_i32,
                    layer.w13_strided_ptrs_w,
                    layer.w13_strided_ptrs_s,
                    buffers["compact_w13_ptrs_w"],
                    buffers["compact_w13_ptrs_s"],
                    buffers["expert_offsets"],
                    buffers["expert_offsets64"],
                    buffers["inv_permuted_idx"],
                    buffers["sorted_expert_ids"],
                    layer.sm70_w13_k_dim,
                    layer.sm70_w13_n_dim,
                    self.group_size,
                    layer.sm70_hidden_logical_size,
                )
            elif use_indexed_w13:
                sm70_ops.fp8_moe_single_token_indexed_dense_w13_sm70_out(
                    buffers["gate_up"],
                    buffers["permuted_input"],
                    x,
                    topk_ids_i32,
                    layer.w13_strided_ptrs_w,
                    layer.w13_strided_ptrs_s,
                    buffers["expert_offsets"],
                    buffers["expert_offsets64"],
                    buffers["inv_permuted_idx"],
                    buffers["sorted_expert_ids"],
                    layer.sm70_w13_k_dim,
                    layer.sm70_w13_n_dim,
                    self.group_size,
                    layer.sm70_hidden_logical_size,
                )
            else:
                sm70_ops.fp8_moe_single_token_dense_w13_sm70_out(
                    buffers["gate_up"],
                    buffers["permuted_input"],
                    x,
                    topk_ids_i32,
                    layer.w13_strided_ptrs_w,
                    layer.w13_strided_ptrs_s,
                    buffers["expert_offsets"],
                    buffers["expert_offsets64"],
                    buffers["inv_permuted_idx"],
                    buffers["sorted_expert_ids"],
                    layer.sm70_w13_k_dim,
                    layer.sm70_w13_n_dim,
                    self.group_size,
                    layer.sm70_hidden_logical_size,
                )
            torch.ops._C.silu_and_mul(buffers["intermediate"], buffers["gate_up"])
            if use_indexed_w2:
                sm70_ops.fp8_moe_single_token_indexed_dense_stage_sm70_out(
                    buffers["sorted_output"],
                    buffers["intermediate"],
                    buffers["expert_offsets"],
                    buffers["sorted_expert_ids"],
                    layer.w2_strided_ptrs_w,
                    layer.w2_strided_ptrs_s,
                    top_k,
                    layer.sm70_w2_k_dim,
                    layer.sm70_w2_n_dim,
                    self.group_size,
                )
            else:
                sm70_ops.fp8_moe_single_token_dense_stage_sm70_out(
                    buffers["sorted_output"],
                    buffers["intermediate"],
                    buffers["expert_offsets"],
                    buffers["sorted_expert_ids"],
                    layer.w2_strided_ptrs_w,
                    layer.w2_strided_ptrs_s,
                    top_k,
                    layer.sm70_w2_k_dim,
                    layer.sm70_w2_n_dim,
                    self.group_size,
                )
            if _single_token_weighted_reduce_enabled():
                _log_runtime_route_once(
                    "SM70 FP8 MoE single-token weighted-reduce path enabled "
                    "(top_k=%d).",
                    top_k,
                )
                sm70_ops.awq_moe_single_token_weighted_reduce_out(
                    buffers["sorted_output"],
                    topk_weights,
                    buffers["inv_permuted_idx"],
                    output,
                    top_k,
                    layer.sm70_hidden_logical_size,
                )
            else:
                torch.ops._moe_C.moe_unpermute(
                    buffers["sorted_output"],
                    topk_weights,
                    buffers["inv_permuted_idx"],
                    buffers["expert_offsets64"][: top_k + 1],
                    top_k,
                    output,
                )
            return output
        if layer.sm70_fp8_moe_permute_with_scratch:
            buffers["permuted_idx"].fill_(total_slots)
            torch.ops._moe_C.moe_permute_with_scratch(
                x,
                topk_ids_i32,
                buffers["token_expert_indices"],
                layer.expert_map,
                layer.global_num_experts,
                layer.local_num_experts,
                top_k,
                buffers["permuted_input"],
                buffers["expert_offsets64"],
                buffers["inv_permuted_idx"],
                buffers["permuted_idx"],
                buffers["sort_workspace"],
                buffers["permuted_experts_id"],
                buffers["sorted_row_idx"],
                buffers["topk_ids_for_sort"],
            )
        else:
            torch.ops._moe_C.moe_permute(
                x,
                topk_ids_i32,
                buffers["token_expert_indices"],
                layer.expert_map,
                layer.global_num_experts,
                layer.local_num_experts,
                top_k,
                buffers["permuted_input"],
                buffers["expert_offsets64"],
                buffers["inv_permuted_idx"],
                buffers["permuted_idx"],
            )
        buffers["expert_offsets"].copy_(buffers["expert_offsets64"], non_blocking=True)

        route_plan = select_sm70_quantized_moe_route(
            batched_enabled=layer.sm70_fp8_moe_batched_gemm,
            num_tokens=num_tokens,
            total_slots=total_slots,
            w13_per_expert_dispatch=(
                layer.sm70_fp8_moe_batched_w13_per_expert_dispatch
            ),
            w2_per_expert_dispatch=(layer.sm70_fp8_moe_batched_w2_per_expert_dispatch),
        )

        if route_plan.w13 == Sm70MoeStageRoute.PER_EXPERT_DISPATCH:
            _log_runtime_route_once(
                "SM70 FP8 MoE batched W13 using per-expert dispatch "
                "selection (experts=%d).",
                layer.sm70_num_experts,
            )
            sm70_ops.fp8_moe_gemm_sm70_per_expert_dispatch_out(
                buffers["gate_up"],
                buffers["permuted_input"],
                buffers["expert_offsets"],
                layer.w13_strided_ptrs_w,
                layer.w13_strided_ptrs_s,
                layer.sm70_num_experts,
                layer.sm70_w13_k_dim,
                layer.sm70_w13_n_dim,
                self.group_size,
                False,
            )
        elif route_plan.w13 == Sm70MoeStageRoute.BATCHED:
            sm70_ops.fp8_moe_gemm_sm70_out(
                buffers["gate_up"],
                buffers["permuted_input"],
                buffers["expert_offsets"],
                layer.w13_strided_ptrs_w,
                layer.w13_strided_ptrs_s,
                layer.sm70_num_experts,
                layer.sm70_w13_k_dim,
                layer.sm70_w13_n_dim,
                self.group_size,
                False,
            )
        else:
            _log_runtime_route_once(
                "SM70 FP8 MoE CUDA-graph-safe dense-stage path enabled (experts=%d).",
                layer.sm70_num_experts,
            )
            sm70_ops.fp8_moe_dense_stage_sm70_out(
                buffers["gate_up"],
                buffers["permuted_input"],
                buffers["expert_offsets"],
                layer._fp8_buf_dense_expert_ids,
                layer.w13_strided_ptrs_w,
                layer.w13_strided_ptrs_s,
                layer.sm70_num_experts,
                layer.sm70_w13_k_dim,
                layer.sm70_w13_n_dim,
                self.group_size,
            )
        torch.ops._C.silu_and_mul(buffers["intermediate"], buffers["gate_up"])
        if route_plan.w2 == Sm70MoeStageRoute.PER_EXPERT_DISPATCH:
            _log_runtime_route_once(
                "SM70 FP8 MoE batched W2 using per-expert dispatch "
                "selection (experts=%d).",
                layer.sm70_num_experts,
            )
            sm70_ops.fp8_moe_gemm_sm70_per_expert_dispatch_out(
                buffers["sorted_output"],
                buffers["intermediate"],
                buffers["expert_offsets"],
                layer.w2_strided_ptrs_w,
                layer.w2_strided_ptrs_s,
                layer.sm70_num_experts,
                layer.sm70_w2_k_dim,
                layer.sm70_w2_n_dim,
                self.group_size,
                False,
            )
        elif route_plan.w2 == Sm70MoeStageRoute.BATCHED:
            sm70_ops.fp8_moe_gemm_sm70_out(
                buffers["sorted_output"],
                buffers["intermediate"],
                buffers["expert_offsets"],
                layer.w2_strided_ptrs_w,
                layer.w2_strided_ptrs_s,
                layer.sm70_num_experts,
                layer.sm70_w2_k_dim,
                layer.sm70_w2_n_dim,
                self.group_size,
                False,
            )
        else:
            sm70_ops.fp8_moe_dense_stage_sm70_out(
                buffers["sorted_output"],
                buffers["intermediate"],
                buffers["expert_offsets"],
                layer._fp8_buf_dense_expert_ids,
                layer.w2_strided_ptrs_w,
                layer.w2_strided_ptrs_s,
                layer.sm70_num_experts,
                layer.sm70_w2_k_dim,
                layer.sm70_w2_n_dim,
                self.group_size,
            )
        torch.ops._moe_C.moe_unpermute(
            buffers["sorted_output"],
            topk_weights,
            buffers["inv_permuted_idx"],
            buffers["expert_offsets64"],
            top_k,
            output,
        )
        if (
            num_tokens == 1
            and layer.sm70_fp8_moe_batched_gemm
            and self.compact_compare_reference
            and hasattr(torch.ops._C, "awq_moe_single_token_exact_layout_prepare")
        ):
            reference_tensors = self._apply_compact_reference_for_compare(
                layer, x, topk_weights, topk_ids_i32, buffers, top_k
            )
            self._maybe_report_compare(
                layer,
                "noncompact",
                reference_tensors,
                {
                    "permuted_input": buffers["permuted_input"],
                    "expert_offsets": buffers["expert_offsets"],
                    "expert_offsets64": buffers["expert_offsets64"],
                    "inv_permuted_idx": buffers["inv_permuted_idx"],
                    "gate_up": buffers["gate_up"],
                    "intermediate": buffers["intermediate"],
                    "sorted_output": buffers["sorted_output"],
                    "output": output,
                },
                topk_ids_i32,
            )
        return output

    def apply_monolithic(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del layer, x, router_logits, input_ids
        raise NotImplementedError("SM70 FP8 MoE base path is not monolithic.")

    def get_fused_moe_quant_config(
        self, layer: RoutedExperts
    ) -> FusedMoEQuantConfig | None:
        del layer
        return None
