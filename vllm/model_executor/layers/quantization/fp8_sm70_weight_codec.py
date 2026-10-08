# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP8 weight preparation and native W13/W2 bindings for shared SM70 MoE."""

from __future__ import annotations

from types import ModuleType
from typing import TYPE_CHECKING

import torch
from torch.nn import Parameter

from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.quantization.utils.sm70_layer_workspaces import (
    LayerWorkspaceView,
)

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.fp8_sm70_moe import Fp8SM70MoEMethod

W13_OPS = {
    "compact": "fp8_moe_single_token_compact_dense_w13_sm70_out",
    "indexed": "fp8_moe_single_token_indexed_dense_w13_sm70_out",
    "active_dense": "fp8_moe_single_token_dense_w13_sm70_out",
    "per_expert": "fp8_moe_gemm_sm70_per_expert_dispatch_out",
    "batched": "fp8_moe_gemm_sm70_out",
    "dense": "fp8_moe_dense_stage_sm70_out",
}
W2_OPS = {
    "indexed": "fp8_moe_single_token_indexed_dense_stage_sm70_out",
    "active_dense": "fp8_moe_single_token_dense_stage_sm70_out",
    "per_expert": "fp8_moe_gemm_sm70_per_expert_dispatch_out",
    "batched": "fp8_moe_gemm_sm70_out",
    "dense": "fp8_moe_dense_stage_sm70_out",
}
CAPABILITIES = {
    "single_token_compact_w13": "_single_token_compact_w13_enabled",
    "single_token_indexed_w13": "_single_token_indexed_w13_enabled",
    "single_token_indexed_w2": "_single_token_indexed_w2_enabled",
    "single_token_weighted_reduce": "_single_token_weighted_reduce_enabled",
    "legacy_single_token_compact": "_legacy_single_token_compact_enabled",
}


class Fp8MoEWeightCodec:
    """Bindings resolve through the legacy module to preserve public patches."""

    def __init__(self, legacy: ModuleType):
        self._legacy = legacy

    @property
    def native_ops(self):
        return self._legacy.sm70_ops

    def policy(self, layer: RoutedExperts) -> LayerWorkspaceView:
        return LayerWorkspaceView(layer, "sm70_fp8_moe_")

    def buffer(self, layer: RoutedExperts, name: str) -> torch.Tensor:
        return getattr(layer, "_fp8_buf_" + name)

    def enabled(self, capability: str) -> bool:
        return getattr(self._legacy, CAPABILITIES[capability])()

    def message(self, suffix: str) -> str:
        return "SM70 FP8 " + suffix

    def log(self, message: str, *args) -> None:
        self._legacy._log_runtime_route_once(self.message(message), *args)

    def gemm_w13(self, mode: str, layer: RoutedExperts, *args) -> None:
        getattr(self.native_ops, W13_OPS[mode])(*args)

    def gemm_w2(self, mode: str, layer: RoutedExperts, *args) -> None:
        getattr(self.native_ops, W2_OPS[mode])(*args)

    def prepare_weights(self, method: Fp8SM70MoEMethod, layer: RoutedExperts) -> None:
        w13 = layer.w13_weight
        w2 = layer.w2_weight
        w13_scale = layer.w13_weight_scale_inv.float()
        w2_scale = layer.w2_weight_scale_inv.float()
        if method.quant_config.activation_scheme == "static":
            w13_input_scale, w2_input_scale = (
                self._legacy.process_fp8_input_tensor_strategy_moe(
                    layer.w13_input_scale, layer.w2_input_scale
                )
            )
            layer.w13_input_scale = w13_input_scale
            layer.w2_input_scale = w2_input_scale
        if not method.block_quant:
            shard_size = layer.intermediate_size_per_partition
            w13, w13_scale = self._legacy.process_fp8_weight_tensor_strategy_moe(
                w13, w13_scale, shard_size, layer.local_num_experts
            )
        num_experts = int(w13.shape[0])
        w13_tm_weights, w13_tm_scales, w13_meta = ([], [], [])
        w2_tm_weights, w2_tm_scales, w2_meta = ([], [], [])
        for expert_id in range(num_experts):
            r13 = self.native_ops.fp8_sm70_prepare(
                w13[expert_id], w13_scale[expert_id], method.group_size
            )
            w13_tm_weights.append(r13[0])
            w13_tm_scales.append(r13[1])
            w13_meta.append(r13[2])
            r2 = self.native_ops.fp8_sm70_prepare(
                w2[expert_id], w2_scale[expert_id], method.group_size
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
        w13_k_ld, w13_q_ld = (int(w13_meta[0][0].item()), int(w13_meta[0][1].item()))
        w2_k_ld, w2_q_ld = (int(w2_meta[0][0].item()), int(w2_meta[0][1].item()))
        w13_ptrs = self.native_ops.awq_moe_build_strided_ptrs(
            layer.w13_tm_weight, layer.w13_tm_scales, w13_k_ld, w13_q_ld, num_experts
        )
        w2_ptrs = self.native_ops.awq_moe_build_strided_ptrs(
            layer.w2_tm_weight, layer.w2_tm_scales, w2_k_ld, w2_q_ld, num_experts
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
        layer.sm70_fp8_moe_batched_gemm = method.use_batched_gemm
        layer.sm70_fp8_moe_batched_w13_per_expert_dispatch = (
            method.use_batched_w13_per_expert_dispatch
        )
        layer.sm70_fp8_moe_batched_w2_per_expert_dispatch = (
            method.use_batched_w2_per_expert_dispatch
        )
        layer.sm70_fp8_moe_permute_with_scratch = method.use_permute_with_scratch
        method._allocate_buffers(layer)
        del layer.w13_weight, layer.w2_weight
        del layer.w13_weight_scale_inv, layer.w2_weight_scale_inv
        self._legacy.logger.info_once(
            "SM70 FP8 MoE TurboMind %s path enabled (%d experts).",
            "batched" if method.use_batched_gemm else "per-expert dense",
            num_experts,
        )
