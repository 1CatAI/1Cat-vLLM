# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared SM70 MoE buffers and execution stages parameterized by weight codec."""

from collections.abc import Callable

import torch

from vllm.model_executor.layers.fused_moe import RoutedExperts, SharedExperts
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.sm70.weight_codec import Sm70MoEWeightCodec
from vllm.model_executor.layers.quantization.sm70_moe_router import (
    Sm70MoeStageRoute,
    select_sm70_quantized_moe_route,
)
from vllm.model_executor.layers.quantization.utils.sm70_layer_workspaces import (
    LayerWorkspaceView,
)


class Sm70MoEMethodBase(FusedMoEMethodBase):
    """Buffer lifecycle and stage scheduling independent of weight encoding.

    Legacy attributes stay on the layer, so graph addresses and patches keep
    their original owner. FP8 is the first consumer. Native GEMM bindings and
    weight preparation belong to the codec; comparison and legacy compact
    callbacks plus other format migrations are subsequent review scopes.
    """

    use_permute_with_scratch: bool
    weight_codec: Sm70MoEWeightCodec
    group_size: int
    compact_compare_reference: bool
    _get_buffers: Callable[..., dict[str, torch.Tensor]]
    _apply_legacy_single_token_compact: Callable[..., torch.Tensor]
    _apply_compact_reference_for_compare: Callable[..., dict[str, torch.Tensor]]
    _maybe_report_compare: Callable[..., None]

    def _allocate_moe_buffers(
        self,
        layer: RoutedExperts,
        *,
        buffer_prefix: str,
        persistent_max_tokens: int,
        empty_weight_dtype: torch.dtype,
        empty_scale_dtype: torch.dtype,
    ) -> None:
        buffers = LayerWorkspaceView(layer, buffer_prefix)
        device = layer.w13_tm_weight.device
        top_k = self.moe.experts_per_token
        persistent_tokens = persistent_max_tokens
        max_slots = persistent_tokens * top_k
        hidden_size = layer.sm70_hidden_logical_size
        num_experts = layer.sm70_num_experts
        buffers.max_tokens = persistent_tokens
        buffers.max_slots = max_slots
        buffers.top_k = top_k
        buffers.output = torch.empty(
            persistent_tokens, hidden_size, dtype=torch.float16, device=device
        )
        buffers.permuted_input = torch.empty(
            max_slots, hidden_size, dtype=torch.float16, device=device
        )
        buffers.intermediate = torch.empty(
            max_slots, layer.sm70_intermediate_size, dtype=torch.float16, device=device
        )
        buffers.gate_up = torch.empty(
            max_slots, layer.sm70_w13_n_dim, dtype=torch.float16, device=device
        )
        buffers.sorted_output = torch.empty(
            max_slots, hidden_size, dtype=torch.float16, device=device
        )
        buffers.expert_offsets = torch.empty(
            num_experts + 1, dtype=torch.int32, device=device
        )
        buffers.expert_offsets64 = torch.empty(
            num_experts + 1, dtype=torch.int64, device=device
        )
        buffers.inv_permuted_idx = torch.empty(
            persistent_tokens, top_k, dtype=torch.int32, device=device
        )
        buffers.topk_ids = torch.empty(
            persistent_tokens, top_k, dtype=torch.int32, device=device
        )
        buffers.token_expert_indices = torch.arange(
            max_slots, dtype=torch.int32, device=device
        ).view(persistent_tokens, top_k)
        buffers.permuted_idx = torch.empty(max_slots, dtype=torch.int32, device=device)
        buffers.sorted_expert_ids = torch.empty(
            max_slots, dtype=torch.int32, device=device
        )
        if self.use_permute_with_scratch:
            sort_workspace_size = torch.ops._moe_C.moe_permute_sort_workspace_size(
                max_slots, layer.global_num_experts
            )
        else:
            sort_workspace_size = 0
        buffers.sort_workspace = torch.empty(
            sort_workspace_size, dtype=torch.int8, device=device
        )
        buffers.permuted_experts_id = torch.empty(
            max_slots, dtype=torch.int32, device=device
        )
        buffers.sorted_row_idx = torch.empty(
            max_slots, dtype=torch.int32, device=device
        )
        buffers.topk_ids_for_sort = torch.empty(
            max_slots, dtype=torch.int32, device=device
        )
        buffers.active_expert_offsets = torch.arange(
            max_slots + 1, dtype=torch.int32, device=device
        )
        buffers.sorted_weights = torch.empty(top_k, dtype=torch.float32, device=device)
        buffers.broadcast_input_indices = torch.empty(
            top_k, dtype=torch.int32, device=device
        )
        buffers.dense_expert_ids = torch.arange(
            num_experts, dtype=torch.int32, device=device
        )
        ptr_row_bytes = int(layer.sm70_ptr_row_bytes)
        buffers.compact_w13_ptrs_w = torch.empty(
            top_k * ptr_row_bytes, dtype=torch.uint8, device=device
        )
        buffers.compact_w13_ptrs_s = torch.empty(
            top_k * ptr_row_bytes, dtype=torch.uint8, device=device
        )
        buffers.legacy_w13_ptrs_w = torch.empty(
            top_k, ptr_row_bytes, dtype=torch.uint8, device=device
        )
        buffers.legacy_w13_ptrs_s = torch.empty(
            top_k, ptr_row_bytes, dtype=torch.uint8, device=device
        )
        buffers.legacy_w2_ptrs_w = torch.empty(
            top_k, ptr_row_bytes, dtype=torch.uint8, device=device
        )
        buffers.legacy_w2_ptrs_s = torch.empty(
            top_k, ptr_row_bytes, dtype=torch.uint8, device=device
        )
        buffers.empty_weight = torch.empty(0, dtype=empty_weight_dtype, device=device)
        buffers.empty_scale = torch.empty(0, dtype=empty_scale_dtype, device=device)

    def _get_moe_buffers(
        self,
        layer: RoutedExperts,
        total_slots: int,
        num_tokens: int,
        *,
        buffer_prefix: str,
    ) -> dict[str, torch.Tensor]:
        buffers = LayerWorkspaceView(layer, buffer_prefix)
        if total_slots <= buffers.max_slots and num_tokens <= buffers.max_tokens:
            return {
                "output": buffers.output[:num_tokens],
                "permuted_input": buffers.permuted_input[:total_slots],
                "intermediate": buffers.intermediate[:total_slots],
                "gate_up": buffers.gate_up[:total_slots],
                "sorted_output": buffers.sorted_output[:total_slots],
                "expert_offsets": buffers.expert_offsets,
                "expert_offsets64": buffers.expert_offsets64,
                "inv_permuted_idx": buffers.inv_permuted_idx[:num_tokens],
                "topk_ids": buffers.topk_ids[:num_tokens],
                "token_expert_indices": buffers.token_expert_indices[:num_tokens],
                "permuted_idx": buffers.permuted_idx[:total_slots],
                "sorted_expert_ids": buffers.sorted_expert_ids[:total_slots],
                "sort_workspace": buffers.sort_workspace,
                "permuted_experts_id": buffers.permuted_experts_id[:total_slots],
                "sorted_row_idx": buffers.sorted_row_idx[:total_slots],
                "topk_ids_for_sort": buffers.topk_ids_for_sort[:total_slots],
                "active_expert_offsets": (
                    buffers.active_expert_offsets[: total_slots + 1]
                ),
                "sorted_weights": buffers.sorted_weights,
                "broadcast_input_indices": buffers.broadcast_input_indices,
                "compact_w13_ptrs_w": buffers.compact_w13_ptrs_w,
                "compact_w13_ptrs_s": buffers.compact_w13_ptrs_s,
                "legacy_w13_ptrs_w": buffers.legacy_w13_ptrs_w,
                "legacy_w13_ptrs_s": buffers.legacy_w13_ptrs_s,
                "legacy_w2_ptrs_w": buffers.legacy_w2_ptrs_w,
                "legacy_w2_ptrs_s": buffers.legacy_w2_ptrs_s,
                "empty_weight": buffers.empty_weight,
                "empty_scale": buffers.empty_scale,
            }

        device = buffers.output.device
        top_k = buffers.top_k
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
            sort_workspace = buffers.sort_workspace
            active_expert_offsets = buffers.active_expert_offsets[: total_slots + 1]
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
            "sorted_weights": buffers.sorted_weights,
            "broadcast_input_indices": buffers.broadcast_input_indices,
            "compact_w13_ptrs_w": buffers.compact_w13_ptrs_w,
            "compact_w13_ptrs_s": buffers.compact_w13_ptrs_s,
            "legacy_w13_ptrs_w": buffers.legacy_w13_ptrs_w,
            "legacy_w13_ptrs_s": buffers.legacy_w13_ptrs_s,
            "legacy_w2_ptrs_w": buffers.legacy_w2_ptrs_w,
            "legacy_w2_ptrs_s": buffers.legacy_w2_ptrs_s,
            "empty_weight": buffers.empty_weight,
            "empty_scale": buffers.empty_scale,
        }

    def _apply_moe(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        codec = self.weight_codec
        policy = codec.policy(layer)
        del shared_experts, shared_experts_input
        if layer.apply_router_weight_on_input:
            raise NotImplementedError(
                codec.message("MoE does not support apply_router_weight_on_input yet.")
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
            and policy.batched_gemm
            and codec.enabled("legacy_single_token_compact")
        ):
            return self._apply_legacy_single_token_compact(
                layer, x, topk_weights, topk_ids_i32, buffers, top_k, output
            )
        if num_tokens == 1 and (not policy.batched_gemm):
            use_compact_w13 = codec.enabled("single_token_compact_w13")
            use_indexed_w13 = not use_compact_w13 and codec.enabled(
                "single_token_indexed_w13"
            )
            use_indexed_w2 = codec.enabled("single_token_indexed_w2")
            codec.log(
                "MoE single-token active-expert dense path enabled "
                "(top_k=%d, experts=%d).",
                top_k,
                layer.sm70_num_experts,
            )
            if use_indexed_w13 or use_indexed_w2:
                codec.log(
                    "MoE single-token indexed dense-stage path enabled "
                    "(top_k=%d, w13=%s, w2=%s).",
                    top_k,
                    use_indexed_w13,
                    use_indexed_w2,
                )
            if use_compact_w13:
                codec.log(
                    "MoE single-token compact grouped W13 path enabled (top_k=%d).",
                    top_k,
                )
                codec.gemm_w13(
                    "compact",
                    layer,
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
                codec.gemm_w13(
                    "indexed",
                    layer,
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
                codec.gemm_w13(
                    "active_dense",
                    layer,
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
                codec.gemm_w2(
                    "indexed",
                    layer,
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
                codec.gemm_w2(
                    "active_dense",
                    layer,
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
            if codec.enabled("single_token_weighted_reduce"):
                codec.log(
                    "MoE single-token weighted-reduce path enabled (top_k=%d).", top_k
                )
                codec.native_ops.awq_moe_single_token_weighted_reduce_out(
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
        if policy.permute_with_scratch:
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
            batched_enabled=policy.batched_gemm,
            num_tokens=num_tokens,
            total_slots=total_slots,
            w13_per_expert_dispatch=policy.batched_w13_per_expert_dispatch,
            w2_per_expert_dispatch=policy.batched_w2_per_expert_dispatch,
        )
        if route_plan.w13 == Sm70MoeStageRoute.PER_EXPERT_DISPATCH:
            codec.log(
                "MoE batched W13 using per-expert dispatch selection (experts=%d).",
                layer.sm70_num_experts,
            )
            codec.gemm_w13(
                "per_expert",
                layer,
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
            codec.gemm_w13(
                "batched",
                layer,
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
            codec.log(
                "MoE CUDA-graph-safe dense-stage path enabled (experts=%d).",
                layer.sm70_num_experts,
            )
            codec.gemm_w13(
                "dense",
                layer,
                buffers["gate_up"],
                buffers["permuted_input"],
                buffers["expert_offsets"],
                codec.buffer(layer, "dense_expert_ids"),
                layer.w13_strided_ptrs_w,
                layer.w13_strided_ptrs_s,
                layer.sm70_num_experts,
                layer.sm70_w13_k_dim,
                layer.sm70_w13_n_dim,
                self.group_size,
            )
        torch.ops._C.silu_and_mul(buffers["intermediate"], buffers["gate_up"])
        if route_plan.w2 == Sm70MoeStageRoute.PER_EXPERT_DISPATCH:
            codec.log(
                "MoE batched W2 using per-expert dispatch selection (experts=%d).",
                layer.sm70_num_experts,
            )
            codec.gemm_w2(
                "per_expert",
                layer,
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
            codec.gemm_w2(
                "batched",
                layer,
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
            codec.gemm_w2(
                "dense",
                layer,
                buffers["sorted_output"],
                buffers["intermediate"],
                buffers["expert_offsets"],
                codec.buffer(layer, "dense_expert_ids"),
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
            and policy.batched_gemm
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
